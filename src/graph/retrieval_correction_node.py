from __future__ import annotations

import asyncio
import math
from typing import Any, cast

from src.agents.base import AgentContext
from src.agents.retrievalCorrectionAgent import (
    QueryRepairAgent,
    QueryRepairResult,
    RetrievalQualityJudge,
    RetrievalQualityReport,
    search_intent_from_dict,
    search_intent_to_dict,
)
from src.agents.searchAgent import SearchIntent, load_search_agent_llm
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.paper_retrieval.models import PaperDocument


DEFAULT_MAX_CORRECTION_ROUNDS = 2
DEFAULT_RETRIEVAL_QUALITY_THRESHOLD = 0.55
DEFAULT_RETRIEVAL_QUALITY_WEIGHTS = {
    "alpha": 0.35,
    "beta": 0.30,
    "gamma": 0.30,
    "delta": 0.15,
}
DEFAULT_SEMANTIC_TARGET_SCORE = 8.0


def run_retrieval_correction_node():
    """生成 Search 后的检索自纠节点，负责判断是否回到 Search。"""

    async def _node(state: State) -> State:
        request = state.get("request")
        if request is None:
            raise ValueError("retrieval correction node missing request")
        constraints = dict(getattr(request, "constraints", {}) or {})
        if not _correction_enabled(constraints):
            return _route_to_read(state, status="disabled", reason="retrieval self-correction disabled")

        timeout_seconds = _positive_float(constraints.get("retrieval_correction_timeout_seconds"), 45.0)
        try:
            return await asyncio.wait_for(_run_correction(state), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return _route_to_read(state, status="timeout", reason="retrieval self-correction timed out")
        except Exception as exc:
            # 自纠模块不能影响原主链路：异常时带着诊断回到旧的 Search -> Read 流程。
            return _route_to_read(state, status="fallback_error", reason=str(exc))

    return _node


def route_after_retrieval_correction(state: State) -> str:
    """LangGraph 条件边：需要修复时回 Search，否则进入 Read。"""

    return "run_search_agent" if state.get("retrieval_correction_route") == "retry_search" else "run_read"


async def _run_correction(state: State) -> State:
    request = state["request"]
    constraints = dict(getattr(request, "constraints", {}) or {})
    papers = list(state.get("search_results") or [])
    search_scores = [dict(item) for item in state.get("search_scores") or [] if isinstance(item, dict)]
    search_summary = dict(state.get("search_summary") or {})
    correction_state = dict(state.get("retrieval_correction") or {})
    repair_attempt_count = _non_negative_int(correction_state.get("repair_attempt_count"), 0)
    max_reloops = _max_correction_round(constraints)
    history = [dict(item) for item in correction_state.get("history") or [] if isinstance(item, dict)]
    reporter = _resolve_reporter(state)

    if reporter is not None:
        reporter.started("正在检查检索结果质量", stage="retrieval_quality_start")

    sample_size = _positive_int(constraints.get("retrieval_correction_sample_size"), 30)
    sample = papers[:sample_size]
    current_intent = search_intent_from_dict(
        state.get("search_intent"),
        topic=str(getattr(request, "topic", "") or ""),
        constraints=constraints,
    )
    retrieval_stats = _build_retrieval_stats(
        papers=papers,
        sample=sample,
        search_summary=search_summary,
        repair_attempt_count=repair_attempt_count,
        max_reloops=max_reloops,
    )

    if not papers:
        report = RetrievalQualityReport(
            passed=False,
            confidence=1.0,
            coverage_stats={"all_required": {"matched": 0, "total": 0}},
            failure_type="no_results",
            diagnostic="当前检索没有返回可供阅读的论文，需要尝试改写 query。",
            recommendations=["放宽或改写检索表达式，确保核心对象和任务同时出现。"],
        )
        judge_diagnostics: JsonObject = {"status": "skipped_no_results"}
    else:
        llm = _resolve_llm(state)
        if llm is None:
            return _route_to_read(state, status="judge_no_llm", reason="retrieval judge llm unavailable")
        judge_diagnostics = {}

        def report_judge_usage(usage: JsonObject) -> None:
            if reporter is not None:
                reporter.progress("检索质量判断模型调用完成", stage="retrieval_quality_judge", **usage)

        report, judge_diagnostics = await RetrievalQualityJudge(
            AgentContext(llm=llm, usage_callback=report_judge_usage)
        ).async_judge(
            topic=str(getattr(request, "topic", "") or ""),
            constraints=constraints,
            intent=current_intent,
            papers=sample,
            retrieval_stats=retrieval_stats,
        )
        if report is None:
            return _route_to_read(
                state,
                status="judge_unavailable",
                reason=str(judge_diagnostics.get("status") or "retrieval judge failed"),
                extra={"judge_diagnostics": judge_diagnostics},
            )
        _apply_quality_floor(report, judged_count=len(sample), constraints=constraints)
    _apply_retrieval_quality_gate(
        report,
        sample=sample,
        search_scores=search_scores,
        search_summary=search_summary,
        constraints=constraints,
    )

    history.append(
        {
            "iteration": repair_attempt_count,
            "search_intent": search_intent_to_dict(current_intent) if current_intent is not None else {},
            "report": report.to_dict(),
            "judge_diagnostics": judge_diagnostics,
        }
    )

    if report.passed:
        if reporter is not None:
            reporter.completed("检索质量通过，进入论文阅读", stage="retrieval_quality_pass")
        return _route_to_read(
            state,
            status="passed",
            reason=report.diagnostic or "retrieval quality sufficient",
            report=report,
            history=history,
            repair_attempt_count=repair_attempt_count,
        )

    if repair_attempt_count >= max_reloops:
        if reporter is not None:
            reporter.completed("检索质量仍不足，已达到自纠上限，回退到原阅读流程", stage="retrieval_quality_fallback")
        return _route_to_read(
            state,
            status="max_loops_exhausted",
            reason=report.diagnostic or "retrieval quality insufficient after repair budget exhausted",
            report=report,
            history=history,
            repair_attempt_count=repair_attempt_count,
        )

    llm = _resolve_llm(state)
    if llm is None:
        return _route_to_read(
            state,
            status="repair_no_llm",
            reason="query repair llm unavailable",
            report=report,
            history=history,
            repair_attempt_count=repair_attempt_count,
        )

    if reporter is not None:
        reporter.progress("检索质量不足，正在改写下一轮 query", stage="query_repair_start")

    def report_repair_usage(usage: JsonObject) -> None:
        if reporter is not None:
            reporter.progress("query 修复模型调用完成", stage="query_repair", **usage)

    repair, repair_diagnostics = await QueryRepairAgent(
        AgentContext(llm=llm, usage_callback=report_repair_usage)
    ).async_repair(
        topic=str(getattr(request, "topic", "") or ""),
        constraints=constraints,
        previous_intent=current_intent,
        quality_report=report,
        retrieval_stats=retrieval_stats,
    )
    if repair is None:
        return _route_to_read(
            state,
            status="repair_unavailable",
            reason=str(repair_diagnostics.get("status") or "query repair failed"),
            report=report,
            history=history,
            repair_attempt_count=repair_attempt_count,
            extra={"repair_diagnostics": repair_diagnostics},
        )

    next_attempt_count = repair_attempt_count + 1
    next_intent = repair.to_search_intent(current_intent, topic=str(getattr(request, "topic", "") or ""))
    repair_payload = repair.to_dict()
    history[-1]["repair"] = repair_payload
    history[-1]["repair_diagnostics"] = repair_diagnostics

    if reporter is not None:
        reporter.completed("query 已修复，准备重新检索", stage="query_repair_done")

    return _route_to_search(
        state,
        report=report,
        repair=repair,
        search_intent=next_intent,
        history=history,
        repair_attempt_count=next_attempt_count,
        max_reloops=max_reloops,
    )


def _route_to_search(
    state: State,
    *,
    report: RetrievalQualityReport,
    repair: QueryRepairResult,
    search_intent: SearchIntent,
    history: list[JsonObject],
    repair_attempt_count: int,
    max_reloops: int,
) -> State:
    updated = dict(state)
    correction = {
        "status": "repairing",
        "route": "retry_search",
        "repair_attempt_count": repair_attempt_count,
        "max_reloops": max_reloops,
        "max_correction_round": max_reloops,
        "remaining_repair_budget": max(max_reloops - repair_attempt_count, 0),
        "latest_report": report.to_dict(),
        "latest_repair": repair.to_dict(),
        "history": history,
    }
    diagnostics = dict(updated.get("diagnostics") or {})
    diagnostics["retrieval_correction"] = correction
    search_summary = dict(updated.get("search_summary") or {})
    search_summary["retrieval_correction"] = {
        "status": "repairing",
        "repair_attempt_count": repair_attempt_count,
        "latest_report": report.to_dict(),
        "latest_repair": repair.to_dict(),
    }
    updated.update(
        retrieval_correction=correction,
        retrieval_correction_route="retry_search",
        search_intent_override=search_intent_to_dict(search_intent),
        diagnostics=diagnostics,
        search_summary=search_summary,
        current_step="retrieval_correction",
    )
    return cast(State, updated)


def _route_to_read(
    state: State,
    *,
    status: str,
    reason: str,
    report: RetrievalQualityReport | None = None,
    history: list[JsonObject] | None = None,
    repair_attempt_count: int | None = None,
    extra: JsonObject | None = None,
) -> State:
    updated = dict(state)
    previous = dict(updated.get("retrieval_correction") or {})
    if repair_attempt_count is None:
        repair_attempt_count = _non_negative_int(previous.get("repair_attempt_count"), 0)
    request_constraints = getattr(updated.get("request"), "constraints", {}) if updated.get("request") is not None else {}
    max_reloops = _max_correction_round(dict(request_constraints or {}))
    correction = {
        "status": status,
        "route": "read",
        "reason": reason,
        "repair_attempt_count": repair_attempt_count,
        "max_reloops": max_reloops,
        "max_correction_round": max_reloops,
        "remaining_repair_budget": max(max_reloops - repair_attempt_count, 0),
        "latest_report": report.to_dict() if report is not None else previous.get("latest_report", {}),
        "history": history if history is not None else list(previous.get("history") or []),
    }
    if extra:
        correction.update(extra)
    diagnostics = dict(updated.get("diagnostics") or {})
    diagnostics["retrieval_correction"] = correction
    search_summary = dict(updated.get("search_summary") or {})
    search_summary["retrieval_correction"] = {
        "status": status,
        "reason": reason,
        "repair_attempt_count": repair_attempt_count,
        "latest_report": correction["latest_report"],
    }
    updated.update(
        retrieval_correction=correction,
        retrieval_correction_route="read",
        search_intent_override={},
        diagnostics=diagnostics,
        search_summary=search_summary,
        current_step="retrieval_correction",
    )
    return cast(State, updated)


def _apply_quality_floor(report: RetrievalQualityReport, *, judged_count: int, constraints: JsonObject) -> None:
    """把 LLM 判断和一个很轻的覆盖率下限合并，避免明显低覆盖被误判为 PASS。"""

    if judged_count <= 0:
        report.passed = False
        report.failure_type = report.failure_type or "no_results"
        return
    all_required = _coverage_entry(report.coverage_stats, "all_required")
    if all_required is None:
        return
    matched = _non_negative_int(all_required.get("matched"), 0)
    total = _non_negative_int(all_required.get("total"), judged_count) or judged_count
    ratio = matched / max(total, 1)
    min_ratio = _positive_float(constraints.get("retrieval_correction_min_all_required_ratio"), 0.2)
    min_count = _positive_int(
        constraints.get("retrieval_correction_min_all_required_count"),
        min(3, max(1, math.ceil(judged_count * min_ratio))),
    )
    if matched < min_count or ratio < min_ratio:
        report.passed = False
        report.failure_type = report.failure_type or "insufficient_all_required_coverage"
        if "all_required" not in report.failed_facets:
            report.failed_facets.append("all_required")


def _apply_retrieval_quality_gate(
    report: RetrievalQualityReport,
    *,
    sample: list[PaperDocument],
    search_scores: list[JsonObject],
    search_summary: JsonObject,
    constraints: JsonObject,
) -> None:
    """按 Q_r 分数决定是否触发 QueryRepair，避免只靠一次模型判断反复自循环。"""

    judge_passed = report.passed
    weights = _retrieval_quality_weights(constraints)
    components = _retrieval_quality_components(
        report,
        sample=sample,
        search_scores=search_scores,
        search_summary=search_summary,
        constraints=constraints,
    )
    score = _unit_float(
        weights["alpha"] * components["Rsemantic"]
        + weights["beta"] * components["Cfacet"]
        + weights["gamma"] * components["Cevidence"]
        - weights["delta"] * components["Dduplicate"],
        0.0,
    )
    threshold = _unit_float(
        constraints.get("retrieval_quality_threshold", constraints.get("retrieval_quality_tau")),
        DEFAULT_RETRIEVAL_QUALITY_THRESHOLD,
    )

    components["judge_passed_before_quality_gate"] = judge_passed
    components["quality_score_passed"] = score >= threshold
    report.quality_score = round(score, 4)
    report.quality_threshold = round(threshold, 4)
    report.quality_components = {key: _round_quality_value(value) for key, value in components.items()}
    report.quality_weights = {key: _round_quality_value(value) for key, value in weights.items()}

    # 中文注释：最终是否回到 Search 由 Q_r 统一决定；LLM Judge 的 passed 作为分项诊断保留。
    report.passed = score >= threshold
    if report.passed:
        if not judge_passed:
            report.diagnostic = _append_diagnostic(
                report.diagnostic,
                f"Retrieval Quality Q_r={score:.3f} 已达到阈值 {threshold:.3f}，不触发 QueryRepair。",
            )
        return

    report.failure_type = report.failure_type or "low_retrieval_quality"
    if "retrieval_quality" not in report.failed_facets:
        report.failed_facets.append("retrieval_quality")
    report.diagnostic = _append_diagnostic(
        report.diagnostic,
        f"Retrieval Quality Q_r={score:.3f} 低于阈值 {threshold:.3f}，触发 QueryRepair。",
    )


def _retrieval_quality_components(
    report: RetrievalQualityReport,
    *,
    sample: list[PaperDocument],
    search_scores: list[JsonObject],
    search_summary: JsonObject,
    constraints: JsonObject,
) -> JsonObject:
    """计算公式里的四个分项，全部归一化到 0 到 1。"""

    return {
        "Rsemantic": _candidate_mean_relevance(sample, search_scores, constraints),
        "Cfacet": _semantic_facet_coverage(report),
        "Cevidence": _high_confidence_relevant_ratio(report),
        "Dduplicate": _duplicate_ratio(sample, search_summary),
    }


def _candidate_mean_relevance(
    sample: list[PaperDocument],
    search_scores: list[JsonObject],
    constraints: JsonObject,
) -> float:
    if not sample:
        return 0.0
    target_score = _positive_float(
        constraints.get("retrieval_quality_semantic_target_score"),
        DEFAULT_SEMANTIC_TARGET_SCORE,
    )
    score_by_key: dict[str, float] = {}
    for item in search_scores:
        score = _non_negative_float(item.get("score"), 0.0)
        paper_payload = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        for key in _paper_lookup_keys_from_payload(paper_payload):
            score_by_key.setdefault(key, score)
    normalized_scores = []
    for paper in sample:
        score = next((score_by_key[key] for key in _paper_lookup_keys(paper) if key in score_by_key), 0.0)
        normalized_scores.append(min(score / target_score, 1.0))
    return sum(normalized_scores) / len(normalized_scores)


def _semantic_facet_coverage(report: RetrievalQualityReport) -> float:
    ratios: list[float] = []
    for facet in report.required_facets:
        if not isinstance(facet, dict):
            continue
        name = str(facet.get("name") or "").strip()
        entry = _coverage_entry(report.coverage_stats, name) if name else None
        if entry is not None:
            ratios.append(_coverage_ratio(entry))
    if not ratios:
        for key, value in report.coverage_stats.items():
            if key == "all_required" or not isinstance(value, dict):
                continue
            ratios.append(_coverage_ratio(value))
    if not ratios:
        all_required = _coverage_entry(report.coverage_stats, "all_required")
        if all_required is not None:
            ratios.append(_coverage_ratio(all_required))
    return sum(ratios) / len(ratios) if ratios else 0.0


def _high_confidence_relevant_ratio(report: RetrievalQualityReport) -> float:
    all_required = _coverage_entry(report.coverage_stats, "all_required")
    if all_required is None:
        return _unit_float(report.confidence, 0.0) if report.passed else 0.0
    return _coverage_ratio(all_required) * _unit_float(report.confidence, 0.0)


def _duplicate_ratio(sample: list[PaperDocument], search_summary: JsonObject) -> float:
    raw_count = _non_negative_int(search_summary.get("raw_candidate_count"), 0)
    deduplicated_count = _non_negative_int(
        search_summary.get("raw_paper_count", search_summary.get("deduplicated_paper_count")),
        len(sample),
    )
    source_duplicate_ratio = 0.0
    if raw_count > 0:
        source_duplicate_ratio = max(raw_count - deduplicated_count, 0) / raw_count
    return max(source_duplicate_ratio, _sample_duplicate_ratio(sample))


def _sample_duplicate_ratio(sample: list[PaperDocument]) -> float:
    if not sample:
        return 0.0
    seen: set[str] = set()
    duplicate_count = 0
    for paper in sample:
        keys = _paper_lookup_keys(paper)
        if any(key in seen for key in keys):
            duplicate_count += 1
            continue
        seen.update(keys)
    return duplicate_count / len(sample)


def _coverage_ratio(entry: JsonObject) -> float:
    matched = _non_negative_int(entry.get("matched"), 0)
    total = _non_negative_int(entry.get("total"), 0)
    if total <= 0:
        return 0.0
    return _unit_float(matched / total, 0.0)


def _paper_lookup_keys(paper: PaperDocument) -> list[str]:
    return _deduplicate_strings(
        [
            f"doi:{str(paper.doi or '').strip().lower()}",
            f"paper_id:{str(paper.paperId or '').strip().lower()}",
            f"id:{str(paper.id or '').strip().lower()}",
            f"title:{' '.join((paper.title or '').strip().lower().split())}",
        ]
    )


def _paper_lookup_keys_from_payload(payload: JsonObject) -> list[str]:
    return _deduplicate_strings(
        [
            f"doi:{str(payload.get('doi') or '').strip().lower()}",
            f"paper_id:{str(payload.get('paperId') or payload.get('paper_id') or '').strip().lower()}",
            f"id:{str(payload.get('id') or '').strip().lower()}",
            f"title:{' '.join(str(payload.get('title') or '').strip().lower().split())}",
        ]
    )


def _retrieval_quality_weights(constraints: JsonObject) -> dict[str, float]:
    raw_weights = constraints.get("retrieval_quality_weights")
    configured = raw_weights if isinstance(raw_weights, dict) else {}
    return {
        "alpha": _non_negative_float(
            constraints.get("retrieval_quality_alpha", configured.get("alpha")),
            DEFAULT_RETRIEVAL_QUALITY_WEIGHTS["alpha"],
        ),
        "beta": _non_negative_float(
            constraints.get("retrieval_quality_beta", configured.get("beta")),
            DEFAULT_RETRIEVAL_QUALITY_WEIGHTS["beta"],
        ),
        "gamma": _non_negative_float(
            constraints.get("retrieval_quality_gamma", configured.get("gamma")),
            DEFAULT_RETRIEVAL_QUALITY_WEIGHTS["gamma"],
        ),
        "delta": _non_negative_float(
            constraints.get("retrieval_quality_delta", configured.get("delta")),
            DEFAULT_RETRIEVAL_QUALITY_WEIGHTS["delta"],
        ),
    }


def _max_correction_round(constraints: JsonObject) -> int:
    return _non_negative_int(
        constraints.get("max_correction_round", constraints.get("retrieval_correction_max_loops")),
        DEFAULT_MAX_CORRECTION_ROUNDS,
    )


def _append_diagnostic(diagnostic: str, note: str) -> str:
    diagnostic = diagnostic.strip()
    return f"{diagnostic} {note}" if diagnostic else note


def _build_retrieval_stats(
    *,
    papers: list[PaperDocument],
    sample: list[PaperDocument],
    search_summary: JsonObject,
    repair_attempt_count: int,
    max_reloops: int,
) -> JsonObject:
    return {
        "candidate_count": len(papers),
        "judged_count": len(sample),
        "raw_candidate_count": search_summary.get("raw_candidate_count"),
        "raw_paper_count": search_summary.get("raw_paper_count"),
        "selected_paper_count": search_summary.get("selected_paper_count"),
        "repair_attempt_count": repair_attempt_count,
        "max_reloops": max_reloops,
    }


def _coverage_entry(stats: JsonObject, key: str) -> JsonObject | None:
    value = stats.get(key)
    return dict(value) if isinstance(value, dict) else None


def _resolve_llm(state: State) -> ProviderSnapshot | None:
    candidate = state.get("retrieval_correction_node_llm") or state.get("search_node_llm")
    if isinstance(candidate, ProviderSnapshot):
        return candidate
    return load_search_agent_llm()


def _resolve_reporter(state: State):
    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("retrieval_correction", "检索自纠")


def _correction_enabled(constraints: JsonObject) -> bool:
    value = constraints.get("enable_retrieval_self_correction", True)
    return not (value is False or str(value).strip().lower() in {"0", "false", "no", "off"})


def _positive_int(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved >= 0 else default


def _positive_float(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _non_negative_float(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved >= 0 else default


def _unit_float(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = default
    return max(0.0, min(1.0, resolved))


def _round_quality_value(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _deduplicate_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
