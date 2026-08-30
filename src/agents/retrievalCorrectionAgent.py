from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.base import AgentContext, AgentSpec, BaseAgent
from src.agents.contracts import JsonObject
from src.agents.searchAgent import SearchIntent, SearchSubtopic
from src.llm import ProviderSnapshot
from src.llm.base import LLMResponse
from src.paper_retrieval.models import PaperDocument


RETRIEVAL_JUDGE_SYSTEM_PROMPT = """
你是论文检索质量评估器。你只判断当前检索结果是否足够支撑后续阅读，不写综述，也不补充外部知识。
你的核心任务是从用户原始语义意图中自动抽取必须同时满足的 required_facets，例如方法、研究对象、任务、应用场景或约束；然后检查候选论文标题和摘要是否覆盖这些 facets。

请严格执行：
1. 不要硬编码任何负面词。false_positive_pattern 必须来自本轮候选论文标题或摘要中反复出现、且偏离用户意图的证据。
2. 每篇论文只能根据给定标题和摘要判断，不能用常识补全。
3. 如果大部分论文只命中宽泛词，但缺少关键对象或任务，应判定 FAIL，并说明 failure_type。
4. PASS 只在足够多论文同时覆盖所有 required_facets 时给出；不要因为检索到了很多论文就 PASS。

输出必须是纯 JSON 对象，不要 Markdown，不要解释文字。结构如下：
{
  "passed": false,
  "confidence": 0.0,
  "required_facets": [{"name": "method", "description": "..."}],
  "coverage_stats": {
    "method": {"matched": 0, "total": 0},
    "all_required": {"matched": 0, "total": 0}
  },
  "failed_facets": ["..."],
  "missing_facets": ["..."],
  "failure_type": "query_drift",
  "false_positive_pattern": ["..."],
  "diagnostic": "...",
  "recommendations": ["..."]
}
""".strip()


QUERY_REPAIR_SYSTEM_PROMPT = """
你是论文检索 query 修复助手。你会收到原始语义意图、上一轮 query、检索质量诊断、缺失 facets 和 false-positive 证据。
你的任务是生成更精确的英文检索表达式，让下一轮检索同时覆盖 required_facets，并减少上一轮误召回模式。

规则：
1. 不要简单复述上一轮 query。必须针对 failed_facets 补充更明确的对象、任务或方法词。
2. excluded_terms 只能来自诊断里的 false_positive_pattern 或明显同义表达，不能凭空扩展。
3. keyword 使用后端可处理的轻量布尔表达式，优先形如 "(a or b) and (c or d)"。
4. 生成 1-3 个互补 subtopics，避免为了凑数制造重复 query。

输出必须是纯 JSON 对象，不要 Markdown，不要解释文字。结构如下：
{
  "subtopics": [
    {"subtopic": "中文说明", "keyword": "(multimodal LLM or multimodal language model) and (molecule generation or molecular design)"}
  ],
  "excluded_terms": ["..."],
  "rationale": "一句话说明如何修复上一轮偏移"
}
""".strip()


@dataclass(slots=True)
class RetrievalQualityReport:
    """保存一次检索质量判断，供图节点决定是否需要重新检索。"""

    passed: bool
    confidence: float = 0.0
    required_facets: list[JsonObject] = field(default_factory=list)
    coverage_stats: JsonObject = field(default_factory=dict)
    failed_facets: list[str] = field(default_factory=list)
    missing_facets: list[str] = field(default_factory=list)
    failure_type: str = ""
    false_positive_pattern: list[str] = field(default_factory=list)
    diagnostic: str = ""
    recommendations: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_threshold: float = 0.0
    quality_components: JsonObject = field(default_factory=dict)
    quality_weights: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """转成普通字典，方便放进 LangGraph state 和 diagnostics。"""

        return asdict(self)


@dataclass(slots=True)
class QueryRepairResult:
    """保存修复后的检索计划，并能转换成 Search 节点可直接执行的 intent。"""

    subtopics: list[SearchSubtopic] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_search_intent(self, previous: SearchIntent | None, *, topic: str) -> SearchIntent:
        """复用上一轮来源、年份和数量约束，只替换 query 相关字段。"""

        keywords = self.keywords or [item.keyword for item in self.subtopics]
        return SearchIntent(
            topic=previous.topic if previous is not None and previous.topic else topic,
            keywords=_deduplicate_terms(keywords),
            subtopics=list(self.subtopics),
            excluded_terms=_deduplicate_terms(
                [*(previous.excluded_terms if previous is not None else []), *self.excluded_terms]
            ),
            year_from=previous.year_from if previous is not None else None,
            year_to=previous.year_to if previous is not None else None,
            max_results=previous.max_results if previous is not None else 60,
            sources=list(previous.sources if previous is not None else []),
        )

    def to_dict(self) -> JsonObject:
        """转成可序列化结构，便于调试和前端展示。"""

        return {
            "subtopics": [asdict(item) for item in self.subtopics],
            "keywords": list(self.keywords),
            "excluded_terms": list(self.excluded_terms),
            "rationale": self.rationale,
        }


class RetrievalQualityJudge(BaseAgent):
    """根据标题和摘要批量判断检索结果是否偏离用户真实意图。"""

    spec = AgentSpec(
        name="retrieval_quality_judge",
        role="screen",
        description="评估当前检索结果是否足够支撑后续阅读。",
        llm_profile="luna_agent",
        input_keys=(),
    )

    def __init__(self, context: AgentContext):
        context.spec = self.spec
        super().__init__(context)

    def _run(self, state: JsonObject) -> JsonObject:
        """同步入口暂不使用，图节点直接调用 async_judge。"""

        raise NotImplementedError("RetrievalQualityJudge only supports async_judge in the graph flow.")

    async def async_judge(
        self,
        *,
        topic: str,
        constraints: JsonObject,
        intent: SearchIntent | None,
        papers: list[PaperDocument],
        retrieval_stats: JsonObject,
        usage_callback: Any | None = None,
    ) -> tuple[RetrievalQualityReport | None, JsonObject]:
        """调用 LLM-as-judge，返回结构化质量报告；解析失败时由图节点回退原流程。"""

        if self.context.llm is None:
            return None, {"status": "no_llm", "message": "retrieval judge llm is not configured"}
        response = await self._call_provider_chat_async(
            self._build_messages(topic, constraints, intent, papers, retrieval_stats)
        )
        self.report_usage(response, usage_callback)
        raw_model_output = response.content or ""
        diagnostics = {
            "status": "ok" if response.ok else "llm_error",
            "raw_model_output": raw_model_output,
            "error_kind": response.error_kind,
            "error_status_code": response.error_status_code,
        }
        if not response.ok:
            return None, diagnostics
        payload = _extract_json_object(raw_model_output)
        if payload is None:
            diagnostics["status"] = "parse_failed"
            return None, diagnostics
        report = self._report_from_payload(payload)
        if report is None:
            diagnostics["status"] = "invalid_schema"
        return report, diagnostics

    async def _call_provider_chat_async(self, messages: list[JsonObject]) -> LLMResponse:
        provider = self.context.llm.provider
        chat = getattr(provider, "chat", None)
        if callable(chat):
            response = chat(messages, temperature=0)
            if inspect.isawaitable(response):
                return await response
            return response
        return await asyncio.to_thread(provider.chat_with_retry, messages, temperature=0)

    def _build_messages(
        self,
        topic: str,
        constraints: JsonObject,
        intent: SearchIntent | None,
        papers: list[PaperDocument],
        retrieval_stats: JsonObject,
    ) -> list[JsonObject]:
        payload = {
            "original_semantic_intent": topic,
            "constraints": dict(constraints or {}),
            "previous_search_intent": _intent_to_dict(intent),
            "retrieval_stats": dict(retrieval_stats),
            "candidate_papers": [_paper_evidence(paper) for paper in papers],
        }
        return [
            {"role": "system", "content": RETRIEVAL_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _report_from_payload(self, payload: JsonObject) -> RetrievalQualityReport | None:
        if not isinstance(payload.get("passed"), bool):
            return None
        return RetrievalQualityReport(
            passed=bool(payload.get("passed")),
            confidence=_clamp_float(payload.get("confidence")),
            required_facets=_json_object_list(payload.get("required_facets")),
            coverage_stats=dict(payload.get("coverage_stats") or {}) if isinstance(payload.get("coverage_stats"), dict) else {},
            failed_facets=_string_list(payload.get("failed_facets")),
            missing_facets=_string_list(payload.get("missing_facets")),
            failure_type=str(payload.get("failure_type") or "").strip(),
            false_positive_pattern=_string_list(payload.get("false_positive_pattern")),
            diagnostic=str(payload.get("diagnostic") or "").strip(),
            recommendations=_string_list(payload.get("recommendations")),
        )


class QueryRepairAgent(BaseAgent):
    """根据质量诊断生成下一轮更精确的检索表达式。"""

    spec = AgentSpec(
        name="query_repair_agent",
        role="search",
        description="根据检索质量诊断修复论文检索 query。",
        llm_profile="luna_agent",
        input_keys=(),
    )

    def __init__(self, context: AgentContext):
        context.spec = self.spec
        super().__init__(context)

    def _run(self, state: JsonObject) -> JsonObject:
        """同步入口暂不使用，图节点直接调用 async_repair。"""

        raise NotImplementedError("QueryRepairAgent only supports async_repair in the graph flow.")

    async def async_repair(
        self,
        *,
        topic: str,
        constraints: JsonObject,
        previous_intent: SearchIntent | None,
        quality_report: RetrievalQualityReport,
        retrieval_stats: JsonObject,
        usage_callback: Any | None = None,
    ) -> tuple[QueryRepairResult | None, JsonObject]:
        """调用修复 Agent 生成新的 SearchIntent 片段。"""

        if self.context.llm is None:
            return None, {"status": "no_llm", "message": "query repair llm is not configured"}
        response = await self._call_provider_chat_async(
            self._build_messages(topic, constraints, previous_intent, quality_report, retrieval_stats)
        )
        self.report_usage(response, usage_callback)
        raw_model_output = response.content or ""
        diagnostics = {
            "status": "ok" if response.ok else "llm_error",
            "raw_model_output": raw_model_output,
            "error_kind": response.error_kind,
            "error_status_code": response.error_status_code,
        }
        if not response.ok:
            return None, diagnostics
        payload = _extract_json_object(raw_model_output)
        if payload is None:
            diagnostics["status"] = "parse_failed"
            return None, diagnostics
        repair = self._repair_from_payload(payload)
        if repair is None:
            diagnostics["status"] = "invalid_schema"
        return repair, diagnostics

    async def _call_provider_chat_async(self, messages: list[JsonObject]) -> LLMResponse:
        provider = self.context.llm.provider
        chat = getattr(provider, "chat", None)
        if callable(chat):
            response = chat(messages, temperature=0)
            if inspect.isawaitable(response):
                return await response
            return response
        return await asyncio.to_thread(provider.chat_with_retry, messages, temperature=0)

    def _build_messages(
        self,
        topic: str,
        constraints: JsonObject,
        previous_intent: SearchIntent | None,
        quality_report: RetrievalQualityReport,
        retrieval_stats: JsonObject,
    ) -> list[JsonObject]:
        payload = {
            "original_semantic_intent": topic,
            "constraints": dict(constraints or {}),
            "previous_search_intent": _intent_to_dict(previous_intent),
            "required_facets": list(quality_report.required_facets),
            "failed_facets": list(quality_report.failed_facets),
            "missing_facets": list(quality_report.missing_facets),
            "false_positive_pattern": list(quality_report.false_positive_pattern),
            "retrieval_quality": {
                "score": quality_report.quality_score,
                "threshold": quality_report.quality_threshold,
                "components": dict(quality_report.quality_components),
                "weights": dict(quality_report.quality_weights),
            },
            "retrieval_statistics": dict(retrieval_stats),
            "diagnostic": quality_report.diagnostic,
        }
        return [
            {"role": "system", "content": QUERY_REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _repair_from_payload(self, payload: JsonObject) -> QueryRepairResult | None:
        subtopics = _subtopics_from_payload(payload.get("subtopics"))
        if not subtopics:
            subtopics = [
                SearchSubtopic(subtopic=f"修复检索 {index}", keyword=query)
                for index, query in enumerate(_string_list(payload.get("queries")), start=1)
            ]
        keywords = _string_list(payload.get("keywords")) or [item.keyword for item in subtopics]
        if not subtopics:
            return None
        return QueryRepairResult(
            subtopics=subtopics,
            keywords=_deduplicate_terms(keywords),
            excluded_terms=_string_list(payload.get("excluded_terms")),
            rationale=str(payload.get("rationale") or "").strip(),
        )


def _intent_to_dict(intent: SearchIntent | None) -> JsonObject:
    if intent is None:
        return {}
    return {
        "topic": intent.topic,
        "keywords": list(intent.keywords),
        "subtopics": [asdict(item) for item in intent.subtopics],
        "excluded_terms": list(intent.excluded_terms),
        "year_from": intent.year_from,
        "year_to": intent.year_to,
        "max_results": intent.max_results,
        "sources": list(intent.sources),
    }


def search_intent_to_dict(intent: SearchIntent) -> JsonObject:
    """给图节点复用的公开转换函数。"""

    return _intent_to_dict(intent)


def search_intent_from_dict(payload: Any, *, topic: str, constraints: JsonObject | None = None) -> SearchIntent | None:
    """从 state 中恢复 SearchIntent；字段缺失时使用当前请求约束兜底。"""

    if not isinstance(payload, dict):
        return None
    constraints = dict(constraints or {})
    subtopics = _subtopics_from_payload(payload.get("subtopics"))
    keywords = _string_list(payload.get("keywords")) or [item.keyword for item in subtopics]
    if not subtopics and keywords:
        subtopics = [SearchSubtopic(subtopic="修复检索", keyword=keywords[0])]
    if not subtopics:
        return None
    return SearchIntent(
        topic=str(payload.get("topic") or topic).strip(),
        keywords=_deduplicate_terms(keywords),
        subtopics=subtopics,
        excluded_terms=_string_list(payload.get("excluded_terms")) or _string_list(constraints.get("excluded_terms")),
        year_from=_optional_int(payload.get("year_from", constraints.get("year_from"))),
        year_to=_optional_int(payload.get("year_to", constraints.get("year_to"))),
        max_results=_positive_int(payload.get("max_results", constraints.get("max_results")), 60),
        sources=_string_list(payload.get("sources")) or _string_list(constraints.get("sources")),
    )


def _subtopics_from_payload(values: Any) -> list[SearchSubtopic]:
    if not isinstance(values, list):
        return []
    subtopics: list[SearchSubtopic] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        subtopic = str(item.get("subtopic") or "").strip()
        keyword = str(item.get("keyword") or "").strip()
        if subtopic and keyword:
            subtopics.append(SearchSubtopic(subtopic=subtopic, keyword=keyword))
    return subtopics


def _paper_evidence(paper: PaperDocument) -> JsonObject:
    abstract = " ".join((paper.abstract or "").split())
    return {
        "paperId": paper.paperId or paper.id,
        "title": paper.title,
        "year": paper.year,
        "source": paper.source,
        "abstract": abstract[:900],
    }


def _extract_json_object(text: str) -> JsonObject | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _json_object_list(value: Any) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return _deduplicate_terms(result)


def _deduplicate_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        text = str(term).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default
