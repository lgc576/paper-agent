from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph import run_graph
from src.agents import ReviewRequest
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowRuntimeContext
from src.graph.runtime_resources import RuntimeConcurrencyLimits, WorkflowRuntimeResources
from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider
from src.llm.base import LLMResponse, normalize_token_usage


JsonObject = dict[str, Any]


RUBRIC_JUDGE_SYSTEM_PROMPT = """
You are an exacting ScholarQA evaluator. Judge only the supplied answer against
the supplied question and rubric. Return one JSON object, with no Markdown.

For every expert ingredient, score from 0.0 to 1.0:
- 1.0 means the answer fully and correctly satisfies the criterion.
- 0.5 means it partially satisfies the criterion.
- 0.0 means it is absent, incorrect, or misleading.

For general criteria, also score from 0.0 to 1.0:
- length: the answer has an appropriate amount of detail for the target range.
- expertise: the answer demonstrates correct scholarly nuance.
- citations: citations are used in a scholarly, useful way.
- excerpts: concrete evidence, examples, or paper-level details are used.

Do not reward unsupported verbosity. Penalize hallucinated facts.

Required JSON schema:
{
  "expert_ingredients": [
    {"name": "criterion_name", "score": 0.0, "reason": "brief reason"}
  ],
  "general_criteria": {
    "length": {"score": 0.0, "reason": "brief reason"},
    "expertise": {"score": 0.0, "reason": "brief reason"},
    "citations": {"score": 0.0, "reason": "brief reason"},
    "excerpts": {"score": 0.0, "reason": "brief reason"}
  }
}
""".strip()


CITATION_JUDGE_SYSTEM_PROMPT = """
You are a citation-audit judge for scientific QA answers. Use only the answer,
the reference list, and the supplied evidence notes. Return one JSON object, no
Markdown.

Identify atomic scientific claims in the answer body. Ignore the bibliography
itself. For each claim:
- needs_citation: true when the claim is scientific, empirical, comparative, or
  literature-specific.
- citation_ids: numeric bracket citations attached to that claim, such as "1".
- supported_citation_ids: cited ids that actually support the claim.
- unsupported_citation_ids: cited ids that do not support the claim.
- unnecessary_citation_ids: cited ids attached to claims that do not need them.

A citation is supported only if the supplied reference/evidence clearly backs
the claim. If evidence is insufficient, mark it unsupported.

Required JSON schema:
{
  "claims": [
    {
      "claim": "atomic claim",
      "needs_citation": true,
      "citation_ids": ["1"],
      "supported_citation_ids": ["1"],
      "unsupported_citation_ids": [],
      "unnecessary_citation_ids": [],
      "verdict": "supported",
      "reason": "brief reason"
    }
  ]
}
""".strip()


QUALITY_JUDGE_SYSTEM_PROMPT = """
You are a ScholarQA holistic quality judge. Score the answer on 1 to 5 integer
scales, using the question and rubric evidence. Return one JSON object, no
Markdown.

Definitions:
- coverage: how completely the answer covers the important aspects of the
  question and rubric.
- relevance: how tightly the answer stays on the asked question.
- organization: how clear, structured, and easy to follow the answer is.

Required JSON schema:
{
  "coverage": {"score": 1, "reason": "brief reason"},
  "relevance": {"score": 1, "reason": "brief reason"},
  "organization": {"score": 1, "reason": "brief reason"}
}
""".strip()


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, usage: Any) -> None:
        normalized = normalize_token_usage(usage)
        self.input_tokens += int(normalized.get("input_tokens") or 0)
        self.output_tokens += int(normalized.get("output_tokens") or 0)

    def to_dict(self) -> JsonObject:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    idx: int | None
    question: str
    metric_config: JsonObject
    ingredients: JsonObject
    qa_metadata: JsonObject
    config_payload: JsonObject


class TokenCountingProvider:
    def __init__(self, provider: Any, counter: TokenUsage):
        self._provider = provider
        self._counter = counter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = await self._maybe_await(self._provider.chat(*args, **kwargs))
        self._counter.add(getattr(response, "usage", None))
        return response

    async def chat_stream(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = await self._maybe_await(self._provider.chat_stream(*args, **kwargs))
        self._counter.add(getattr(response, "usage", None))
        return response

    async def embed(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._maybe_await(self._provider.embed(*args, **kwargs))
        self._counter.add(getattr(response, "usage", None))
        return response

    def chat_with_retry(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = self._provider.chat_with_retry(*args, **kwargs)
        self._counter.add(getattr(response, "usage", None))
        return response

    def chat_stream_with_retry(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = self._provider.chat_stream_with_retry(*args, **kwargs)
        self._counter.add(getattr(response, "usage", None))
        return response

    def embed_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        response = self._provider.embed_with_retry(*args, **kwargs)
        self._counter.add(getattr(response, "usage", None))
        return response

    async def aclose(self) -> None:
        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await self._maybe_await(close())

    async def _maybe_await(self, value: Any) -> Any:
        if hasattr(value, "__await__"):
            return await value
        return value


class RuntimeEventCollector:
    def __init__(self) -> None:
        self.events: list[JsonObject] = []

    def emit(self, event: JsonObject) -> JsonObject:
        self.events.append(copy.deepcopy(event))
        return event


class DeepSeekJudge:
    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        timeout_s: float,
        max_retries: int,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.usage = TokenUsage()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
            timeout=timeout_s,
            max_retries=max_retries,
        )

    async def close(self) -> None:
        await self.client.close()

    async def judge_json(
        self,
        *,
        system_prompt: str,
        payload: JsonObject,
        max_tokens: int,
    ) -> JsonObject:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        self.usage.add(response_dict.get("usage"))
        content = ((response_dict.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        parsed = extract_json_object(content)
        if parsed is None:
            raise ValueError(f"judge returned non-JSON output: {content[:500]}")
        return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Paper-Agent on ScholarQA-CS.")
    parser.add_argument("--data-dir", default="data/scholarqa_cs")
    parser.add_argument("--output-dir", default="data/evaluation/scholarqa_cs")
    parser.add_argument("--run-dir", default=None, help="Resume or write results in an existing run directory.")
    parser.add_argument("--model-config", default="config/model.json")
    parser.add_argument("--system-config", default="config/system.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--variants", nargs="+", default=["without_loop", "with_loop"])
    parser.add_argument("--sources", nargs="+", default=["arxiv"])
    parser.add_argument("--max-results", type=int, default=15)
    parser.add_argument("--deep-read-limit", type=int, default=15)
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--retrieval-correction-max-loops", type=int, default=1)
    parser.add_argument("--retrieval-correction-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--agent-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--agent-provider-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--agent-provider-max-retries", type=int, default=1)
    parser.add_argument("--paper-task-concurrency", type=int, default=3)
    parser.add_argument("--read-model-concurrency", type=int, default=1)
    parser.add_argument("--judge-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--judge-api-base-env", default="DEEPSEEK_API_BASE")
    parser.add_argument("--judge-model-env", default="DEEPSEEK_JUDGE_MODEL")
    parser.add_argument("--judge-provider", default="deepseek")
    parser.add_argument("--judge-config-agent", default="solar_agent")
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-answer-char-limit", type=int, default=32000)
    parser.add_argument("--judge-evidence-char-limit", type=int, default=22000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--resume-state", default=None, help="Resume read/write/eval from a saved state or checkpoint JSON.")
    parser.add_argument("--eval-only-state", default=None, help="State JSON to use when rerunning judge without the agent.")
    parser.add_argument("--eval-only-answer", default=None, help="Answer Markdown file to judge without rerunning the agent.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-agent-reasoning-effort",
        action="store_true",
        help="Keep reasoning_effort from config/model.json for agent calls.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = load_benchmark_cases(Path(args.data_dir))
    cases = select_cases(cases, limit=args.limit, case_ids=set(args.case_id or []))
    validate_single_file_mode(args, cases)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_dir.name
    answers_dir = run_dir / "answers"
    states_dir = run_dir / "states"
    run_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    if args.save_state:
        states_dir.mkdir(parents=True, exist_ok=True)

    judge_connection = resolve_judge_connection(args)
    run_config = {
        "run_id": run_id,
        "case_count": len(cases),
        "variants": list(args.variants),
        "sources": list(args.sources),
        "max_results": args.max_results,
        "deep_read_limit": args.deep_read_limit,
        "year_from": args.year_from,
        "year_to": args.year_to,
        "language": args.language,
        "retrieval_correction_max_loops": args.retrieval_correction_max_loops,
        "agent_provider_timeout_seconds": args.agent_provider_timeout_seconds,
        "agent_provider_max_retries": args.agent_provider_max_retries,
        "paper_task_concurrency": args.paper_task_concurrency,
        "read_model_concurrency": args.read_model_concurrency,
        "judge_provider": args.judge_provider,
        "judge_api_base": judge_connection["api_base"],
        "judge_model": judge_connection["model"],
        "resume_state": args.resume_state,
        "eval_only_state": args.eval_only_state,
        "eval_only_answer": args.eval_only_answer,
        "dry_run": args.dry_run,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "selected_cases": len(cases), "config": run_config}, ensure_ascii=False, indent=2))
        return

    judge = DeepSeekJudge(
        api_key=judge_connection["api_key"],
        api_base=judge_connection["api_base"],
        model=judge_connection["model"],
        timeout_s=args.judge_timeout_seconds,
        max_retries=args.judge_max_retries,
    )

    result_path = run_dir / "case_results.jsonl"
    records: list[JsonObject] = []
    completed_keys = load_completed_keys(result_path) if args.skip_existing else set()

    try:
        with result_path.open("a", encoding="utf-8") as out:
            for case_index, case in enumerate(cases, start=1):
                for variant in args.variants:
                    key = f"{case.case_id}:{variant}"
                    if key in completed_keys:
                        continue
                    print(f"[{case_index}/{len(cases)}] running {case.case_id} {variant}", flush=True)
                    record = await evaluate_case_variant(
                        case,
                        variant=variant,
                        args=args,
                        judge=judge,
                        run_dir=run_dir,
                        answers_dir=answers_dir,
                        states_dir=states_dir,
                    )
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    records.append(record)
    finally:
        await judge.close()

    if args.skip_existing and result_path.exists():
        records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    summary = summarize_records(records)
    summary["run_id"] = run_id
    summary["run_dir"] = str(run_dir)
    summary["judge_usage"] = judge.usage.to_dict()
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


async def evaluate_case_variant(
    case: BenchmarkCase,
    *,
    variant: str,
    args: argparse.Namespace,
    judge: DeepSeekJudge,
    run_dir: Path,
    answers_dir: Path,
    states_dir: Path,
) -> JsonObject:
    started_at = time.perf_counter()
    if args.eval_only_state or args.eval_only_answer:
        return await evaluate_case_variant_from_files(
            case,
            variant=variant,
            args=args,
            judge=judge,
            started_at=started_at,
        )

    agent_usage = TokenUsage()
    event_collector = RuntimeEventCollector()
    turn_id = safe_id(f"{case.case_id}_{variant}")
    runtime_resources = WorkflowRuntimeResources(
        RuntimeConcurrencyLimits(
            paper_task_concurrency=max(1, int(args.paper_task_concurrency)),
            read_model_concurrency=max(1, int(args.read_model_concurrency)),
        )
    )
    runtime = WorkflowRuntimeContext(
        session_key="scholarqa_eval",
        turn_id=turn_id,
        run_id=turn_id,
        sync_port=InlineWorkflowSyncPort(
            event_collector.emit,
            session_key="scholarqa_eval",
            turn_id=turn_id,
            run_id=turn_id,
            workflow_name="scholarqa_eval",
        ),
        resources=runtime_resources,
    )
    state_overrides: JsonObject = {}
    snapshots: list[ProviderSnapshot] = []
    try:
        state_overrides, snapshots = build_agent_state_overrides(args, agent_usage)
        if args.resume_state:
            state_overrides["read_resume_checkpoint"] = load_resume_checkpoint(resolve_input_path(args.resume_state))
        constraints = build_constraints(args, variant)
        request = ReviewRequest(topic=case.question, constraints=constraints, language=args.language)
        result = await asyncio.wait_for(
            run_graph(request, runtime=runtime, state_overrides=state_overrides),
            timeout=args.agent_timeout_seconds,
        )
        agent_status = "ok"
        agent_error = ""
        state = result.state
        answer_text = extract_answer_text(state)
    except Exception as exc:
        agent_status = "error"
        agent_error = format_exception(exc)
        state = state_from_runtime_events(event_collector.events, agent_error=agent_error)
        answer_text = ""
    finally:
        await close_snapshots(snapshots)
        await runtime_resources.aclose()

    answer_path = answers_dir / f"{safe_id(case.case_id)}__{variant}.md"
    answer_path.write_text(answer_text, encoding="utf-8")
    if args.save_state and state:
        state_path = states_dir / f"{safe_id(case.case_id)}__{variant}.json"
        state_path.write_text(json.dumps(to_jsonable(state), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        state_path = None

    eval_started_usage = copy.deepcopy(judge.usage)
    if not answer_text.strip():
        evaluation = empty_evaluation("empty_answer")
    else:
        try:
            evaluation = await evaluate_answer(case, answer_text, state, judge, args)
        except Exception as exc:
            evaluation = empty_evaluation("judge_error")
            evaluation["error"] = format_exception(exc)
    eval_usage = TokenUsage(
        input_tokens=judge.usage.input_tokens - eval_started_usage.input_tokens,
        output_tokens=judge.usage.output_tokens - eval_started_usage.output_tokens,
    )

    read_relevance = mean_read_relevance(state)
    self_metrics = self_retrieval_metrics(state)
    duration_s = round(time.perf_counter() - started_at, 3)
    total_usage = TokenUsage(
        input_tokens=agent_usage.input_tokens + eval_usage.input_tokens,
        output_tokens=agent_usage.output_tokens + eval_usage.output_tokens,
    )

    return {
        "case_id": case.case_id,
        "idx": case.idx,
        "question": case.question,
        "variant": variant,
        "agent_status": agent_status,
        "agent_error": agent_error,
        "duration_seconds": duration_s,
        "answer_path": str(answer_path),
        "state_path": str(state_path) if state_path is not None else None,
        "answer_char_count": len(answer_text),
        "agent_usage": agent_usage.to_dict(),
        "evaluation_usage": eval_usage.to_dict(),
        "total_usage": total_usage.to_dict(),
        "read_relevance": read_relevance,
        "self_retrieval": self_metrics,
        "retrieval_correction": extract_retrieval_correction(state),
        "evaluation": evaluation,
    }


async def evaluate_case_variant_from_files(
    case: BenchmarkCase,
    *,
    variant: str,
    args: argparse.Namespace,
    judge: DeepSeekJudge,
    started_at: float,
) -> JsonObject:
    state_path = resolve_input_path(str(args.eval_only_state))
    answer_path = resolve_input_path(str(args.eval_only_answer))
    state = load_json_object(state_path)
    answer_text = answer_path.read_text(encoding="utf-8")

    eval_started_usage = copy.deepcopy(judge.usage)
    if not answer_text.strip():
        evaluation = empty_evaluation("empty_answer")
    else:
        try:
            evaluation = await evaluate_answer(case, answer_text, state, judge, args)
        except Exception as exc:
            evaluation = empty_evaluation("judge_error")
            evaluation["error"] = format_exception(exc)
    eval_usage = TokenUsage(
        input_tokens=judge.usage.input_tokens - eval_started_usage.input_tokens,
        output_tokens=judge.usage.output_tokens - eval_started_usage.output_tokens,
    )
    agent_usage = TokenUsage()

    return {
        "case_id": case.case_id,
        "idx": case.idx,
        "question": case.question,
        "variant": variant,
        "agent_status": "skipped_eval_only",
        "agent_error": "",
        "duration_seconds": round(time.perf_counter() - started_at, 3),
        "answer_path": str(answer_path),
        "state_path": str(state_path),
        "answer_char_count": len(answer_text),
        "agent_usage": agent_usage.to_dict(),
        "evaluation_usage": eval_usage.to_dict(),
        "total_usage": eval_usage.to_dict(),
        "read_relevance": mean_read_relevance(state),
        "self_retrieval": self_retrieval_metrics(state),
        "retrieval_correction": extract_retrieval_correction(state),
        "evaluation": evaluation,
    }


def validate_single_file_mode(args: argparse.Namespace, cases: list[BenchmarkCase]) -> None:
    eval_only = bool(args.eval_only_state or args.eval_only_answer)
    if eval_only and not (args.eval_only_state and args.eval_only_answer):
        raise ValueError("--eval-only-state and --eval-only-answer must be used together.")
    if args.resume_state and eval_only:
        raise ValueError("--resume-state cannot be combined with --eval-only-state/--eval-only-answer.")
    if (args.resume_state or eval_only) and (len(cases) != 1 or len(args.variants) != 1):
        raise ValueError("--resume-state and --eval-only-* require exactly one selected case and one variant.")


def load_resume_checkpoint(path: Path) -> JsonObject:
    payload = load_json_object(path)
    checkpoint = payload.get("read_resume_checkpoint")
    if isinstance(checkpoint, dict):
        return copy.deepcopy(checkpoint)
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("read_checkpoint"), dict):
        return copy.deepcopy(diagnostics["read_checkpoint"])
    if looks_like_read_checkpoint(payload):
        return payload
    raise ValueError(f"state file does not contain read_resume_checkpoint: {path}")


def looks_like_read_checkpoint(payload: JsonObject) -> bool:
    return any(key in payload for key in ("request", "search_results", "read_results", "pending_paper", "pending_read_result"))


def load_json_object(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_input_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def state_from_runtime_events(events: list[JsonObject], *, agent_error: str) -> JsonObject:
    checkpoint = latest_checkpoint_from_runtime_events(events)
    diagnostics: JsonObject = {
        "agent_error": {"message": agent_error},
        "runtime_event_tail": compact_runtime_events(events[-50:]),
    }
    if not checkpoint:
        return {
            "current_step": "agent_error",
            "diagnostics": diagnostics,
        }

    diagnostics["read_checkpoint"] = checkpoint
    return {
        "request": dict(checkpoint.get("request") or {}),
        "search_results": list(checkpoint.get("search_results") or []),
        "read_results": list(checkpoint.get("read_results") or []),
        "read_artifact_refs": list(checkpoint.get("read_artifact_refs") or []),
        "read_resume_checkpoint": checkpoint,
        "read_paper_statuses": list(checkpoint.get("paper_runtime_statuses") or []),
        "diagnostics": diagnostics,
        "current_step": str(checkpoint.get("current_step") or "agent_error"),
    }


def latest_checkpoint_from_runtime_events(events: list[JsonObject]) -> JsonObject:
    for event in reversed(events):
        metadata = event.get("metadata") if isinstance(event, dict) else {}
        if not isinstance(metadata, dict):
            continue
        checkpoint = metadata.get("checkpoint")
        if isinstance(checkpoint, dict):
            return copy.deepcopy(checkpoint)
    return {}


def compact_runtime_events(events: list[JsonObject]) -> list[JsonObject]:
    compacted: list[JsonObject] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        metadata = dict(event.get("metadata") or {}) if isinstance(event.get("metadata"), dict) else {}
        if "checkpoint" in metadata:
            metadata["checkpoint"] = "<saved in read_resume_checkpoint>"
        compacted.append(
            {
                "event": event.get("event"),
                "id": event.get("id"),
                "parent_id": event.get("parent_id"),
                "type": event.get("type"),
                "title": event.get("title"),
                "status": event.get("status"),
                "show_content": event.get("show_content"),
                "metadata": metadata,
                "created_at": event.get("created_at"),
                "updated_at": event.get("updated_at"),
            }
        )
    return compacted


async def evaluate_answer(
    case: BenchmarkCase,
    answer_text: str,
    state: JsonObject,
    judge: DeepSeekJudge,
    args: argparse.Namespace,
) -> JsonObject:
    errors: JsonObject = {}

    rubric_payload = {
        "question": case.question,
        "answer": truncate(answer_text, args.judge_answer_char_limit),
        "rubric": build_rubric_payload(case),
    }
    try:
        rubric_raw = await judge.judge_json(
            system_prompt=RUBRIC_JUDGE_SYSTEM_PROMPT,
            payload=rubric_payload,
            max_tokens=args.judge_max_tokens,
        )
        rubric = score_rubric(case, rubric_raw)
    except Exception as exc:
        rubric = empty_rubric_evaluation()
        errors["rubric"] = format_exception(exc)

    citation_payload = {
        "question": case.question,
        "answer": truncate(strip_references_section(answer_text), args.judge_answer_char_limit),
        "references": build_reference_payload(state),
        "evidence_notes": truncate_evidence(build_evidence_payload(state), args.judge_evidence_char_limit),
    }
    try:
        citation_raw = await judge.judge_json(
            system_prompt=CITATION_JUDGE_SYSTEM_PROMPT,
            payload=citation_payload,
            max_tokens=args.judge_max_tokens,
        )
        citation = score_citations(citation_raw)
    except Exception as exc:
        citation = empty_citation_evaluation()
        errors["citation"] = format_exception(exc)

    quality_payload = {
        "question": case.question,
        "answer": truncate(answer_text, args.judge_answer_char_limit),
        "rubric": build_rubric_payload(case),
    }
    try:
        quality_raw = await judge.judge_json(
            system_prompt=QUALITY_JUDGE_SYSTEM_PROMPT,
            payload=quality_payload,
            max_tokens=args.judge_max_tokens,
        )
        quality = score_quality(quality_raw)
    except Exception as exc:
        quality = empty_quality_evaluation("judge_error")
        errors["quality"] = format_exception(exc)

    status = "ok" if not errors else "judge_error" if len(errors) == 3 else "partial_judge_error"
    result = {
        "status": status,
        "rubric_correctness": rubric,
        "citation": citation,
        "quality": quality,
    }
    if errors:
        result["errors"] = errors
    return result


def build_agent_state_overrides(args: argparse.Namespace, counter: TokenUsage) -> tuple[JsonObject, list[ProviderSnapshot]]:
    model_path = Path(args.model_config)
    if not model_path.exists():
        raise FileNotFoundError(f"model config not found: {model_path}")
    raw_config = json.loads(model_path.read_text(encoding="utf-8"))
    if not args.keep_agent_reasoning_effort:
        raw_config = copy.deepcopy(raw_config)
        for agent in dict(raw_config.get("agents") or {}).values():
            if isinstance(agent, dict):
                agent["reasoning_effort"] = None
                agent["reasoningEffort"] = None
    raw_config = copy.deepcopy(raw_config)
    for provider in dict(raw_config.get("providers") or {}).values():
        if not isinstance(provider, dict):
            continue
        provider["timeout_s"] = max(1.0, float(args.agent_provider_timeout_seconds))
        provider["max_retries"] = max(1, int(args.agent_provider_max_retries))
    system = SystemConfig.load(args.system_config)
    model_config = ModelConfig.from_dict(raw_config, system)

    profile_by_node = {
        "search_node_llm": "luna_agent",
        "retrieval_correction_node_llm": "luna_agent",
        "read_node_llm": "default_agent",
        "analysis_node_llm": "solar_agent",
        "writing_outline_node_llm": "default_agent",
        "writing_node_llm": "default_agent",
    }
    snapshots: list[ProviderSnapshot] = []
    overrides: JsonObject = {}
    for node_key, profile in profile_by_node.items():
        snapshot = make_provider(model_config, profile, timeout_s=max(1.0, float(args.agent_provider_timeout_seconds)))
        wrapped = ProviderSnapshot(
            provider=TokenCountingProvider(snapshot.provider, counter),
            model=snapshot.model,
            context_window_tokens=snapshot.context_window_tokens,
            signature=snapshot.signature,
        )
        snapshots.append(wrapped)
        overrides[node_key] = wrapped
    return overrides, snapshots


def build_constraints(args: argparse.Namespace, variant: str) -> JsonObject:
    if variant not in {"with_loop", "without_loop"}:
        raise ValueError(f"unknown variant: {variant}")
    constraints: JsonObject = {
        "sources": list(args.sources),
        "max_results": int(args.max_results),
        "deep_read_limit": int(args.deep_read_limit),
        "enable_retrieval_self_correction": variant == "with_loop",
        "retrieval_correction_max_loops": int(args.retrieval_correction_max_loops),
        "retrieval_correction_timeout_seconds": float(args.retrieval_correction_timeout_seconds),
        "retrieval_correction_sample_size": int(args.max_results),
        "answer_language": args.language,
        "evaluation_task": "ScholarQA-CS",
    }
    if args.year_from is not None:
        constraints["year_from"] = args.year_from
    if args.year_to is not None:
        constraints["year_to"] = args.year_to
    return constraints


def resolve_judge_connection(args: argparse.Namespace) -> JsonObject:
    api_key = os.getenv(args.judge_api_key_env)
    api_base = args.judge_api_base or os.getenv(args.judge_api_base_env)
    model = args.judge_model or os.getenv(args.judge_model_env)

    config_path = Path(args.model_config)
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            providers = dict(raw.get("providers") or {})
            agents = dict(raw.get("agents") or {})
            agent = agents.get(args.judge_config_agent)
            provider_name = args.judge_provider
            if isinstance(agent, dict):
                provider_name = str(agent.get("provider") or provider_name)
                model = model or str(agent.get("model_name") or agent.get("model") or "")
            provider = providers.get(provider_name)
            if isinstance(provider, dict):
                api_key = api_key or provider.get("api_key") or os.getenv(str(provider.get("api_key_env") or ""))
                api_base = api_base or provider.get("api_base") or provider.get("apiBase")
        except Exception:
            pass

    resolved = {
        "api_key": api_key or "",
        "api_base": api_base or "https://api.deepseek.com/v1",
        "model": model or "deepseek-v4-pro",
    }
    if not resolved["api_key"]:
        raise ValueError(
            f"DeepSeek judge API key is missing. Set {args.judge_api_key_env} "
            "or provide it in config/model.json."
        )
    return resolved


def load_benchmark_cases(data_dir: Path) -> list[BenchmarkCase]:
    configs = json.loads((data_dir / "test_configs_snippets.json").read_text(encoding="utf-8"))
    output_by_question = read_jsonl_by_key(data_dir / "output_snippets.jsonl", "question")
    metadata_by_question = read_jsonl_by_key(data_dir / "qa_metadata_all.jsonl", "question")
    seen_questions: set[str] = set()
    cases: list[BenchmarkCase] = []
    for item in configs:
        question = str(item.get("initial_prompt") or "").strip()
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        qa_metadata = metadata_by_question.get(question, {})
        output_payload = output_by_question.get(question, {})
        case_id = str(item.get("case_id") or qa_metadata.get("folder") or safe_id(question)[:16])
        idx = optional_int(qa_metadata.get("idx"))
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                idx=idx,
                question=question,
                metric_config=dict(item.get("metric_config") or {}),
                ingredients=dict(output_payload.get("ingredients") or {}),
                qa_metadata=dict(qa_metadata),
                config_payload=dict(item),
            )
        )
    return cases


def read_jsonl_by_key(path: Path, key: str) -> dict[str, JsonObject]:
    result: dict[str, JsonObject] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            lookup = str(payload.get(key) or "").strip()
            if lookup and lookup not in result:
                result[lookup] = payload
    return result


def select_cases(cases: list[BenchmarkCase], *, limit: int | None, case_ids: set[str]) -> list[BenchmarkCase]:
    selected = [case for case in cases if not case_ids or case.case_id in case_ids]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def build_rubric_payload(case: BenchmarkCase) -> JsonObject:
    metric = dict(case.metric_config.get("config") or {})
    expert_items = []
    for item in list(metric.get("other_properties") or []):
        if not isinstance(item, dict):
            continue
        expert_items.append(
            {
                "name": str(item.get("name") or f"ingredient_{len(expert_items)}"),
                "criterion": str(item.get("criterion") or ""),
                "weight": float_or_default(item.get("weight"), 0.0),
                "evidence": list(item.get("evidence") or []),
            }
        )
    if not expert_items:
        for group_name in ("most_important", "nice_to_have"):
            for index, item in enumerate(list(case.ingredients.get(group_name) or [])):
                if isinstance(item, dict):
                    expert_items.append(
                        {
                            "name": f"{group_name}_{index}",
                            "criterion": str(item.get("text") or ""),
                            "weight": 1.0,
                            "evidence": list(item.get("snippets") or []),
                        }
                    )
    return {
        "expert_items": expert_items,
        "general_criteria": {
            "low_length": metric.get("low_length"),
            "high_length": metric.get("high_length"),
            "length_weight": metric.get("length_weight", 0.05),
            "expertise_weight": metric.get("expertise_weight", 0.05),
            "citations_weight": metric.get("citations_weight", 0.2),
            "excerpts_weight": metric.get("excerpts_weight", 0.1),
        },
    }


def score_rubric(case: BenchmarkCase, raw: JsonObject) -> JsonObject:
    rubric = build_rubric_payload(case)
    expert_items = list(rubric.get("expert_items") or [])
    expected_names = {str(item["name"]): item for item in expert_items}
    model_scores = {}
    for item in list(raw.get("expert_ingredients") or []):
        if isinstance(item, dict):
            model_scores[str(item.get("name") or "")] = item

    scored_items = []
    weighted_sum = 0.0
    weight_sum = 0.0
    for item in expert_items:
        name = str(item["name"])
        weight = float_or_default(item.get("weight"), 0.0)
        if weight <= 0:
            weight = 1.0
        score = clamp01((model_scores.get(name) or {}).get("score"))
        weighted_sum += score * weight
        weight_sum += weight
        scored_items.append(
            {
                "name": name,
                "criterion": item.get("criterion"),
                "weight": weight,
                "score": score,
                "reason": str((model_scores.get(name) or {}).get("reason") or ""),
            }
        )
    expert_score = weighted_sum / weight_sum if weight_sum else 0.0

    general_raw = dict(raw.get("general_criteria") or {})
    general_weights = dict((rubric.get("general_criteria") or {}))
    general_items = []
    general_weight_sum = 0.0
    general_weighted_sum = 0.0
    for name in ("length", "expertise", "citations", "excerpts"):
        payload = dict(general_raw.get(name) or {})
        score = clamp01(payload.get("score"))
        weight = float_or_default(general_weights.get(f"{name}_weight"), 0.0)
        if weight <= 0:
            weight = 1.0
        general_weight_sum += weight
        general_weighted_sum += score * weight
        general_items.append(
            {
                "name": name,
                "weight": weight,
                "score": score,
                "reason": str(payload.get("reason") or ""),
            }
        )
    general_score = general_weighted_sum / general_weight_sum if general_weight_sum else 0.0
    weighted_correctness = 0.6 * expert_score + 0.4 * general_score
    return {
        "weighted_correctness": round(weighted_correctness, 4),
        "expert_ingredients_score": round(expert_score, 4),
        "general_criteria_score": round(general_score, 4),
        "expert_ingredients": scored_items,
        "general_criteria": general_items,
        "unmatched_expert_names": sorted(set(model_scores) - set(expected_names)),
    }


def score_citations(raw: JsonObject) -> JsonObject:
    claims = [dict(item) for item in list(raw.get("claims") or []) if isinstance(item, dict)]
    claims_needing = 0
    claims_with_supported = 0
    citation_total = 0
    supported_necessary = 0
    unsupported_total = 0
    unnecessary_total = 0
    normalized_claims = []
    for claim in claims:
        needs = bool(claim.get("needs_citation"))
        citation_ids = string_list(claim.get("citation_ids"))
        supported_ids = string_list(claim.get("supported_citation_ids"))
        unsupported_ids = string_list(claim.get("unsupported_citation_ids"))
        unnecessary_ids = string_list(claim.get("unnecessary_citation_ids"))
        citation_total += len(citation_ids)
        unsupported_total += len(unsupported_ids)
        unnecessary_total += len(unnecessary_ids)
        if needs:
            claims_needing += 1
            if supported_ids:
                claims_with_supported += 1
            supported_necessary += len([cid for cid in supported_ids if cid in citation_ids])
        normalized_claims.append(
            {
                "claim": str(claim.get("claim") or ""),
                "needs_citation": needs,
                "citation_ids": citation_ids,
                "supported_citation_ids": supported_ids,
                "unsupported_citation_ids": unsupported_ids,
                "unnecessary_citation_ids": unnecessary_ids,
                "verdict": str(claim.get("verdict") or ""),
                "reason": str(claim.get("reason") or ""),
            }
        )
    precision = supported_necessary / citation_total if citation_total else 0.0
    recall = claims_with_supported / claims_needing if claims_needing else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4),
        "claims_needing_citation": claims_needing,
        "claims_with_supported_citation": claims_with_supported,
        "citation_instances": citation_total,
        "supported_necessary_citations": supported_necessary,
        "unsupported_citations": unsupported_total,
        "unnecessary_citations": unnecessary_total,
        "claims": normalized_claims,
    }


def score_quality(raw: JsonObject) -> JsonObject:
    result = {}
    for name in ("coverage", "relevance", "organization"):
        payload = dict(raw.get(name) or {})
        result[name] = {
            "score": clamp_int(payload.get("score"), 1, 5),
            "reason": str(payload.get("reason") or ""),
        }
    return result


def empty_rubric_evaluation() -> JsonObject:
    return {
        "weighted_correctness": 0.0,
        "expert_ingredients_score": 0.0,
        "general_criteria_score": 0.0,
        "expert_ingredients": [],
        "general_criteria": [],
    }


def empty_citation_evaluation() -> JsonObject:
    return {
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "citation_f1": 0.0,
        "claims_needing_citation": 0,
        "claims_with_supported_citation": 0,
        "citation_instances": 0,
        "supported_necessary_citations": 0,
        "unsupported_citations": 0,
        "unnecessary_citations": 0,
        "claims": [],
    }


def empty_quality_evaluation(status: str) -> JsonObject:
    return {
        "coverage": {"score": 1, "reason": status},
        "relevance": {"score": 1, "reason": status},
        "organization": {"score": 1, "reason": status},
    }


def empty_evaluation(status: str) -> JsonObject:
    return {
        "status": status,
        "rubric_correctness": empty_rubric_evaluation(),
        "citation": empty_citation_evaluation(),
        "quality": empty_quality_evaluation(status),
    }


def summarize_records(records: list[JsonObject]) -> JsonObject:
    by_variant: dict[str, list[JsonObject]] = {}
    for record in records:
        by_variant.setdefault(str(record.get("variant")), []).append(record)

    summary: JsonObject = {"variants": {}, "paired_deltas": {}, "self_retrieval": {}}
    for variant, items in by_variant.items():
        eval_ok_items = [item for item in items if dict(item.get("evaluation") or {}).get("status") == "ok"]
        evaluation_status_counts: dict[str, int] = {}
        for item in items:
            status = str(dict(item.get("evaluation") or {}).get("status") or "missing")
            evaluation_status_counts[status] = evaluation_status_counts.get(status, 0) + 1
        summary["variants"][variant] = {
            "case_count": len(items),
            "ok_count": sum(1 for item in items if item.get("agent_status") == "ok"),
            "agent_ok_count": sum(1 for item in items if item.get("agent_status") == "ok"),
            "eval_ok_count": len(eval_ok_items),
            "judge_error_count": evaluation_status_counts.get("judge_error", 0),
            "partial_judge_error_count": evaluation_status_counts.get("partial_judge_error", 0),
            "empty_answer_count": evaluation_status_counts.get("empty_answer", 0),
            "evaluation_status_counts": evaluation_status_counts,
            "rubric_correctness": mean_metric(eval_ok_items, ["evaluation", "rubric_correctness", "weighted_correctness"]),
            "expert_ingredients_score": mean_metric(eval_ok_items, ["evaluation", "rubric_correctness", "expert_ingredients_score"]),
            "general_criteria_score": mean_metric(eval_ok_items, ["evaluation", "rubric_correctness", "general_criteria_score"]),
            "citation_precision": mean_metric(eval_ok_items, ["evaluation", "citation", "citation_precision"]),
            "citation_recall": mean_metric(eval_ok_items, ["evaluation", "citation", "citation_recall"]),
            "citation_f1": mean_metric(eval_ok_items, ["evaluation", "citation", "citation_f1"]),
            "coverage": mean_metric(eval_ok_items, ["evaluation", "quality", "coverage", "score"]),
            "relevance": mean_metric(eval_ok_items, ["evaluation", "quality", "relevance", "score"]),
            "organization": mean_metric(eval_ok_items, ["evaluation", "quality", "organization", "score"]),
            "read_relevance_mean": mean_metric(items, ["read_relevance", "mean_score"]),
            "agent_tokens": sum_metric(items, ["agent_usage", "total_tokens"]),
            "evaluation_tokens": sum_metric(items, ["evaluation_usage", "total_tokens"]),
            "total_tokens": sum_metric(items, ["total_usage", "total_tokens"]),
        }

    paired = build_paired_records(records)
    if paired:
        eval_ok_paired = [
            pair
            for pair in paired
            if dict(pair["with_loop"].get("evaluation") or {}).get("status") == "ok"
            and dict(pair["without_loop"].get("evaluation") or {}).get("status") == "ok"
        ]
        summary["paired_deltas"]["paired_case_count"] = len(paired)
        summary["paired_deltas"]["paired_eval_ok_count"] = len(eval_ok_paired)
        eval_delta_fields = {
            "rubric_correctness": ["evaluation", "rubric_correctness", "weighted_correctness"],
            "citation_f1": ["evaluation", "citation", "citation_f1"],
            "coverage": ["evaluation", "quality", "coverage", "score"],
            "relevance": ["evaluation", "quality", "relevance", "score"],
            "organization": ["evaluation", "quality", "organization", "score"],
        }
        for name, path in eval_delta_fields.items():
            deltas = [numeric_at(pair["with_loop"], path) - numeric_at(pair["without_loop"], path) for pair in eval_ok_paired]
            summary["paired_deltas"][f"with_loop_minus_without_loop_{name}"] = round(mean(deltas), 4) if deltas else None

        runtime_delta_fields = {
            "read_relevance_mean": ["read_relevance", "mean_score"],
            "total_tokens": ["total_usage", "total_tokens"],
        }
        for name, path in runtime_delta_fields.items():
            deltas = [numeric_at(pair["with_loop"], path) - numeric_at(pair["without_loop"], path) for pair in paired]
            summary["paired_deltas"][f"with_loop_minus_without_loop_{name}"] = round(mean(deltas), 4)

    with_loop_records = by_variant.get("with_loop", [])
    fail_first = 0
    fail_then_pass = 0
    quality_gains = []
    semantic_gains = []
    for record in with_loop_records:
        metrics = dict(record.get("self_retrieval") or {})
        if metrics.get("q0_failed"):
            fail_first += 1
            if metrics.get("q1_passed"):
                fail_then_pass += 1
        if metrics.get("quality_gain_q1_minus_q0") is not None:
            quality_gains.append(float(metrics["quality_gain_q1_minus_q0"]))
        if metrics.get("semantic_gain_q1_minus_q0") is not None:
            semantic_gains.append(float(metrics["semantic_gain_q1_minus_q0"]))
    summary["self_retrieval"] = {
        "correction_success_numerator": fail_then_pass,
        "correction_success_denominator": fail_first,
        "correction_success_rate": round(fail_then_pass / fail_first, 4) if fail_first else None,
        "retrieval_quality_gain_q1_minus_q0": round(mean(quality_gains), 4) if quality_gains else None,
        "retrieval_semantic_gain_q1_minus_q0": round(mean(semantic_gains), 4) if semantic_gains else None,
        "paired_reader_relevance_gain_after_minus_before": summary["paired_deltas"].get(
            "with_loop_minus_without_loop_read_relevance_mean"
        ),
    }
    return summary


def build_paired_records(records: list[JsonObject]) -> list[JsonObject]:
    by_case: dict[str, dict[str, JsonObject]] = {}
    for record in records:
        by_case.setdefault(str(record.get("case_id")), {})[str(record.get("variant"))] = record
    return [
        {"case_id": case_id, "with_loop": variants["with_loop"], "without_loop": variants["without_loop"]}
        for case_id, variants in by_case.items()
        if "with_loop" in variants and "without_loop" in variants
    ]


def self_retrieval_metrics(state: JsonObject) -> JsonObject:
    correction = extract_retrieval_correction(state)
    history = [dict(item) for item in list(correction.get("history") or []) if isinstance(item, dict)]
    first = dict((history[0].get("report") or {}) if history else {})
    second = dict((history[1].get("report") or {}) if len(history) > 1 else {})
    q0_failed = bool(history) and not bool(first.get("passed"))
    q1_passed = bool(second) and bool(second.get("passed"))
    quality_gain = None
    semantic_gain = None
    if first and second:
        quality_gain = round(float_or_default(second.get("quality_score"), 0.0) - float_or_default(first.get("quality_score"), 0.0), 4)
        first_components = dict(first.get("quality_components") or {})
        second_components = dict(second.get("quality_components") or {})
        semantic_gain = round(
            float_or_default(second_components.get("Rsemantic"), 0.0)
            - float_or_default(first_components.get("Rsemantic"), 0.0),
            4,
        )
    return {
        "q0_failed": q0_failed,
        "q1_passed": q1_passed,
        "correction_success": bool(q0_failed and q1_passed),
        "history_length": len(history),
        "repair_attempt_count": optional_int(correction.get("repair_attempt_count")) or 0,
        "final_status": str(correction.get("status") or ""),
        "quality_gain_q1_minus_q0": quality_gain,
        "semantic_gain_q1_minus_q0": semantic_gain,
    }


def mean_read_relevance(state: JsonObject) -> JsonObject:
    read_results = [dict(item) for item in list(state.get("read_results") or []) if isinstance(item, dict)]
    scores = []
    for item in read_results:
        relevance = dict(item.get("relevance") or {})
        if relevance.get("score") is not None:
            scores.append(float_or_default(relevance.get("score"), 0.0))
    return {
        "paper_count": len(read_results),
        "scored_count": len(scores),
        "mean_score": round(mean(scores), 4) if scores else 0.0,
    }


def extract_answer_text(state: JsonObject) -> str:
    metadata = dict(state.get("assistant_message_metadata") or {})
    final_markdown = str(metadata.get("final_markdown") or "").strip()
    if final_markdown:
        return final_markdown
    writing_report = dict(state.get("writing_report") or metadata.get("writing_report") or {})
    content_markdown = str(writing_report.get("content_markdown") or "").strip()
    if content_markdown:
        return content_markdown
    return str(state.get("assistant_message") or "").strip()


def build_reference_payload(state: JsonObject) -> list[JsonObject]:
    writing_report = dict(state.get("writing_report") or {})
    refs = []
    for item in list(writing_report.get("references") or []):
        if not isinstance(item, dict):
            continue
        metadata = dict(item.get("metadata") or {})
        refs.append(
            {
                "id": str(item.get("index") or ""),
                "paperId": str(item.get("paperId") or metadata.get("paperId") or metadata.get("id") or ""),
                "citation": str(item.get("citation") or ""),
                "title": str(metadata.get("title") or ""),
                "abstract": str(metadata.get("abstract") or "")[:1200],
            }
        )
    return refs


def build_evidence_payload(state: JsonObject) -> list[JsonObject]:
    evidence = []
    for item in list(state.get("read_results") or []):
        if not isinstance(item, dict):
            continue
        paper = dict(item.get("paper") or {})
        note = dict(item.get("note") or {})
        extraction = dict(item.get("extraction") or {})
        evidence.append(
            {
                "paperId": str(paper.get("paperId") or paper.get("id") or ""),
                "title": str(paper.get("title") or ""),
                "abstract": str(paper.get("abstract") or "")[:900],
                "note": {
                    "main_question": note.get("main_question"),
                    "methods": note.get("methods"),
                    "datasets": note.get("datasets"),
                    "contributions": note.get("contributions"),
                    "limitations": note.get("limitations"),
                    "main_results": note.get("main_results"),
                    "short_summary": note.get("short_summary"),
                },
                "extraction": {
                    "research_topic": extraction.get("research_topic"),
                    "research_object": extraction.get("research_object"),
                    "methods": extraction.get("methods"),
                    "conclusions": extraction.get("conclusions"),
                    "contributions": extraction.get("contributions"),
                    "limitations": extraction.get("limitations"),
                },
            }
        )
    return evidence


def truncate_evidence(evidence: list[JsonObject], char_limit: int) -> list[JsonObject]:
    result = []
    used = 0
    for item in evidence:
        text = json.dumps(item, ensure_ascii=False)
        if used + len(text) > char_limit and result:
            break
        if len(text) > char_limit:
            item = {"paperId": item.get("paperId"), "title": item.get("title"), "abstract": truncate(str(item.get("abstract") or ""), char_limit)}
            text = json.dumps(item, ensure_ascii=False)
        result.append(item)
        used += len(text)
    return result


def extract_retrieval_correction(state: JsonObject) -> JsonObject:
    diagnostics = dict(state.get("diagnostics") or {})
    correction = diagnostics.get("retrieval_correction")
    if isinstance(correction, dict):
        return dict(correction)
    correction = state.get("retrieval_correction")
    return dict(correction) if isinstance(correction, dict) else {}


async def close_snapshots(snapshots: list[ProviderSnapshot]) -> None:
    for snapshot in snapshots:
        try:
            await snapshot.aclose()
        except Exception:
            pass


def load_completed_keys(result_path: Path) -> set[str]:
    keys = set()
    if not result_path.exists():
        return keys
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        keys.add(f"{payload.get('case_id')}:{payload.get('variant')}")
    return keys


def strip_references_section(text: str) -> str:
    return re.split(r"\n#{1,6}\s*(references|bibliography|参考文献)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]


def extract_json_object(text: str) -> JsonObject | None:
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
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def numeric_at(payload: JsonObject, path: list[str]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    return float_or_default(value, 0.0)


def mean_metric(items: list[JsonObject], path: list[str]) -> float:
    values = [numeric_at(item, path) for item in items]
    return round(mean(values), 4) if values else 0.0


def sum_metric(items: list[JsonObject], path: list[str]) -> int:
    return int(sum(numeric_at(item, path) for item in items))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float_or_default(value, 0.0)))


def clamp_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def truncate(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    return text[:char_limit] + "\n[TRUNCATED]"


def safe_id(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return text.strip("._-") or "case"


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items() if serializable_key(str(key))}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serializable_key(key: str) -> bool:
    return key not in {
        "runtime_context",
        "session_repo",
        "search_node_service",
        "search_node_llm",
        "retrieval_correction_node_llm",
        "read_node_llm",
        "analysis_node_llm",
        "writing_outline_node_llm",
        "writing_node_llm",
        "search_node_sink",
    }


if __name__ == "__main__":
    asyncio.run(main())
