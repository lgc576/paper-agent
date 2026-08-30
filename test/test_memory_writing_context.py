import json
import asyncio
import tempfile
import unittest

from src.agents.writingAgent import _abstract_messages
from src.agents import ReviewRequest
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowRuntimeContext
from src.graph.search_node import run_search_agent_node
from src.llm.base import LLMResponse
from src.llm.factory import ProviderSnapshot
from src.models.sessions import SessionRecord
from src.paper_retrieval.models import PaperDocument, SearchResponse
from src.services.memory import memory_context_for_record
from src.services.paper_workflow_runtime import _constraints_from_frame


class _Backend:
    """只提供 memory_store_for_repo 需要的存储根目录。"""

    def __init__(self, storage_root):
        self.storage_root = storage_root


class _Repo:
    """只提供 memory_store_for_repo 需要的最小属性。"""

    def __init__(self, storage_root):
        self.backend = _Backend(storage_root)


class _FakeProvider:
    """给 search 节点返回固定检索计划，避免测试依赖真实模型。"""

    async def chat(self, messages, **kwargs):
        return LLMResponse(
            content=json.dumps(
                {
                    "research_topic": "大语言模型在分子扩散领域的应用",
                    "writing_context": {
                        "role": "分子扩散领域的专家",
                        "style": "简洁直白，通顺严谨，学术规范",
                    },
                    "subtopics": [
                        {
                            "subtopic": "分子扩散任务中的大语言模型应用",
                            "keyword": "(large language model or LLM) and (molecular diffusion or molecule diffusion)",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            finish_reason="stop",
        )


class _SearchService:
    """返回一篇可通过搜索节点粗筛的论文。"""

    async def async_search(self, **kwargs):
        return SearchResponse(
            query=kwargs.get("query") or "",
            sources_used=["openalex"],
            source_results={"openalex": 1},
            papers=[
                PaperDocument(
                    id="paper-1",
                    paperId="paper-1",
                    title="Large Language Models for Molecular Diffusion",
                    authors=["Tester"],
                    abstract="Large language models are applied to molecular diffusion tasks.",
                    source="openalex",
                )
            ],
        )


class MemoryWritingContextTest(unittest.TestCase):
    def test_session_style_context_is_extracted_without_durable_prompt(self):
        """验证用户隐式写作风格能作为当前会话约束进入后续节点。"""

        record = SessionRecord(
            key="session-1",
            messages=[
                {
                    "role": "user",
                    "content": "你是一位分子扩散领域的专家，需要用简洁直白，通顺严谨，学术规范的语言为我调研大语言模型在分子扩散领域的应用",
                }
            ],
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        memory_context = memory_context_for_record(_Repo(temp_dir.name), record, "继续写正文")

        self.assertEqual(memory_context["current_writing_context"]["role"], "分子扩散领域的专家")
        self.assertEqual(memory_context["current_writing_context"]["style"], "简洁直白，通顺严谨，学术规范")

    def test_workflow_constraints_receive_session_writing_context(self):
        """验证入口层把会话写作约束放进 ReviewRequest.constraints。"""

        constraints = _constraints_from_frame(
            {
                "constraints": {},
                "memory_context": {
                    "current_writing_context": {
                        "role": "分子扩散领域的专家",
                        "style": "简洁直白，通顺严谨，学术规范",
                    }
                },
            }
        )

        self.assertEqual(constraints["current_writing_context"]["role"], "分子扩散领域的专家")
        self.assertEqual(constraints["current_writing_context"]["style"], "简洁直白，通顺严谨，学术规范")

    def test_abstract_prompt_receives_writing_context(self):
        """验证摘要模型也能看到正文使用的同一份写作约束。"""

        messages = _abstract_messages(
            topic="大语言模型在分子扩散领域的应用",
            sections=[{"section_title": "方法", "content": "正文"}],
            word_count=300,
            memory_context="本轮写作约束：\n- 写作风格：简洁直白，通顺严谨，学术规范",
        )
        payload = json.loads(messages[1]["content"])

        self.assertIn("简洁直白", payload["写作约束与记忆"])

    def test_search_runtime_event_outputs_writing_context(self):
        """验证检索条件事件会输出本轮识别到的写作身份和风格。"""

        events = []
        runtime = WorkflowRuntimeContext(
            session_key="session-1",
            turn_id="turn-1",
            sync_port=InlineWorkflowSyncPort(
                lambda event: events.append(event) or event,
                session_key="session-1",
                turn_id="turn-1",
            ),
        )
        llm = ProviderSnapshot(
            provider=_FakeProvider(),
            model="fake-model",
            context_window_tokens=4096,
            signature="fake",
        )

        asyncio.run(
            run_search_agent_node()(
                {
                    "request": ReviewRequest(topic="请调研大语言模型在分子扩散领域的应用"),
                    "runtime_context": runtime,
                    "search_node_llm": llm,
                    "search_node_service": _SearchService(),
                }
            )
        )
        intent_events = [
            event
            for event in events
            if event.get("event") == "runtime_event"
            and event.get("id") == "turn-1:search:plan_search"
            and dict(event.get("metadata") or {}).get("stage") == "intent_ready"
        ]

        self.assertTrue(intent_events)
        detail = dict(intent_events[-1]["detail_content"])
        self.assertIn("large language model", detail["keywords"][0])
        self.assertEqual(detail["writing_context"]["role"], "分子扩散领域的专家")
        self.assertEqual(detail["writing_context"]["style"], "简洁直白，通顺严谨，学术规范")


if __name__ == "__main__":
    unittest.main()
