from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from src.models.sessions import utc_now


JsonObject = dict[str, Any]

MEMORY_SCHEMA_VERSION = 2
MAX_NOTES = 12
MAX_CONTEXT_NOTES = 5
MAX_MEMORY_PROMPT_CHARS = 2200
MAX_WRITING_CONTEXT_CHARS = 900
SESSION_STYLE_SCAN_LIMIT = 8
SECRET_TEXT_PATTERN = re.compile(r"(?i)(api[_ -]?key|token|secret|password|sk-[A-Za-z0-9_-]{8,})")


class UserMemoryStore:
    """保存跨会话用户记忆的小文件仓库。

    中文说明：
    会话表负责保存完整聊天记录；这里另存一份很小的用户记忆，只放“以后还可能用到”的偏好、
    对话风格和最近几轮任务摘要。这样下一轮请求可以带上有用背景，又不会把整段历史塞进模型。
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def render_context(self, user_id: str, current_text: str) -> JsonObject:
        """取出当前请求可能用到的记忆，并整理成可放进提示词的短文本。"""

        state = self.load(user_id)
        conversation_rules = _rank_texts(state["conversation_rules"], current_text)[:MAX_CONTEXT_NOTES]
        preferences = _rank_texts(state["preferences"], current_text)[:MAX_CONTEXT_NOTES]
        style_notes = _rank_texts(state["style_notes"], current_text)[:MAX_CONTEXT_NOTES]
        recent_turns = _rank_turns(state["conversation_notes"], current_text)[:3]

        lines = ["用户长期记忆："]
        if conversation_rules:
            lines.append("- 对话规则/规范：")
            lines.extend(f"  - {item}" for item in conversation_rules)
        if preferences:
            lines.append("- 用户偏好：")
            lines.extend(f"  - {item}" for item in preferences)
        if style_notes:
            lines.append("- 对话风格：")
            lines.extend(f"  - {item}" for item in style_notes)
        if recent_turns:
            lines.append("- 相关旧对话：")
            for item in recent_turns:
                lines.append(f"  - 用户曾问：{item['user']}")
                if item.get("assistant"):
                    lines.append(f"    上次回复：{item['assistant']}")
        if len(lines) == 1:
            return {"enabled": True, "user_id": user_id, "prompt": ""}
        lines.append("- 使用方式：只在这些记忆和当前问题相关时采用；如果当前问题明确给了新要求，以当前问题为准。")
        prompt = _clip_block("\n".join(lines), MAX_MEMORY_PROMPT_CHARS)
        return {
            "enabled": True,
            "user_id": user_id,
            "prompt": prompt,
            "conversation_rules": conversation_rules,
            "preferences": preferences,
            "style_notes": style_notes,
            "recent_turns": recent_turns,
        }

    def record_turn(self, user_id: str, user_text: str, assistant_text: str, status: str = "completed") -> None:
        """把一轮对话沉淀成短记忆。

        中文说明：
        这里不用模型做总结，先用规则提取明确偏好和简短对话摘要。好处是启动快、依赖少、不会因为
        记忆总结模型不可用而影响主流程。
        """

        clean_user_text = _clean_text(user_text, 500)
        if not clean_user_text or _looks_secret(clean_user_text):
            return

        state = self.load(user_id)
        for text in _extract_conversation_rules(clean_user_text):
            _append_unique(state["conversation_rules"], text)
        for text in _extract_preferences(clean_user_text):
            _append_unique(state["preferences"], text)
        for text in _extract_style_notes(clean_user_text):
            _append_unique(state["style_notes"], text)

        state["conversation_notes"].append(
            {
                "user": _clean_text(clean_user_text, 160),
                "assistant": _clean_text(_first_useful_line(assistant_text), 180),
                "status": str(status or "completed"),
                "created_at": utc_now(),
            }
        )
        state["conversation_notes"] = state["conversation_notes"][-MAX_NOTES:]
        state["updated_at"] = utc_now()
        self.save(user_id, state)

    def load(self, user_id: str) -> JsonObject:
        """读取用户记忆；文件不存在或损坏时返回空记忆，主流程继续运行。"""

        path = self._path(user_id)
        if not path.exists():
            return _default_state(user_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return _default_state(user_id)
        return _normalize_state(user_id, payload)

    def save(self, user_id: str, state: JsonObject) -> None:
        """原子写入记忆文件，避免程序中断时留下半截 JSON。"""

        payload = _normalize_state(user_id, state)
        path = self._path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(path)

    def _path(self, user_id: str) -> Path:
        return self.root / f"{_safe_user_id(user_id)}.json"


def memory_store_for_repo(repo: Any) -> UserMemoryStore:
    """按当前会话仓库的位置放置用户记忆文件。"""

    backend = getattr(repo, "backend", None)
    storage_root = getattr(backend, "storage_root", None)
    root = Path(storage_root) if storage_root else Path("data")
    return UserMemoryStore(root / "memory")


def memory_context_for_record(repo: Any, record: Any, current_text: str) -> JsonObject:
    """从会话记录推导用户编号，并读取本轮要注入的记忆。"""

    user_id = str(getattr(record, "user_id", "") or "local-user")
    context = memory_store_for_repo(repo).render_context(user_id, current_text)
    writing_context = _session_writing_context(record, current_text)
    if writing_context:
        # 中文说明：这类“你是一位……需要用……”通常只约束当前会话的论文写作。
        # 它来自会话消息，不写入用户级长期记忆；会话被删除后自然过期。
        context["current_writing_context"] = writing_context
    return context


def record_turn_memory(repo: Any, record: Any, user_text: str, assistant_text: str, status: str = "completed") -> None:
    """会话结束后沉淀用户记忆；失败不影响正常聊天落库。"""

    try:
        user_id = str(getattr(record, "user_id", "") or "local-user")
        memory_store_for_repo(repo).record_turn(user_id, user_text, assistant_text, status=status)
    except Exception:
        return


def memory_prompt_from_constraints(constraints: JsonObject | None) -> str:
    """从请求约束里取出可直接放入提示词的记忆文本。"""

    if not isinstance(constraints, dict):
        return ""
    memory_context = constraints.get("memory_context")
    if not isinstance(memory_context, dict):
        return ""
    return str(memory_context.get("prompt") or "").strip()


def writing_prompt_from_constraints(constraints: JsonObject | None) -> str:
    """整理写作节点需要遵守的本轮约束和长期记忆。"""

    if not isinstance(constraints, dict):
        return ""
    parts = [_current_writing_context_prompt(constraints.get("current_writing_context"))]
    memory_prompt = memory_prompt_from_constraints(constraints)
    if memory_prompt:
        parts.append(memory_prompt)
    return _clip_block("\n\n".join(part for part in parts if part), MAX_MEMORY_PROMPT_CHARS)


def research_constraints_from_constraints(constraints: JsonObject | None) -> JsonObject:
    """给检索和阅读节点使用的约束，去掉只影响写作表达的内容。"""

    if not isinstance(constraints, dict):
        return {}
    return {
        key: value
        for key, value in constraints.items()
        if key not in {"memory_context", "current_writing_context"}
    }


def _default_state(user_id: str) -> JsonObject:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "user_id": user_id,
        "conversation_rules": [],
        "preferences": [],
        "style_notes": [],
        "conversation_notes": [],
        "updated_at": utc_now(),
    }


def _normalize_state(user_id: str, payload: Any) -> JsonObject:
    if not isinstance(payload, dict):
        payload = {}
    state = _default_state(user_id)
    state["conversation_rules"] = _clean_text_list(payload.get("conversation_rules"))
    state["preferences"] = _clean_text_list(payload.get("preferences"))
    state["style_notes"] = _clean_text_list(payload.get("style_notes"))
    state["conversation_notes"] = _clean_turns(payload.get("conversation_notes"))
    state["updated_at"] = str(payload.get("updated_at") or state["updated_at"])
    return state


def _clean_turns(value: Any) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    turns: list[JsonObject] = []
    for item in value[-MAX_NOTES:]:
        if not isinstance(item, dict):
            continue
        user = _clean_text(item.get("user"), 160)
        if not user:
            continue
        turns.append(
            {
                "user": user,
                "assistant": _clean_text(item.get("assistant"), 180),
                "status": str(item.get("status") or "completed"),
                "created_at": str(item.get("created_at") or utc_now()),
            }
        )
    return turns


def _clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[-MAX_NOTES:]:
        text = _clean_text(item, 220)
        if text and not _looks_secret(text) and text not in result:
            result.append(text)
    return result


def _extract_conversation_rules(text: str) -> list[str]:
    patterns = (
        r"(?:对话规则|对话规范|回答规则|回复规则|输出规范|写作规范)[:：]?\s*(.+)",
        r"(?:以后|后续|之后).{0,12}(?:回答|回复|输出|写作).+",
    )
    return _extract_by_patterns(text, patterns)


def _extract_preferences(text: str) -> list[str]:
    patterns = (
        r"(?:记住|请记住|保存|记录|长期记忆)[:：]?\s*(.+)",
        r"(?:偏好|Preference)[:：]\s*(.+)",
        r"(?:以后|后续|之后).{0,8}(?:请|帮我|默认)?(.+)",
    )
    return _extract_by_patterns(text, patterns)


def _extract_style_notes(text: str) -> list[str]:
    patterns = (
        r"(?:记住|请记住|保存|记录|长期记忆|以后|后续|之后|默认).{0,16}(?:回答|回复|输出|写作).{0,12}(?:风格|语气|格式)[:：]?\s*(.+)",
        r"(?:以后|后续|之后|默认).{0,20}(?:简洁|详细|中文|英文|正式|口语|学术|列表|表格).+",
    )
    return _extract_by_patterns(text, patterns)


def _extract_by_patterns(text: str, patterns: tuple[str, ...]) -> list[str]:
    results: list[str] = []
    for line in [item.strip() for item in re.split(r"[\r\n]+", text) if item.strip()]:
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).strip() if match.lastindex else match.group(0).strip()
            value = _clean_text(value, 220)
            if value and not _looks_secret(value) and value not in results:
                results.append(value)
            break
    return results


def _current_writing_context_prompt(value: Any) -> str:
    """把本轮写作约束渲染成短提示词。"""

    if not isinstance(value, dict):
        return ""
    lines = ["本轮写作约束："]
    role = _clean_text(value.get("role"), 160)
    style = _clean_text(value.get("style"), 180)
    if role:
        lines.append(f"- 写作身份：{role}")
    if style:
        lines.append(f"- 写作风格：{style}")
    if len(lines) == 1:
        return ""
    lines.append("- 优先级：这些约束只约束本轮大纲、正文和摘要；不参与论文检索关键词。")
    return _clip_block("\n".join(lines), MAX_WRITING_CONTEXT_CHARS)


def _session_writing_context(record: Any, current_text: str) -> JsonObject:
    """从当前会话最近几条用户消息中提取写作身份和语言风格。"""

    context: JsonObject = {}
    messages = list(getattr(record, "messages", []) or [])
    user_texts = [
        str(message.get("content") or "")
        for message in messages[-SESSION_STYLE_SCAN_LIMIT:]
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    user_texts.append(str(current_text or ""))
    for text in user_texts:
        extracted = _writing_context_from_text(text)
        if extracted.get("role"):
            context["role"] = extracted["role"]
        if extracted.get("style"):
            context["style"] = extracted["style"]
    return context


def _writing_context_from_text(text: str) -> JsonObject:
    """解析“你是一位专家、用某种语言风格写作”这类直白要求。"""

    raw_text = str(text or "")
    if _looks_secret(raw_text):
        return {}
    context: JsonObject = {}
    role_match = re.search(
        r"(?:你是|你作为|请你作为|作为)(?:一位|一个)?(?P<role>[^，,。；;\n]{2,80}?)(?=，|,|。|；|;|需要|请|用|帮我|为我)",
        raw_text,
    )
    if role_match:
        context["role"] = _clean_text(role_match.group("role"), 160)

    style_match = re.search(
        r"(?:需要|请|请你|帮我)?(?:用|以|采用)(?P<style>[^。；;\n]{2,140}?)(?:的语言|的风格|的语气)",
        raw_text,
    )
    if style_match:
        context["style"] = _clean_text(style_match.group("style"), 180)
    return context


def _rank_texts(items: list[str], query: str) -> list[str]:
    query_tokens = _tokens(query)
    return sorted(
        items,
        key=lambda item: (len(_tokens(item) & query_tokens), items.index(item)),
        reverse=True,
    )


def _rank_turns(items: list[JsonObject], query: str) -> list[JsonObject]:
    """只召回和当前问题有重合的旧对话，避免最近历史无脑进入提示词。"""

    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    ranked: list[tuple[tuple[int, int], JsonObject]] = []
    for index, item in enumerate(items):
        text = f"{item.get('user') or ''} {item.get('assistant') or ''}"
        overlap = len(_tokens(text) & query_tokens)
        if overlap:
            ranked.append(((overlap, index), item))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in ranked]


def _tokens(text: str) -> set[str]:
    raw_text = str(text)
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", raw_text)}
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", raw_text)
    tokens.update(chinese_chars)
    tokens.update("".join(chinese_chars[index : index + 2]) for index in range(len(chinese_chars) - 1))
    return tokens


def _append_unique(items: list[str], text: str) -> None:
    if text in items:
        items.remove(text)
    items.append(text)
    del items[:-MAX_NOTES]


def _first_useful_line(text: str) -> str:
    for line in str(text or "").splitlines():
        clean = _clean_text(line, 180)
        if clean:
            return clean
    return ""


def _clean_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _clip_block(text: str, limit: int) -> str:
    """截短整段提示词，但保留换行，方便模型仍按条目阅读。"""

    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _looks_secret(text: str) -> bool:
    return bool(SECRET_TEXT_PATTERN.search(str(text or "")))


def _safe_user_id(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(user_id or "local-user")).strip("._")
    return safe or "local-user"
