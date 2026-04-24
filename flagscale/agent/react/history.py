"""Conversation history management with context window protection."""

import json
import logging

from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 ASCII chars per token, ~1.5 CJK chars per token."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿' or '가' <= c <= '힯' or '぀' <= c <= 'ヿ')
    ascii_chars = len(text) - cjk
    return ascii_chars // 4 + int(cjk * 1.5) + 1


def _message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens in a single message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += _estimate_tokens(json.dumps(block, ensure_ascii=False))
            else:
                total += _estimate_tokens(str(block))
        return total
    return _estimate_tokens(json.dumps(msg, ensure_ascii=False))


def _is_tool_result(msg: Dict[str, Any]) -> bool:
    """Check if a message is a tool result (OpenAI role=tool or Anthropic tool_result block)."""
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def _has_tool_use(msg: Dict[str, Any]) -> bool:
    """Check if an assistant message contains tool_use blocks."""
    if msg.get("tool_calls"):
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return False


class HistoryManager:
    """Manages conversation history to stay within context limits.

    Strategy: when total tokens exceed max_context_tokens, first truncate tool
    results in older messages, then drop oldest message pairs if still over.
    Always preserves assistant(tool_use) + user(tool_result) pairing.
    """

    def __init__(self, max_context_tokens: int = 100000):
        self.max_context_tokens = max_context_tokens
        self._messages: List[Dict[str, Any]] = []

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._messages

    def append(self, message: Dict[str, Any]):
        self._messages.append(message)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return messages, truncating old tool results if over budget."""
        total = sum(_message_tokens(m) for m in self._messages)
        if total <= self.max_context_tokens:
            return _validate_tool_pairs(list(self._messages))

        logger.info("History exceeds budget (%d > %d tokens), truncating old tool results", total, self.max_context_tokens)

        keep_recent = min(10, max(len(self._messages) - 2, 1))
        result = []

        for i, msg in enumerate(self._messages):
            is_recent = (i >= len(self._messages) - keep_recent)
            if msg.get("role") == "system":
                result.append(msg)
            elif is_recent:
                result.append(msg)
            else:
                result.append(_truncate_message(msg))

        total = sum(_message_tokens(m) for m in result)
        if total > self.max_context_tokens:
            logger.info("Still over budget after truncation (%d tokens), dropping old message pairs", total)
            result = _drop_old_pairs(result, self.max_context_tokens)

        new_total = sum(_message_tokens(m) for m in result)
        logger.info("After truncation: %d tokens", new_total)
        return _validate_tool_pairs(result)

    def clear(self):
        self._messages.clear()


def _truncate_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Replace long content in tool results with a summary placeholder."""
    content = msg.get("content", "")

    if isinstance(content, str) and len(content) > 500:
        role = msg.get("role", "")
        if role == "tool":
            return {**msg, "content": f"[truncated tool result, {len(content)} chars]"}

    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and len(inner) > 500:
                    new_blocks.append({**block, "content": f"[truncated tool result, {len(inner)} chars]"})
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)
        return {**msg, "content": new_blocks}

    return msg


def _validate_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove orphaned tool_use or tool_result blocks to prevent API 400 errors."""
    result = list(messages)
    i = 0
    while i < len(result):
        msg = result[i]
        if msg.get("role") == "assistant" and _has_tool_use(msg):
            # Must be followed by a tool_result
            if i + 1 >= len(result) or not _is_tool_result(result[i + 1]):
                result.pop(i)
                continue
        elif _is_tool_result(msg):
            # Must be preceded by an assistant with tool_use
            if i == 0 or not (result[i - 1].get("role") == "assistant" and _has_tool_use(result[i - 1])):
                result.pop(i)
                continue
        i += 1
    return result


def _drop_old_pairs(messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    """Drop oldest non-system messages in pairs to stay within budget.

    Preserves assistant(tool_use) + user(tool_result) pairing by always
    dropping them together.
    """
    total = sum(_message_tokens(m) for m in messages)
    if total <= budget:
        return messages

    result = list(messages)
    i = 0
    while i < len(result) and total > budget:
        if result[i].get("role") == "system":
            i += 1
            continue

        # Check if this is an assistant message with tool_use followed by tool_result
        if (result[i].get("role") == "assistant" and _has_tool_use(result[i])
                and i + 1 < len(result) and _is_tool_result(result[i + 1])):
            total -= _message_tokens(result[i]) + _message_tokens(result[i + 1])
            result.pop(i)
            result.pop(i)
        # Check if this is a tool_result (orphaned or Anthropic format) — skip, handle with its assistant
        elif _is_tool_result(result[i]):
            total -= _message_tokens(result[i])
            result.pop(i)
        else:
            total -= _message_tokens(result[i])
            result.pop(i)

    return result
