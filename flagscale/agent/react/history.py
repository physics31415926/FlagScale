"""Conversation history management with context window protection."""

import json
import logging

from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for mixed CJK/English."""
    return len(text) // 4 + 1


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


class HistoryManager:
    """Manages conversation history to stay within context limits.

    Strategy: when total tokens exceed max_context_tokens, truncate tool
    results in older messages (keeping the most recent ones intact).
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
            return list(self._messages)

        logger.info("History exceeds budget (%d > %d tokens), truncating old tool results", total, self.max_context_tokens)

        # Keep at most the last 10 messages untouched, but always truncate
        # at least the non-system, non-recent messages
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

        new_total = sum(_message_tokens(m) for m in result)
        logger.info("After truncation: %d tokens", new_total)
        return result

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
