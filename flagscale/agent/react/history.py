"""Conversation history management with context window protection."""

import json
import logging

from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = (
    "Summarize the key conclusions, decisions, and findings from this conversation segment. "
    "PRESERVE with high fidelity:\n"
    "- File paths that were read or modified\n"
    "- Error messages and their root causes\n"
    "- Solutions and workarounds found\n"
    "- Decisions made and their rationale\n"
    "- Current approach/strategy and what phase we're in\n"
    "- What was ruled out and why\n"
    "Be specific about technical details (function names, error messages, version numbers, shapes). "
    "Keep the summary under 1000 tokens."
)

COMPACTION_NOTICE = (
    "<context-compacted>\n"
    "Previous context was compacted. A summary of dropped content is available in "
    "<context-summary> above.\n"
    "If you need details that aren't in the summary, re-read the relevant files "
    "rather than assuming you remember.\n"
    "</context-compacted>"
)

MAX_SUMMARY_TOKENS = 4000
TRUNCATE_THRESHOLD = 2000
KEEP_RECENT = 12  # Reduced from 20 to limit recent message token consumption
AGING_WINDOW = 10
AGING_THRESHOLD = 800


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 ASCII chars per token, ~1.5 CJK chars per token."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿' or '가' <= c <= '힯' or '぀' <= c <= 'ヿ')
    ascii_chars = len(text) - cjk
    return ascii_chars // 4 + int(cjk * 1.5) + 1


def _smart_truncate(content: str, max_chars: int = 600) -> str:
    """Truncate preserving structure: first lines + error tail + summary."""
    if len(content) <= max_chars:
        return content
    lines = content.splitlines()

    error_tail = _extract_error_tail(content, max_chars=400)
    if error_tail:
        head = "\n".join(lines[:3])
        return f"{head}\n[... {len(lines)} lines, {len(content)} chars ...]\n{error_tail}"

    if len(lines) > 15:
        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-5:])
        return f"{head}\n[... {len(lines) - 10} lines omitted, {len(content)} chars total ...]\n{tail}"

    return content[:max_chars] + f"\n[... truncated, {len(content)} chars total]"


def _age_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Truncate a single message's tool results if they're large."""
    content = msg.get("content", "")

    if isinstance(content, str) and len(content) > AGING_THRESHOLD:
        if msg.get("role") == "tool":
            return {**msg, "content": _smart_truncate(content)}

    if isinstance(content, list):
        new_blocks = []
        changed = False
        for block in content:
            if (isinstance(block, dict) and block.get("type") == "tool_result"
                    and isinstance(block.get("content", ""), str)
                    and len(block["content"]) > AGING_THRESHOLD):
                new_blocks.append({**block, "content": _smart_truncate(block["content"])})
                changed = True
            else:
                new_blocks.append(block)
        if changed:
            return {**msg, "content": new_blocks}

    return msg


def _age_tool_results(messages: List[Dict[str, Any]], keep_recent: int = AGING_WINDOW) -> List[Dict[str, Any]]:
    """Proactively truncate old tool results to save context budget."""
    if len(messages) <= keep_recent:
        return messages
    cutoff = len(messages) - keep_recent
    result = []
    for i, msg in enumerate(messages):
        if i >= cutoff:
            result.append(msg)
            continue
        if msg.get("role") == "system":
            result.append(msg)
            continue
        result.append(_age_message(msg))
    return result


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


def _extract_text(msg: Dict[str, Any]) -> str:
    """Extract readable text from a message for summarization."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        parts.append(inner[:500])
                elif block.get("type") == "tool_use":
                    name = block.get("name", "")
                    inp = block.get("input", {})
                    parts.append(f"[tool_use: {name}({json.dumps(inp, ensure_ascii=False)[:200]})]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


class HistoryManager:
    """Manages conversation history to stay within context limits.

    Strategy: when total tokens exceed max_context_tokens:
    1. Truncate tool results in older messages (threshold: 2000 chars)
    2. If still over, collect messages to drop, generate LLM summary, then drop
    3. Insert summary as <context-summary> that survives future compactions
    4. Notify agent that compaction happened
    """

    # Progressive compaction: each successive compaction targets a lower ratio,
    # leaving more buffer so the next compaction is delayed longer.
    _COMPACTION_RATIOS = [0.80, 0.70, 0.60, 0.50]

    def __init__(self, max_context_tokens: int = 100000):
        self.max_context_tokens = max_context_tokens
        self._messages: List[Dict[str, Any]] = []
        self._full_log: List[Dict[str, Any]] = []
        self._last_compacted_from = None
        self._last_compacted_to = None
        self._actual_input_tokens = None
        self._last_inflation_ratio = 1.0  # Preserve inflation ratio across compactions
        self._summarizer: Optional[Callable[[str], str]] = None
        self._accumulated_summary: str = ""
        self._compaction_anchors: List[str] = []
        self._compaction_happened = False
        self._compaction_count = 0

    def set_summarizer(self, callback: Callable[[str], str]):
        """Inject LLM summarization callback. Signature: (text) -> summary_string."""
        self._summarizer = callback

    def set_compaction_anchors(self, anchors: List[str]):
        """Set anchors that MUST be preserved in the next compaction summary."""
        self._compaction_anchors = anchors[:10]

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._messages

    @property
    def full_log(self) -> List[Dict[str, Any]]:
        """Complete uncompacted message history for export/archival."""
        return self._full_log

    @property
    def compaction_happened(self) -> bool:
        """True if the last get_messages() call triggered compaction."""
        return self._compaction_happened

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    @property
    def last_compaction_ratio(self) -> Optional[float]:
        """The ratio used in the most recent compaction, or None."""
        if self._compaction_count == 0:
            return None
        idx = min(self._compaction_count - 1, len(self._COMPACTION_RATIOS) - 1)
        return self._COMPACTION_RATIOS[idx]

    def get_context_pressure(self) -> float:
        """Return current context usage as a ratio (0.0 to 1.0+)."""
        if self.max_context_tokens <= 0:
            return 0.0
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        total = max(estimated, actual)
        return total / self.max_context_tokens

    def _get_inflation_ratio(self) -> float:
        """Return ratio of actual API tokens to local estimate.

        When the API reports actual_input_tokens, local estimates can be off
        by 1.5-2x.  This ratio lets compaction target a *local-estimate*
        budget that, after inflation, lands at the real target.

        Falls back to last known ratio if actual_input_tokens is not available.
        """
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        if actual > 0 and estimated > 0:
            ratio = max(actual / estimated, 1.0)
            self._last_inflation_ratio = ratio  # Remember for future use
            return ratio
        return self._last_inflation_ratio  # Use last known ratio as fallback

    def force_compact(self, target_ratio: float = 0.50) -> bool:
        """Force compaction to a target ratio. Returns True if compaction occurred."""
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        current = max(estimated, actual)
        inflation = self._get_inflation_ratio()

        # Target in *real* tokens, then deflate to local-estimate space
        real_target = int(self.max_context_tokens * target_ratio)
        local_target = int(real_target / inflation)

        if current <= real_target:
            return False

        logger.warning(
            "Force compact: estimated=%d, actual=%d, inflation=%.2f, "
            "real_target=%d, local_target=%d",
            estimated, actual, inflation, real_target, local_target,
        )

        keep_recent = min(KEEP_RECENT, max(len(self._messages) - 2, 1))
        result = []
        for i, msg in enumerate(self._messages):
            is_recent = (i >= len(self._messages) - keep_recent)
            if msg.get("role") == "system":
                result.append(msg)
            elif is_recent:
                result.append(msg)
            else:
                result.append(_truncate_message(msg))

        new_estimated = sum(_message_tokens(m) for m in result)
        if new_estimated > local_target:
            to_drop, to_keep = _collect_droppable(result, local_target)
            if to_drop and self._summarizer:
                summary_text = self._build_summary_input(to_drop, self._compaction_anchors)
                self._compaction_anchors = []
                try:
                    new_summary = self._summarizer(summary_text)
                    self._merge_summary(new_summary)
                except Exception as e:
                    logger.warning("Summarizer failed during force compact: %s", e)
            result = to_keep

        result = self._inject_summary(result)
        self._messages = result
        # Keep _actual_input_tokens to preserve inflation ratio memory
        self._compaction_count += 1
        final_estimated = sum(_message_tokens(m) for m in self._messages)
        logger.info("Force compact done: %d -> %d estimated tokens (≈%d real)",
                    estimated, final_estimated, int(final_estimated * inflation))
        return True

    def append(self, message: Dict[str, Any]):
        self._messages.append(message)
        self._full_log.append(message)

    def report_actual_tokens(self, input_tokens: int):
        """Feed back the actual input_tokens from the API response."""
        self._actual_input_tokens = input_tokens

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return messages, compacting with LLM summary if over budget."""
        self._messages = _age_tool_results(self._messages, keep_recent=AGING_WINDOW)
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        total = max(estimated, actual)
        inflation = self._get_inflation_ratio()
        self._last_compacted_from = None
        self._last_compacted_to = None
        self._compaction_happened = False
        if total <= self.max_context_tokens:
            return _validate_tool_pairs(list(self._messages))

        logger.info("History exceeds budget (estimated=%d, actual=%d, inflation=%.2f, limit=%d), compacting",
                     estimated, actual, inflation, self.max_context_tokens)
        original_total = total

        # Dynamic target: compress harder each successive time
        ratio_idx = min(self._compaction_count, len(self._COMPACTION_RATIOS) - 1)
        ratio = self._COMPACTION_RATIOS[ratio_idx]
        real_target = int(self.max_context_tokens * ratio)
        local_target = int(real_target / inflation)
        logger.info("Compaction #%d, target ratio=%.0f%%, real_budget=%d, local_budget=%d",
                     self._compaction_count + 1, ratio * 100, real_target, local_target)

        keep_recent = min(KEEP_RECENT, max(len(self._messages) - 2, 1))

        # Step 1: truncate old tool results (threshold raised to 2000 chars)
        result = []
        for i, msg in enumerate(self._messages):
            is_recent = (i >= len(self._messages) - keep_recent)
            if msg.get("role") == "system":
                result.append(msg)
            elif is_recent:
                result.append(msg)
            else:
                result.append(_truncate_message(msg))

        new_estimated = sum(_message_tokens(m) for m in result)

        # Step 2: if still over target, collect messages to drop and summarize them
        if new_estimated > local_target:
            logger.info("Still over target after truncation (%d tokens, local_target=%d), summarizing and dropping",
                        new_estimated, local_target)
            to_drop, to_keep = _collect_droppable(result, local_target)

            if to_drop and self._summarizer:
                summary_text = self._build_summary_input(to_drop, self._compaction_anchors)
                self._compaction_anchors = []
                try:
                    new_summary = self._summarizer(summary_text)
                    self._merge_summary(new_summary)
                    logger.info("Generated compaction summary (%d chars)", len(new_summary))
                except Exception as e:
                    logger.warning("Summarizer failed, dropping without summary: %s", e)

            result = to_keep

        # Step 3: inject accumulated summary after system message
        result = self._inject_summary(result)

        # Step 4: hard ceiling — if still over budget, aggressively truncate recent messages
        new_estimated = sum(_message_tokens(m) for m in result)
        if new_estimated > local_target:
            logger.warning(
                "Hard ceiling: still %d tokens after drop (local_target=%d), truncating recent",
                new_estimated, local_target,
            )
            result = self._hard_ceiling_truncate(result, local_target)

        new_estimated = sum(_message_tokens(m) for m in result)
        logger.info("After compaction #%d (ratio=%.0f%%): %d -> %d estimated (≈%d real)",
                     self._compaction_count + 1, ratio * 100, estimated, new_estimated,
                     int(new_estimated * inflation))
        self._messages = result
        # Keep _actual_input_tokens to preserve inflation ratio memory
        self._last_compacted_from = original_total
        self._last_compacted_to = new_estimated
        self._compaction_happened = True
        self._compaction_count += 1
        return _validate_tool_pairs(list(self._messages))

    def _build_summary_input(self, messages: List[Dict[str, Any]], anchors: Optional[List[str]] = None) -> str:
        """Build text input for the summarizer from messages about to be dropped."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            text = _extract_text(msg)
            if text.strip():
                parts.append(f"[{role}] {text}")
        combined = "\n---\n".join(parts)
        if len(combined) > 32000:
            combined = combined[:32000] + "\n[... truncated for summarization ...]"
        anchor_section = ""
        if anchors:
            anchor_section = (
                "\n\nMANDATORY ANCHORS — these MUST appear verbatim in your summary:\n"
                + "\n".join(f"- {a}" for a in anchors[:10])
                + "\n"
            )
        return f"{SUMMARIZE_PROMPT}{anchor_section}\n\n---\nConversation segment:\n{combined}"

    def _merge_summary(self, new_summary: str):
        """Merge new summary into accumulated summary, keeping total under limit."""
        if self._accumulated_summary:
            merged = f"{self._accumulated_summary}\n\n---\n\n{new_summary}"
        else:
            merged = new_summary
        # Trim if over token limit
        while _estimate_tokens(merged) > MAX_SUMMARY_TOKENS and "\n\n---\n\n" in merged:
            # Drop the oldest section
            _, _, merged = merged.partition("\n\n---\n\n")
        self._accumulated_summary = merged

    def _inject_summary(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert or update the <context-summary> message after the system message."""
        if not self._accumulated_summary:
            return messages

        summary_msg = {
            "role": "user",
            "content": f"<context-summary>\n{self._accumulated_summary}\n</context-summary>"
        }

        result = []
        inserted = False
        for msg in messages:
            # Remove any existing context-summary message
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.startswith("<context-summary>"):
                    continue
            result.append(msg)
            # Insert after system message
            if not inserted and msg.get("role") == "system":
                result.append(summary_msg)
                inserted = True

        if not inserted:
            result.insert(0, summary_msg)

        return result

    def _hard_ceiling_truncate(self, messages: List[Dict[str, Any]], local_target: int) -> List[Dict[str, Any]]:
        """Emergency truncation when normal compaction fails to reach target.

        Keeps system/summary messages at the front, then fills from the most
        recent messages backward with aggressive truncation.
        """
        head = []
        body = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            is_summary = isinstance(content, str) and content.startswith("<context-summary>")
            if role == "system" or is_summary:
                head.append(msg)
            else:
                body.append(msg)

        head_tokens = sum(_message_tokens(m) for m in head)
        budget = local_target - head_tokens

        kept = []
        used = 0
        for msg in reversed(body):
            truncated = _truncate_message(msg, max_chars=400)
            t = _message_tokens(truncated)
            if used + t > budget:
                break
            kept.append(truncated)
            used += t

        kept.reverse()
        result = head + kept
        logger.warning("Hard ceiling kept %d/%d messages, %d tokens",
                      len(result), len(messages), head_tokens + used)
        return result

    def clear(self):
        self._messages.clear()
        self._full_log.clear()
        self._accumulated_summary = ""


def _extract_error_tail(content: str, max_chars: int = 1500) -> str:
    """Extract error/traceback portion from tool output for preservation during truncation."""
    lines = content.splitlines()
    error_start = -1
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(kw in lower for kw in ('traceback', 'error:', 'exception:', 'fatal:', 'failed')):
            if error_start < 0:
                error_start = i
    if error_start >= 0:
        error_text = "\n".join(lines[error_start:])
        if len(error_text) > max_chars:
            error_text = error_text[-max_chars:]
        return error_text
    return ""


def _truncate_message(msg: Dict[str, Any], max_chars: int = TRUNCATE_THRESHOLD) -> Dict[str, Any]:
    """Replace long content in tool results with a summary placeholder.
    Preserves error/traceback content to avoid losing diagnostic information."""
    content = msg.get("content", "")

    if isinstance(content, str) and len(content) > max_chars:
        role = msg.get("role", "")
        if role == "tool":
            error_tail = _extract_error_tail(content)
            if error_tail:
                return {**msg, "content": f"[truncated tool result, {len(content)} chars. Error preserved:]\n{error_tail}"}
            return {**msg, "content": f"[truncated tool result, {len(content)} chars]"}

    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and len(inner) > max_chars:
                    error_tail = _extract_error_tail(inner)
                    if error_tail:
                        new_blocks.append({**block, "content": f"[truncated tool result, {len(inner)} chars. Error preserved:]\n{error_tail}"})
                    else:
                        new_blocks.append({**block, "content": f"[truncated tool result, {len(inner)} chars]"})
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)
        return {**msg, "content": new_blocks}

    return msg


def _merge_user_messages(msg1: Dict[str, Any], msg2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two user messages into one, handling both string and list content."""
    c1 = msg1.get("content", "")
    c2 = msg2.get("content", "")
    blocks1 = c1 if isinstance(c1, list) else [{"type": "text", "text": c1}]
    blocks2 = c2 if isinstance(c2, list) else [{"type": "text", "text": c2}]
    return {"role": "user", "content": blocks1 + blocks2}


def _validate_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove orphaned tool_use or tool_result blocks to prevent API 400 errors."""
    result = list(messages)
    i = 0
    while i < len(result):
        msg = result[i]
        if msg.get("role") == "assistant" and _has_tool_use(msg):
            if i + 1 >= len(result) or not _is_tool_result(result[i + 1]):
                result.pop(i)
                continue
        elif _is_tool_result(msg):
            if i == 0 or not (result[i - 1].get("role") == "assistant" and _has_tool_use(result[i - 1])):
                result.pop(i)
                continue
        i += 1

    # Merge consecutive user messages (Anthropic requires strict role alternation)
    i = 0
    while i < len(result) - 1:
        if result[i].get("role") == "user" and result[i + 1].get("role") == "user":
            result[i] = _merge_user_messages(result[i], result[i + 1])
            result.pop(i + 1)
        else:
            i += 1

    return result


def _collect_droppable(messages: List[Dict[str, Any]], budget: int):
    """Separate messages into (to_drop, to_keep) lists to fit within budget.

    Keeps system messages and recent messages. Drops oldest non-system messages
    in proper pairs (assistant+tool_result together).
    """
    total = sum(_message_tokens(m) for m in messages)
    if total <= budget:
        return [], messages

    to_drop = []
    to_keep = list(messages)
    i = 0
    while i < len(to_keep) and total > budget:
        if to_keep[i].get("role") == "system":
            i += 1
            continue

        if (to_keep[i].get("role") == "assistant" and _has_tool_use(to_keep[i])
                and i + 1 < len(to_keep) and _is_tool_result(to_keep[i + 1])):
            total -= _message_tokens(to_keep[i]) + _message_tokens(to_keep[i + 1])
            to_drop.append(to_keep.pop(i))
            to_drop.append(to_keep.pop(i))
        elif _is_tool_result(to_keep[i]):
            total -= _message_tokens(to_keep[i])
            to_drop.append(to_keep.pop(i))
        else:
            total -= _message_tokens(to_keep[i])
            to_drop.append(to_keep.pop(i))

    return to_drop, to_keep
