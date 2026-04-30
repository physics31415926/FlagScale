"""Tests for HistoryManager."""

from flagscale.agent.react.history import (
    HistoryManager, _estimate_tokens, _message_tokens,
    _is_tool_result, _has_tool_use, _collect_droppable,
)


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_short(self):
        assert _estimate_tokens("hello") >= 1

    def test_proportional(self):
        short = _estimate_tokens("a" * 100)
        long = _estimate_tokens("a" * 1000)
        assert long > short

    def test_cjk_higher_than_ascii(self):
        ascii_text = "a" * 100
        cjk_text = "你" * 100
        assert _estimate_tokens(cjk_text) > _estimate_tokens(ascii_text)

    def test_cjk_chars_counted_as_1_5_tokens(self):
        cjk_text = "你好世界"
        tokens = _estimate_tokens(cjk_text)
        assert tokens >= 6  # 4 chars * 1.5 = 6

    def test_mixed_cjk_ascii(self):
        text = "Hello 你好 World 世界"
        tokens = _estimate_tokens(text)
        ascii_only = "Hello  World "
        cjk_only = "你好世界"
        assert tokens >= int(len(cjk_only) * 1.5) + len(ascii_only) // 4

    def test_japanese_counted(self):
        text = "こんにちは"
        tokens = _estimate_tokens(text)
        assert tokens >= 7  # 5 * 1.5 = 7.5

    def test_korean_counted(self):
        text = "안녕하세요"
        tokens = _estimate_tokens(text)
        assert tokens >= 7  # 5 * 1.5 = 7.5


class TestHelpers:
    def test_is_tool_result_openai(self):
        assert _is_tool_result({"role": "tool", "content": "result"})

    def test_is_tool_result_anthropic(self):
        msg = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]}
        assert _is_tool_result(msg)

    def test_is_tool_result_normal_user(self):
        assert not _is_tool_result({"role": "user", "content": "hello"})

    def test_has_tool_use_openai(self):
        msg = {"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}]}
        assert _has_tool_use(msg)

    def test_has_tool_use_anthropic(self):
        msg = {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "shell", "input": {}}]}
        assert _has_tool_use(msg)

    def test_has_tool_use_text_only(self):
        msg = {"role": "assistant", "content": "just text"}
        assert not _has_tool_use(msg)


class TestHistoryManager:
    def test_append_and_get(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "You are helpful."})
        hm.append({"role": "user", "content": "Hi"})
        msgs = hm.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"

    def test_full_log_preserved(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hi"})
        assert len(hm.full_log) == 2

    def test_truncation_on_budget(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert any(m["role"] == "user" and m["content"] == "recent" for m in msgs)

    def test_compaction_flag(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "x" * 5000})
        hm.get_messages()
        assert hm.compaction_happened

    def test_no_compaction_under_budget(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hi"})
        hm.get_messages()
        assert not hm.compaction_happened

    def test_summarizer_called_on_compaction(self):
        called = []
        def fake_summarizer(text):
            called.append(text)
            return "Summary: stuff happened"

        hm = HistoryManager(max_context_tokens=100)
        hm.set_summarizer(fake_summarizer)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert len(called) > 0
        # Summary should be injected
        summary_msgs = [m for m in msgs if isinstance(m.get("content", ""), str) and "<context-summary>" in m["content"]]
        assert len(summary_msgs) == 1

    def test_orphaned_tool_result_removed(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "orphan"})
        hm.append({"role": "user", "content": "hi"})
        msgs = hm.get_messages()
        assert not any(m.get("role") == "tool" for m in msgs)

    def test_anthropic_tool_pair_preserved(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "shell", "input": {}}]})
        hm.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert len(msgs) == 4

    def test_anthropic_orphan_removed(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "orphan"}]})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        assert len(msgs) == 2


class TestCollectDroppable:
    def test_preserves_system(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 10000},
        ]
        _, kept = _collect_droppable(messages, budget=10)
        assert kept[0]["role"] == "system"

    def test_under_budget_no_change(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        dropped, kept = _collect_droppable(messages, budget=100000)
        assert len(kept) == 2
        assert len(dropped) == 0

    def test_drops_openai_tool_pair(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""},
            {"role": "tool", "tool_call_id": "1", "content": "x" * 5000},
            {"role": "user", "content": "recent"},
        ]
        dropped, kept = _collect_droppable(messages, budget=10)
        assert kept[-1]["content"] == "recent"
        assert not any(m.get("role") == "tool" for m in kept)

    def test_fallback_drops_when_still_over(self):
        """Even after truncation, if still over budget, drop old pairs."""
        hm = HistoryManager(max_context_tokens=50)
        hm.append({"role": "system", "content": "s"})
        # Old pair
        hm.append({"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""})
        hm.append({"role": "tool", "tool_call_id": "1", "content": "x" * 5000})
        # Recent
        hm.append({"role": "user", "content": "hi"})
        msgs = hm.get_messages()
        # Should not contain the old tool result
        assert not any(m.get("role") == "tool" and "xxxxx" in m.get("content", "") for m in msgs)
