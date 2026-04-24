"""Tests for HistoryManager."""

from flagscale.agent.react.history import (
    HistoryManager, _estimate_tokens, _message_tokens,
    _is_tool_result, _has_tool_use, _drop_old_pairs,
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

    def test_no_truncation_under_limit(self):
        hm = HistoryManager(max_context_tokens=100000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "hello"})
        hm.append({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "shell", "input": {}}]})
        hm.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "short result"}]})
        msgs = hm.get_messages()
        tool_msg = [m for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), list) and any(b.get("type") == "tool_result" for b in m["content"])][0]
        assert tool_msg["content"][0]["content"] == "short result"

    def test_truncation_over_limit(self):
        # keep_recent = min(10, max(len-2, 1)); need len>=13 so tool_result at i=2 is not recent
        hm = HistoryManager(max_context_tokens=500)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "shell", "input": {}}]})
        hm.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 10000}]})
        for i in range(11):
            hm.append({"role": "user", "content": f"msg {i}"})
        msgs = hm.get_messages()
        tool_msgs = [m for m in msgs if isinstance(m.get("content"), list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])]
        assert tool_msgs and "truncated" in tool_msgs[0]["content"][0]["content"]

    def test_recent_messages_preserved(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        for i in range(5):
            hm.append({"role": "user", "content": f"msg {i}"})
        msgs = hm.get_messages()
        assert msgs[-1]["content"] == "msg 4"

    def test_clear(self):
        hm = HistoryManager()
        hm.append({"role": "user", "content": "hi"})
        hm.clear()
        assert len(hm.messages) == 0

    def test_anthropic_tool_result_truncation(self):
        hm = HistoryManager(max_context_tokens=500)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "assistant", "content": [{"type": "tool_use", "id": "123", "name": "shell", "input": {}}]})
        hm.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "123", "content": "y" * 10000}
            ],
        })
        for i in range(11):
            hm.append({"role": "user", "content": f"msg {i}"})
        msgs = hm.get_messages()
        tool_msgs = [m for m in msgs if isinstance(m.get("content"), list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])]
        assert tool_msgs and "truncated" in tool_msgs[0]["content"][0]["content"]


class TestDropOldPairs:
    def test_drops_assistant_tool_result_pair(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "shell", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]},
            {"role": "user", "content": "recent"},
        ]
        result = _drop_old_pairs(messages, budget=10)
        roles = [m["role"] for m in result]
        assert "system" in roles
        assert result[-1]["content"] == "recent"

    def test_preserves_system(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 10000},
        ]
        result = _drop_old_pairs(messages, budget=10)
        assert result[0]["role"] == "system"

    def test_under_budget_no_change(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = _drop_old_pairs(messages, budget=100000)
        assert len(result) == 2

    def test_drops_openai_tool_pair(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "tool_calls": [{"id": "1", "name": "shell"}], "content": ""},
            {"role": "tool", "tool_call_id": "1", "content": "x" * 5000},
            {"role": "user", "content": "recent"},
        ]
        result = _drop_old_pairs(messages, budget=10)
        assert result[-1]["content"] == "recent"
        assert not any(m.get("role") == "tool" for m in result)

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
