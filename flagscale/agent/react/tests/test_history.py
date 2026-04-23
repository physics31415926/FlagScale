"""Tests for HistoryManager."""

from flagscale.agent.react.history import HistoryManager, _estimate_tokens, _message_tokens


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_short(self):
        assert _estimate_tokens("hello") >= 1

    def test_proportional(self):
        short = _estimate_tokens("a" * 100)
        long = _estimate_tokens("a" * 1000)
        assert long > short


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
        hm.append({"role": "tool", "content": "short result"})
        msgs = hm.get_messages()
        assert msgs[2]["content"] == "short result"

    def test_truncation_over_limit(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "tool", "content": "x" * 10000})
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        # The old tool result should be truncated
        tool_msg = [m for m in msgs if m["role"] == "tool"][0]
        assert "truncated" in tool_msg["content"]

    def test_recent_messages_preserved(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        for i in range(5):
            hm.append({"role": "user", "content": f"msg {i}"})
        # Most recent should be preserved as-is
        msgs = hm.get_messages()
        assert msgs[-1]["content"] == "msg 4"

    def test_clear(self):
        hm = HistoryManager()
        hm.append({"role": "user", "content": "hi"})
        hm.clear()
        assert len(hm.messages) == 0

    def test_anthropic_tool_result_truncation(self):
        hm = HistoryManager(max_context_tokens=100)
        hm.append({"role": "system", "content": "sys"})
        hm.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "123", "content": "y" * 10000}
            ],
        })
        hm.append({"role": "user", "content": "recent"})
        msgs = hm.get_messages()
        block = msgs[1]["content"][0]
        assert "truncated" in block["content"]
