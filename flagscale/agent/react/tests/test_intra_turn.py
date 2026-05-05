"""Tests for intra-turn sliding window compaction."""

import pytest

from flagscale.agent.react.history import HistoryManager


def _make_tool_exchange(tool_name, command, output, tool_id="t1"):
    """Create a typical assistant tool_use + user tool_result pair."""
    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": {"command": command} if tool_name == "shell" else {"path": command},
            }
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": output,
            }
        ],
    }
    return assistant_msg, user_msg


class TestCompactIntraTurn:

    def test_no_compact_when_few_messages(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Run training"})
        h.append({"role": "assistant", "content": "Starting training."})
        assert h.compact_intra_turn(keep_last=4) is False

    def test_compact_reduces_messages(self):
        h = HistoryManager(max_context_tokens=100000)
        # Real user message (turn start)
        h.append({"role": "user", "content": "Monitor the training"})

        # Simulate 10 tool exchanges
        for i in range(10):
            a, u = _make_tool_exchange("shell", f"tail -5 log.txt", f"step {i}\n", f"t{i}")
            h.append(a)
            h.append(u)

        original_count = len(h.messages)
        assert original_count == 21  # 1 user + 10*(assistant+user)

        result = h.compact_intra_turn(keep_last=4)
        assert result is True
        # Should have: user_msg + summary + last 4 messages
        assert len(h.messages) <= 6
        # Summary should contain turn-progress marker
        found_summary = False
        for msg in h.messages:
            if isinstance(msg.get("content"), str) and "<turn-progress>" in msg["content"]:
                found_summary = True
                break
        assert found_summary

    def test_compact_preserves_recent_messages(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Check GPU status"})

        for i in range(8):
            a, u = _make_tool_exchange("shell", f"nvidia-smi", f"GPU {i}: 50%\n", f"t{i}")
            h.append(a)
            h.append(u)

        # Last exchange
        last_a, last_u = _make_tool_exchange("shell", "nvidia-smi", "GPU 7: 100%\n", "tlast")
        h.append(last_a)
        h.append(last_u)

        h.compact_intra_turn(keep_last=4)

        # The last 4 messages should be preserved exactly
        assert h.messages[-1]["content"][0]["content"] == "GPU 7: 100%\n"

    def test_compact_extracts_errors(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Start training"})

        # Normal exchanges
        for i in range(5):
            a, u = _make_tool_exchange("shell", "tail log.txt", "normal output\n", f"t{i}")
            h.append(a)
            h.append(u)

        # Error exchange
        a, u = _make_tool_exchange("shell", "tail log.txt", "ERROR: CUDA OOM\n", "terr")
        h.append(a)
        h.append(u)

        # More normal
        for i in range(4):
            a, u = _make_tool_exchange("shell", "nvidia-smi", "ok\n", f"tn{i}")
            h.append(a)
            h.append(u)

        h.compact_intra_turn(keep_last=4)

        # Find summary and check it contains the error
        for msg in h.messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "<turn-progress>" in content:
                assert "ERROR" in content or "CUDA" in content
                break
        else:
            pytest.fail("No turn-progress summary found")

    def test_compact_extracts_actions(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Setup environment"})

        commands = ["pip install torch", "conda activate env", "python train.py"]
        for i, cmd in enumerate(commands):
            a, u = _make_tool_exchange("shell", cmd, f"done {i}\n", f"t{i}")
            h.append(a)
            h.append(u)

        # Add more to trigger compaction
        for i in range(5):
            a, u = _make_tool_exchange("shell", "echo ok", "ok\n", f"tx{i}")
            h.append(a)
            h.append(u)

        h.compact_intra_turn(keep_last=4)

        for msg in h.messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "<turn-progress>" in content:
                assert "shell:" in content
                break

    def test_repeated_compact_doesnt_nest(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Long task"})

        # First batch
        for i in range(10):
            a, u = _make_tool_exchange("shell", f"cmd{i}", f"out{i}\n", f"t{i}")
            h.append(a)
            h.append(u)

        h.compact_intra_turn(keep_last=4)
        count_after_first = len(h.messages)

        # Second batch
        for i in range(10):
            a, u = _make_tool_exchange("shell", f"cmd2_{i}", f"out2_{i}\n", f"t2{i}")
            h.append(a)
            h.append(u)

        h.compact_intra_turn(keep_last=4)
        count_after_second = len(h.messages)

        # Should still be compact
        assert count_after_second <= 7  # summary + keep_last + maybe original user msg

    def test_find_turn_start_skips_tool_results(self):
        h = HistoryManager(max_context_tokens=100000)
        h.append({"role": "user", "content": "Do the thing"})

        for i in range(6):
            a, u = _make_tool_exchange("shell", f"cmd{i}", f"out{i}\n", f"t{i}")
            h.append(a)
            h.append(u)

        # _find_turn_start should find the original "Do the thing" message
        start = h._find_turn_start()
        assert h.messages[start]["content"] == "Do the thing"
