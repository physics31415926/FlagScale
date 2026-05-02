"""Tests for poll-mode detection and execution."""

import time
import unittest
from unittest.mock import MagicMock, patch

from flagscale.agent.react.agent import ReactAgent
from flagscale.agent.react.config import AgentConfig


class TestPollOutputChanged(unittest.TestCase):
    """Test _poll_output_changed static method."""

    def test_identical_output(self):
        assert not ReactAgent._poll_output_changed("hello world", "hello world")

    def test_same_lines_different_order(self):
        old = "line1\nline2\nline3"
        new = "line1\nline2\nline3"
        assert not ReactAgent._poll_output_changed(old, new)

    def test_new_lines_added(self):
        old = "line1\nline2"
        new = "line1\nline2\nline3_new"
        assert ReactAgent._poll_output_changed(old, new)

    def test_significant_length_change(self):
        old = "a" * 100
        new = "a" * 115
        assert ReactAgent._poll_output_changed(old, new)

    def test_minor_length_change_same_lines(self):
        old = "line1\nline2"
        new = "line1\nline2"
        assert not ReactAgent._poll_output_changed(old, new)

    def test_empty_to_content(self):
        assert ReactAgent._poll_output_changed("", "new content")

    def test_both_empty(self):
        assert not ReactAgent._poll_output_changed("", "")

    def test_whitespace_only_difference(self):
        old = "line1\nline2\n"
        new = "line1\nline2\n\n"
        assert not ReactAgent._poll_output_changed(old, new)


class TestDetectPollPattern(unittest.TestCase):
    """Test _detect_poll_pattern method."""

    def _make_agent(self, **kwargs):
        config = AgentConfig(
            provider="openai", model="test", api_key="test-key",
            poll_detect_window=2, **kwargs,
        )
        with patch.object(ReactAgent, '__init__', lambda self, cfg: None):
            agent = ReactAgent.__new__(ReactAgent)
            agent.config = config
            agent._recent_iters = []
        return agent

    def test_no_iterations(self):
        agent = self._make_agent()
        assert not agent._detect_poll_pattern()

    def test_too_few_iterations(self):
        agent = self._make_agent()
        agent._recent_iters = [
            {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
             "llm_output_tokens": 50, "tool_elapsed": 0.2},
        ]
        assert not agent._detect_poll_pattern()

    def test_matching_pattern(self):
        agent = self._make_agent()
        entry = {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
                 "llm_output_tokens": 50, "tool_elapsed": 0.2}
        agent._recent_iters = [entry, entry]
        assert agent._detect_poll_pattern()

    def test_different_commands(self):
        agent = self._make_agent()
        agent._recent_iters = [
            {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
             "llm_output_tokens": 50, "tool_elapsed": 0.2},
            {"tool_name": "shell", "command": "tail -5 log.txt", "output": "...",
             "llm_output_tokens": 50, "tool_elapsed": 0.2},
        ]
        assert not agent._detect_poll_pattern()

    def test_high_output_tokens_breaks_pattern(self):
        agent = self._make_agent()
        agent._recent_iters = [
            {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
             "llm_output_tokens": 50, "tool_elapsed": 0.2},
            {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
             "llm_output_tokens": 300, "tool_elapsed": 0.2},
        ]
        assert not agent._detect_poll_pattern()

    def test_slow_tool_breaks_pattern(self):
        agent = self._make_agent()
        agent._recent_iters = [
            {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
             "llm_output_tokens": 50, "tool_elapsed": 0.2},
            {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
             "llm_output_tokens": 50, "tool_elapsed": 10},
        ]
        assert not agent._detect_poll_pattern()

    def test_none_entry_breaks_pattern(self):
        agent = self._make_agent()
        agent._recent_iters = [
            None,
            {"tool_name": "shell", "command": "wc -l log.txt", "output": "10",
             "llm_output_tokens": 50, "tool_elapsed": 0.2},
        ]
        assert not agent._detect_poll_pattern()


class TestRecordIteration(unittest.TestCase):
    """Test _record_iteration method."""

    def _make_agent(self):
        with patch.object(ReactAgent, '__init__', lambda self, cfg: None):
            agent = ReactAgent.__new__(ReactAgent)
            agent.config = AgentConfig(
                provider="openai", model="test", api_key="test-key",
                poll_detect_window=2,
            )
            agent._recent_iters = []
        return agent

    def test_single_shell_recorded(self):
        agent = self._make_agent()
        tool_calls = [{"name": "shell", "arguments": {"command": "ls -la"}}]
        agent._record_iteration(tool_calls, ["file1\nfile2"], 50, [0.3])
        assert len(agent._recent_iters) == 1
        assert agent._recent_iters[0]["command"] == "ls -la"

    def test_non_shell_recorded_as_none(self):
        agent = self._make_agent()
        tool_calls = [{"name": "read_file", "arguments": {"path": "/tmp/x"}}]
        agent._record_iteration(tool_calls, ["content"], 50, [0.1])
        assert len(agent._recent_iters) == 1
        assert agent._recent_iters[0] is None

    def test_multiple_tools_recorded_as_none(self):
        agent = self._make_agent()
        tool_calls = [
            {"name": "shell", "arguments": {"command": "ls"}},
            {"name": "shell", "arguments": {"command": "pwd"}},
        ]
        agent._record_iteration(tool_calls, ["a", "b"], 50, [0.1, 0.2])
        assert agent._recent_iters[0] is None

    def test_window_trimming(self):
        agent = self._make_agent()
        for i in range(5):
            tool_calls = [{"name": "shell", "arguments": {"command": f"cmd{i}"}}]
            agent._record_iteration(tool_calls, [f"out{i}"], 50, [0.1])
        assert len(agent._recent_iters) == 2
        assert agent._recent_iters[0]["command"] == "cmd3"


class TestRunPollMode(unittest.TestCase):
    """Test _run_poll_mode method."""

    def _make_agent(self):
        with patch.object(ReactAgent, '__init__', lambda self, cfg: None):
            agent = ReactAgent.__new__(ReactAgent)
            agent.config = AgentConfig(
                provider="openai", model="test", api_key="test-key",
                poll_interval=1, poll_max_duration=5,
            )
            agent._recent_iters = []
            agent.tool_registry = MagicMock()
        return agent

    @patch('flagscale.agent.react.agent.display')
    def test_output_changes_immediately(self, mock_display):
        agent = self._make_agent()
        agent.tool_registry.execute.return_value = "ERROR: something broke\nnew output line"
        output, count, elapsed, reason, routine = agent._run_poll_mode(
            "wc -l log.txt", "old output", "call-1")
        assert reason == "changed"
        assert count == 1
        assert "new output" in output

    @patch('flagscale.agent.react.agent.display')
    def test_timeout_when_no_change(self, mock_display):
        agent = self._make_agent()
        agent.config.poll_interval = 1
        agent.config.poll_max_duration = 3
        agent.tool_registry.execute.return_value = "same output"
        output, count, elapsed, reason, routine = agent._run_poll_mode(
            "wc -l log.txt", "same output", "call-1")
        assert reason == "timeout"
        assert count >= 1

    @patch('flagscale.agent.react.agent.display')
    def test_routine_change_absorbed(self, mock_display):
        """Routine changes (e.g., line count +1) should not break poll."""
        agent = self._make_agent()
        agent.config.poll_interval = 1
        agent.config.poll_max_duration = 3
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return f"{24 + call_count[0]} train.log"
            return "26 train.log"
        agent.tool_registry.execute.side_effect = side_effect
        output, count, elapsed, reason, routine = agent._run_poll_mode(
            "wc -l train.log", "24 train.log", "call-1")
        assert reason == "timeout"
        assert routine == 2


class TestReplaceLastToolResult(unittest.TestCase):
    """Test _replace_last_tool_result method."""

    def _make_agent(self):
        with patch.object(ReactAgent, '__init__', lambda self, cfg: None):
            agent = ReactAgent.__new__(ReactAgent)
            from flagscale.agent.react.history import HistoryManager
            agent.history = HistoryManager()
        return agent

    def test_replace_openai_format(self):
        agent = self._make_agent()
        agent.history.append({"role": "assistant", "content": "checking..."})
        agent.history.append({"role": "tool", "tool_call_id": "1", "content": "old"})
        agent._replace_last_tool_result(
            {"role": "tool", "tool_call_id": "1", "content": "new"})
        assert agent.history.messages[-1]["content"] == "new"

    def test_replace_anthropic_format(self):
        agent = self._make_agent()
        agent.history.append({"role": "assistant", "content": [
            {"type": "text", "text": "checking..."}]})
        agent.history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "1", "content": "old"}]})
        agent._replace_last_tool_result(
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "new"}]})
        last = agent.history.messages[-1]
        assert last["content"][0]["content"] == "new"

    def test_no_tool_result_appends(self):
        agent = self._make_agent()
        agent.history.append({"role": "assistant", "content": "hello"})
        agent._replace_last_tool_result(
            {"role": "tool", "tool_call_id": "1", "content": "result"})
        assert len(agent.history.messages) == 2
        assert agent.history.messages[-1]["content"] == "result"


class TestAnnotationsMatch(unittest.TestCase):
    """Test _annotations_match static method."""

    def test_both_empty(self):
        assert ReactAgent._annotations_match([], [])

    def test_identical(self):
        a = ["Training process running", "GPU utilization OK"]
        assert ReactAgent._annotations_match(a, list(a))

    def test_different_order(self):
        a = ["A", "B"]
        b = ["B", "A"]
        assert ReactAgent._annotations_match(a, b)

    def test_different_content(self):
        a = ["Training process running"]
        b = ["Training process stopped"]
        assert not ReactAgent._annotations_match(a, b)

    def test_different_length(self):
        a = ["A"]
        b = ["A", "B"]
        assert not ReactAgent._annotations_match(a, b)

    def test_whitespace_ignored(self):
        a = ["  Training running  "]
        b = ["Training running"]
        assert ReactAgent._annotations_match(a, b)

    def test_one_empty(self):
        assert not ReactAgent._annotations_match([], ["something"])


class TestDedupAnnotations(unittest.TestCase):
    """Test _dedup_annotations method."""

    def _make_agent(self):
        with patch.object(ReactAgent, '__init__', lambda self, cfg: None):
            agent = ReactAgent.__new__(ReactAgent)
            agent._last_result_annotations = []
        return agent

    def test_first_call_returns_annotations(self):
        agent = self._make_agent()
        result = agent._dedup_annotations(["Training running"])
        assert result == ["Training running"]

    def test_duplicate_suppressed(self):
        agent = self._make_agent()
        agent._dedup_annotations(["Training running"])
        result = agent._dedup_annotations(["Training running"])
        assert result == []

    def test_different_annotations_returned(self):
        agent = self._make_agent()
        agent._dedup_annotations(["Training running"])
        result = agent._dedup_annotations(["Training stopped"])
        assert result == ["Training stopped"]

    def test_empty_annotations_returns_empty(self):
        agent = self._make_agent()
        assert agent._dedup_annotations([]) == []


class TestHealthCheckAutoVocab(unittest.TestCase):
    """Test auto vocab_size detection in _health_check."""

    def test_loss_near_ln_32000(self):
        import math
        from flagscale.agent.react.tools.find_log import _health_check
        metrics = {"last_loss": {"lm_loss": math.log(32000)}, "iterations": [1]}
        warnings = _health_check(metrics, vocab_size=0)
        assert any("32000" in w for w in warnings)

    def test_loss_near_ln_128256(self):
        import math
        from flagscale.agent.react.tools.find_log import _health_check
        metrics = {"last_loss": {"ce_loss": 11.7}, "iterations": [1]}
        warnings = _health_check(metrics, vocab_size=0)
        assert any("128256" in w for w in warnings)

    def test_normal_loss_no_warning(self):
        from flagscale.agent.react.tools.find_log import _health_check
        metrics = {"last_loss": {"lm_loss": 2.5}, "iterations": [1]}
        warnings = _health_check(metrics, vocab_size=0)
        assert not any("random" in w.lower() for w in warnings)

    def test_explicit_vocab_size_still_works(self):
        import math
        from flagscale.agent.react.tools.find_log import _health_check
        metrics = {"last_loss": {"lm_loss": math.log(32000) * 0.85}, "iterations": [1]}
        warnings = _health_check(metrics, vocab_size=32000)
        assert any("32000" in w for w in warnings)


if __name__ == "__main__":
    unittest.main()
