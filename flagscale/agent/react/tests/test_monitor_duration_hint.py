"""Tests for monitor duration adaptive hints in _check_monitor_duration."""

import pytest
from unittest.mock import Mock
from flagscale.agent.react.agent import WorkerAgent
from flagscale.agent.react.config import AgentConfig


@pytest.fixture
def agent():
    config = AgentConfig(provider="mock", model="test")
    a = WorkerAgent.__new__(WorkerAgent)
    a.config = config
    a.session_memory = Mock()
    a.history = Mock()
    a._inject_message = Mock()
    return a


def _tc(args: dict) -> dict:
    return {"name": "monitor", "arguments": args}


class TestCheckMonitorDuration:
    def test_fires_on_timeout_with_default_duration(self, agent):
        tc = _tc({"file": "/tmp/train.log", "duration": 300})
        result = "Monitoring timed out after 300s. No pattern matched."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is not None
        assert "[Monitor]" in hint
        assert "duration=600" in hint

    def test_fires_on_duration_reached_signal(self, agent):
        tc = _tc({"output_dir": "/outputs/run1", "duration": 120})
        result = "Duration reached. Process still running."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is not None
        assert "duration=240" in hint

    def test_fires_on_timeout_reached_signal(self, agent):
        tc = _tc({"command": "pgrep python", "duration": 60})
        result = "Timeout reached. No success pattern found."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is not None
        assert "duration=120" in hint

    def test_caps_suggested_duration_at_1800(self, agent):
        tc = _tc({"file": "/tmp/log.txt", "duration": 1200})
        result = "Timed out after 1200s."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is not None
        assert "duration=1800" in hint

    def test_no_hint_when_process_died(self, agent):
        tc = _tc({"file": "/tmp/log.txt", "duration": 300})
        result = "Timed out. Process died — no further output expected."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is None

    def test_no_hint_on_success_exit(self, agent):
        tc = _tc({"file": "/tmp/log.txt", "duration": 300})
        result = "Success pattern matched at step 100. Training healthy."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is None

    def test_no_hint_on_error_exit(self, agent):
        tc = _tc({"file": "/tmp/log.txt", "duration": 300})
        result = "Error detected in stderr: CUDA out of memory."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is None

    def test_no_hint_for_non_monitor_tool(self, agent):
        tc = {"name": "shell", "arguments": {"command": "sleep 300"}}
        result = "Timed out."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is None

    def test_no_hint_for_non_string_result(self, agent):
        tc = _tc({"file": "/tmp/log.txt", "duration": 300})
        hint = agent._check_monitor_duration(tc, {"error": "something"})
        assert hint is None

    def test_dedup_same_target_not_warned_twice(self, agent):
        tc = _tc({"output_dir": "/outputs/run1", "duration": 300})
        result = "Timed out after 300s."
        h1 = agent._check_monitor_duration(tc, result)
        h2 = agent._check_monitor_duration(tc, result)
        assert h1 is not None
        assert h2 is None

    def test_different_targets_both_warned(self, agent):
        r = "Timed out after 300s."
        h1 = agent._check_monitor_duration(
            _tc({"output_dir": "/outputs/run1", "duration": 300}), r
        )
        h2 = agent._check_monitor_duration(
            _tc({"output_dir": "/outputs/run2", "duration": 300}), r
        )
        assert h1 is not None
        assert h2 is not None

    def test_output_dir_label_in_hint(self, agent):
        tc = _tc({"output_dir": "/outputs/qwen3_run", "duration": 300})
        result = "Timed out after 300s."
        hint = agent._check_monitor_duration(tc, result)
        assert "output_dir=`/outputs/qwen3_run`" in hint

    def test_file_label_in_hint(self, agent):
        tc = _tc({"file": "/tmp/train.log", "duration": 300})
        result = "Timed out after 300s."
        hint = agent._check_monitor_duration(tc, result)
        assert "file=`/tmp/train.log`" in hint

    def test_command_label_in_hint(self, agent):
        tc = _tc({"command": "pgrep -f torchrun", "duration": 300})
        result = "Timed out after 300s."
        hint = agent._check_monitor_duration(tc, result)
        assert "command=`pgrep -f torchrun`" in hint

    def test_uses_default_300_when_duration_not_specified(self, agent):
        tc = _tc({"file": "/tmp/log.txt"})
        result = "Timed out after 300s."
        hint = agent._check_monitor_duration(tc, result)
        assert hint is not None
        assert "duration=600" in hint

    def test_suggests_target_step_and_success_pattern(self, agent):
        tc = _tc({"output_dir": "/outputs/run1", "duration": 300})
        result = "Timed out after 300s."
        hint = agent._check_monitor_duration(tc, result)
        assert "target_step" in hint
        assert "success_pattern" in hint


class TestMonitorHintIntegration:
    def test_hint_injected_via_inject_efficiency_hints(self, agent):
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        tool_calls = [_tc({"output_dir": "/outputs/run1", "duration": 300})]
        results = ["Timed out after 300s. Process still running."]

        agent._inject_efficiency_hints(tool_calls, results)

        assert agent._inject_message.called
        call_args = agent._inject_message.call_args[0][0]
        assert "[Monitor]" in call_args

    def test_no_hint_injected_on_success(self, agent):
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        tool_calls = [_tc({"output_dir": "/outputs/run1", "duration": 300})]
        results = ["Success pattern matched. Training healthy."]

        agent._inject_efficiency_hints(tool_calls, results)

        assert not agent._inject_message.called
