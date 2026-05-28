"""Tests for error pattern auto-memorization."""

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
    a.turn_count = 1
    a._error_pattern_history = []
    return a


class TestExtractErrorSnippet:
    def test_extracts_error_lines(self, agent):
        result = "Some output\nError: file not found\nTraceback: line 42\nMore output"
        snippet = agent._extract_error_snippet(result)
        assert "Error: file not found" in snippet
        assert "Traceback: line 42" in snippet

    def test_truncates_long_snippets(self, agent):
        result = "Error: " + "x" * 500
        snippet = agent._extract_error_snippet(result, max_chars=100)
        assert len(snippet) <= 100

    def test_handles_no_error_keywords(self, agent):
        result = "Normal output without errors"
        snippet = agent._extract_error_snippet(result)
        assert snippet == "Normal output without errors"


class TestTrackErrorPattern:
    def test_records_error_event(self, agent):
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        tool_calls = [{"name": "shell", "arguments": {"command": "python test.py"}}]
        results = ["Error: ModuleNotFoundError: No module named 'torch'"]

        agent._track_error_pattern(tool_calls, results)

        assert len(agent._error_pattern_history) == 1
        assert agent._error_pattern_history[0]["phase"] == "error"
        assert "ModuleNotFoundError" in agent._error_pattern_history[0]["error_snippet"]

    def test_records_fix_event_after_error(self, agent):
        """Fix event is recorded; if the fix result also signals success, memorization fires."""
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        # Turn 1: error
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "python test.py"}}],
            ["Error: ModuleNotFoundError: No module named 'torch'"],
        )
        agent.turn_count += 1
        assert len(agent._error_pattern_history) == 1
        assert agent._error_pattern_history[0]["phase"] == "error"

        # Turn 2: fix via edit_file — "Successfully edited" triggers success branch
        # so the state machine memorizes and clears in the same turn
        agent._track_error_pattern(
            [{"name": "edit_file", "arguments": {"path": "requirements.txt"}}],
            ["Successfully edited requirements.txt"],
        )

        # State was cleared after memorization
        assert len(agent._error_pattern_history) == 0
        assert agent.session_memory.put.called

    def test_auto_memorizes_on_success(self, agent):
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        # Turn 1: error
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "python test.py"}}],
            ["Error: ModuleNotFoundError: No module named 'torch'"],
        )
        agent.turn_count += 1

        # Turn 2: fix
        agent._track_error_pattern(
            [{"name": "edit_file", "arguments": {"path": "requirements.txt", "old_string": "foo", "new_string": "bar"}}],
            ["Successfully edited"],
        )
        agent.turn_count += 1

        # Turn 3: success
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "python test.py"}}],
            ["All tests passed. Success!"],
        )

        # Should have called session_memory.put
        assert agent.session_memory.put.called
        call_args = agent.session_memory.put.call_args[1]
        assert call_args["mem_type"] == "finding"
        assert "ModuleNotFoundError" in call_args["content"]

    def test_clears_history_after_memorization(self, agent):
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        # error → fix → success
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "cmd1"}}],
            ["Error: something failed"],
        )
        agent.turn_count += 1
        agent._track_error_pattern(
            [{"name": "edit_file", "arguments": {"path": "fix.py"}}],
            ["Edited"],
        )
        agent.turn_count += 1
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "cmd1"}}],
            ["Success! All done."],
        )

        # History should be cleared after memorization
        assert len(agent._error_pattern_history) == 0

    def test_no_memorization_without_fix(self, agent):
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        # error → success (no fix in between)
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "cmd1"}}],
            ["Error: failed"],
        )
        agent.turn_count += 1
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "cmd1"}}],
            ["Success!"],
        )

        # Should NOT memorize — no fix event
        assert not agent.session_memory.put.called

    def test_shell_fix_only_if_command_changed(self, agent):
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        # error
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "python test.py"}}],
            ["Error: failed"],
        )
        agent.turn_count += 1

        # Same command again — should NOT count as fix
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "python test.py"}}],
            ["Error: failed again"],
        )

        fix_events = [e for e in agent._error_pattern_history if e["phase"] == "fix"]
        assert len(fix_events) == 0

    def test_shell_fix_if_command_changed(self, agent):
        """Shell command with different cmd is recorded as fix; if result signals success, memorizes."""
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        # error
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "python test.py"}}],
            ["Error: failed"],
        )
        agent.turn_count += 1

        # Different command with success result — fix recorded then memorized
        agent._track_error_pattern(
            [{"name": "shell", "arguments": {"command": "pip install torch"}}],
            ["Successfully installed torch"],
        )

        # State cleared after memorization
        assert len(agent._error_pattern_history) == 0
        assert agent.session_memory.put.called

    def test_limits_history_to_10_events(self, agent):
        """History is capped at 10 total events to avoid unbounded growth."""
        agent.history.get_messages = Mock(return_value=[])
        agent.session_memory.list_entries = Mock(return_value=[])

        # Generate 20 pure error events (same command each time so no fix is recorded)
        for i in range(20):
            agent._track_error_pattern(
                [{"name": "read_file", "arguments": {"path": f"/file{i}.py"}}],
                [f"Error {i}: file not found"],
            )
            agent.turn_count += 1

        # Should keep only the last 10
        assert len(agent._error_pattern_history) <= 10


class TestAutoMemorizeErrorPattern:
    def test_generates_valid_memory_key(self, agent):
        error_event = {
            "error_snippet": "ModuleNotFoundError: No module named 'torch'",
            "tool": "shell",
            "args": {"command": "python test.py"},
        }
        fix_event = {
            "tool": "shell",
            "args": {"command": "pip install torch"},
        }

        agent._auto_memorize_error_pattern(error_event, fix_event, "shell", "Success")

        assert agent.session_memory.put.called
        key = agent.session_memory.put.call_args[1]["key"]
        assert key.startswith("workaround_")
        assert len(key) <= 60

    def test_content_includes_error_and_fix(self, agent):
        error_event = {
            "error_snippet": "FileNotFoundError: config.yaml not found",
            "tool": "shell",
            "args": {"command": "python train.py"},
        }
        fix_event = {
            "tool": "write_file",
            "args": {"path": "config.yaml", "content": "..."},
        }

        agent._auto_memorize_error_pattern(error_event, fix_event, "shell", "Success")

        content = agent.session_memory.put.call_args[1]["content"]
        assert "FileNotFoundError" in content
        assert "write_file" in content
        assert "config.yaml" in content

    def test_handles_edit_file_fix(self, agent):
        error_event = {
            "error_snippet": "SyntaxError: invalid syntax",
            "tool": "shell",
            "args": {"command": "python script.py"},
        }
        fix_event = {
            "tool": "edit_file",
            "args": {"path": "script.py", "old_string": "print x", "new_string": "print(x)"},
        }

        agent._auto_memorize_error_pattern(error_event, fix_event, "shell", "Success")

        content = agent.session_memory.put.call_args[1]["content"]
        assert "edit_file" in content
        assert "print x" in content
        assert "print(x)" in content

    def test_handles_shell_fix(self, agent):
        error_event = {
            "error_snippet": "Permission denied",
            "tool": "shell",
            "args": {"command": "cat /root/file.txt"},
        }
        fix_event = {
            "tool": "shell",
            "args": {"command": "sudo cat /root/file.txt"},
        }

        agent._auto_memorize_error_pattern(error_event, fix_event, "shell", "Success")

        content = agent.session_memory.put.call_args[1]["content"]
        assert "sudo cat /root/file.txt" in content

    def test_does_not_crash_on_memory_write_failure(self, agent):
        agent.session_memory.put.side_effect = Exception("Memory write failed")

        error_event = {"error_snippet": "Error", "tool": "shell", "args": {}}
        fix_event = {"tool": "shell", "args": {"command": "fix"}}

        # Should not raise
        agent._auto_memorize_error_pattern(error_event, fix_event, "shell", "Success")
