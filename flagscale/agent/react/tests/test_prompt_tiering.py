"""Tests for system prompt tiering (Layer 4)."""

import pytest
from collections import deque
from unittest.mock import MagicMock, patch

from flagscale.agent.react.agent import (
    SYSTEM_PROMPT_CORE,
    SYSTEM_PROMPT_OPTIONAL,
    ReactAgent,
)


class TestSystemPromptCore:

    def test_core_prompt_has_all_placeholders(self):
        placeholders = [
            "{skills}", "{cwd}", "{plan_context}", "{memory_context}",
            "{workspace_context}", "{situational_context}",
            "{optional_sections}", "{skill_context}",
        ]
        for p in placeholders:
            assert p in SYSTEM_PROMPT_CORE, f"Missing placeholder: {p}"

    def test_core_prompt_formats_without_error(self):
        result = SYSTEM_PROMPT_CORE.format(
            skills="- test-skill: does things",
            cwd="/tmp",
            plan_context="",
            memory_context="",
            workspace_context="",
            situational_context="",
            optional_sections="",
            skill_context="",
        )
        assert "FlagScale Agent" in result
        assert "/tmp" in result

    def test_core_prompt_no_unescaped_braces(self):
        """Ensure regex patterns like {4-7} are properly escaped as {{4-7}}."""
        # This should not raise KeyError
        try:
            SYSTEM_PROMPT_CORE.format(
                skills="", cwd="", plan_context="", memory_context="",
                workspace_context="", situational_context="",
                optional_sections="", skill_context="",
            )
        except (KeyError, ValueError) as e:
            pytest.fail(f"Unescaped brace in SYSTEM_PROMPT_CORE: {e}")


class TestOptionalSections:

    def test_all_sections_are_strings(self):
        for name, content in SYSTEM_PROMPT_OPTIONAL.items():
            assert isinstance(content, str), f"Section '{name}' is not a string"
            assert len(content) > 10, f"Section '{name}' is too short"

    def test_optional_sections_have_headers(self):
        for name, content in SYSTEM_PROMPT_OPTIONAL.items():
            assert content.startswith("##"), f"Section '{name}' should start with ##"


class TestGetOptionalSections:

    def _make_agent_stub(self):
        agent = MagicMock()
        agent._turn_iteration_count = 0
        agent._last_tool_calls_deque = deque(maxlen=5)
        agent._porting_mode = False
        agent._last_tool_had_error = False
        agent._consecutive_train_failures = 0
        agent._last_compaction_count = 0
        agent._PHASE_TOOL_SETS = ReactAgent._PHASE_TOOL_SETS
        agent._CORE_TOOLS = ReactAgent._CORE_TOOLS
        agent._extra_tools_next_iter = set()
        agent.task_plan = MagicMock()
        agent.task_plan.get_active.return_value = None
        agent._detect_tool_phase = ReactAgent._detect_tool_phase.__get__(agent)
        agent._get_optional_sections = ReactAgent._get_optional_sections.__get__(agent)
        return agent

    def test_first_iteration_includes_guidance(self):
        agent = self._make_agent_stub()
        agent._turn_iteration_count = 0
        sections = agent._get_optional_sections()
        assert "planning" in sections
        assert "memory_rules" in sections
        assert "experiment" in sections

    def test_later_iteration_minimal(self):
        agent = self._make_agent_stub()
        agent._turn_iteration_count = 10
        sections = agent._get_optional_sections()
        # No active plan, no errors, no porting — should be minimal
        assert "planning" not in sections
        assert "porting" not in sections
        assert "user_commands" not in sections

    def test_active_plan_includes_planning(self):
        agent = self._make_agent_stub()
        agent._turn_iteration_count = 10
        agent.task_plan.get_active.return_value = {"name": "test-plan"}
        sections = agent._get_optional_sections()
        assert "planning" in sections

    def test_porting_mode_includes_porting(self):
        agent = self._make_agent_stub()
        agent._turn_iteration_count = 10
        agent._porting_mode = True
        sections = agent._get_optional_sections()
        assert "porting" in sections

    def test_errors_include_decision(self):
        agent = self._make_agent_stub()
        agent._turn_iteration_count = 10
        agent._last_tool_had_error = True
        sections = agent._get_optional_sections()
        assert "decision" in sections

    def test_user_commands_only_first_iteration(self):
        agent = self._make_agent_stub()
        agent._turn_iteration_count = 1
        sections = agent._get_optional_sections()
        assert "user_commands" in sections

        agent._turn_iteration_count = 2
        sections = agent._get_optional_sections()
        assert "user_commands" not in sections

    def test_monitoring_phase_minimal(self):
        agent = self._make_agent_stub()
        agent._turn_iteration_count = 10
        agent._last_tool_calls_deque.append("shell")
        agent._last_tool_calls_deque.append("monitor")
        sections = agent._get_optional_sections()
        # Monitoring phase shouldn't add planning or experiment
        assert "planning" not in sections
        assert "experiment" not in sections
