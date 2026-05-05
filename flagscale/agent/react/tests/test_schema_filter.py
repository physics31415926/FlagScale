"""Tests for phase-based schema filtering (Layer 3)."""

import pytest
from collections import deque
from unittest.mock import MagicMock, patch

from flagscale.agent.react.tools import ToolRegistry
from flagscale.agent.react.tools.base import Tool


class DummyTool(Tool):
    """Minimal tool for testing schema filtering."""

    def __init__(self, name, description="test tool"):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        return "ok"


class TestToolRegistryFiltered:

    def test_to_schemas_filtered_returns_subset(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        reg.register(DummyTool("read_file"))
        reg.register(DummyTool("write_file"))
        reg.register(DummyTool("monitor"))
        reg.register(DummyTool("plan_create"))

        schemas = reg.to_schemas_filtered("openai", {"shell", "monitor"})
        names = {s["function"]["name"] for s in schemas}
        assert names == {"shell", "monitor"}

    def test_to_schemas_filtered_empty_set(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        schemas = reg.to_schemas_filtered("openai", set())
        assert schemas == []

    def test_to_schemas_filtered_unknown_names_ignored(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        schemas = reg.to_schemas_filtered("openai", {"shell", "nonexistent"})
        assert len(schemas) == 1

    def test_to_schemas_filtered_anthropic_format(self):
        reg = ToolRegistry()
        reg.register(DummyTool("shell"))
        reg.register(DummyTool("monitor"))
        schemas = reg.to_schemas_filtered("anthropic", {"shell"})
        assert len(schemas) == 1
        assert schemas[0]["name"] == "shell"


class TestPhaseDetection:
    """Test _detect_tool_phase logic in isolation."""

    def _make_agent_stub(self):
        """Create a minimal object with the phase detection method."""
        from flagscale.agent.react.agent import ReactAgent

        # We'll test the method logic directly by creating a mock
        agent = MagicMock()
        agent._last_tool_calls_deque = deque(maxlen=5)
        agent._PHASE_TOOL_SETS = ReactAgent._PHASE_TOOL_SETS
        agent._CORE_TOOLS = ReactAgent._CORE_TOOLS
        agent._extra_tools_next_iter = set()
        agent._detect_tool_phase = ReactAgent._detect_tool_phase.__get__(agent)
        agent._get_filtered_schemas = ReactAgent._get_filtered_schemas.__get__(agent)
        return agent

    def test_default_phase_when_empty(self):
        agent = self._make_agent_stub()
        assert agent._detect_tool_phase() == "default"

    def test_monitoring_phase(self):
        agent = self._make_agent_stub()
        agent._last_tool_calls_deque.append("shell")
        agent._last_tool_calls_deque.append("monitor")
        assert agent._detect_tool_phase() == "monitoring"

    def test_planning_phase(self):
        agent = self._make_agent_stub()
        agent._last_tool_calls_deque.append("plan_create")
        agent._last_tool_calls_deque.append("plan_update")
        agent._last_tool_calls_deque.append("shell")
        assert agent._detect_tool_phase() == "planning"

    def test_training_phase(self):
        agent = self._make_agent_stub()
        agent._last_tool_calls_deque.append("workspace_experiment")
        agent._last_tool_calls_deque.append("parse_training_metrics")
        assert agent._detect_tool_phase() == "training"

    def test_default_phase_mixed_tools(self):
        agent = self._make_agent_stub()
        agent._last_tool_calls_deque.append("shell")
        agent._last_tool_calls_deque.append("read_file")
        agent._last_tool_calls_deque.append("write_file")
        assert agent._detect_tool_phase() == "default"

    def test_filtered_schemas_monitoring(self):
        agent = self._make_agent_stub()
        reg = ToolRegistry()
        for name in ["shell", "read_file", "monitor", "write_file", "plan_create",
                     "parse_training_metrics", "workspace_experiment"]:
            reg.register(DummyTool(name))
        agent.tool_registry = reg
        agent.provider = MagicMock()
        agent.provider.schema_format = "openai"

        schemas = agent._get_filtered_schemas("monitoring")
        names = {s["function"]["name"] for s in schemas}
        # monitoring set: monitor, shell, read_file, parse_training_metrics
        assert "monitor" in names
        assert "shell" in names
        assert "read_file" in names
        assert "write_file" not in names
        assert "plan_create" not in names

    def test_filtered_schemas_default_returns_all(self):
        agent = self._make_agent_stub()
        reg = ToolRegistry()
        for name in ["shell", "read_file", "monitor", "write_file", "plan_create"]:
            reg.register(DummyTool(name))
        agent.tool_registry = reg
        agent.provider = MagicMock()
        agent.provider.schema_format = "openai"

        schemas = agent._get_filtered_schemas("default")
        assert len(schemas) == 5

    def test_extra_tools_next_iter_included(self):
        agent = self._make_agent_stub()
        agent._extra_tools_next_iter = {"write_file"}
        reg = ToolRegistry()
        for name in ["shell", "read_file", "monitor", "write_file"]:
            reg.register(DummyTool(name))
        agent.tool_registry = reg
        agent.provider = MagicMock()
        agent.provider.schema_format = "openai"

        schemas = agent._get_filtered_schemas("monitoring")
        names = {s["function"]["name"] for s in schemas}
        assert "write_file" in names
