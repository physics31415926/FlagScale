"""Tests for efficiency hints (batching and memory)."""

import pytest
from unittest.mock import Mock
from flagscale.agent.react.agent import WorkerAgent
from flagscale.agent.react.config import AgentConfig


@pytest.fixture
def mock_agent():
    """Create a minimal WorkerAgent with mocked dependencies."""
    config = AgentConfig(provider="mock", model="test")
    agent = WorkerAgent.__new__(WorkerAgent)
    agent.config = config
    agent.session_memory = Mock()
    agent.history = Mock()
    agent._inject_message = Mock()
    return agent


def test_batching_hint_fires_after_3_single_calls(mock_agent):
    """Batching hint should fire when 3+ consecutive turns each had 1 tool call."""
    # Simulate history with 3 single-tool assistant turns
    mock_agent.history.get_messages = Mock(return_value=[
        {"role": "user", "content": "task 1"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file"}]},
        {"role": "user", "content": "result 1"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "shell"}]},
        {"role": "user", "content": "result 2"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file"}]},
    ])
    
    tool_calls = [{"name": "read_file", "arguments": {}}]
    results = ["some output"]
    
    mock_agent._inject_efficiency_hints(tool_calls, results)
    
    # Should inject batching hint
    assert mock_agent._inject_message.called
    call_args = mock_agent._inject_message.call_args[0][0]
    assert "[Efficiency]" in call_args
    assert "batch" in call_args.lower()


def test_batching_hint_not_fired_for_multi_tool_turn(mock_agent):
    """Batching hint should NOT fire if current turn has multiple tool calls."""
    mock_agent.history.get_messages = Mock(return_value=[
        {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file"}]},
    ])
    
    tool_calls = [
        {"name": "read_file", "arguments": {}},
        {"name": "shell", "arguments": {}},
    ]
    results = ["output1", "output2"]
    
    mock_agent._inject_efficiency_hints(tool_calls, results)
    
    # Should NOT inject batching hint (current turn already batched)
    if mock_agent._inject_message.called:
        call_args = mock_agent._inject_message.call_args[0][0]
        assert "[Efficiency]" not in call_args or "batch" not in call_args.lower()


def test_memory_hint_fires_for_large_file(mock_agent):
    """Memory hint should fire when reading a large file (>100 lines) not in memory."""
    mock_agent.session_memory.list_entries = Mock(return_value=[])
    # Empty history so batching hint doesn't fire
    mock_agent.history.get_messages = Mock(return_value=[])

    tool_calls = [{"name": "read_file", "arguments": {"path": "/tmp/large.py"}}]
    results = ["[/tmp/large.py] lines 1-50 of 250\n...content..."]

    mock_agent._inject_efficiency_hints(tool_calls, results)

    # Should inject memory hint
    assert mock_agent._inject_message.called
    call_args = mock_agent._inject_message.call_args[0][0]
    assert "[Memory]" in call_args
    assert "250 lines" in call_args
    assert "memory_write" in call_args


def test_memory_hint_not_fired_for_small_file(mock_agent):
    """Memory hint should NOT fire for files <100 lines."""
    mock_agent.session_memory.list_entries = Mock(return_value=[])
    mock_agent.history.get_messages = Mock(return_value=[])

    tool_calls = [{"name": "read_file", "arguments": {"path": "/tmp/small.py"}}]
    results = ["[/tmp/small.py] lines 1-50 of 50\n...content..."]

    mock_agent._inject_efficiency_hints(tool_calls, results)

    # Should NOT inject memory hint
    assert not mock_agent._inject_message.called


def test_memory_hint_not_fired_if_already_memorized(mock_agent):
    """Memory hint should NOT fire if the file is already in memory."""
    mock_agent.session_memory.list_entries = Mock(return_value=[
        {"key": "large_py_structure", "content": "large.py Contains class definitions..."}
    ])
    mock_agent.history.get_messages = Mock(return_value=[])

    tool_calls = [{"name": "read_file", "arguments": {"path": "/tmp/large.py"}}]
    results = ["[/tmp/large.py] lines 1-50 of 250\n...content..."]

    mock_agent._inject_efficiency_hints(tool_calls, results)

    # Should NOT inject memory hint (already memorized)
    assert not mock_agent._inject_message.called


def test_count_recent_single_tool_turns(mock_agent):
    """Test _count_recent_single_tool_turns helper."""
    mock_agent.history.get_messages = Mock(return_value=[
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file"}]},
        {"role": "user", "content": "result"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "shell"}]},
        {"role": "user", "content": "result"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file"}]},
        {"role": "user", "content": "result"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "read_file"},
            {"type": "tool_use", "name": "shell"},
        ]},  # Multi-tool turn breaks the streak
        {"role": "user", "content": "result"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file"}]},
    ])
    
    count = mock_agent._count_recent_single_tool_turns(lookback=5)
    
    # Should count 1 (only the last single-tool turn before the multi-tool break)
    assert count == 1
