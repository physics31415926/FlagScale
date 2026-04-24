"""Tests for ReactAgent core loop."""

import json
import os
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from flagscale.agent.react.agent import ReactAgent, _PluginShellTool
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.memory import SessionMemory


def _make_config(tmp_path, **overrides):
    defaults = dict(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        max_iterations=5,
        max_cost=0.0,
        session_dir=str(tmp_path / "sessions"),
        skill_dirs=[str(tmp_path / "skills")],
        plugin_tool_dirs=[],
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


class FakeStreamEvent:
    """Helper to build streaming events."""
    @staticmethod
    def text(content):
        return {"type": "text", "content": content}

    @staticmethod
    def tool_start(id, name):
        return {"type": "tool_start", "id": id, "name": name}

    @staticmethod
    def tool_delta(id, args_delta):
        return {"type": "tool_delta", "id": id, "arguments_delta": args_delta}

    @staticmethod
    def usage(input_tokens=100, output_tokens=50):
        return {"type": "usage", "input_tokens": input_tokens, "output_tokens": output_tokens}

    @staticmethod
    def done():
        return {"type": "done"}


def _mock_provider_text_only(text="Hello!"):
    """Create a mock provider that returns a text-only response."""
    provider = MagicMock()
    provider.schema_format = "openai"

    def stream_fn(messages, tools):
        yield FakeStreamEvent.text(text)
        yield FakeStreamEvent.usage()
        yield FakeStreamEvent.done()

    provider.chat_stream.side_effect = stream_fn
    provider.format_assistant_message.return_value = {"role": "assistant", "content": text}
    provider.format_tool_result.side_effect = lambda tid, content: {"role": "tool", "tool_call_id": tid, "content": content}
    return provider


def _mock_provider_with_tool_call(tool_name="shell", tool_args='{"command": "echo hi"}', tool_result_text="Then done."):
    """Create a mock provider: first call returns tool use, second returns text."""
    provider = MagicMock()
    provider.schema_format = "openai"
    call_count = [0]

    def stream_fn(messages, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            yield FakeStreamEvent.tool_start("tc1", tool_name)
            yield FakeStreamEvent.tool_delta("tc1", tool_args)
            yield FakeStreamEvent.usage()
            yield FakeStreamEvent.done()
        else:
            yield FakeStreamEvent.text(tool_result_text)
            yield FakeStreamEvent.usage()
            yield FakeStreamEvent.done()

    provider.chat_stream.side_effect = stream_fn
    provider.format_assistant_message.side_effect = lambda r: {"role": "assistant", "content": r.get("content", "")}
    provider.format_tool_result.side_effect = lambda tid, content: {"role": "tool", "tool_call_id": tid, "content": content}
    return provider


class TestReactAgentLoop:
    def test_text_only_response(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only("Hello!")

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        agent.provider = provider
        agent.history.append({"role": "user", "content": "Hi"})
        agent._react_loop()

        assert agent._turn_count == 1
        assert provider.chat_stream.call_count == 1

    def test_tool_call_dispatches(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_with_tool_call("shell", '{"command": "echo hi"}')

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        agent.provider = provider
        agent.history.append({"role": "user", "content": "run echo"})
        agent._react_loop()

        assert provider.chat_stream.call_count == 2

    def test_max_iterations_stops(self, tmp_path):
        config = _make_config(tmp_path, max_iterations=2)
        provider = MagicMock()
        provider.schema_format = "openai"

        def always_tool(messages, tools):
            yield FakeStreamEvent.tool_start("tc1", "shell")
            yield FakeStreamEvent.tool_delta("tc1", '{"command": "echo loop"}')
            yield FakeStreamEvent.usage()
            yield FakeStreamEvent.done()

        provider.chat_stream.side_effect = always_tool
        provider.format_assistant_message.side_effect = lambda r: {"role": "assistant", "content": ""}
        provider.format_tool_result.side_effect = lambda tid, c: {"role": "tool", "tool_call_id": tid, "content": c}

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        agent.provider = provider
        agent.history.append({"role": "user", "content": "loop"})
        with patch("builtins.input", return_value="n"):
            agent._react_loop()

        assert provider.chat_stream.call_count == 2

    def test_autosave_written_after_turn(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only("Done.")

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        agent.provider = provider
        agent.history.append({"role": "user", "content": "test"})
        agent._react_loop()

        autosave_path = agent._autosave_path()
        assert os.path.isfile(autosave_path)
        with open(autosave_path) as f:
            data = json.load(f)
        assert data["id"] == "autosave"
        assert data["metadata"]["turns"] == 1

    def test_exit_clears_autosave(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only("Done.")

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        agent.provider = provider
        agent.history.append({"role": "user", "content": "test"})
        agent._react_loop()

        autosave_path = agent._autosave_path()
        assert os.path.isfile(autosave_path)

        agent._exit()
        assert not os.path.isfile(autosave_path)


class TestResultJudge:
    """Test _result_judge with mocked LLM provider."""

    def test_clean_output_returns_empty(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.return_value = {"content": '{"annotations": [], "severity": "info"}'}
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        result = agent._result_judge("ls -la", "total 0\ndrwxr-xr-x 2 root root", 0.1)
        assert result == []
        provider.chat.assert_called_once()

    def test_calls_llm_on_error_output(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.return_value = {"content": '{"annotations": ["CUDA driver mismatch"], "severity": "error"}'}
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        result = agent._result_judge("python train.py", "RuntimeError: CUDA error: driver", 5.0)
        assert len(result) == 1
        assert "CUDA" in result[0]

    def test_calls_llm_on_long_duration(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.return_value = {"content": '{"annotations": ["Command took unusually long"], "severity": "warning"}'}
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        result = agent._result_judge("rm -rf /tmp/cache", "removed", 130.0)
        assert len(result) >= 1

    def test_graceful_on_llm_failure(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.side_effect = Exception("API error")
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        result = agent._result_judge("pip install torch", "Successfully installed torch", 30.0)
        assert result == []


class TestSkillJudge:
    """Test _skill_judge with mocked LLM provider."""

    def test_returns_matching_skill(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.return_value = {"content": '{"skills": ["train-run"]}'}
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        agent.skill_manager = MagicMock()
        agent.skill_manager.list_skills.return_value = [
            {"name": "train-run", "description": "Launch and manage training runs"},
            {"name": "env-setup", "description": "Setup environment"},
        ]
        result = agent._skill_judge("train Qwen3-0.6B on 8 GPUs")
        assert result == ["train-run"]

    def test_returns_empty_when_no_match(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.return_value = {"content": '{"skills": []}'}
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        agent.skill_manager = MagicMock()
        agent.skill_manager.list_skills.return_value = [
            {"name": "train-run", "description": "Launch and manage training runs"},
        ]
        result = agent._skill_judge("what time is it")
        assert result == []

    def test_filters_invalid_skill_names(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.return_value = {"content": '{"skills": ["train-run", "nonexistent-skill"]}'}
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        agent.skill_manager = MagicMock()
        agent.skill_manager.list_skills.return_value = [
            {"name": "train-run", "description": "Launch and manage training runs"},
        ]
        result = agent._skill_judge("train a model")
        assert result == ["train-run"]

    def test_graceful_on_llm_failure(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.side_effect = Exception("API error")
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        agent.skill_manager = MagicMock()
        agent.skill_manager.list_skills.return_value = [
            {"name": "train-run", "description": "Launch and manage training runs"},
        ]
        result = agent._skill_judge("train a model")
        assert result == []

    def test_uses_conversation_context(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        provider.chat.return_value = {"content": '{"skills": ["train-run"]}'}
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        agent.skill_manager = MagicMock()
        agent.skill_manager.list_skills.return_value = [
            {"name": "train-run", "description": "Launch and manage training runs"},
        ]
        agent.history.append({"role": "user", "content": "I want to train Qwen3"})
        agent.history.append({"role": "assistant", "content": "Sure, let me help."})
        result = agent._skill_judge("start it on 8 GPUs")
        assert result == ["train-run"]
        call_args = provider.chat.call_args[0][0][0]["content"]
        assert "train Qwen3" in call_args


class TestProxyConnectivity:
    """Test _test_proxy method."""

    def test_proxy_success(self, tmp_path, capsys):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        fake_result = MagicMock()
        fake_result.stdout = "200"
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result):
            agent._test_proxy("http://proxy:8080")
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "200" in captured.out

    def test_proxy_failure(self, tmp_path, capsys):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        fake_result = MagicMock()
        fake_result.stdout = ""
        fake_result.stderr = "Could not resolve proxy"
        with patch("subprocess.run", return_value=fake_result):
            agent._test_proxy("http://badproxy:9999")
        captured = capsys.readouterr()
        assert "✗" in captured.out

    def test_proxy_timeout(self, tmp_path, capsys):
        import subprocess
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("curl", 20)):
            agent._test_proxy("http://slowproxy:8080")
        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "timed out" in captured.out


class TestCheckAutosave:
    def test_resume_restores_state(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        # Write a fake autosave
        autosave_path = agent._autosave_path()
        os.makedirs(os.path.dirname(autosave_path), exist_ok=True)
        data = {
            "id": "autosave",
            "timestamp": time.time(),
            "metadata": {
                "turns": 3,
                "loaded_skills": ["train-run"],
                "input_tokens": 5000,
                "output_tokens": 800,
            },
            "messages": [
                {"role": "user", "content": "start training"},
                {"role": "assistant", "content": "OK, starting..."},
            ],
        }
        with open(autosave_path, "w") as f:
            json.dump(data, f)

        with patch("builtins.input", return_value="y"):
            agent._check_autosave()

        assert agent._turn_count == 3
        assert "train-run" in agent._loaded_skills
        assert agent._session_input_tokens == 5000

    def test_decline_clears_autosave(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        autosave_path = agent._autosave_path()
        os.makedirs(os.path.dirname(autosave_path), exist_ok=True)
        data = {
            "id": "autosave",
            "timestamp": time.time(),
            "metadata": {"turns": 1, "loaded_skills": [], "input_tokens": 0, "output_tokens": 0},
            "messages": [{"role": "user", "content": "hi"}],
        }
        with open(autosave_path, "w") as f:
            json.dump(data, f)

        with patch("builtins.input", return_value="n"):
            agent._check_autosave()

        assert not os.path.isfile(autosave_path)
        assert agent._turn_count == 0

    def test_no_autosave_file(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        agent._check_autosave()
        assert agent._turn_count == 0


class TestPluginShellToolInjection:
    def test_shlex_quote_prevents_injection(self, tmp_path):
        marker = tmp_path / "pwned.txt"
        spec = {
            "name": "test_tool",
            "command": "echo {input}",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
        }
        tool = _PluginShellTool(spec)
        tool.execute(input=f"; touch {marker}")
        assert not marker.exists()

    def test_normal_input_works(self):
        spec = {
            "name": "test_tool",
            "command": "echo {msg}",
            "parameters": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
            },
        }
        tool = _PluginShellTool(spec)
        result = tool.execute(msg="hello world")
        assert "hello world" in result


class TestEnsureMemoryWritten:
    """Tests for LLM-based memory judge in _ensure_memory_written."""

    def _make_agent(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider_text_only()
        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)
        agent.provider = provider
        # Use isolated memory directory to avoid cross-test pollution
        agent.session_memory = SessionMemory(str(tmp_path / "memory"))
        return agent

    def _populate_conversation(self, agent, user_msg="Please check the status and memory usage of all GPUs on the server", assistant_msg="All 8 GPUs are idle with 0% memory utilization"):
        agent._turn_count = 2
        agent.history.append({"role": "user", "content": user_msg})
        agent.history.append({"role": "assistant", "content": assistant_msg})

    def test_llm_says_save(self, tmp_path):
        agent = self._make_agent(tmp_path)
        self._populate_conversation(agent)
        llm_response = json.dumps({"save": True, "key": "gpu_status", "type": "finding", "content": "All 8 GPUs idle"})
        agent.provider.chat.return_value = {"content": llm_response, "tool_calls": None}

        agent._ensure_memory_written()

        entries = agent.session_memory.list_entries()
        assert len(entries) == 1
        assert entries[0]["key"] == "gpu_status"
        assert entries[0]["type"] == "finding"
        assert entries[0]["content"] == "All 8 GPUs idle"

    def test_llm_says_skip(self, tmp_path):
        agent = self._make_agent(tmp_path)
        self._populate_conversation(agent, user_msg="do not remember this conversation")
        agent.provider.chat.return_value = {"content": '{"save": false}', "tool_calls": None}

        agent._ensure_memory_written()

        entries = agent.session_memory.list_entries()
        assert len(entries) == 0

    def test_llm_error_skips_silently(self, tmp_path):
        agent = self._make_agent(tmp_path)
        self._populate_conversation(agent)
        agent.provider.chat.side_effect = Exception("API timeout")

        agent._ensure_memory_written()

        entries = agent.session_memory.list_entries()
        assert len(entries) == 0

    def test_llm_returns_invalid_json_skips(self, tmp_path):
        agent = self._make_agent(tmp_path)
        self._populate_conversation(agent)
        agent.provider.chat.return_value = {"content": "I think we should save this", "tool_calls": None}

        agent._ensure_memory_written()

        entries = agent.session_memory.list_entries()
        assert len(entries) == 0

    def test_skips_when_memory_already_written(self, tmp_path):
        agent = self._make_agent(tmp_path)
        self._populate_conversation(agent)
        # Simulate LLM having already written memory this session
        agent.session_memory.put("existing", "finding", "already here", agent._session_id)

        agent._ensure_memory_written()

        # provider.chat should NOT have been called for judging
        agent.provider.chat.assert_not_called()

    def test_skips_short_sessions(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent._turn_count = 1  # Only 1 turn
        agent.history.append({"role": "user", "content": "hi"})

        agent._ensure_memory_written()

        agent.provider.chat.assert_not_called()

    def test_invalid_type_defaults_to_context(self, tmp_path):
        agent = self._make_agent(tmp_path)
        self._populate_conversation(agent)
        llm_response = json.dumps({"save": True, "key": "test", "type": "bogus_type", "content": "some content"})
        agent.provider.chat.return_value = {"content": llm_response, "tool_calls": None}

        agent._ensure_memory_written()

        entries = agent.session_memory.list_entries()
        assert len(entries) == 1
        assert entries[0]["type"] == "context"
