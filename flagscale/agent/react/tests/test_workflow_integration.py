"""Integration tests: verify skill_judge selects correct skills for various user inputs."""

import os

import pytest

from unittest.mock import patch, MagicMock

from flagscale.agent.react.agent import ReactAgent
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.skills import SkillManager


SKILLS_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "skills"
)
SKILLS_DIR = os.path.normpath(SKILLS_DIR)


def _make_config(tmp_path):
    return AgentConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-key",
        max_iterations=5,
        max_cost=0.0,
        session_dir=str(tmp_path / "sessions"),
        skill_dirs=[SKILLS_DIR],
        plugin_tool_dirs=[],
    )


def _mock_provider():
    provider = MagicMock()
    provider.supports_tools = True
    provider.format_system_message.return_value = {"role": "system", "content": "sys"}
    provider.format_user_message.return_value = {"role": "user", "content": "msg"}
    provider.format_assistant_message.return_value = {"role": "assistant", "content": "ok"}
    provider.format_tool_result.return_value = {"role": "tool", "content": "result"}
    return provider


SKILL_TRIGGER_CASES = [
    # (user_input, expected_skill_in_response)
    ("help me install FlagScale and set up the training environment", "env-setup"),
    ("detect the hardware topology on this server", "topo-detect"),
    ("port Qwen3-0.6B model to FlagScale", "model-porter"),
    ("preprocess the training data into Megatron binary format", "data-prep"),
    ("generate the training YAML config with TP=4 PP=2", "train-config"),
    ("start training on 8 GPUs", "train-run"),
    ("check the training loss and monitor for anomalies", "train-monitor"),
    ("reproduce the original training results as baseline", "reproduce"),
    ("align the loss curves between original and FlagScale", "precision-alignment"),
]


class TestSkillJudgeIntegration:
    """Test that _skill_judge returns the expected skill for realistic user inputs.

    These tests use the real skill list from the skills directory but mock the
    LLM response to verify the filtering and validation logic.
    """

    @pytest.mark.parametrize("user_input,expected_skill", SKILL_TRIGGER_CASES)
    def test_skill_judge_returns_expected(self, tmp_path, user_input, expected_skill):
        config = _make_config(tmp_path)
        provider = _mock_provider()
        provider.chat.return_value = {"content": f'{{"skills": ["{expected_skill}"]}}'}

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        result = agent._skill_judge(user_input)
        assert expected_skill in result, (
            f"Expected '{expected_skill}' for input '{user_input}', got {result}"
        )

    def test_skill_judge_filters_nonexistent(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider()
        provider.chat.return_value = {"content": '{"skills": ["train-run", "fake-skill"]}'}

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        result = agent._skill_judge("start training")
        assert "train-run" in result
        assert "fake-skill" not in result

    def test_skill_judge_empty_for_irrelevant(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider()
        provider.chat.return_value = {"content": '{"skills": []}'}

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        result = agent._skill_judge("what is the weather today")
        assert result == []

    def test_auto_load_skills_adds_to_loaded_set(self, tmp_path):
        config = _make_config(tmp_path)
        provider = _mock_provider()
        provider.chat.return_value = {"content": '{"skills": ["env-setup"]}'}

        with patch("flagscale.agent.react.agent.get_provider", return_value=provider):
            agent = ReactAgent(config)

        assert "env-setup" not in agent._loaded_skills
        agent._auto_load_skills("install FlagScale environment on this server")
        assert "env-setup" in agent._loaded_skills


class TestSkillManagerIntegration:
    """Test SkillManager with real skill files."""

    def test_all_skills_listed(self):
        mgr = SkillManager([SKILLS_DIR])
        skills = mgr.list_skills()
        names = {s["name"] for s in skills}
        expected = {
            "env-setup", "topo-detect", "model-porter", "data-prep",
            "train-config", "train-run", "train-monitor", "reproduce",
            "precision-alignment",
        }
        assert expected.issubset(names)

    def test_all_skills_have_keywords(self):
        mgr = SkillManager([SKILLS_DIR])
        for skill in mgr.list_skills():
            assert len(skill.get("keywords", [])) > 0, (
                f"Skill '{skill['name']}' has no keywords"
            )

    def test_skill_load_returns_xml_wrapped(self):
        mgr = SkillManager([SKILLS_DIR])
        for skill_info in mgr.list_skills():
            name = skill_info["name"]
            content = mgr.load(name)
            assert content.startswith(f'<skill name="{name}">'), (
                f"Skill '{name}' load output missing XML wrapper"
            )
            assert content.endswith("</skill>")
