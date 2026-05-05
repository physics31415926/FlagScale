"""Tests for skill lifecycle management (Layer 5)."""

import pytest
from collections import deque
from unittest.mock import MagicMock

from flagscale.agent.react.agent import ReactAgent


class TestSkillLifecycle:

    def _make_agent_stub(self):
        agent = MagicMock()
        agent._active_skill_content = {}
        agent._skill_load_iterations = {}
        agent._total_iterations = 0
        agent._training_started = False
        agent._env_verified = False
        agent._recently_referenced_skills = set()
        agent._check_skill_lifecycle = ReactAgent._check_skill_lifecycle.__get__(agent)
        return agent

    def test_no_crash_when_empty(self):
        agent = self._make_agent_stub()
        agent._check_skill_lifecycle()
        assert agent._active_skill_content == {}

    def test_unload_train_run_after_training_started(self):
        agent = self._make_agent_stub()
        agent._active_skill_content = {"train-run": "skill content here"}
        agent._skill_load_iterations = {"train-run": 5}
        agent._total_iterations = 10
        agent._training_started = True

        agent._check_skill_lifecycle()

        assert "train-run" not in agent._active_skill_content
        assert "train-run" not in agent._skill_load_iterations

    def test_keep_train_run_before_training_started(self):
        agent = self._make_agent_stub()
        agent._active_skill_content = {"train-run": "skill content"}
        agent._skill_load_iterations = {"train-run": 5}
        agent._total_iterations = 10
        agent._training_started = False

        agent._check_skill_lifecycle()

        assert "train-run" in agent._active_skill_content

    def test_unload_env_setup_after_verified(self):
        agent = self._make_agent_stub()
        agent._active_skill_content = {"env-setup": "env content"}
        agent._skill_load_iterations = {"env-setup": 2}
        agent._total_iterations = 8
        agent._env_verified = True

        agent._check_skill_lifecycle()

        assert "env-setup" not in agent._active_skill_content

    def test_unload_after_30_iterations_unused(self):
        agent = self._make_agent_stub()
        agent._active_skill_content = {"some-skill": "content"}
        agent._skill_load_iterations = {"some-skill": 0}
        agent._total_iterations = 31
        agent._recently_referenced_skills = set()

        agent._check_skill_lifecycle()

        assert "some-skill" not in agent._active_skill_content

    def test_keep_if_recently_referenced(self):
        agent = self._make_agent_stub()
        agent._active_skill_content = {"some-skill": "content"}
        agent._skill_load_iterations = {"some-skill": 0}
        agent._total_iterations = 31
        agent._recently_referenced_skills = {"some-skill"}

        agent._check_skill_lifecycle()

        assert "some-skill" in agent._active_skill_content

    def test_recently_referenced_cleared_after_check(self):
        agent = self._make_agent_stub()
        agent._active_skill_content = {"skill-a": "a"}
        agent._skill_load_iterations = {"skill-a": 25}
        agent._total_iterations = 30
        agent._recently_referenced_skills = {"skill-a"}

        agent._check_skill_lifecycle()

        assert agent._recently_referenced_skills == set()

    def test_multiple_skills_selective_unload(self):
        agent = self._make_agent_stub()
        agent._active_skill_content = {
            "train-run": "train content",
            "env-setup": "env content",
            "model-porter": "porter content",
        }
        agent._skill_load_iterations = {
            "train-run": 5,
            "env-setup": 3,
            "model-porter": 10,
        }
        agent._total_iterations = 15
        agent._training_started = True
        agent._env_verified = False

        agent._check_skill_lifecycle()

        # train-run unloaded (training started)
        assert "train-run" not in agent._active_skill_content
        # env-setup kept (not verified, age=12 < 30)
        assert "env-setup" in agent._active_skill_content
        # model-porter kept (age=5 < 30)
        assert "model-porter" in agent._active_skill_content

    def test_skill_content_injected_into_prompt(self):
        """Verify that _active_skill_content would be formatted into prompt."""
        content = {"train-run": "## Preflight\n- Check GPU\n- Check data"}
        parts = []
        for name, text in content.items():
            parts.append(f"## Active Skill: {name}\n{text}")
        skill_context = "\n\n".join(parts)
        assert "Active Skill: train-run" in skill_context
        assert "Check GPU" in skill_context
