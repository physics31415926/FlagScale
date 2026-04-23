"""Tests for SkillManager."""

import os

import pytest

from flagscale.agent.react.skills import SkillManager


@pytest.fixture
def skill_dirs(tmp_path):
    """Create two skill directories with test skills."""
    dir1 = tmp_path / "builtin"
    dir2 = tmp_path / "user"
    dir1.mkdir()
    dir2.mkdir()

    # Skill in builtin dir
    s1 = dir1 / "my_skill"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        "---\nname: my_skill\ndescription: A test skill\n---\nDo something useful."
    )

    # Skill in user dir (overrides builtin)
    s2 = dir2 / "my_skill"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(
        "---\nname: my_skill\ndescription: User override\n---\nUser version."
    )

    # Another skill only in user dir
    s3 = dir2 / "extra"
    s3.mkdir()
    (s3 / "SKILL.md").write_text(
        "---\nname: extra\ndescription: Extra skill\n---\nExtra content."
    )

    return [str(dir1), str(dir2)]


class TestSkillManager:
    def test_list_skills(self, skill_dirs):
        mgr = SkillManager(skill_dirs)
        skills = mgr.list_skills()
        names = {s["name"] for s in skills}
        assert "my_skill" in names
        assert "extra" in names

    def test_load_priority(self, skill_dirs):
        """Later directories take priority."""
        mgr = SkillManager(skill_dirs)
        content = mgr.load("my_skill")
        assert "User version" in content

    def test_load_missing(self, skill_dirs):
        mgr = SkillManager(skill_dirs)
        with pytest.raises(FileNotFoundError):
            mgr.load("nonexistent")

    def test_empty_dirs(self):
        mgr = SkillManager(["/nonexistent/path"])
        assert mgr.list_skills() == []

    def test_parse_frontmatter_no_yaml(self):
        meta, body = SkillManager._parse_frontmatter("Just plain text")
        assert meta == {}
        assert body == "Just plain text"

    def test_parse_frontmatter_valid(self):
        content = "---\nname: test\ndescription: desc\n---\nBody here."
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta["name"] == "test"
        assert body == "Body here."

    def test_parse_frontmatter_bad_yaml(self):
        content = "---\n: invalid: yaml: {{{\n---\nBody."
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta == {}
        assert body == "Body."
