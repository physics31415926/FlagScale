"""Tests for PathValidationGuard."""

import os
import tempfile
import pytest
from flagscale.agent.react.guard.path_validation import PathValidationGuard
from flagscale.agent.react.guard import GuardContext, GuardVerdict


def _ctx(cmd: str) -> GuardContext:
    return GuardContext(tool_name="shell", tool_args={"command": cmd})


# ── Path extraction ────────────────────────────────────────────────────────────

class TestFindMissingPaths:
    def test_unix_absolute_missing(self):
        g = PathValidationGuard()
        result = g._find_missing_paths("ls /definitely/does/not/exist/xyz123")
        assert "/definitely/does/not/exist/xyz123" in result

    def test_unix_absolute_existing(self):
        g = PathValidationGuard()
        # /tmp always exists on Unix; on Windows this test is skipped
        if not os.path.exists("/tmp"):
            pytest.skip("No /tmp on this platform")
        result = g._find_missing_paths("ls /tmp")
        assert "/tmp" not in result

    def test_windows_absolute_missing(self):
        g = PathValidationGuard()
        result = g._find_missing_paths(r"dir C:\NoSuchDir\xyz123")
        assert any("xyz123" in p for p in result)

    def test_explicit_relative_missing(self):
        g = PathValidationGuard()
        result = g._find_missing_paths("cat ./no_such_file_xyz.txt")
        assert any("no_such_file_xyz" in p for p in result)

    def test_existing_file_not_flagged(self):
        g = PathValidationGuard()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        try:
            result = g._find_missing_paths(f"cat {tmp}")
            assert tmp not in result
        finally:
            os.unlink(tmp)

    def test_no_paths_in_simple_command(self):
        g = PathValidationGuard()
        result = g._find_missing_paths("echo hello world")
        assert result == []

    def test_url_not_flagged(self):
        g = PathValidationGuard()
        result = g._find_missing_paths("curl https://example.com/api/v1/data")
        assert result == []


# ── Guard verdict ──────────────────────────────────────────────────────────────

class TestCheckPre:
    def test_inject_for_missing_path(self):
        g = PathValidationGuard()
        verdict = g.check_pre(_ctx("cat /no/such/path/xyz123"))
        assert verdict is not None
        assert verdict.action == "inject_msg"
        assert "[PathCheck]" in verdict.message
        assert "xyz123" in verdict.message

    def test_no_verdict_for_existing_path(self):
        g = PathValidationGuard()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        try:
            verdict = g.check_pre(_ctx(f"cat {tmp}"))
            assert verdict is None
        finally:
            os.unlink(tmp)

    def test_no_verdict_for_mkdir(self):
        """mkdir is a creation command — should not warn about missing target."""
        g = PathValidationGuard()
        verdict = g.check_pre(_ctx("mkdir /tmp/new_dir_xyz"))
        assert verdict is None

    def test_no_verdict_for_touch(self):
        g = PathValidationGuard()
        verdict = g.check_pre(_ctx("touch /tmp/new_file_xyz.txt"))
        assert verdict is None

    def test_no_verdict_for_redirect_create(self):
        """Redirect > creates a file — should not warn."""
        g = PathValidationGuard()
        verdict = g.check_pre(_ctx("echo hello > /tmp/output_xyz.txt"))
        assert verdict is None

    def test_no_verdict_for_cp_source_missing(self):
        """cp is a creation command — skip even if source is missing."""
        g = PathValidationGuard()
        verdict = g.check_pre(_ctx("cp /missing/source /tmp/dest"))
        assert verdict is None

    def test_no_verdict_for_empty_command(self):
        g = PathValidationGuard()
        verdict = g.check_pre(GuardContext(tool_name="shell", tool_args={}))
        assert verdict is None

    def test_no_verdict_for_non_shell_tool(self):
        """Guard only activates on shell tool."""
        g = PathValidationGuard()
        ctx = GuardContext(tool_name="read_file", tool_args={"path": "/no/such/path/xyz"})
        # check_pre is only called by GuardRegistry for activate_on_tools match
        # but we test the guard directly — it should still check tool_args["command"]
        # which is absent, so no verdict
        verdict = g.check_pre(ctx)
        assert verdict is None

    def test_dedup_same_path_not_warned_twice(self):
        """Same missing path should only trigger one warning per session."""
        g = PathValidationGuard()
        v1 = g.check_pre(_ctx("cat /no/such/path/xyz123"))
        v2 = g.check_pre(_ctx("ls /no/such/path/xyz123"))
        assert v1 is not None
        assert v2 is None  # Already warned — no repeat

    def test_different_missing_paths_both_warned(self):
        """Two different missing paths should each get a warning."""
        g = PathValidationGuard()
        v1 = g.check_pre(_ctx("cat /no/such/path/aaa111"))
        v2 = g.check_pre(_ctx("cat /no/such/path/bbb222"))
        assert v1 is not None
        assert v2 is not None

    def test_message_contains_path_check_prefix(self):
        g = PathValidationGuard()
        verdict = g.check_pre(_ctx("python /no/such/script_xyz.py"))
        assert verdict is not None
        assert verdict.reason == "missing_path_in_command"

    def test_reset_turn_does_not_clear_warned_set(self):
        """reset_turn should NOT clear warned paths — dedup persists across turns."""
        g = PathValidationGuard()
        g.check_pre(_ctx("cat /no/such/path/xyz123"))
        g.reset_turn()
        v = g.check_pre(_ctx("cat /no/such/path/xyz123"))
        assert v is None  # Still deduped after reset_turn
