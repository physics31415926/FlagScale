"""Tests for loop detection and tool-call deduplication (loop_detect.py)."""

import pytest
from unittest.mock import MagicMock

from flagscale.agent.react.loop_detect import LoopDetectMixin


class FakeAgent(LoopDetectMixin):
    """Minimal stub that provides the state LoopDetectMixin expects."""

    def __init__(self):
        self._recent_tool_calls = []
        self._tool_call_cache = {}
        self._files_read_this_session = set()
        self._files_written_this_session = set()


class TestGetToolCallKey:

    def setup_method(self):
        self.agent = FakeAgent()

    def test_shell_key(self):
        key = self.agent._get_tool_call_key("shell", {"command": "ls -la"})
        assert key == ("shell", "ls -la")

    def test_read_file_key(self):
        key = self.agent._get_tool_call_key("read_file", {"path": "/a/b.py", "start_line": 10, "end_line": 50})
        assert key == ("read_file", "/a/b.py", 10, 50)

    def test_read_file_key_defaults(self):
        key = self.agent._get_tool_call_key("read_file", {"path": "/a/b.py"})
        assert key == ("read_file", "/a/b.py", 0, 0)

    def test_write_file_key(self):
        key = self.agent._get_tool_call_key("write_file", {"path": "/x/y.py"})
        assert key == ("write_file", "/x/y.py")

    def test_edit_file_key_includes_old_string_hash(self):
        key = self.agent._get_tool_call_key("edit_file", {"file_path": "/x.py", "old_string": "hello"})
        assert key[0] == "edit_file"
        assert key[1] == "/x.py"
        assert isinstance(key[2], int)

    def test_load_skill_key(self):
        key = self.agent._get_tool_call_key("load_skill", {"name": "train-run"})
        assert key == ("load_skill", "train-run")

    def test_generic_tool_key(self):
        key = self.agent._get_tool_call_key("monitor", {"output_dir": "/outputs/exp1", "duration": 60})
        assert key[0] == "monitor"
        assert "output_dir=/outputs/exp1" in key[1]


class TestCheckLoopDetection:

    def setup_method(self):
        self.agent = FakeAgent()

    def test_no_loop_below_threshold(self):
        for _ in range(2):
            result = self.agent._check_loop_detection("shell", {"command": "cat log.txt"})
        assert result == ""

    def test_loop_detected_at_threshold(self):
        for _ in range(2):
            self.agent._check_loop_detection("shell", {"command": "cat log.txt"})
        result = self.agent._check_loop_detection("shell", {"command": "cat log.txt"})
        assert "LOOP DETECTION" in result
        assert "3 times" in result

    def test_different_commands_no_loop(self):
        for i in range(5):
            result = self.agent._check_loop_detection("shell", {"command": f"cmd_{i}"})
        assert result == ""

    def test_window_eviction(self):
        # Fill window with different commands
        for i in range(10):
            self.agent._check_loop_detection("shell", {"command": f"unique_{i}"})
        # Now repeat — old entries evicted, so count resets
        result = self.agent._check_loop_detection("shell", {"command": "unique_0"})
        assert result == ""

    def test_mixed_tools_no_false_positive(self):
        self.agent._check_loop_detection("shell", {"command": "ls"})
        self.agent._check_loop_detection("read_file", {"path": "/a.py"})
        self.agent._check_loop_detection("shell", {"command": "ls"})
        result = self.agent._check_loop_detection("read_file", {"path": "/a.py"})
        assert result == ""


class TestCheckDuplicateRead:

    def setup_method(self):
        self.agent = FakeAgent()

    def test_no_cache_returns_none(self):
        result = self.agent._check_duplicate_read("read_file", {"path": "/a.py"})
        assert result is None

    def test_cached_read_returns_cached(self):
        self.agent._tool_call_cache[("read_file", "/a.py", "", "")] = "cached content"
        result = self.agent._check_duplicate_read("read_file", {"path": "/a.py"})
        assert result == "cached content"

    def test_cached_read_with_lines(self):
        self.agent._tool_call_cache[("read_file", "/a.py", "10", "20")] = "lines 10-20"
        result = self.agent._check_duplicate_read("read_file", {"path": "/a.py", "start_line": "10", "end_line": "20"})
        assert result == "lines 10-20"

    def test_memory_write_cached(self):
        self.agent._tool_call_cache[("memory_write", "my_key")] = "written"
        result = self.agent._check_duplicate_read("memory_write", {"key": "my_key"})
        assert result == "written"

    def test_non_cacheable_tool_returns_none(self):
        result = self.agent._check_duplicate_read("shell", {"command": "ls"})
        assert result is None

    def test_empty_path_returns_none(self):
        result = self.agent._check_duplicate_read("read_file", {"path": ""})
        assert result is None

    def test_empty_memory_key_returns_none(self):
        result = self.agent._check_duplicate_read("memory_write", {"key": ""})
        assert result is None


class TestCacheToolResult:

    def setup_method(self):
        self.agent = FakeAgent()

    def test_cache_read_file(self):
        self.agent._cache_tool_result("read_file", {"path": "/x.py"}, "file content")
        assert self.agent._tool_call_cache[("read_file", "/x.py", "", "")] == "file content"

    def test_no_cache_on_error(self):
        self.agent._cache_tool_result("read_file", {"path": "/x.py"}, "ERROR: file not found")
        assert ("read_file", "/x.py", "", "") not in self.agent._tool_call_cache

    def test_cache_memory_write(self):
        self.agent._cache_tool_result("memory_write", {"key": "k1"}, "OK: saved")
        assert self.agent._tool_call_cache[("memory_write", "k1")] == "OK: saved"

    def test_no_cache_shell(self):
        self.agent._cache_tool_result("shell", {"command": "ls"}, "output")
        assert len(self.agent._tool_call_cache) == 0

    def test_no_cache_empty_path(self):
        self.agent._cache_tool_result("read_file", {"path": ""}, "content")
        assert len(self.agent._tool_call_cache) == 0
