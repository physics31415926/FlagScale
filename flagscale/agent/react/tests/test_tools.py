"""Tests for agent tools."""

import os
import tempfile

import pytest

from flagscale.agent.react.tools.base import Tool
from flagscale.agent.react.tools.edit_file import EditFileTool
from flagscale.agent.react.tools.read_file import ReadFileTool
from flagscale.agent.react.tools.shell import ShellTool
from flagscale.agent.react.tools.write_file import WriteFileTool
from flagscale.agent.react.tools import ToolRegistry


class TestReadFileTool:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        tool = ReadFileTool()
        assert tool.execute(path=str(f)) == "hello world"

    def test_read_missing_file(self):
        tool = ReadFileTool()
        result = tool.execute(path="/nonexistent/path/file.txt")
        assert result.startswith("ERROR:")

    def test_schema_openai(self):
        tool = ReadFileTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"

    def test_schema_anthropic(self):
        tool = ReadFileTool()
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "read_file"
        assert "input_schema" in schema


class TestWriteFileTool:
    def test_write_new_file(self, tmp_path):
        f = tmp_path / "out.txt"
        tool = WriteFileTool()
        result = tool.execute(path=str(f), content="test content")
        assert "Successfully" in result
        assert f.read_text() == "test content"

    def test_write_creates_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "out.txt"
        tool = WriteFileTool()
        tool.execute(path=str(f), content="nested")
        assert f.read_text() == "nested"


class TestEditFileTool:
    def test_edit_replaces(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("foo = 1\nbar = 2\n")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="foo = 1", new_string="foo = 42")
        assert "Successfully" in result
        assert "foo = 42" in f.read_text()

    def test_edit_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello")
        tool = EditFileTool()
        result = tool.execute(path=str(f), old_string="missing", new_string="x")
        assert result.startswith("ERROR:")

    def test_edit_missing_file(self):
        tool = EditFileTool()
        result = tool.execute(path="/nonexistent", old_string="a", new_string="b")
        assert result.startswith("ERROR:")


class TestShellTool:
    def test_basic_command(self):
        tool = ShellTool(require_confirm=False)
        result = tool.execute(command="echo hello")
        assert "hello" in result

    def test_timeout(self):
        tool = ShellTool(timeout=1, require_confirm=False)
        result = tool.execute(command="sleep 10")
        assert "timed out" in result

    def test_dangerous_command_blocked(self):
        tool = ShellTool(check_dangerous=True, require_confirm=False)
        result = tool.execute(command="rm -rf /")
        assert result.startswith("FATAL:")

    def test_dangerous_check_disabled(self):
        tool = ShellTool(check_dangerous=False, timeout=1, require_confirm=False)
        result = tool.execute(command="echo safe")
        assert "safe" in result

    def test_confirm_denied(self):
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: False)
        result = tool.execute(command="rm /tmp/test_file")
        assert "DENIED" in result

    def test_confirm_approved(self):
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: True)
        result = tool.execute(command="rm /tmp/nonexistent_flagscale_test_xyz")
        assert "DENIED" not in result

    def test_confirm_not_triggered_for_safe_commands(self):
        called = []
        tool = ShellTool(require_confirm=True, confirm_fn=lambda cmd: (called.append(1), False)[1])
        tool.execute(command="echo safe")
        assert len(called) == 0


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        tool = reg.get("read_file")
        assert tool.name == "read_file"

    def test_get_missing(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_execute_truncates(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100000)
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        result = reg.execute("read_file", path=str(f))
        assert len(result) < 100000
        assert "truncated" in result

    def test_to_schemas(self):
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        reg.register(ShellTool())
        schemas = reg.to_schemas("openai")
        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)
