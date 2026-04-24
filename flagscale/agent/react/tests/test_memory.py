"""Tests for the session memory system."""

import os
import time

import pytest

from flagscale.agent.react.memory import SessionMemory
from flagscale.agent.react.tools.memory_write import MemoryWriteTool
from flagscale.agent.react.tools.memory_read import MemoryReadTool


@pytest.fixture
def memory_dir(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def memory(memory_dir):
    return SessionMemory(memory_dir, ttl_days=7)


class TestSessionMemory:
    def test_put_and_get(self, memory):
        memory.put("k1", "finding", "TP=8 causes OOM", "sess1")
        entry = memory.get("k1")
        assert entry is not None
        assert entry["key"] == "k1"
        assert entry["type"] == "finding"
        assert entry["content"] == "TP=8 causes OOM"
        assert entry["session_id"] == "sess1"

    def test_get_missing(self, memory):
        assert memory.get("nonexistent") is None

    def test_put_overwrites(self, memory):
        memory.put("k1", "finding", "old content", "sess1")
        memory.put("k1", "decision", "new content", "sess2")
        entry = memory.get("k1")
        assert entry["type"] == "decision"
        assert entry["content"] == "new content"
        assert entry["session_id"] == "sess2"

    def test_delete(self, memory):
        memory.put("k1", "finding", "content", "sess1")
        assert memory.delete("k1") is True
        assert memory.get("k1") is None
        assert memory.delete("k1") is False

    def test_list_entries(self, memory):
        memory.put("a", "finding", "fact a", "s1")
        memory.put("b", "decision", "choice b", "s1")
        entries = memory.list_entries()
        assert len(entries) == 2
        keys = {e["key"] for e in entries}
        assert keys == {"a", "b"}

    def test_list_entries_empty(self, memory):
        assert memory.list_entries() == []

    def test_clear(self, memory):
        memory.put("a", "finding", "x", "s1")
        memory.put("b", "todo", "y", "s1")
        count = memory.clear()
        assert count == 2
        assert memory.list_entries() == []

    def test_clear_by_type(self, memory):
        memory.put("a", "finding", "fact a", "s1")
        memory.put("b", "todo", "task b", "s1")
        memory.put("c", "finding", "fact c", "s1")
        memory.put("d", "context", "ctx d", "s1")
        count = memory.clear_by_type("finding")
        assert count == 2
        remaining = memory.list_entries()
        assert len(remaining) == 2
        keys = {e["key"] for e in remaining}
        assert keys == {"b", "d"}

    def test_clear_by_type_returns_zero_for_unknown(self, memory):
        memory.put("a", "finding", "fact", "s1")
        count = memory.clear_by_type("nonexistent")
        assert count == 0
        assert len(memory.list_entries()) == 1

    def test_clear_by_type(self, memory):
        memory.put("a", "finding", "fact a", "s1")
        memory.put("b", "context", "ctx b", "s1")
        memory.put("c", "finding", "fact c", "s1")
        memory.put("d", "todo", "task d", "s1")
        count = memory.clear_by_type("finding")
        assert count == 2
        remaining = memory.list_entries()
        assert len(remaining) == 2
        remaining_types = {e["type"] for e in remaining}
        assert "finding" not in remaining_types

    def test_clear_by_type_returns_zero_for_unknown(self, memory):
        memory.put("a", "finding", "fact", "s1")
        count = memory.clear_by_type("nonexistent")
        assert count == 0
        assert len(memory.list_entries()) == 1

    def test_ttl_expiry(self, memory_dir):
        memory = SessionMemory(memory_dir, ttl_days=0)
        memory.put("k1", "finding", "content", "s1")
        time.sleep(0.1)
        assert memory.get("k1") is None

    def test_recent_returns_newest_first(self, memory):
        memory.put("old", "finding", "old fact", "s1")
        time.sleep(0.05)
        memory.put("new", "finding", "new fact", "s1")
        entries = memory.recent(max_tokens=1000)
        assert len(entries) == 2
        assert entries[0]["key"] == "new"
        assert entries[1]["key"] == "old"

    def test_recent_respects_budget(self, memory):
        memory.put("a", "finding", "x" * 2000, "s1")
        time.sleep(0.05)
        memory.put("b", "finding", "short", "s1")
        entries = memory.recent(max_tokens=100)
        assert len(entries) == 1
        assert entries[0]["key"] == "b"

    def test_key_with_special_chars(self, memory):
        memory.put("my/key with spaces", "context", "content", "s1")
        entry = memory.get("my/key with spaces")
        assert entry is not None
        assert entry["content"] == "content"


class TestMemoryTools:
    def test_memory_write_tool(self, memory):
        tool = MemoryWriteTool(memory, "sess1")
        result = tool.execute(key="test", type="finding", content="test content")
        assert "Memorized" in result
        assert "[finding]" in result
        entry = memory.get("test")
        assert entry is not None
        assert entry["content"] == "test content"

    def test_memory_read_tool_hit(self, memory):
        memory.put("test", "decision", "use TP=4", "s1")
        tool = MemoryReadTool(memory)
        result = tool.execute(key="test")
        assert "[decision]" in result
        assert "use TP=4" in result

    def test_memory_read_tool_miss(self, memory):
        tool = MemoryReadTool(memory)
        result = tool.execute(key="nonexistent")
        assert "No memory found" in result
