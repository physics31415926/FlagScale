"""Tests for the project knowledge cache."""

import os
import tempfile
import time

import pytest

from flagscale.agent.react.cache import KnowledgeCache
from flagscale.agent.react.tools.cache_read import CacheReadTool
from flagscale.agent.react.tools.cache_write import CacheWriteTool


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def cache(cache_dir):
    return KnowledgeCache(cache_dir, ttl_days=7)


@pytest.fixture
def source_file(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text("hello world")
    return str(p)


class TestKnowledgeCache:
    def test_put_and_get(self, cache, source_file):
        cache.put("test_key", "test description", "cached content", [source_file])
        entry = cache.get("test_key")
        assert entry is not None
        assert entry["content"] == "cached content"
        assert entry["description"] == "test description"
        assert entry["key"] == "test_key"

    def test_get_missing(self, cache):
        assert cache.get("nonexistent") is None

    def test_get_stale_hash(self, cache, source_file):
        cache.put("k", "desc", "content", [source_file])
        assert cache.get("k") is not None
        with open(source_file, "w") as f:
            f.write("modified content")
        assert cache.get("k") is None

    def test_get_stale_ttl(self, cache_dir, source_file):
        cache = KnowledgeCache(cache_dir, ttl_days=0)
        cache.put("k", "desc", "content", [source_file])
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_delete(self, cache, source_file):
        cache.put("k", "desc", "content", [source_file])
        assert cache.delete("k") is True
        assert cache.get("k") is None
        assert cache.delete("k") is False

    def test_list_entries(self, cache, source_file):
        cache.put("a", "desc a", "content a", [source_file])
        cache.put("b", "desc b", "content b", [source_file])
        entries = cache.list_entries()
        assert len(entries) == 2
        keys = {e["key"] for e in entries}
        assert keys == {"a", "b"}
        assert all(e["valid"] for e in entries)

    def test_list_entries_empty(self, cache):
        assert cache.list_entries() == []

    def test_clear(self, cache, source_file):
        cache.put("a", "desc", "content", [source_file])
        cache.put("b", "desc", "content", [source_file])
        count = cache.clear()
        assert count == 2
        assert cache.list_entries() == []

    def test_query_keyword_match(self, cache, source_file):
        cache.put("env_dependencies", "training environment dependencies", "deps info", [source_file])
        cache.put("model_configs", "model configuration patterns", "config info", [source_file])
        results = cache.query("environment dependencies")
        assert len(results) >= 1
        assert results[0]["key"] == "env_dependencies"

    def test_query_no_match(self, cache, source_file):
        cache.put("env_dependencies", "training environment dependencies", "deps info", [source_file])
        results = cache.query("completely unrelated xyz")
        assert len(results) == 0

    def test_file_hash_missing_source(self, cache):
        cache.put("k", "desc", "content", ["/nonexistent/file.txt"])
        entry = cache.get("k")
        assert entry is not None

    def test_file_hash_missing_after_delete(self, cache, source_file):
        cache.put("k", "desc", "content", [source_file])
        assert cache.get("k") is not None
        os.remove(source_file)
        assert cache.get("k") is None


class TestCacheTools:
    def test_cache_write_tool(self, cache, source_file):
        tool = CacheWriteTool(cache)
        result = tool.execute(key="test", description="test desc", content="test content", sources=[source_file])
        assert "Cached knowledge 'test'" in result
        entry = cache.get("test")
        assert entry is not None
        assert entry["content"] == "test content"

    def test_cache_read_tool_hit(self, cache, source_file):
        cache.put("test", "desc", "cached data", [source_file])
        tool = CacheReadTool(cache)
        result = tool.execute(key="test")
        assert result == "cached data"

    def test_cache_read_tool_miss(self, cache):
        tool = CacheReadTool(cache)
        result = tool.execute(key="nonexistent")
        assert "Cache miss" in result
