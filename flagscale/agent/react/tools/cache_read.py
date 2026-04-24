"""Cache read tool — retrieve cached project knowledge."""

from flagscale.agent.react.tools.base import Tool


class CacheReadTool(Tool):
    name = "cache_read"
    description = (
        "Read a specific cached knowledge entry by key. "
        "Use when you know a cache entry exists and want to retrieve it. "
        "Returns the cached content if valid, or an error if not found/stale."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key of the cache entry to read.",
            },
        },
        "required": ["key"],
    }

    def __init__(self, cache):
        self._cache = cache

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        entry = self._cache.get(key)
        if entry is None:
            return f"Cache miss: no valid entry for '{key}'."
        return entry.get("content", "")
