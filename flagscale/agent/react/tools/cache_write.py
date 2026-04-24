"""Cache write tool — save project knowledge for future reuse."""

from flagscale.agent.react.tools.base import Tool


class CacheWriteTool(Tool):
    name = "cache_write"
    description = (
        "Cache a project knowledge summary for future reuse across conversations. "
        "Use after analyzing project files to answer structural questions "
        "(dependencies, architecture, config patterns, directory layout, etc.). "
        "Provide the source file paths that were read to produce this knowledge — "
        "the cache will auto-invalidate when those files change."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "A short identifier for this knowledge entry (e.g. 'env_dependencies', 'model_configs', 'parallel_strategies').",
            },
            "description": {
                "type": "string",
                "description": "One-line description of what this entry covers, used for future keyword matching.",
            },
            "content": {
                "type": "string",
                "description": "The knowledge summary to cache (markdown format).",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths that were read to produce this knowledge. Used for cache invalidation.",
            },
        },
        "required": ["key", "description", "content", "sources"],
    }

    def __init__(self, cache):
        self._cache = cache

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        description = kwargs["description"]
        content = kwargs["content"]
        sources = kwargs.get("sources", [])
        try:
            self._cache.put(key, description, content, sources)
            return f"Cached knowledge '{key}' ({len(content)} chars, {len(sources)} source files)."
        except Exception as e:
            return f"ERROR: Failed to cache: {e}"
