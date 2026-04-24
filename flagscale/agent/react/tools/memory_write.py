"""Memory write tool — save key findings, decisions, and todos."""

from flagscale.agent.react.tools.base import Tool


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = (
        "Save a key finding, decision, or todo for future sessions. "
        "Use to record important discoveries, choices made, or pending work "
        "so the agent remembers them across conversations. "
        "Writing the same key updates the existing entry. "
        "Prioritize recording: file/weight/env paths, numerical results (loss, throughput, alignment diffs), "
        "component analysis outcomes (SUPPORTED/ADAPTABLE/CUSTOM), blockers and workarounds, "
        "and anything hard to re-derive. Skip things easily re-read from code or configs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short identifier for this memory (e.g. 'aquila70b_tp_oom', 'parallel_strategy_final', 'todo_test_ep').",
            },
            "type": {
                "type": "string",
                "enum": ["finding", "decision", "todo", "context"],
                "description": "Memory type: finding (discovered fact), decision (choice made), todo (pending work), context (background info).",
            },
            "content": {
                "type": "string",
                "description": "The memory content. Keep concise — one fact per entry.",
            },
        },
        "required": ["key", "type", "content"],
    }

    def __init__(self, memory, session_id: str = ""):
        self._memory = memory
        self._session_id = session_id

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        mem_type = kwargs["type"]
        content = kwargs["content"]
        try:
            self._memory.put(key, mem_type, content, self._session_id)
            return f"Memorized [{mem_type}] '{key}' ({len(content)} chars)."
        except Exception as e:
            return f"ERROR: Failed to save memory: {e}"
