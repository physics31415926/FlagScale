"""Memory write tool — save key findings, decisions, and todos."""

from flagscale.agent.react.tools.base import Tool


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = (
        "Save a key finding, decision, or todo for future sessions. "
        "Use to record important discoveries, choices made, or pending work "
        "so the agent remembers them across conversations. "
        "Writing the same key updates the existing entry. "
        "Use 'supersedes' to delete old entries that this new one replaces. "
        "Entries are automatically associated with the current task from workspace current.yaml. "
        "Prioritize recording: env quirks and tool incompatibilities, "
        "file/weight/env paths, version constraints, framework-specific gotchas, "
        "numerical results (loss, throughput), workarounds that took trial-and-error to find, "
        "and anything hard to re-derive. "
        "PROACTIVE RULE: after any unexpected failure that required a workaround, "
        "immediately memorize it if a future session could hit the same issue. "
        "SUPERSEDE RULE: when new information contradicts, completes, or replaces older memories, "
        "use 'supersedes' to list the old key(s) to delete. This applies to ALL memory types:\n"
        "  - finding: new analysis contradicts old conclusion\n"
        "  - decision: choice was reversed or refined\n"
        "  - todo: task completed, abandoned, or superseded by new approach\n"
        "  - context: background info became outdated\n"
        "Keeping stale memories misleads future sessions. When in doubt, supersede. "
        "Do NOT use memory for: experiment records (use workspace_experiment), "
        "current session state (use workspace_current), or information easily re-read from files/configs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short identifier, lowercase alphanumeric and underscores only, 2-80 chars (e.g. 'aquila70b_tp_oom', 'parallel_strategy_final'). NO error messages, hashes, or special characters.",
            },
            "type": {
                "type": "string",
                "enum": ["finding", "decision", "todo", "context"],
                "description": "Memory type: finding (discovered fact), decision (choice made), todo (pending work), context (background info).",
            },
            "content": {
                "type": "string",
                "description": "The memory content. Be clear and specific — include the exact error, flag, or version number so future sessions can act on it directly.",
            },
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of old memory keys that this entry replaces. Those entries will be deleted.",
            },
        },
        "required": ["key", "type", "content"],
    }

    def __init__(self, memory, session_id: str = "", workspace_manager=None):
        self._memory = memory
        self._session_id = session_id
        self._workspace_manager = workspace_manager

    def _get_current_task(self) -> str:
        if self._workspace_manager:
            return self._workspace_manager.get_current_task()
        return ""

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        mem_type = kwargs["type"]
        content = kwargs["content"]
        supersedes = kwargs.get("supersedes", [])
        task = self._get_current_task()

        from flagscale.agent.react.memory import SessionMemory

        if not SessionMemory.is_valid_key(key):
            sanitized = SessionMemory.sanitize_key(key)
            if not sanitized or not SessionMemory.is_valid_key(sanitized):
                return (
                    f"ERROR: Invalid memory key '{key}'. "
                    "Key must be 2-80 chars, lowercase alphanumeric and underscores only "
                    "(e.g. 'bagel_tp_oom', 'parallel_strategy_final'). "
                    "Do not use error messages, hashes, or special characters as keys."
                )
            return (
                f"ERROR: Invalid memory key '{key}'. "
                f"Suggested key: '{sanitized}'. "
                "Key must be 2-80 chars, lowercase alphanumeric and underscores only."
            )

        try:
            deleted = []
            for old_key in supersedes:
                if self._memory.delete(old_key):
                    deleted.append(old_key)

            self._memory.put(key, mem_type, content, self._session_id, task=task)
            task_info = f" [task: {task}]" if task else ""
            supersede_info = f" Superseded: {', '.join(deleted)}." if deleted else ""
            return f"Memorized [{mem_type}] '{key}' ({len(content)} chars).{task_info}{supersede_info}"
        except Exception as e:
            return f"ERROR: Failed to save memory: {e}"
