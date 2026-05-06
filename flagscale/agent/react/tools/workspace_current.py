"""Workspace current state tool — manage current.yaml."""

from flagscale.agent.react.tools.base import Tool


class WorkspaceCurrentTool(Tool):
    name = "workspace_current"
    description = (
        "Read or update current.yaml (current task state). "
        "This file tracks: task (what you're working on), status (running/blocked/completed), "
        "current_experiment (name of active experiment), blockers (list of issues), "
        "next_steps (list of actions), context (list of key facts). "
        "Use 'read' to see current state. Use 'update' to modify specific fields. "
        "Use 'set_task' at session start to define the task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "update", "set_task"],
                "description": "read: return current.yaml content. update: modify fields. set_task: set task field.",
            },
            "task": {"type": "string", "description": "Task description (for set_task or update)."},
            "status": {
                "type": "string",
                "enum": ["running", "blocked", "completed", "starting"],
                "description": "Current status (for update).",
            },
            "current_experiment": {"type": "string", "description": "Name of active experiment (for update)."},
            "blockers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of blocking issues (for update).",
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of next actions (for update).",
            },
            "context": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key facts/context (for update).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, workspace_manager):
        self._manager = workspace_manager

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]

        if action == "read":
            current = self._manager.read_current()
            if not current:
                return "(current.yaml does not exist yet)"
            import yaml
            return yaml.dump(current, allow_unicode=True, default_flow_style=False, sort_keys=False)

        elif action == "set_task":
            task = kwargs.get("task")
            if not task:
                return "ERROR: task required for set_task action."
            return self._manager.update_current(task=task, status="starting")

        elif action == "update":
            update_fields = {}
            for key in ("task", "status", "current_experiment", "blockers", "next_steps", "context"):
                if key in kwargs:
                    update_fields[key] = kwargs[key]
            if not update_fields:
                return "ERROR: No fields to update. Provide at least one of: task, status, current_experiment, blockers, next_steps, context."
            return self._manager.update_current(**update_fields)

        return f"ERROR: Unknown action '{action}'."
