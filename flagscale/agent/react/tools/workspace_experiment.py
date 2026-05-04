"""Workspace experiment tool — manage per-experiment YAML files."""

from flagscale.agent.react.tools.base import Tool


class WorkspaceExperimentTool(Tool):
    name = "workspace_experiment"
    description = (
        "Manage experiment records. Each experiment has its own YAML file with: "
        "name, purpose, hypothesis, config, dir, attempts (append-only list), status, root_cause, learnings. "
        "Use 'create' to start a new experiment. Use 'add_attempt' to record each try. "
        "Use 'finalize' when done (set status, root_cause, learnings). "
        "Use 'read' to load a specific experiment. Use 'list' to see all experiments."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "add_attempt", "update_last_attempt", "finalize", "read", "list"],
                "description": "Action to perform.",
            },
            "name": {"type": "string", "description": "Experiment name (required for create/add_attempt/finalize/read)."},
            "purpose": {"type": "string", "description": "Experiment purpose (for create)."},
            "hypothesis": {"type": "string", "description": "Expected outcome (for create)."},
            "config": {
                "type": "object",
                "description": "Experiment config: hardware, parallelism, precision, batch_size, etc. (for create).",
            },
            "dir": {"type": "string", "description": "Experiment directory path (for create)."},
            "change": {"type": "string", "description": "What changed in this attempt (for add_attempt)."},
            "result": {"type": "string", "description": "Attempt result (for add_attempt/update_last_attempt)."},
            "status": {
                "type": "string",
                "enum": ["running", "failed", "completed", "paused"],
                "description": "Final status (for finalize).",
            },
            "root_cause": {"type": "string", "description": "Root cause of failure (for finalize, if failed)."},
            "learnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key learnings from this experiment (for finalize).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, workspace_manager):
        self._manager = workspace_manager

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]

        if action == "create":
            name = kwargs.get("name")
            purpose = kwargs.get("purpose", "")
            hypothesis = kwargs.get("hypothesis", "")
            config = kwargs.get("config", {})
            exp_dir = kwargs.get("dir", "")
            if not name:
                return "ERROR: name required for create."
            return self._manager.create_experiment(name, purpose, hypothesis, config, exp_dir)

        elif action == "add_attempt":
            name = kwargs.get("name")
            change = kwargs.get("change", "")
            result = kwargs.get("result", "")
            if not name:
                return "ERROR: name required for add_attempt."
            return self._manager.add_attempt(name, change, result)

        elif action == "update_last_attempt":
            name = kwargs.get("name")
            result = kwargs.get("result", "")
            if not name:
                return "ERROR: name required for update_last_attempt."
            return self._manager.update_last_attempt(name, result)

        elif action == "finalize":
            name = kwargs.get("name")
            status = kwargs.get("status", "completed")
            root_cause = kwargs.get("root_cause")
            learnings = kwargs.get("learnings", [])
            if not name:
                return "ERROR: name required for finalize."
            return self._manager.finalize_experiment(name, status, root_cause, learnings)

        elif action == "read":
            name = kwargs.get("name")
            if not name:
                return "ERROR: name required for read."
            exp = self._manager.read_experiment(name)
            if not exp:
                return f"Experiment '{name}' not found."
            import yaml
            return yaml.dump(exp, allow_unicode=True, default_flow_style=False, sort_keys=False)

        elif action == "list":
            experiments = self._manager.list_experiments()
            if not experiments:
                return "(no experiments yet)"
            lines = [f"- {e['name']} ({e['status']})" for e in experiments]
            return "\n".join(lines)

        return f"ERROR: Unknown action '{action}'."
