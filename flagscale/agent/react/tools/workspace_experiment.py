"""Workspace experiment tool — manage per-experiment YAML files."""

from flagscale.agent.react.tools.base import Tool


class WorkspaceExperimentTool(Tool):
    name = "workspace_experiment"
    description = (
        "Manage experiment records. Each experiment has its own YAML file.\n"
        "Experiment level: name, purpose, hypothesis, base_config, base_dir, status, root_cause, learnings.\n"
        "Attempt level: change (what you modified), config (this run's config), "
        "output_dir (unique per attempt), result (outcome).\n"
        "Flow: create → add_attempt (before EACH launch) → update_last_attempt (after result) → finalize.\n"
        "The gate will BLOCK training launch unless a pending attempt exists."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "add_attempt", "update_last_attempt", "finalize", "read", "list"],
                "description": "Action to perform.",
            },
            "name": {"type": "string", "description": "Experiment name (required for all actions except list)."},
            "purpose": {"type": "string", "description": "Why this experiment exists (for create)."},
            "hypothesis": {"type": "string", "description": "What you expect to happen and why (for create)."},
            "base_config": {
                "type": "object",
                "description": "Initial/baseline config: model, TP, DP, batch_size, key flags (for create).",
            },
            "base_dir": {"type": "string", "description": "Initial log directory for this experiment (for create)."},
            "change": {
                "type": "string",
                "description": "What changed in this attempt vs previous (for add_attempt). First attempt: 'initial run'.",
            },
            "config": {
                "type": "object",
                "description": "Full config for THIS attempt: model, TP, DP, batch_size, key flags, etc. (for add_attempt).",
            },
            "output_dir": {
                "type": "string",
                "description": "Unique output directory for THIS attempt's results/logs (for add_attempt). Must differ from all previous attempts.",
            },
            "result": {"type": "string", "description": "Attempt result (for update_last_attempt)."},
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
            base_config = kwargs.get("base_config", kwargs.get("config", {}))
            base_dir = kwargs.get("base_dir", "") or kwargs.get("dir", "")
            # Engineering layer enforcement: ALL fields required
            missing = []
            if not name:
                missing.append("name")
            if not purpose:
                missing.append("purpose")
            if not hypothesis:
                missing.append("hypothesis")
            if not base_config:
                missing.append("base_config")
            if not base_dir:
                missing.append("base_dir")
            if missing:
                return (
                    f"ERROR: create requires ALL fields. Missing or empty: {', '.join(missing)}.\n"
                    "Required: name, purpose, hypothesis, base_config (initial config dict), base_dir (initial log dir).\n"
                    "Every field must contain meaningful content, not placeholders."
                )
            return self._manager.create_experiment(name, purpose, hypothesis, base_config, base_dir)

        elif action == "add_attempt":
            name = kwargs.get("name")
            change = kwargs.get("change", "")
            config = kwargs.get("config", {})
            output_dir = kwargs.get("output_dir", "")
            # Engineering layer enforcement: ALL attempt fields required
            missing = []
            if not name:
                missing.append("name")
            if not change:
                missing.append("change")
            if not config:
                missing.append("config")
            if not output_dir:
                missing.append("output_dir")
            if missing:
                return (
                    f"ERROR: add_attempt requires ALL fields. Missing or empty: {', '.join(missing)}.\n"
                    "Required: name, change (what you modified), config (this run's full config), "
                    "output_dir (where this run's logs/checkpoints go).\n"
                    "For first attempt, change='initial run'. Config must include key params for this run."
                )
            return self._manager.add_attempt(name, change, output_dir, config)

        elif action == "update_last_attempt":
            name = kwargs.get("name")
            result = kwargs.get("result", "")
            if not name:
                return "ERROR: name required for update_last_attempt."
            if not result:
                return "ERROR: result required for update_last_attempt. What happened?"
            return self._manager.update_last_attempt(name, result)

        elif action == "finalize":
            name = kwargs.get("name")
            status = kwargs.get("status", "completed")
            root_cause = kwargs.get("root_cause")
            learnings = kwargs.get("learnings", [])
            if not name:
                return "ERROR: name required for finalize."
            if not learnings:
                return (
                    "ERROR: finalize requires non-empty 'learnings' list.\n"
                    "What did you learn from this experiment? Include at least one concrete takeaway."
                )
            if status == "failed" and not root_cause:
                return (
                    "ERROR: finalize with status='failed' requires 'root_cause'.\n"
                    "What was the root cause of the failure?"
                )
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
