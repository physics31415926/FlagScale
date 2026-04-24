"""Plan update tool — modify task plan steps and status."""

from flagscale.agent.react.tools.base import Tool


class PlanUpdateTool(Tool):
    name = "plan_update"
    description = (
        "Update the active task plan: mark steps done/skipped, add new steps, "
        "replan, or complete/abandon the plan. Use to track progress as you work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["step_done", "step_doing", "step_skip", "add_steps", "complete", "abandon"],
                "description": "What to do: step_done/step_doing/step_skip (update a step), add_steps (insert new steps), complete/abandon (finish the plan).",
            },
            "step_id": {
                "type": "integer",
                "description": "Step number to update (for step_done/step_doing/step_skip).",
            },
            "notes": {
                "type": "string",
                "description": "Notes or reason for the update.",
            },
            "new_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New step descriptions (for add_steps).",
            },
            "after_step_id": {
                "type": "integer",
                "description": "Insert new steps after this step (for add_steps). Omit to append at end.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, task_plan):
        self._plan = task_plan

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]
        try:
            if action == "step_done":
                step_id = kwargs.get("step_id")
                if not step_id:
                    return "ERROR: step_id required for step_done."
                self._plan.update_step(step_id, "done", kwargs.get("notes", ""))
            elif action == "step_doing":
                step_id = kwargs.get("step_id")
                if not step_id:
                    return "ERROR: step_id required for step_doing."
                self._plan.update_step(step_id, "doing", kwargs.get("notes", ""))
            elif action == "step_skip":
                step_id = kwargs.get("step_id")
                if not step_id:
                    return "ERROR: step_id required for step_skip."
                self._plan.skip_step(step_id, kwargs.get("notes", ""))
            elif action == "add_steps":
                new_steps = kwargs.get("new_steps", [])
                if not new_steps:
                    return "ERROR: new_steps required for add_steps."
                self._plan.add_steps(new_steps, kwargs.get("after_step_id"))
            elif action == "complete":
                self._plan.complete()
            elif action == "abandon":
                self._plan.abandon(kwargs.get("notes", ""))
            else:
                return f"ERROR: Unknown action '{action}'."
            return self._plan.summary()
        except Exception as e:
            return f"ERROR: {e}"
