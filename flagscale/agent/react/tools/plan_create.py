"""Plan create tool — create a structured task plan."""

from flagscale.agent.react.tools.base import Tool


class PlanCreateTool(Tool):
    name = "plan_create"
    description = (
        "Create a task plan with ordered steps for complex multi-step work. "
        "Use when starting environment setup, model porting, training runs, "
        "or any task with 3+ sequential steps. Only one plan can be active at a time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short plan title, e.g. 'ESPnet LibriSpeech training reproduction'.",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of step descriptions.",
            },
        },
        "required": ["title", "steps"],
    }

    def __init__(self, task_plan, session_id: str = ""):
        self._plan = task_plan
        self._session_id = session_id

    def execute(self, **kwargs) -> str:
        title = kwargs["title"]
        steps = kwargs["steps"]
        if not steps:
            return "ERROR: At least one step is required."
        try:
            plan = self._plan.create(title, steps, self._session_id)
            return f"Plan created.\n\n{self._plan.summary()}"
        except Exception as e:
            return f"ERROR: {e}"
