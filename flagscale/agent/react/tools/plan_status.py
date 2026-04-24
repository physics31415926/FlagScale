"""Plan status tool — show current plan progress."""

from flagscale.agent.react.tools.base import Tool


class PlanStatusTool(Tool):
    name = "plan_status"
    description = (
        "Show the current task plan and progress. "
        "Use at the start of a turn to check where you left off, "
        "or after completing steps to see what's next."
    )
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, task_plan):
        self._plan = task_plan

    def execute(self, **kwargs) -> str:
        return self._plan.summary()
