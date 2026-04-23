"""Load skill tool."""

from flagscale.agent.react.tools.base import Tool


class LoadSkillTool(Tool):
    name = "load_skill"
    description = "Load a skill by name. Returns the skill content that provides specialized instructions."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the skill to load.",
            },
        },
        "required": ["name"],
    }

    def __init__(self, skill_manager):
        self._skill_manager = skill_manager

    def execute(self, **kwargs) -> str:
        name = kwargs["name"]
        try:
            return self._skill_manager.load(name)
        except Exception as e:
            return f"ERROR: loading skill '{name}': {e}"
