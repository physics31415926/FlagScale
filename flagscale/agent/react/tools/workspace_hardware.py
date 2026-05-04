"""Workspace hardware tool — manage hardware.yaml."""

from flagscale.agent.react.tools.base import Tool


class WorkspaceHardwareTool(Tool):
    name = "workspace_hardware"
    description = (
        "Read or write hardware info (GPU count, model, VRAM, driver version, etc.). "
        "This is static info that rarely changes. Written once at session start, "
        "injected into system prompt automatically."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write"],
                "description": "read: return hardware info. write: overwrite hardware info.",
            },
            "content": {
                "type": "object",
                "description": "Hardware info dict (for write). Example: {gpus: '8× A800-SXM4-80GB', vram: '640GB total', driver: 'compatible with CUDA 12.4'}",
            },
        },
        "required": ["action"],
    }

    def __init__(self, workspace_manager):
        self._manager = workspace_manager

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]

        if action == "read":
            hw = self._manager.read_hardware()
            if not hw:
                return "(no hardware info recorded yet)"
            import yaml
            return yaml.dump(hw, allow_unicode=True, default_flow_style=False)

        elif action == "write":
            content = kwargs.get("content")
            if not content:
                return "ERROR: content required for write action."
            if not isinstance(content, dict):
                return "ERROR: content must be a dict."
            return self._manager.write_hardware(content)

        return f"ERROR: Unknown action '{action}'."
