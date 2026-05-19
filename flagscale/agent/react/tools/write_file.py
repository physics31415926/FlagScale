"""Write file tool."""

import os

from flagscale.agent.react.tools.base import Tool, EFFECT_WRITE_FS
from flagscale.agent.react.tools.read_file import get_file_cache

# -- Paths that should never be written by the agent --
_PROTECTED_PATHS = frozenset({
    os.path.expanduser("~/.bashrc"),
    os.path.expanduser("~/.profile"),
    os.path.expanduser("~/.bash_profile"),
    os.path.expanduser("~/.zshrc"),
    os.path.expanduser("~/.ssh/authorized_keys"),
})


def _is_protected_path(path: str) -> bool:
    """Check if path is protected from agent writes."""
    resolved = os.path.abspath(os.path.realpath(path))
    if resolved in _PROTECTED_PATHS:
        return True
    if resolved.startswith("/etc/") and not resolved.startswith("/etc/apt/"):
        return True
    if resolved.startswith("/boot/"):
        return True
    return False


class WriteFileTool(Tool):
    name = "write_file"
    effects = EFFECT_WRITE_FS
    description = "Create or overwrite a file at the given path with the provided content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to write.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def execute(self, **kwargs) -> str:
        path = kwargs["path"]
        content = kwargs["content"]

        if _is_protected_path(path):
            return f"ERROR: Cannot write to protected system path: {path}"

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            get_file_cache().invalidate(os.path.abspath(path))
            get_file_cache().invalidate(path)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"ERROR: {e}"
