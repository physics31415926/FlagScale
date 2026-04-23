"""Write file tool."""

import os

from flagscale.agent.react.tools.base import Tool


class WriteFileTool(Tool):
    name = "write_file"
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
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"ERROR: {e}"
