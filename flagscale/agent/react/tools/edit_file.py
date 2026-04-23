"""Edit file tool — exact string replacement."""

from flagscale.agent.react.tools.base import Tool


class EditFileTool(Tool):
    name = "edit_file"
    description = "Edit a file by replacing an exact string match. The old_string must match exactly (including whitespace)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact string to find and replace.",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement string.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    def execute(self, **kwargs) -> str:
        path = kwargs["path"]
        old_string = kwargs["old_string"]
        new_string = kwargs["new_string"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_string not in content:
                return f"ERROR: old_string not found in {path}"

            count = content.count(old_string)
            new_content = content.replace(old_string, new_string, 1)

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            msg = f"Successfully edited {path}"
            if count > 1:
                msg += f" (replaced first of {count} occurrences)"
            return msg
        except FileNotFoundError:
            return f"ERROR: file not found: {path}"
        except Exception as e:
            return f"ERROR: {e}"
