"""Read file tool with line range support."""

import os

from flagscale.agent.react.tools.base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a file. Supports line ranges for large files. "
        "Returns content with line numbers for easy reference."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to read.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-based). Default: 1",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (inclusive). Default: end of file. Max 500 lines per call.",
            },
            "numbered": {
                "type": "boolean",
                "description": "Include line numbers. Default: true",
            },
        },
        "required": ["path"],
    }

    MAX_LINES = 500

    def execute(self, **kwargs) -> str:
        path = kwargs["path"]
        start = kwargs.get("start_line", 1)
        end = kwargs.get("end_line")
        numbered = kwargs.get("numbered", True)

        if not os.path.exists(path):
            return f"ERROR: File not found: {path}"
        if os.path.isdir(path):
            return f"ERROR: Path is a directory: {path}"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"ERROR: {e}"

        total = len(lines)
        start = max(1, start)
        if end is None:
            end = min(start + self.MAX_LINES - 1, total)
        end = min(end, total)

        if start > total:
            return f"File has {total} lines, requested start_line={start} is past end."

        selected = lines[start - 1:end]
        truncated = end < total and (end - start + 1) >= self.MAX_LINES

        if numbered:
            width = len(str(end))
            output_lines = [f"{i:{width}d}| {line}" for i, line in enumerate(selected, start=start)]
        else:
            output_lines = selected

        header = f"[{path}] lines {start}-{end} of {total}"
        result = header + "\n" + "".join(output_lines)

        if truncated:
            result += f"\n... truncated at {self.MAX_LINES} lines. Use start_line={end + 1} to continue."

        return result
