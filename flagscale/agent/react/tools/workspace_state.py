"""Workspace state tool — persist and retrieve current work context."""

import os
import time

from flagscale.agent.react.tools.base import Tool


def _default_state_path():
    return os.path.join(os.getcwd(), ".flagscale", "workspace_state.md")


class WorkspaceStateTool(Tool):
    name = "workspace_state"
    description = (
        "Read or update the workspace state file (.flagscale/workspace_state.md). "
        "Use this to persist what you're working on, key findings, and next steps "
        "so context survives across sessions. The state file is injected into the "
        "system prompt on startup."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append"],
                "description": "read: return current state. write: overwrite state. append: add to state.",
            },
            "content": {
                "type": "string",
                "description": "Content to write or append (for write/append actions).",
            },
            "section": {
                "type": "string",
                "description": "Optional section header to update (for write action). Updates only that section.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, state_path: str = ""):
        self._path = state_path or _default_state_path()

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]
        if action == "read":
            return self._read()
        elif action == "write":
            content = kwargs.get("content", "")
            section = kwargs.get("section")
            if not content:
                return "ERROR: content required for write action."
            return self._write(content, section)
        elif action == "append":
            content = kwargs.get("content", "")
            if not content:
                return "ERROR: content required for append action."
            return self._append(content)
        return f"ERROR: Unknown action '{action}'."

    def _read(self) -> str:
        if not os.path.isfile(self._path):
            return "(no workspace state file exists yet)"
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"ERROR reading state: {e}"

    def _write(self, content: str, section: str = None) -> str:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        if section and os.path.isfile(self._path):
            return self._update_section(section, content)
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            header = f"<!-- Updated: {ts} -->\n"
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(header + content + "\n")
            return f"Workspace state written ({len(content)} chars)."
        except Exception as e:
            return f"ERROR writing state: {e}"

    def _append(self, content: str) -> str:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(f"\n<!-- Appended: {ts} -->\n{content}\n")
            return f"Appended to workspace state ({len(content)} chars)."
        except Exception as e:
            return f"ERROR appending state: {e}"

    def _update_section(self, section: str, content: str) -> str:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return f"ERROR reading state: {e}"

        header = f"## {section}"
        lines = text.split("\n")
        new_lines = []
        in_section = False
        replaced = False
        for line in lines:
            if line.strip().startswith("## "):
                if in_section:
                    in_section = False
                if line.strip() == header:
                    new_lines.append(header)
                    new_lines.append(content)
                    new_lines.append("")
                    in_section = True
                    replaced = True
                    continue
            if not in_section:
                new_lines.append(line)

        if not replaced:
            new_lines.append("")
            new_lines.append(header)
            new_lines.append(content)
            new_lines.append("")

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        final = f"<!-- Updated: {ts} -->\n" + "\n".join(new_lines)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(final)
            return f"Section '{section}' updated."
        except Exception as e:
            return f"ERROR writing state: {e}"
