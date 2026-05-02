"""Workspace state tool — persist and retrieve current work context."""

import os
import time

from flagscale.agent.react.tools.base import Tool


def _default_state_path():
    return os.path.join(os.path.expanduser("~"), ".flagscale", "workspace_state.md")


class WorkspaceStateTool(Tool):
    name = "workspace_state"
    description = (
        "Read or update the workspace state file (.flagscale/workspace_state.md). "
        "This is the EXPERIMENT REGISTRY and session state. Use it for: experiment entries "
        "(purpose, config, result, reflection), current hardware info, active blockers, "
        "and session summaries. Do NOT duplicate information here that belongs in memory "
        "(cross-session facts like env paths or version decisions). "
        "The state file is injected into the system prompt on startup."
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
        if section:
            # Validate experiment content before any write
            if section.lower() == "experiments":
                validation_error = self._validate_experiment_content(content)
                if validation_error:
                    return validation_error
            if os.path.isfile(self._path):
                return self._update_section(section, content)
            # File doesn't exist — create with proper section structure
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                structured = f"<!-- Updated: {ts} -->\n## {section}\n{content}\n"
                with open(self._path, "w", encoding="utf-8") as f:
                    f.write(structured)
                return f"Section '{section}' created ({len(content)} chars)."
            except Exception as e:
                return f"ERROR writing state: {e}"
        # Guard: if file already has structured sections, refuse bare overwrite
        if os.path.isfile(self._path) and not section:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    existing = f.read()
                if "\n## " in existing and len(existing) > 200:
                    return (
                        "ERROR: workspace_state already has structured sections. "
                        "Use section='<name>' to update a specific section, or "
                        "action='append' to add content. Bare write would destroy "
                        "existing experiment records and other data."
                    )
            except Exception:
                pass
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
        # Catch experiment entries written via append instead of section='Experiments'
        if "### " in content and any(kw in content.lower() for kw in ("purpose", "config", "result")):
            return (
                "ERROR: This looks like an experiment entry. Use "
                "workspace_state(action='write', section='Experiments', content=...) "
                "instead of append. The section parameter ensures the entry is placed "
                "under the ## Experiments header."
            )
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
        old_section_lines = []
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
            if in_section:
                old_section_lines.append(line)
            else:
                new_lines.append(line)

        if not replaced:
            new_lines.append("")
            new_lines.append(header)
            new_lines.append(content)
            new_lines.append("")

        # Warn if new content is much shorter than old (possible data loss)
        old_content = "\n".join(old_section_lines).strip()
        data_loss_warning = ""
        if (old_content
                and len(content.strip()) < len(old_content) * 0.5
                and len(old_content) > 100):
            data_loss_warning = (
                f" WARNING: new content ({len(content)} chars) is much shorter "
                f"than old content ({len(old_content)} chars) in section "
                f"'{section}'. If this section contains experiment records, "
                f"make sure ALL previous entries are preserved in the new content."
            )

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        # Strip old timestamp comments to prevent accumulation
        cleaned = [l for l in new_lines if not l.strip().startswith("<!-- Updated:")]
        final = f"<!-- Updated: {ts} -->\n" + "\n".join(cleaned)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(final)
            return f"Section '{section}' updated.{data_loss_warning}"
        except Exception as e:
            return f"ERROR writing state: {e}"

    @staticmethod
    def _validate_experiment_content(content: str) -> str:
        """Validate experiment content format. Returns error message or empty string."""
        stripped = content.strip()
        if stripped.startswith("## "):
            return (
                "ERROR: Do not include '## Experiments' header in the content — "
                "the tool adds it automatically via section='Experiments'. "
                "Your content should start with '### <exp_name> (status)' directly."
            )
        if "### " not in stripped:
            return (
                "ERROR: Experiment content must use '### ' sub-headers for each entry. "
                "Expected format:\n"
                "### exp_name (running)\n"
                "- **Purpose**: ...\n"
                "- **Config**: ...\n"
                "- **Dir**: ...\n"
                "- **Result**: (pending)\n\n"
                "Use section='Experiments' and provide ### entries as content."
            )
        return ""
