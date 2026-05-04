"""Workspace manager — manages current state, experiments, and hardware info."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class WorkspaceManager:
    """Manages workspace state split into current.yaml, hardware.yaml, and per-experiment files."""

    def __init__(self, workspace_dir: str = ""):
        self._dir = workspace_dir or os.path.join(Path.home(), ".flagscale", "workspace_state")
        self._current_path = os.path.join(self._dir, "current.yaml")
        self._hardware_path = os.path.join(self._dir, "hardware.yaml")
        self._experiments_dir = os.path.join(self._dir, "experiments")

    # ── Current State ───────────────────────────────────────────────────

    def read_current(self) -> Dict:
        """Read current.yaml. Returns empty dict if not exists."""
        if not os.path.isfile(self._current_path):
            return {}
        try:
            with open(self._current_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def update_current(self, **kwargs) -> str:
        """Update specific fields in current.yaml.

        Supported fields: task, status, current_experiment, blockers, next_steps, context, session_summary.
        """
        os.makedirs(self._dir, exist_ok=True)
        current = self.read_current()

        for key, value in kwargs.items():
            if key in ("task", "status", "current_experiment", "session_summary"):
                current[key] = value
            elif key in ("blockers", "next_steps", "context"):
                # These are lists
                if not isinstance(value, list):
                    return f"ERROR: {key} must be a list"
                current[key] = value
            else:
                return f"ERROR: Unknown field '{key}'"

        current["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(self._current_path, "w", encoding="utf-8") as f:
                yaml.dump(current, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Updated current.yaml: {', '.join(kwargs.keys())}"
        except Exception as e:
            return f"ERROR: Failed to update current.yaml: {e}"

    def get_current_task(self) -> str:
        """Get the task field from current.yaml. Returns empty string if not set."""
        current = self.read_current()
        return current.get("task", "")

    def get_current_experiment(self) -> str:
        """Get the current_experiment field from current.yaml. Returns empty string if not set."""
        current = self.read_current()
        return current.get("current_experiment", "")

    # ── Hardware ────────────────────────────────────────────────────────

    def read_hardware(self) -> Dict:
        """Read hardware.yaml. Returns empty dict if not exists."""
        if not os.path.isfile(self._hardware_path):
            return {}
        try:
            with open(self._hardware_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def write_hardware(self, content: Dict) -> str:
        """Write hardware.yaml."""
        os.makedirs(self._dir, exist_ok=True)
        try:
            with open(self._hardware_path, "w", encoding="utf-8") as f:
                yaml.dump(content, f, allow_unicode=True, default_flow_style=False)
            return "Hardware info written."
        except Exception as e:
            return f"ERROR: Failed to write hardware.yaml: {e}"

    # ── Experiments ─────────────────────────────────────────────────────

    def _experiment_path(self, name: str) -> str:
        safe_name = name.replace("/", "_").replace(" ", "_")
        return os.path.join(self._experiments_dir, f"{safe_name}.yaml")

    def create_experiment(self, name: str, purpose: str, hypothesis: str, config: Dict, exp_dir: str) -> str:
        """Create a new experiment file."""
        os.makedirs(self._experiments_dir, exist_ok=True)
        path = self._experiment_path(name)

        if os.path.isfile(path):
            return f"ERROR: Experiment '{name}' already exists. Use add_attempt to update it."

        experiment = {
            "name": name,
            "purpose": purpose,
            "hypothesis": hypothesis,
            "config": config,
            "dir": exp_dir,
            "attempts": [],
            "status": "running",
            "root_cause": None,
            "learnings": [],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(experiment, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Experiment '{name}' created."
        except Exception as e:
            return f"ERROR: Failed to create experiment: {e}"

    def read_experiment(self, name: str) -> Optional[Dict]:
        """Read a specific experiment file. Returns None if not exists."""
        path = self._experiment_path(name)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def add_attempt(self, name: str, change: str, result: str) -> str:
        """Append an attempt to an experiment."""
        exp = self.read_experiment(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."

        attempt_id = len(exp.get("attempts", [])) + 1
        attempt = {
            "id": attempt_id,
            "change": change,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        exp.setdefault("attempts", []).append(attempt)

        try:
            path = self._experiment_path(name)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Attempt #{attempt_id} added to '{name}'."
        except Exception as e:
            return f"ERROR: Failed to add attempt: {e}"

    def update_last_attempt(self, name: str, result: str) -> str:
        """Update the result of the last attempt."""
        exp = self.read_experiment(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."

        attempts = exp.get("attempts", [])
        if not attempts:
            return f"ERROR: No attempts in experiment '{name}'."

        attempts[-1]["result"] = result

        try:
            path = self._experiment_path(name)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Last attempt in '{name}' updated."
        except Exception as e:
            return f"ERROR: Failed to update attempt: {e}"

    def finalize_experiment(self, name: str, status: str, root_cause: Optional[str], learnings: List[str]) -> str:
        """Finalize an experiment with status, root_cause, and learnings."""
        exp = self.read_experiment(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."

        exp["status"] = status
        exp["root_cause"] = root_cause
        exp["learnings"] = learnings

        try:
            path = self._experiment_path(name)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Experiment '{name}' finalized with status '{status}'."
        except Exception as e:
            return f"ERROR: Failed to finalize experiment: {e}"

    def list_experiments(self) -> List[Dict]:
        """List all experiments (name + status only)."""
        if not os.path.isdir(self._experiments_dir):
            return []

        experiments = []
        for fname in sorted(os.listdir(self._experiments_dir)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._experiments_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    exp = yaml.safe_load(f)
                    experiments.append({
                        "name": exp.get("name", fname[:-5]),
                        "status": exp.get("status", "unknown"),
                    })
            except Exception:
                continue
        return experiments

    # ── Migration ───────────────────────────────────────────────────────

    def migrate_from_markdown(self, old_path: str) -> str:
        """Migrate from old workspace_state.md to new structure."""
        if not os.path.isfile(old_path):
            return "ERROR: Old workspace_state.md not found."

        try:
            with open(old_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"ERROR: Failed to read old file: {e}"

        # Parse sections
        sections = {}
        lines = content.split("\n")
        current_section = None
        current_lines = []

        for line in lines:
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        # Create current.yaml from Session Summary
        summary = sections.get("Session Summary", "")
        current = {
            "task": "Migrated from old workspace_state.md",
            "status": "unknown",
            "session_summary": summary,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(self._dir, exist_ok=True)
        with open(self._current_path, "w", encoding="utf-8") as f:
            yaml.dump(current, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # Create hardware.yaml from Hardware section
        hardware_text = sections.get("Hardware", "")
        if hardware_text:
            hardware = {"info": hardware_text}
            with open(self._hardware_path, "w", encoding="utf-8") as f:
                yaml.dump(hardware, f, allow_unicode=True, default_flow_style=False)

        # Parse Experiments section into individual files
        experiments_text = sections.get("Experiments", "")
        if experiments_text and "### " in experiments_text:
            os.makedirs(self._experiments_dir, exist_ok=True)
            entries = experiments_text.split("### ")[1:]
            for entry in entries:
                lines = entry.split("\n")
                first_line = lines[0].strip()
                # Extract name from "exp_name (status)" format
                if "(" in first_line:
                    name = first_line.split("(")[0].strip()
                    status = first_line.split("(")[1].rstrip(")").strip()
                else:
                    name = first_line
                    status = "unknown"

                exp_content = "\n".join(lines[1:]).strip()
                experiment = {
                    "name": name,
                    "purpose": "Migrated from old workspace_state.md",
                    "hypothesis": "",
                    "config": {},
                    "dir": "",
                    "attempts": [],
                    "status": status,
                    "root_cause": None,
                    "learnings": [],
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "migrated_content": exp_content,
                }
                path = self._experiment_path(name)
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(experiment, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        return f"Migrated from {old_path}. Created current.yaml, hardware.yaml, and {len(entries) if experiments_text else 0} experiment files."
