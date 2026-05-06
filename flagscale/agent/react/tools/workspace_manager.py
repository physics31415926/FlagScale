"""Workspace manager — manages current state, session state, experiments, and hardware info.

File layout under ~/.flagscale/workspace_state/:
  current.yaml       — cross-session persistent state (task, context, current_experiment, recent_sessions)
  session_state.json — ephemeral in-session state (crash recovery + compaction resume)
  hardware.yaml      — hardware info (GPUs, driver, etc.)
  experiments/       — per-experiment YAML files
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class WorkspaceManager:
    """Manages workspace state: current.yaml, session_state.json, hardware.yaml, experiments/."""

    _MAX_RECENT_SESSIONS = 3

    def __init__(self, workspace_dir: str = ""):
        self._dir = workspace_dir or os.path.join(Path.home(), ".flagscale", "workspace_state")
        self._current_path = os.path.join(self._dir, "current.yaml")
        self._hardware_path = os.path.join(self._dir, "hardware.yaml")
        self._session_state_path = os.path.join(self._dir, "session_state.json")
        self._experiments_dir = os.path.join(self._dir, "experiments")

    # ── Current State (cross-session persistent) ───────────────────────

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

        Supported fields: task, status, current_experiment, blockers, next_steps, context.
        """
        os.makedirs(self._dir, exist_ok=True)
        current = self.read_current()

        for key, value in kwargs.items():
            if key in ("task", "status", "current_experiment"):
                current[key] = value
            elif key in ("blockers", "next_steps", "context"):
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
        """Get the task field from current.yaml."""
        current = self.read_current()
        return current.get("task", "")

    def write_current(self, data: Dict) -> str:
        """Write full current.yaml (used by auto-update mechanisms)."""
        os.makedirs(self._dir, exist_ok=True)
        try:
            with open(self._current_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return "OK"
        except Exception as e:
            return f"ERROR: {e}"

    def get_current_experiment(self) -> str:
        """Get the current_experiment field from current.yaml."""
        current = self.read_current()
        return current.get("current_experiment", "")

    # ── Recent Sessions (stored in current.yaml) ──────────────────────

    def append_session_summary(self, session_id: str, task: str, summary: str, metadata: str = "") -> str:
        """Append a session summary to current.yaml's recent_sessions list. Keeps last N."""
        os.makedirs(self._dir, exist_ok=True)
        current = self.read_current()

        entry = {
            "session_id": session_id,
            "task": task[:100],
            "summary": summary[:150] if summary else "",
            "metadata": metadata,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        recent = current.get("recent_sessions", [])
        if not isinstance(recent, list):
            recent = []
        recent.append(entry)
        if len(recent) > self._MAX_RECENT_SESSIONS:
            recent = recent[-self._MAX_RECENT_SESSIONS:]
        current["recent_sessions"] = recent
        current["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(self._current_path, "w", encoding="utf-8") as f:
                yaml.dump(current, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return "OK"
        except Exception as e:
            return f"ERROR: {e}"

    def get_recent_sessions(self, n: int = 3) -> List[Dict]:
        """Get the N most recent session summaries from current.yaml."""
        current = self.read_current()
        recent = current.get("recent_sessions", [])
        if not isinstance(recent, list):
            return []
        return recent[-n:]

    # ── Session State (ephemeral, in-session only) ─────────────────────

    def write_session_state(self, state: Dict) -> str:
        """Write session_state.json — unified crash recovery + compaction resume state."""
        os.makedirs(self._dir, exist_ok=True)
        state["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["generated_ts"] = time.time()
        try:
            with open(self._session_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            return "OK"
        except Exception as e:
            return f"ERROR: {e}"

    def read_session_state(self) -> Dict:
        """Read session_state.json. Returns empty dict if not exists."""
        if not os.path.isfile(self._session_state_path):
            return {}
        try:
            with open(self._session_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def clear_session_state(self):
        """Remove session_state.json (called at session end or new session start)."""
        try:
            if os.path.isfile(self._session_state_path):
                os.remove(self._session_state_path)
        except Exception:
            pass

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

    def create_experiment(self, name: str, purpose: str, hypothesis: str, base_config: Dict, base_dir: str) -> str:
        """Create a new experiment file.

        base_config: initial/baseline config for this experiment.
        base_dir: initial log directory for this experiment.
        Subsequent attempts record their own config and output_dir.
        """
        os.makedirs(self._experiments_dir, exist_ok=True)
        path = self._experiment_path(name)

        if os.path.isfile(path):
            return f"ERROR: Experiment '{name}' already exists. Use add_attempt to update it."

        experiment = {
            "name": name,
            "purpose": purpose,
            "hypothesis": hypothesis,
            "base_config": base_config,
            "base_dir": base_dir,
            "attempts": [],
            "events": [],
            "status": "running",
            "root_cause": None,
            "learnings": [],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(experiment, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            self.update_current(current_experiment=name)
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

    def find_latest_experiment(self) -> str:
        """Find the most recently modified experiment file. Returns name or empty string."""
        if not os.path.isdir(self._experiments_dir):
            return ""
        try:
            files = [f for f in os.listdir(self._experiments_dir) if f.endswith(".yaml")]
            if not files:
                return ""
            latest = max(files, key=lambda f: os.path.getmtime(os.path.join(self._experiments_dir, f)))
            return latest.replace(".yaml", "")
        except Exception:
            return ""

    def add_attempt(self, name: str, change: str, output_dir: str, config: Optional[Dict] = None, result: str = "") -> str:
        """Append an attempt to an experiment.

        Each attempt records: what changed, where output goes, config for this run.
        Result starts empty (pending) and gets filled by update_last_attempt.
        """
        exp = self.read_experiment(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."

        # Enforce unique output_dir across all attempts
        existing_dirs = [a.get("output_dir", "") for a in exp.get("attempts", [])]
        if output_dir in existing_dirs:
            return (
                f"ERROR: output_dir '{output_dir}' already used by a previous attempt.\n"
                "Each attempt MUST have a unique output_dir. Use a timestamp or attempt number suffix."
            )

        attempt_id = len(exp.get("attempts", [])) + 1
        attempt = {
            "id": attempt_id,
            "change": change,
            "config": config or {},
            "output_dir": output_dir,
            "result": result or "pending",
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
        """Update the result of the most recent attempt."""
        exp = self.read_experiment(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."
        attempts = exp.get("attempts", [])
        if not attempts:
            return f"ERROR: No attempts in '{name}'."
        attempts[-1]["result"] = result
        try:
            path = self._experiment_path(name)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Last attempt in '{name}' updated."
        except Exception as e:
            return f"ERROR: Failed to update attempt: {e}"

    def has_pending_attempt(self, name: str) -> bool:
        """Check if the latest attempt is still pending (no result yet)."""
        exp = self.read_experiment(name)
        if not exp:
            return False
        attempts = exp.get("attempts", [])
        if not attempts:
            return False
        return attempts[-1].get("result") == "pending"

    def add_event(self, name: str, event_type: str, detail: str) -> str:
        """Append a lightweight event to the experiment's event log.

        Events are non-training actions: code changes, kills, verification steps, etc.
        They don't have config/output_dir — those are for attempts (actual training runs).
        """
        exp = self.read_experiment(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."

        event = {
            "type": event_type,
            "detail": detail[:300],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        exp.setdefault("events", []).append(event)

        try:
            path = self._experiment_path(name)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Event '{event_type}' recorded."
        except Exception as e:
            return f"ERROR: Failed to add event: {e}"

    def finalize_experiment(self, name: str, status: str, root_cause: Optional[str], learnings: List[str]) -> str:
        """Finalize an experiment with status, root_cause, and learnings."""
        exp = self.read_experiment(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."

        exp["status"] = status
        exp["root_cause"] = root_cause
        exp["learnings"] = learnings
        exp["finalized"] = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            path = self._experiment_path(name)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return f"Experiment '{name}' finalized as '{status}'."
        except Exception as e:
            return f"ERROR: Failed to finalize experiment: {e}"

    def list_experiments(self) -> List[Dict]:
        """List all experiments with basic info."""
        if not os.path.isdir(self._experiments_dir):
            return []
        results = []
        try:
            for f in sorted(os.listdir(self._experiments_dir)):
                if not f.endswith(".yaml"):
                    continue
                path = os.path.join(self._experiments_dir, f)
                with open(path, "r", encoding="utf-8") as fh:
                    exp = yaml.safe_load(fh) or {}
                results.append({
                    "name": exp.get("name", f.replace(".yaml", "")),
                    "status": exp.get("status", "unknown"),
                    "attempts": len(exp.get("attempts", [])),
                    "created": exp.get("created", ""),
                })
        except Exception:
            pass
        return results

    # ── Migration ──────────────────────────────────────────────────────

    def migrate_from_old_format(self):
        """Migrate old session_history.yaml and .agent_state.json/snapshot.yaml to new format."""
        migrated = []

        # Migrate session_history.yaml → current.yaml recent_sessions
        old_history = os.path.join(self._dir, "session_history.yaml")
        if os.path.isfile(old_history):
            try:
                with open(old_history, "r", encoding="utf-8") as f:
                    history = yaml.safe_load(f) or []
                if isinstance(history, list) and history:
                    current = self.read_current()
                    recent = history[-self._MAX_RECENT_SESSIONS:]
                    current["recent_sessions"] = recent
                    self.write_current(current)
                    migrated.append("session_history.yaml → current.yaml:recent_sessions")
                os.remove(old_history)
            except Exception:
                pass

        # Remove old .agent_state.json and snapshot.yaml (superseded by session_state.json)
        old_agent_state = os.path.join(self._dir, ".agent_state.json")
        old_snapshot = os.path.join(self._dir, "snapshot.yaml")
        for old_file in (old_agent_state, old_snapshot):
            if os.path.isfile(old_file):
                try:
                    os.remove(old_file)
                    migrated.append(f"removed {os.path.basename(old_file)}")
                except Exception:
                    pass

        return migrated
