"""Experiment manager — manages per-experiment YAML files."""

import logging
import os
import time
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages experiment records under a session-specific directory.

    Schema:
        Experiment (top-level):
            name, purpose, hypothesis, status, created, attempts[],
            root_cause, learnings[], finalized_at

        Attempt:
            timestamp, change, hardware{gpus, gpu_type, driver?, cuda?},
            config{model, tp, dp, pp?, global_batch_size, seq_length,
                   train_iters, precision, ...},
            output_dir, result
    """

    _CONFIG_REQUIRED_KEYS = ("model", "tp", "dp")

    def __init__(self, experiments_dir: str):
        self._dir = experiments_dir

    def _path(self, name: str) -> str:
        safe = name.replace("/", "_").replace(" ", "_")
        return os.path.join(self._dir, f"{safe}.yaml")

    def create(self, name: str, purpose: str, hypothesis: str = "") -> str:
        if os.path.isfile(self._path(name)):
            return f"ERROR: Experiment '{name}' already exists."
        os.makedirs(self._dir, exist_ok=True)
        exp = {
            "name": name,
            "purpose": purpose,
            "hypothesis": hypothesis,
            "status": "running",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attempts": [],
            "root_cause": "",
            "learnings": [],
            "finalized_at": "",
        }
        self._save(name, exp)
        return f"Experiment '{name}' created."

    def read(self, name: str) -> Optional[Dict]:
        path = self._path(name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or None

    def _save(self, name: str, exp: Dict):
        with open(self._path(name), "w", encoding="utf-8") as f:
            yaml.dump(exp, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def add_attempt(self, name: str, change: str, hardware: Dict = None,
                    config: Dict = None, output_dir: str = "") -> str:
        exp = self.read(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."
        config = config or {}
        warnings = self._validate_attempt_config(config)
        attempt = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "change": change,
            "hardware": hardware or {},
            "config": config,
            "output_dir": output_dir,
            "result": "(pending)",
        }
        exp.setdefault("attempts", []).append(attempt)
        exp["status"] = "running"
        self._save(name, exp)
        msg = f"Attempt #{len(exp['attempts'])} added to '{name}'."
        if warnings:
            msg += f"\nWARNING: {warnings}"
        return msg

    def _validate_attempt_config(self, config: Dict) -> str:
        """Warn if attempt config is missing key training parameters."""
        if not config:
            return "config is empty — should contain model, tp, dp, global_batch_size, etc."
        missing = [k for k in self._CONFIG_REQUIRED_KEYS if k not in config]
        if missing:
            return f"config missing recommended fields: {', '.join(missing)}"
        # Reject non-training keys mixed into config
        non_config_keys = {"reason", "fix", "note", "description", "change"}
        bad_keys = non_config_keys & set(config.keys())
        if bad_keys:
            return (f"config contains non-config fields: {', '.join(bad_keys)}. "
                    "Use the 'change' parameter for descriptions, keep config for training parameters only.")
        return ""

    def update_last_attempt(self, name: str, result: str) -> str:
        exp = self.read(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."
        attempts = exp.get("attempts", [])
        if not attempts:
            return f"ERROR: No attempts in '{name}'."
        attempts[-1]["result"] = result
        self._save(name, exp)
        return f"Updated last attempt result for '{name}'."

    def finalize(self, name: str, status: str, root_cause: str = "",
                 learnings: List[str] = None) -> str:
        exp = self.read(name)
        if not exp:
            return f"ERROR: Experiment '{name}' not found."
        exp["status"] = status
        exp["root_cause"] = root_cause
        exp["learnings"] = learnings or []
        exp["finalized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save(name, exp)
        return f"Experiment '{name}' finalized as '{status}'."

    def list(self) -> List[Dict]:
        if not os.path.isdir(self._dir):
            return []
        results = []
        for f in sorted(os.listdir(self._dir)):
            if not f.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, f)
            try:
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

    def get_current_experiment(self) -> str:
        """Return the name of the most recent running experiment, or ''."""
        for exp_info in reversed(self.list()):
            if exp_info.get("status") == "running":
                return exp_info["name"]
        return ""
