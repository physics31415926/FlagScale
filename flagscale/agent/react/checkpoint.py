"""Checkpoint capture and auto-persistence mixin."""

import logging
import os
import re

logger = logging.getLogger(__name__)


class CheckpointMixin:
    """Records training events and auto-persists state on key actions."""

    def _checkpoint_training_launch(self, cmd: str, result: str):
        """Checkpoint: training launched successfully."""
        return ""

    def _checkpoint_training_failure(self, cmd: str, result: str):
        """Checkpoint: training failed. Record to experiment."""
        current_exp = self._experiment_manager.get_current_experiment()
        if not current_exp:
            return ""

        error_summary = self._extract_error_summary(result)
        warning = ""
        try:
            self._experiment_manager.update_last_attempt(current_exp, f"FAILED: {error_summary}")
        except Exception as e:
            logger.warning("Failed to update experiment attempt: %s", e)
            warning = f"\n⚠️ Experiment update failed: {e}. Failure not recorded.\n"

        return warning

    def _check_hydra_cache_stale(self, cmd: str) -> str:
        """Check if config files were edited but hydra output dir still has old cache."""
        config_edits = [
            f for f in self._files_written_this_session
            if re.search(r'\.(yaml|yml)$', f) and re.search(r'conf|config|example', f, re.I)
        ]
        if not config_edits:
            return ""

        output_dir = ""
        m = re.search(r'outputs/(\S+)', cmd)
        if m:
            output_dir = m.group(0)
        else:
            exp_name = self._experiment_manager.get_current_experiment()
            if exp_name:
                output_dir = f"outputs/{exp_name}"

        if not output_dir:
            return ""

        return (
            f"\n⚠️ HYDRA CACHE WARNING: You edited config files ({', '.join(os.path.basename(f) for f in config_edits[:3])}) "
            f"but FlagScale may use cached config from a previous run in {output_dir}/hydra/. "
            f"If the training fails with the SAME error as before, delete {output_dir}/hydra/ and relaunch. "
            f"Or pass `--config-dir` to force Hydra to re-resolve.\n"
        )

    def _checkpoint_new_error(self, error_signature: str, full_error: str):
        """Checkpoint: new unique error encountered."""
        if not hasattr(self, "_seen_errors"):
            self._seen_errors = set()

        if error_signature in self._seen_errors:
            return ""
        self._seen_errors.add(error_signature)
        return ""

    def _extract_error_summary(self, result: str) -> str:
        """Extract first meaningful error line from tool result."""
        lines = result.split("\n")
        # Skip monitor status headers ("Monitor result: ...") and section labels ("Events:", "Recent output:")
        skip_prefixes = ("Monitor result:", "Events:", "Recent output:", "Latest metrics:")
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(p) for p in skip_prefixes):
                continue
            if not stripped or stripped.startswith("["):
                continue
            if any(kw in stripped.lower() for kw in ("error", "exception", "failed", "traceback", "oom", "cuda")):
                clean = stripped.lstrip("  ")
                return clean[:200] if len(clean) <= 200 else clean[:197] + "..."
        # Fallback: first non-empty, non-header line
        for line in lines:
            stripped = line.strip()
            if stripped and not any(stripped.startswith(p) for p in skip_prefixes):
                return stripped[:200]
        return "Unknown error"

    def _record_verification_advance(self, new_stage, command_snippet):
        """Auto-record verification stage advancement to workspace experiment."""
        try:
            exp_name = self._experiment_manager.get_current_experiment()
            if exp_name:
                logger.info("Auto-recorded verification advance: %s", new_stage)
        except Exception as e:
            logger.warning("Failed to auto-record verification advance: %s", e)

    # ── Auto-Persistence Layer ─────────────────────────────────────────

    def _auto_persist_on_event(self, tool_name, arguments, result, error):
        """Automatic persistence after key events — no agent decision needed."""
        try:
            cmd = arguments.get("command", "") if tool_name == "shell" else ""

            if tool_name == "shell" and self._is_training_launch(cmd):
                if error or (result and ("Error" in result or "Traceback" in result)):
                    self._auto_record_training_attempt(cmd, result, success=False)
                elif result and ("iteration" in result.lower() or "loss" in result.lower()):
                    self._auto_record_training_attempt(cmd, result, success=True)

            if tool_name == "shell" and re.search(r'pkill|kill\s+-?\d|killall', cmd):
                self._auto_record_kill_event(cmd)

            if tool_name in ("write_file", "edit_file") and not error:
                path = arguments.get("path", "") or arguments.get("file_path", "")
                if path and self._porting_mode and re.search(r'model|train|layer|attention|mlp|embed', path, re.I):
                    self._auto_record_code_change(path)

            if tool_name == "plan_update" and arguments.get("status") == "done":
                self._auto_bind_plan_to_experiment(arguments)

        except Exception as e:
            logger.debug("Auto-persist failed: %s", e)

    def _auto_record_training_attempt(self, cmd, result, success):
        """Record training result to the current pending attempt."""
        exp_name = self._experiment_manager.get_current_experiment()
        if not exp_name:
            return
        if success:
            lines = (result or "").split("\n")
            info_lines = [l for l in lines if "iteration" in l.lower() or "loss" in l.lower()][:3]
            summary = "\n".join(info_lines)[:200] if info_lines else "Training started successfully"
            self._experiment_manager.update_last_attempt(exp_name, f"SUCCESS: {summary}")
        else:
            err_lines = []
            for line in (result or "").split("\n"):
                if "error" in line.lower() or "traceback" in line.lower() or "assert" in line.lower():
                    err_lines.append(line.strip())
            error_summary = "\n".join(err_lines[-3:])[:200] if err_lines else "Unknown error"
            self._experiment_manager.update_last_attempt(exp_name, f"FAIL: {error_summary}")

    def _auto_record_kill_event(self, cmd):
        """Record kill event — no-op after workspace_manager removal."""
        pass

    def _auto_record_code_change(self, path):
        """Record model code change — no-op after workspace_manager removal."""
        pass

    def _auto_bind_plan_to_experiment(self, arguments):
        """When plan step is done, auto-add experiment attempt — no-op after workspace_manager removal."""
        pass
