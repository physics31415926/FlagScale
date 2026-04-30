"""Find latest FlagScale training logs with intelligent rank scanning."""

import math
import os
import re
import subprocess
import time

from flagscale.agent.react.tools.base import Tool


def _last_sorted_subdir(parent: str, key=None):
    """Return the last subdirectory under parent when sorted by key."""
    if not os.path.isdir(parent):
        return ""
    entries = [e for e in os.listdir(parent) if os.path.isdir(os.path.join(parent, e))]
    if not entries:
        return ""
    entries.sort(key=key)
    return os.path.join(parent, entries[-1])


def _numeric_key(name: str):
    """Extract trailing number for sorting: 'attempt_2' -> 2, '7' -> 7."""
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else 0


def _tail(path: str, n: int = 50) -> str:
    try:
        out = subprocess.run(
            ["tail", f"-{n}", path],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout or "(empty)"
    except Exception as e:
        return f"ERROR reading: {e}"


def _parse_megatron_metrics(text: str) -> dict:
    """Extract training metrics from Megatron log output."""
    metrics = {"iterations": [], "last_iter": None, "last_loss": {}, "anomalies": []}
    for line in text.splitlines():
        m = re.search(r'iteration\s+(\d+)', line, re.IGNORECASE)
        if not m:
            continue
        iteration = int(m.group(1))
        metrics["iterations"].append(iteration)
        metrics["last_iter"] = iteration
        for field_pattern in [
            (r'lm loss[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'lm_loss'),
            (r'ce[_ ]?loss[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'ce_loss'),
            (r'loss[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'loss'),
            (r'grad[ _]norm[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'grad_norm'),
            (r'num[_ ]zeros[:\s]+([\d.]+(?:E[+-]?\d+)?)', 'num_zeros'),
        ]:
            fm = re.search(field_pattern[0], line, re.IGNORECASE)
            if fm:
                try:
                    metrics["last_loss"][field_pattern[1]] = float(fm.group(1))
                except ValueError:
                    pass
    return metrics


def _health_check(metrics: dict, vocab_size: int = 0) -> list:
    """Run training health checks on parsed metrics."""
    warnings = []
    loss_val = (
        metrics["last_loss"].get("ce_loss")
        or metrics["last_loss"].get("lm_loss")
        or metrics["last_loss"].get("loss")
    )
    if loss_val is not None and vocab_size > 0:
        random_loss = math.log(vocab_size)
        if loss_val > random_loss * 0.8:
            warnings.append(
                f"WARNING: loss={loss_val:.4f} ~ ln({vocab_size})={random_loss:.2f} "
                f"-> model may be outputting random. Check: weights loaded? forward pass correct?"
            )
    grad_norm = metrics["last_loss"].get("grad_norm")
    if grad_norm is not None and grad_norm == 0:
        warnings.append("WARNING: grad_norm=0 -> gradients not flowing. Check loss computation and frozen params.")
    num_zeros = metrics["last_loss"].get("num_zeros")
    if num_zeros is not None and num_zeros > 1e9:
        warnings.append(f"WARNING: num_zeros={num_zeros:.2e} -> most gradients are zero.")
    return warnings


class FindLatestLogTool(Tool):
    name = "find_latest_log"
    description = (
        "Find and display the latest FlagScale training log. "
        "Scans ALL ranks to find the one with training metrics (loss/iteration). "
        "Returns structured output: loss rank stdout, error ranks stderr, health checks. "
        "Megatron prints metrics on the LAST pipeline rank, not rank 0."
    )
    parameters = {
        "type": "object",
        "properties": {
            "experiment": {
                "type": "string",
                "description": "Experiment name (e.g. qwen3_0_6b_train) or full experiment directory path",
            },
            "log_type": {
                "type": "string",
                "enum": ["stdout", "stderr", "both"],
                "description": "Which log to show. Default: both",
            },
            "lines": {
                "type": "integer",
                "description": "Number of tail lines per log file. Default: 50",
            },
            "vocab_size": {
                "type": "integer",
                "description": "Model vocab size for health check (e.g. 32000, 128256). If provided, checks if loss ~ ln(vocab_size).",
            },
        },
        "required": ["experiment"],
    }

    def __init__(self, outputs_dir: str = ""):
        self._outputs_dir = outputs_dir or os.path.join(os.getcwd(), "outputs")

    def execute(self, **kwargs) -> str:
        experiment = kwargs["experiment"]
        log_type = kwargs.get("log_type", "both")
        lines = kwargs.get("lines", 50)
        vocab_size = kwargs.get("vocab_size", 0)

        attempt_dir = self._find_attempt_dir(experiment)
        if attempt_dir.startswith("ERROR"):
            return attempt_dir

        rank_dirs = self._list_rank_dirs(attempt_dir)
        if not rank_dirs:
            return f"ERROR: No rank directories found in {attempt_dir}"

        loss_rank, loss_content, loss_metrics = self._find_loss_rank(rank_dirs, lines)
        error_ranks = self._find_error_ranks(rank_dirs)

        parts = [f"Experiment: {experiment}", f"Attempt dir: {attempt_dir}", f"Ranks: 0-{len(rank_dirs)-1}"]

        if log_type in ("stdout", "both"):
            if loss_rank is not None:
                rank_num = os.path.basename(loss_rank)
                parts.append(f"\n=== Loss rank (rank {rank_num}, stdout) ===")
                parts.append(f"Path: {os.path.join(loss_rank, 'stdout.log')}")
                parts.append(loss_content)
                if loss_metrics["last_iter"] is not None:
                    summary_parts = [f"Latest iteration: {loss_metrics['last_iter']}"]
                    for k, v in loss_metrics["last_loss"].items():
                        summary_parts.append(f"{k}: {v}")
                    parts.append("\n--- Metrics summary ---")
                    parts.append(", ".join(summary_parts))
                health_warnings = _health_check(loss_metrics, vocab_size)
                if health_warnings:
                    parts.append("\n--- Health warnings ---")
                    parts.extend(health_warnings)
            else:
                parts.append("\n=== No rank found with training metrics (loss/iteration) ===")
                last_rank = rank_dirs[-1]
                rank_num = os.path.basename(last_rank)
                stdout_path = os.path.join(last_rank, "stdout.log")
                if os.path.isfile(stdout_path):
                    parts.append(f"\nFallback: last rank {rank_num} stdout:")
                    parts.append(f"Path: {stdout_path}")
                    parts.append(_tail(stdout_path, lines))

        if error_ranks and log_type in ("stderr", "both"):
            parts.append("\n=== Error ranks ===")
            for rank_dir, stderr_content in error_ranks[:5]:
                rank_num = os.path.basename(rank_dir)
                parts.append(f"\n--- rank {rank_num} stderr ---")
                parts.append(f"Path: {os.path.join(rank_dir, 'stderr.log')}")
                parts.append(stderr_content)

        if not error_ranks and log_type in ("stderr", "both"):
            parts.append("\n=== No stderr errors detected ===")

        return "\n".join(parts)

    def _find_attempt_dir(self, experiment: str) -> str:
        """Resolve experiment to the latest attempt directory."""
        if os.path.isdir(experiment):
            exp_dir = experiment
        else:
            exp_dir = os.path.join(self._outputs_dir, experiment)
        if not os.path.isdir(exp_dir):
            return f"ERROR: Experiment directory not found: {exp_dir}"

        details_dir = os.path.join(exp_dir, "logs", "details")
        if not os.path.isdir(details_dir):
            return f"ERROR: No logs/details directory in {exp_dir}"

        node_dir = _last_sorted_subdir(details_dir)
        if not node_dir:
            return f"ERROR: No node directories in {details_dir}"
        ts_dir = _last_sorted_subdir(node_dir)
        if not ts_dir:
            return f"ERROR: No timestamp directories in {node_dir}"
        run_dir = _last_sorted_subdir(ts_dir)
        if not run_dir:
            return f"ERROR: No run directories in {ts_dir}"
        attempt_dir = _last_sorted_subdir(run_dir, key=_numeric_key)
        if not attempt_dir:
            return f"ERROR: No attempt directories in {run_dir}"
        return attempt_dir

    def _list_rank_dirs(self, attempt_dir: str) -> list:
        """List all rank directories sorted numerically."""
        if not os.path.isdir(attempt_dir):
            return []
        entries = [
            os.path.join(attempt_dir, e)
            for e in os.listdir(attempt_dir)
            if os.path.isdir(os.path.join(attempt_dir, e)) and e.isdigit()
        ]
        entries.sort(key=lambda p: int(os.path.basename(p)))
        return entries

    def _find_loss_rank(self, rank_dirs: list, lines: int):
        """Scan all ranks to find the one printing training metrics. Check last rank first."""
        search_order = list(reversed(rank_dirs))
        for rank_dir in search_order:
            stdout_path = os.path.join(rank_dir, "stdout.log")
            if not os.path.isfile(stdout_path):
                continue
            content = _tail(stdout_path, lines)
            if re.search(r'iteration\s+\d+', content, re.IGNORECASE):
                metrics = _parse_megatron_metrics(content)
                return rank_dir, content, metrics
        return None, "", {"iterations": [], "last_iter": None, "last_loss": {}, "anomalies": []}

    def _find_error_ranks(self, rank_dirs: list) -> list:
        """Find ranks with non-empty stderr containing errors."""
        error_ranks = []
        for rank_dir in rank_dirs:
            stderr_path = os.path.join(rank_dir, "stderr.log")
            if not os.path.isfile(stderr_path):
                continue
            size = os.path.getsize(stderr_path)
            if size == 0:
                continue
            content = _tail(stderr_path, 10)
            if any(kw in content.lower() for kw in ["error", "exception", "traceback", "fault", "killed", "oom"]):
                error_ranks.append((rank_dir, content))
        return error_ranks
