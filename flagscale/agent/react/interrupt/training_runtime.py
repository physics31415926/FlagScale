"""TrainingRuntimeInterrupt — monitor enforcement, heartbeat, hang/kill-retry/zombie detection,
auto-restart strategy, and multi-node health check reminders.

Mirrors v1's _check_monitor_after_launch_gate, _check_training_hang,
_check_kill_retry_loop, _check_gpu_zombie_escalation, and
_check_source_reading_gate.

Enhancements (P1-3):
- Heartbeat: tracks turns since last monitor, injects GPU-check reminders periodically
- Auto-restart: on crash, suggests specific config modifications (lower batch_size,
  reduce tp, add gradient_checkpointing) based on failure signature
- Multi-node health check: when multi-GPU config detected, injects nccl-tests and
  connectivity check instructions

activate_on: {"is_training"} — only for training scenes.
"""

from __future__ import annotations

import re
import time

from .base import Interrupt, Intervention, Observation


# ── Auto-restart config templates ──
_AUTO_RESTART_STRATEGIES = {
    "oom": [
        ("global_batch_size", "reduce by 50%", "halve"),
        ("gradient_accumulation_steps", "increase to compensate", "double_gas"),
        ("recompute_activations", "true", "enable_recompute"),
    ],
    "nccl": [
        ("NCCL_IB_DISABLE", "1", "disable_ib"),
        ("NCCL_SOCKET_IFNAME", "eth0", "set_nic"),
        ("NCCL_DEBUG", "INFO", "enable_nccl_debug"),
    ],
    "cuda": [
        ("precision", "bf16 → fp16", "downgrade_precision"),
        ("deterministic_mode", "false", "disable_deterministic"),
    ],
    "default": [
        ("global_batch_size", "reduce by 50%", "halve_batch"),
        ("tp", "reduce if >1", "reduce_tp"),
        ("gradient_checkpointing", "true", "enable_ckpt"),
    ],
}


class TrainingRuntimeInterrupt(Interrupt):
    """Training lifecycle management and monitoring enforcement.

    activate_on: {"is_training"} — only for training scenes.
    """

    name = "training_runtime"
    activate_on = {"is_training"}
    priority = 50

    # ── Self-owned state ──
    _awaiting_monitor: bool = False
    _monitor_gate_block_count: int = 0
    _consecutive_train_failures: int = 0
    _last_train_failure_reasons: list[str] = []
    _kill_retry_timestamps: list[float] = []
    _training_launch_timestamps: list[float] = []
    _source_reads_since_last_failure: int = 0
    _training_started: bool = False
    # ── Heartbeat state (P1-3) ──
    _turns_since_last_monitor: int = 0
    _turns_since_last_gpu_check: int = 0
    _last_launch_output_dir: str = ""
    # ── Multi-node state (P1-3) ──
    _multi_node_warned: bool = False

    # ── Thresholds ──
    _MONITOR_GATE_MAX_BLOCKS = 5
    _KILL_RETRY_WINDOW = 120  # seconds
    _KILL_RETRY_MAX = 3
    _TRAINING_HANG_SECONDS = 180
    _MIN_SOURCE_READS_BEFORE_FIX = 2
    # ── Heartbeat thresholds (P1-3) ──
    _HEARTBEAT_MONITOR_INTERVAL = 3   # inject reminder every N turns without monitor
    _HEARTBEAT_GPU_CHECK_INTERVAL = 5  # inject GPU check reminder every N turns

    def check_pre(self, obs: Observation) -> Intervention | None:
        # --- Heartbeat: track turns since last monitor (P1-3) ---
        if self._training_started:
            self._turns_since_last_monitor += 1
            self._turns_since_last_gpu_check += 1

        # --- Monitor enforcement ---
        if self._awaiting_monitor:
            if obs.tool_name == "monitor":
                self._awaiting_monitor = False
                self._monitor_gate_block_count = 0
                return None
            if obs.tool_name in ("plan_update", "workspace_experiment", "read_file"):
                return None  # Allowed

            # Allow read-only diagnostic shell commands
            if obs.tool_name == "shell":
                cmd = obs.tool_args.get("command", "")
                if self._is_read_only_diagnostic(cmd):
                    return None

            # Auto-clear after max blocks (prevents permanent deadlock)
            self._monitor_gate_block_count += 1
            if self._monitor_gate_block_count >= self._MONITOR_GATE_MAX_BLOCKS:
                self._awaiting_monitor = False
                self._monitor_gate_block_count = 0
                return None

            return Intervention(
                action="block",
                message=(
                    "[MONITOR GATE — COMMAND NOT EXECUTED]\n\n"
                    "After launching training, you MUST call monitor() to observe "
                    "the process. Training was just launched — use "
                    "monitor(output_dir=...) to watch for errors or progress "
                    "before doing anything else.\n\n"
                    "Read-only commands (pgrep, ps, cat, ls) are allowed for diagnostics."
                ),
                reason="monitor required after train launch",
            )

        # --- Source reading gate: after 2+ training failures, read source before writing fixes ---
        if self._consecutive_train_failures >= 2:
            if obs.tool_name in ("write_file", "edit_file"):
                if self._source_reads_since_last_failure < self._MIN_SOURCE_READS_BEFORE_FIX:
                    target = obs.tool_args.get("path", "") or obs.tool_args.get("file_path", "")
                    if not target or any(ext in target for ext in (".yaml", ".yml", ".md", ".txt", ".json")):
                        return None  # Config/doc files are exempt
                    return Intervention(
                        action="inject_msg",
                        message=(
                            f"\n[SOURCE READING REQUIRED] You have "
                            f"{self._consecutive_train_failures} consecutive failures "
                            f"but have only read {self._source_reads_since_last_failure} "
                            f"framework source files since the last failure.\n"
                            "Before writing another fix, read the UPSTREAM implementation:\n"
                            "- Find the actual Megatron / TransformerEngine / FlagScale code path involved\n"
                            "- Understand what the framework expects (args, shapes, dtypes, return values)\n"
                            "- Then write a fix based on what you learned, not on guessing"
                        ),
                        reason="source reading required before fix",
                    )

        # --- Heartbeat: periodic GPU check reminder (P1-3) ---
        if self._training_started and not self._awaiting_monitor:
            if self._turns_since_last_gpu_check >= self._HEARTBEAT_GPU_CHECK_INTERVAL:
                self._turns_since_last_gpu_check = 0
                return Intervention(
                    action="inject_msg",
                    message=(
                        "[HEARTBEAT] Training was launched "
                        f"{self._turns_since_last_monitor} turns ago. "
                        "Check GPU utilization and process health:\n"
                        f"  nvidia-smi\n"
                        "  ps aux | grep python\n"
                        "  tail -50 <output_dir>/log.txt\n\n"
                        "If GPU util = 0% and process exists but no output, "
                        "training may be hung — kill and diagnose."
                    ),
                    reason="periodic gpu health check",
                )

            if self._turns_since_last_monitor >= self._HEARTBEAT_MONITOR_INTERVAL:
                if self._last_launch_output_dir:
                    return Intervention(
                        action="inject_msg",
                        message=(
                            "[HEARTBEAT] No monitor call in "
                            f"{self._turns_since_last_monitor} turns. "
                            "Run monitor(output_dir=...) to check training progress, "
                            "loss curve, throughput, and error logs."
                        ),
                        reason="monitor overdue",
                    )
                else:
                    return Intervention(
                        action="inject_msg",
                        message=(
                            "[HEARTBEAT] No monitor call in "
                            f"{self._turns_since_last_monitor} turns. "
                            "Training is running but not being observed. "
                            "Check nvidia-smi for GPU utilization and look for "
                            "progress indicators in the training log."
                        ),
                        reason="monitor overdue",
                    )

        return None

    def check_post(self, obs: Observation) -> Intervention | None:
        # --- Track monitor calls for heartbeat (P1-3) ---
        if obs.tool_name == "monitor":
            self._turns_since_last_monitor = 0
            self._turns_since_last_gpu_check = 0

        # --- Detect training launch ---
        if obs.tool_name == "shell":
            cmd = obs.tool_args.get("command", "")
            if self._is_training_command(cmd):
                now = time.time()
                self._training_launch_timestamps.append(now)
                self._awaiting_monitor = True
                self._training_started = True
                self._turns_since_last_monitor = 0
                self._turns_since_last_gpu_check = 0
                self._last_launch_output_dir = obs.tool_args.get("output_dir", "")

                # --- Multi-node health check (P1-3) ---
                multi_node_msg = self._check_multi_node_setup(obs)
                if multi_node_msg and not self._multi_node_warned:
                    self._multi_node_warned = True
                    return Intervention(
                        action="inject_msg",
                        message=multi_node_msg,
                        reason="multi-node health check reminder",
                    )

            # Detect kill-retry loops
            if self._is_kill_command(cmd):
                self._kill_retry_timestamps.append(time.time())
                # Prune old entries
                cutoff = time.time() - self._KILL_RETRY_WINDOW
                self._kill_retry_timestamps = [
                    t for t in self._kill_retry_timestamps if t > cutoff
                ]
                if len(self._kill_retry_timestamps) >= self._KILL_RETRY_MAX:
                    return Intervention(
                        action="inject_msg",
                        message=(
                            "[TrainingRuntime] Kill-retry loop detected — "
                            f"{len(self._kill_retry_timestamps)} kill commands in "
                            f"{self._KILL_RETRY_WINDOW}s. "
                            "Diagnose the root cause before restarting."
                        ),
                        reason="kill-retry loop",
                    )

        # --- Track training failures (with auto-restart strategy, P1-3) ---
        if obs.tool_result and self._is_training_failure(obs.tool_result):
            self._consecutive_train_failures += 1
            self._last_train_failure_reasons.append(obs.tool_result[-300:])
            self._source_reads_since_last_failure = 0

            # Determine failure category for auto-restart strategy
            failure_lower = obs.tool_result.lower()
            strategy = _AUTO_RESTART_STRATEGIES["default"]
            if "oom" in failure_lower or "out of memory" in failure_lower:
                strategy = _AUTO_RESTART_STRATEGIES["oom"]
            elif "nccl" in failure_lower:
                strategy = _AUTO_RESTART_STRATEGIES["nccl"]
            elif "cuda" in failure_lower:
                strategy = _AUTO_RESTART_STRATEGIES["cuda"]

            strategy_lines = "\n".join(
                f"  - {k}: {desc}" for k, desc, _ in strategy
            )
            restart_msg = (
                f"\n[AUTO-RESTART STRATEGY] Detected failure category, "
                f"suggested config modifications before next attempt:\n"
                f"{strategy_lines}\n"
                "After applying fixes, call add_attempt() with new config, "
                "then relaunch training."
            )

            # Auto-compare: diff last two attempts' configs to pinpoint what changed
            compare_msg = ""
            if obs.current_experiment_name and obs.experiment_diff_fn:
                try:
                    diff_result = obs.experiment_diff_fn(obs.current_experiment_name)
                    if diff_result.get("diffs"):
                        compare_msg = (
                            f"\n\n[AUTO-COMPARE] Config diffs between last two attempts "
                            f"of '{obs.current_experiment_name}':\n"
                            f"{diff_result['summary']}\n"
                            "Review which config change likely caused this failure."
                        )
                except Exception:
                    pass

            if self._consecutive_train_failures >= 3:
                return Intervention(
                    action="escalate",
                    message=(
                        f"[TrainingRuntime] {self._consecutive_train_failures} "
                        "consecutive training failures. The current configuration "
                        "will not succeed without changes. Diagnose root cause "
                        f"before retrying.{restart_msg}{compare_msg}"
                    ),
                    reason="consecutive training failures",
                )
            elif compare_msg:
                return Intervention(
                    action="inject_msg",
                    message=(compare_msg.strip() + restart_msg),
                    reason="config diff and restart strategy after failure",
                )
        else:
            # Track source code reading for the source-reading gate
            if obs.tool_name == "read_file" and self._consecutive_train_failures > 0:
                path = obs.tool_args.get("path", "") or obs.tool_args.get("file_path", "")
                if path and path.endswith(".py"):
                    self._source_reads_since_last_failure += 1

        # --- GPU zombie detection ---
        if obs.tool_name == "shell" and obs.tool_result:
            zombie_msg = self._check_zombie_gpu(obs.tool_args.get("command", ""), obs.tool_result)
            if zombie_msg:
                return Intervention(
                    action="inject_msg",
                    message=zombie_msg,
                    reason="gpu zombie process detected",
                )

        return None

    def check_training_hang(self, cmd: str, result: str, elapsed: float) -> Intervention | None:
        """Check if a training command has hung (called after timeout)."""
        if not self._is_training_command(cmd):
            return None
        if elapsed < self._TRAINING_HANG_SECONDS:
            return None

        # Check if output contains progress indicators
        has_progress = bool(re.search(
            r'iteration|step|loss|throughput|iter',
            result, re.IGNORECASE,
        ))
        if not has_progress:
            return Intervention(
                action="inject_msg",
                message=(
                    f"[TrainingRuntime] Training process has been running for "
                    f"{elapsed:.0f}s with no progress indicators. The process may be "
                    f"hung. Check process status with pgrep/nvidia-smi and consider "
                    f"killing and restarting with modified configuration."
                ),
                reason=f"training hang: {elapsed:.0f}s no progress",
            )
        return None

    def reset_turn(self):
        """Heartbeat and multi-node state persist across turns (training lifecycle)."""

    # ── Multi-node check (P1-3) ────────────────────────────────────────────

    @staticmethod
    def _check_multi_node_setup(obs: Observation) -> str | None:
        """Detect multi-node config and generate health check instructions."""
        # Check for multi-node indicators in the command or config
        cmd = obs.tool_args.get("command", "")
        config = obs.tool_args.get("config", {})
        args = obs.tool_args.get("args", {})

        # Multi-node signals: nnodes > 1, hostfile, node_rank, or tp*dp > gpus_per_node
        is_multi_node = False
        indicators = []

        if "nnodes" in cmd or "node_rank" in cmd or "hostfile" in cmd:
            is_multi_node = True
            indicators.append("command line contains multi-node args")

        nnodes = config.get("nnodes") or args.get("nnodes")
        if nnodes and int(nnodes) > 1:
            is_multi_node = True
            indicators.append(f"nnodes={nnodes}")

        # Check tp*dp vs typical single-node GPU count
        tp = config.get("tp") or args.get("tp") or 1
        dp = config.get("dp") or args.get("dp") or 1
        if int(tp) * int(dp) > 8:  # >8 GPUs typical single-node max
            is_multi_node = True
            indicators.append(f"tp={tp}, dp={dp} ({(int(tp) * int(dp))} GPUs)")

        if not is_multi_node:
            return None

        return (
            "\n[MULTI-NODE HEALTH CHECK] Multi-node training detected "
            f"({' ; '.join(indicators)}). Before launching, verify:\n\n"
            "1. NCCL allreduce bandwidth:\n"
            "   mpirun -np <ngpus> -H <node1>:<ngpus_per>,<node2>:<ngpus_per> \\\n"
            "     -x NCCL_IB_DISABLE=0 -x NCCL_DEBUG=INFO \\\n"
            "     all_reduce_perf -b 8M -e 128M -f 2 -g 1\n\n"
            "2. Inter-node SSH (passwordless):\n"
            "   for host in <nodelist>; do ssh $host hostname; done\n\n"
            "3. Shared storage writability:\n"
            "   mpirun -np <nnodes> -H <nodelist> touch /shared/test_write && rm /shared/test_write\n\n"
            "4. GPU visibility:\n"
            "   mpirun -np <nnodes> -H <nodelist> nvidia-smi --query-gpu=name,memory.total --format=csv"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_read_only_diagnostic(cmd: str) -> bool:
        return bool(re.match(
            r'\s*(grep|find|cat|ls|head|tail|wc|file|stat|which|type|echo|'
            r'pwd|env|printenv|hostname|uname|date|id|whoami|ps|pgrep|'
            r'nvidia-smi|rocminfo)\b',
            cmd,
        ))

    @staticmethod
    def _is_training_command(cmd: str) -> bool:
        return bool(re.search(
            r'(?<!python -m )'
            r'(torchrun|deepspeed|flagscale\s+train|'
            r'python.*pretrain|python.*train|python.*finetune|'
            r'mpirun|horovodrun)',
            cmd, re.IGNORECASE,
        ))

    @staticmethod
    def _is_kill_command(cmd: str) -> bool:
        return bool(re.search(r'\b(kill|pkill|killall)\b', cmd))

    @staticmethod
    def _is_training_failure(result: str) -> bool:
        lower = result.lower()
        return any(kw in lower for kw in [
            "traceback", "runtimeerror", "exitcode=1", "exitcode 1",
            "oom", "out of memory", "cuda error", "nccl error",
            "bus error", "segfault", "killed",
        ])

    @staticmethod
    def _check_zombie_gpu(cmd: str, result: str) -> str | None:
        """Detect GPU zombie processes and provide escalation strategy."""
        zombie_indicators = [
            "nvidia-smi" in cmd and "No running processes found" not in result,
            "process.*still.*running" in result.lower(),
            "cannot.*allocate.*memory" in result.lower() and "still.*in use" in result.lower(),
        ]
        if not any(zombie_indicators):
            return None
        return (
            "\n[GPU ZOMBIE WARNING] Possible zombie GPU processes detected. "
            "Action plan:\n"
            "1. Identify: nvidia-smi | grep python ; pgrep -a python\n"
            "2. Kill: kill -9 <PID> (for each zombie process)\n"
            "3. Verify: nvidia-smi should show 0MiB memory used\n"
            "4. If zombies persist: fuser -v /dev/nvidia* to find PIDs"
        )
