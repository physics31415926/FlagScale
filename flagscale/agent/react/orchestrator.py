"""Orchestrator — incremental routing with three execution modes.

1. Single Worker: simple task → one WorkerAgent
2. SubtaskRunner: complex multi-stage task → serial pipeline with DAG
3. BatchRunner: independent experiments → parallel workers
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .config import AgentConfig
from .profile import PROFILES, WorkerProfile
from .scene import PRESETS, ScenePreset
from .agent import WorkerAgent, WorkerResult


# ── SubtaskDefinition ─────────────────────────────────────────────────────

@dataclass
class SubtaskDefinition:
    """A single subtask in a multi-stage pipeline.

    Supports DAG dependencies: some subtasks can run in parallel
    and others depend on multiple upstream subtasks.
    """
    id: str
    description: str
    profile_name: str
    upstream_keys: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


# ── SubtaskRunner ─────────────────────────────────────────────────────────

class SubtaskRunner:
    """Executes a subtask DAG with isolated histories.

    NOT a Multi-Agent framework. Each Worker has independent HistoryManager.
    """

    @staticmethod
    def _topological_batches(
        subtasks: list[SubtaskDefinition],
    ) -> list[list[SubtaskDefinition]]:
        """Group subtasks into batches that can run in parallel.

        Each batch contains subtasks whose dependencies are all satisfied.
        Within a batch, subtasks are independent and can run concurrently.
        """
        remaining = {s.id: s for s in subtasks}
        completed: set[str] = set()
        batches: list[list[SubtaskDefinition]] = []

        while remaining:
            ready = [
                s for s in remaining.values()
                if all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                unresolved = {s.id: s.depends_on for s in remaining.values()}
                raise ValueError(f"Unresolvable DAG dependencies: {unresolved}")
            batches.append(ready)
            for s in ready:
                del remaining[s.id]
                completed.add(s.id)

        return batches

    # ── Regex patterns for task splitting ────────────────────────────────

    _SUBTASK_PATTERNS: list[tuple[str, str]] = [
        ("env|环境|conda|pip|setup|搭建|install|部署|deploy|cluster|集群",
         "train|训练|run|启动|launch|复现|reproduce|verify|验证"),
        ("migrate|迁移|port|porting",
         "train|训练|run|启动|verify|验证"),
        ("源码|source|clone|git|download|下载",
         "train|训练|config|配置|launch|启动"),
        ("复现|reproduce",
         "源码|source|config|配置|环境|env|搭建|setup"),
    ]

    _BATCH_PATTERNS: list[str] = [
        r"对比|compare|分别|分别跑|同时跑|都试试|试一下.*和|试一下.*跟",
        r"跑.*个.*(?:配置|参数|实验|组)",
        r"(?:哪个|哪种).*(?:更好|更快|更优|更稳定)",
    ]

    # ── Subtask templates ─────────────────────────────────────────────────

    _TEMPLATES: dict[str, list[SubtaskDefinition]] = {
        "env_and_reproduce": [
            SubtaskDefinition(
                id="env_setup",
                description="Detect hardware, create environment, install dependencies",
                profile_name="env-setup",
                depends_on=[],
            ),
            SubtaskDefinition(
                id="data_prep",
                description="Download and preprocess training data",
                profile_name="training-reproduce",
                depends_on=[],
            ),
            SubtaskDefinition(
                id="reproduce",
                description="Clone reference implementation, understand code and model architecture",
                profile_name="training-reproduce",
                upstream_keys=[
                    "env_path", "cuda_version", "gpu_count", "gpu_type",
                    "data_path", "tokenizer_path",
                ],
                depends_on=["env_setup", "data_prep"],
            ),
            SubtaskDefinition(
                id="train",
                description="Generate FlagScale config and launch training",
                profile_name="training-reproduce",
                upstream_keys=["source_repo", "model_arch", "config_params"],
                depends_on=["reproduce"],
            ),
            SubtaskDefinition(
                id="verify",
                description="Monitor training and verify reproduction success",
                profile_name="training-reproduce",
                upstream_keys=["output_dir", "experiment_name"],
                depends_on=["train"],
            ),
        ],
        "model_migration": [
            SubtaskDefinition(
                id="analyze",
                description="Analyze source model structure, framework, and checkpoint format",
                profile_name="model-migration",
                depends_on=[],
            ),
            SubtaskDefinition(
                id="implement",
                description="Implement Megatron-native model and checkpoint converter",
                profile_name="model-migration",
                upstream_keys=["source_model_arch", "checkpoint_layout"],
                depends_on=["analyze"],
            ),
            SubtaskDefinition(
                id="verify",
                description="Verify forward/backward/distributed correctness",
                profile_name="model-migration",
                upstream_keys=["model_path", "converter_path"],
                depends_on=["implement"],
            ),
        ],
    }

    @classmethod
    def should_split_to_subtasks(cls, user_input: str) -> bool:
        """Check if task spans multiple skill domains → needs context isolation."""
        for stage1_kw, stage2_kw in cls._SUBTASK_PATTERNS:
            if re.search(stage1_kw, user_input, re.I) and re.search(stage2_kw, user_input, re.I):
                return True
        return False

    @classmethod
    def should_batch(cls, user_input: str) -> bool:
        """Check if task wants to compare multiple variants."""
        for pat in cls._BATCH_PATTERNS:
            if re.search(pat, user_input, re.I):
                return True
        return False

    def run(
        self,
        template_name: str,
        user_input: str,
        orchestrator: "Orchestrator",
    ) -> WorkerResult:
        """Execute subtask DAG with topological batching."""
        subtasks = self._TEMPLATES[template_name]
        batches = self._topological_batches(subtasks)
        upstream: dict[str, str] = {}

        for batch in batches:
            if len(batch) == 1:
                sub = batch[0]
                context = self._build_upstream_summary(sub.upstream_keys, upstream)
                worker = orchestrator._create_worker(sub.profile_name)
                task = self._build_task(sub.description, user_input, context)
                result = worker.execute(task)
                if result.status == "failed":
                    return result
                upstream.update(result.artifacts)
            else:
                def _run_subtask(sub):
                    context = self._build_upstream_summary(sub.upstream_keys, upstream)
                    worker = orchestrator._create_worker(sub.profile_name)
                    task = self._build_task(sub.description, user_input, context)
                    return sub.id, worker.execute(task)

                batch_results: dict[str, WorkerResult] = {}
                with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as pool:
                    futures = {pool.submit(_run_subtask, s): s for s in batch}
                    for future in as_completed(futures):
                        sub_id, result = future.result()
                        batch_results[sub_id] = result

                for sub in batch:
                    result = batch_results.get(sub.id)
                    if result is None:
                        return WorkerResult(status="failed",
                            summary=f"Subtask {sub.id} returned no result")
                    if result.status == "failed":
                        return result
                    upstream.update(result.artifacts)

        return WorkerResult(
            status="success",
            summary="All subtasks completed",
            artifacts=upstream,
        )

    @staticmethod
    def _build_upstream_summary(keys: list[str], upstream: dict) -> str:
        """Build concise summary from upstream results. NOT full history."""
        lines = ["Previous stage results:"]
        for k in keys:
            if k in upstream:
                lines.append(f"  {k}: {upstream[k]}")
        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _build_task(description: str, user_input: str, context: str) -> str:
        """Build the task prompt for a subtask Worker."""
        parts = [description]
        if context:
            parts.append(f"\nContext from previous stages:\n{context}")
        parts.append(f"\nOriginal request: {user_input}")
        return "\n".join(parts)


# ── BatchRunner ────────────────────────────────────────────────────────────

class BatchRunner:
    """Execute same-type work with different parameters in parallel.

    NOT multi-agent — these are independent workers with isolated histories.
    Each uses the same WorkerProfile but different task descriptions.
    """

    def run(
        self,
        profile_name: str,
        tasks: list[str],
        orchestrator: "Orchestrator",
    ) -> list[WorkerResult]:
        """Run multiple independent workers in parallel."""

        def _run_one(task: str) -> WorkerResult:
            worker = orchestrator._create_worker(profile_name)
            return worker.execute(task)

        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
            ordered = list(pool.map(_run_one, tasks))
        return ordered

    @staticmethod
    def summarize(results: list[WorkerResult]) -> str:
        """Compare results across parallel runs."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"  Run {i}: {r.status} — {r.summary}")
        return "\n".join(lines)


# ── Orchestrator ───────────────────────────────────────────────────────────

class Orchestrator:
    """Entry point: routes user requests to the right execution mode.

    Infrastructure components (provider, tool_registry, skill_manager,
    session_memory, task_plan) are injected at construction time and
    shared across all workers. Each worker gets its own HistoryManager
    for context isolation.
    """

    def __init__(
        self,
        provider=None,
        tool_registry=None,
        skill_manager=None,
        session_memory=None,
        task_plan=None,
        experiment_manager=None,
    ):
        self.profiles: dict[str, WorkerProfile] = PROFILES
        self.presets: dict[str, ScenePreset] = PRESETS
        self.subtask_runner = SubtaskRunner()
        self.batch_runner = BatchRunner()
        self.scene: ScenePreset | None = None

        # Shared infrastructure
        self.provider = provider
        self.tool_registry = tool_registry
        self.skill_manager = skill_manager
        self.session_memory = session_memory
        self.task_plan = task_plan
        self.experiment_manager = experiment_manager

    def handle(self, user_input: str) -> str:
        """Handle a user request. Route to single Worker / SubtaskRunner / BatchRunner."""
        # 1. Detect scene
        self.scene = self._refine_scene(user_input)

        # 2. Check for batch (parallel experiments)
        if self.subtask_runner.should_batch(user_input):
            tasks = self._extract_batch_tasks(user_input)
            if len(tasks) >= 2:
                profile = self._pick_profile(user_input)
                results = self.batch_runner.run(profile, tasks, self)
                return self._format_batch_response(results)

        # 3. Check for multi-subtask (complex pipeline)
        if self.subtask_runner.should_split_to_subtasks(user_input):
            template = self._pick_template(user_input)
            result = self.subtask_runner.run(template, user_input, self)
            return self._format_response(result)

        # 4. Single worker
        profile_name = self._pick_profile(user_input)
        worker = self._create_worker(profile_name)
        result = worker.execute(user_input)
        return self._format_response(result)

    def _refine_scene(self, user_input: str) -> ScenePreset:
        """Auto-detect scene, optionally confirm with user."""
        return ScenePreset.auto_detect(user_input=user_input)

    def _pick_template(self, user_input: str) -> str:
        """Pick the right subtask template based on user intent."""
        if re.search(r"环境|env|搭建|setup|deploy", user_input, re.I):
            return "env_and_reproduce"
        if re.search(r"迁移|migrate|port", user_input, re.I):
            return "model_migration"
        return "env_and_reproduce"

    def _pick_profile(self, user_input: str) -> str:
        """Pick WorkerProfile based on user intent."""
        if re.search(r"迁移|migrate|port", user_input, re.I):
            if re.search(r"chip|ascend|昇腾|kunlun|dcu", user_input, re.I):
                return "chip-migration"
            return "model-migration"
        if re.search(r"推理|inference|deploy|部署|serving|vllm|sglang", user_input, re.I):
            return "inference-deploy"
        if re.search(r"环境|env|setup|install|conda|pip", user_input, re.I):
            return "env-setup"
        return "training-reproduce"

    def _create_worker(self, profile_name: str) -> WorkerAgent:
        """Create a fresh WorkerAgent with shared infrastructure.

        Context isolation: each worker gets its OWN HistoryManager.
        All other infrastructure is shared (injected at Orchestrator init).
        """
        profile = self.profiles[profile_name]

        # Build constraints from profile's scene_constraints
        constraints = set(profile.scene_constraints)
        if self.scene:
            constraints |= self.scene.constraints

        worker_scene = ScenePreset(
            name=profile.name,
            mode="training",
            chip_type=self.scene.chip_type if self.scene else "nvidia",
            chip_vendor_sdk=self.scene.chip_vendor_sdk if self.scene else "cuda",
            target_framework="megatron-core",
            source_framework="",
            default_precision="bf16",
            network_topology="single_node",
            constraints=constraints,
        )

        return WorkerAgent(
            config=AgentConfig(),
            scene=worker_scene,
            _provider=self.provider,
            _tool_registry=self.tool_registry,
            _skill_manager=self.skill_manager,
            _session_memory=self.session_memory,
            _task_plan=self.task_plan,
            _experiment_manager=self.experiment_manager,
        )

    def _extract_batch_tasks(self, user_input: str) -> list[str]:
        """Extract individual task descriptions for batch execution."""
        # Simple heuristic: split on common separators
        tasks = re.split(r"[;；]", user_input)
        if len(tasks) < 2:
            tasks = re.split(r"和|跟|对比|分别", user_input)
        return [t.strip() for t in tasks if t.strip()]

    @staticmethod
    def _format_response(result: WorkerResult) -> str:
        """Format WorkerResult for user display."""
        if result.status == "success":
            return f"✓ {result.summary}"
        return f"✗ [{result.status}] {result.summary}"

    @staticmethod
    def _format_batch_response(results: list[WorkerResult]) -> str:
        """Format batch results for user display."""
        summary = BatchRunner.summarize(results)
        return f"Batch comparison:\n{summary}"
