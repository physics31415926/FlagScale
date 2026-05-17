"""Scene preset — parameterizes WorkerAgent behavior by scenario.

ScenePreset replaces free-form SceneContext dataclass construction.
Users select a preset (or auto-detect), then optionally override fields.

Key design: constraints set 是机器可消费的标记:
- WorkerProfile.scene_constraints 声明"我在这些 constraint 下才激活"
- Interrupt.activate_on 声明"我在这些 constraint 下才生效"
- Checklist 根据 constraints 决定激活哪些检查项
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class ScenePreset:
    """A named preset that bundles typical scene parameters."""

    name: str  # "megatron-training-nvidia"
    mode: str  # "training" | "inference_serving" | "inference_engine"

    # Hardware
    chip_type: str  # "nvidia" | "ascend" | "kunlun" | "dcu" | "mthreads"
    chip_vendor_sdk: str  # "cuda" | "ascend" | "kunlunxin" | "rocm"

    # Framework
    target_framework: str  # "megatron-core" | "flagscale+vllm" | "flagscale+sglang"
    source_framework: str  # "" = not migrating

    # Precision
    default_precision: str  # "bf16" | "fp16" | "fp8"

    # Network
    network_topology: str  # "single_node" | "multi_node_ib" | "multi_node_roce"

    # Constraints — machine-consumable tags parameterizing behavior
    constraints: set[str] = field(default_factory=set)

    @classmethod
    def auto_detect(cls, cwd: str | None = None, user_input: str = "") -> "ScenePreset":
        """Pure function: detect scene from environment. No LLM needed."""
        cwd = cwd or os.getcwd()

        constraints: set[str] = set()

        # Detect chip type from env
        chip_type = "nvidia"
        chip_vendor_sdk = "cuda"
        if os.environ.get("ASCEND_HOME"):
            chip_type = "ascend"
            chip_vendor_sdk = "ascend"
        elif os.environ.get("ROCM_PATH"):
            chip_type = "dcu"
            chip_vendor_sdk = "rocm"

        # Detect mode from user_input
        mode = "training"
        if re.search(r"inference|serving|推理|部署|vllm|sglang", user_input, re.IGNORECASE):
            mode = "inference_serving"
            constraints.add("is_inference")
        else:
            constraints.add("is_training")

        # Detect migration intent
        if re.search(r"迁移|migrate|port|porting|from.*megatron|from.*deepspeed", user_input, re.IGNORECASE):
            constraints.add("is_migration")
            if chip_type != "nvidia":
                constraints.add("is_chip_migration")

        # Detect multi-node
        if re.search(r"多节点|multi.node|集群|cluster|slurm", user_input, re.IGNORECASE):
            constraints.add("requires_multi_node")
            network_topology = "multi_node_ib"
        else:
            network_topology = "single_node"

        # Detect RL
        if re.search(r"RL|reinforcement|强化学习|PPO|GRPO|reward", user_input, re.IGNORECASE):
            constraints.add("is_rl")

        # Determine frameworks
        source = ""
        if re.search(r"from\s+megatron|原来是?megatron|megatron.*迁移", user_input, re.IGNORECASE):
            source = "megatron"
        elif re.search(r"deepspeed|from\s+DS|deepspeed.*迁移", user_input, re.IGNORECASE):
            source = "deepspeed"
        elif re.search(r"fsdp|from\s+FSDP", user_input, re.IGNORECASE):
            source = "fsdp"
        elif re.search(r"vllm|vLLM", user_input, re.IGNORECASE):
            source = "vllm"

        target = "megatron-core"
        if mode == "inference_serving":
            target = "flagscale+vllm"

        # Precision: favor bf16 on NVIDIA, fp16 on domestic chips
        precision = "bf16" if chip_type == "nvidia" else "fp16"

        name = f"{target.split('+')[0]}-{mode}-{chip_type}"
        if source:
            name += f"-from-{source}"

        return cls(
            name=name,
            mode=mode,
            chip_type=chip_type,
            chip_vendor_sdk=chip_vendor_sdk,
            target_framework=target,
            source_framework=source,
            default_precision=precision,
            network_topology=network_topology,
            constraints=constraints,
        )


# ── Preset library ────────────────────────────────────────────────────────

PRESETS: dict[str, ScenePreset] = {
    "megatron-training-nvidia": ScenePreset(
        name="megatron-training-nvidia",
        mode="training",
        chip_type="nvidia",
        chip_vendor_sdk="cuda",
        target_framework="megatron-core",
        source_framework="",
        default_precision="bf16",
        network_topology="single_node",
        constraints={"is_training"},
    ),
    "megatron-training-ascend": ScenePreset(
        name="megatron-training-ascend",
        mode="training",
        chip_type="ascend",
        chip_vendor_sdk="ascend",
        target_framework="megatron-core",
        source_framework="",
        default_precision="fp16",
        network_topology="single_node",
        constraints={"is_training", "is_chip_migration", "flash_attn_no_ascend"},
    ),
    "vllm-inference-nvidia": ScenePreset(
        name="vllm-inference-nvidia",
        mode="inference_serving",
        chip_type="nvidia",
        chip_vendor_sdk="cuda",
        target_framework="flagscale+vllm",
        source_framework="",
        default_precision="fp16",
        network_topology="single_node",
        constraints={"is_inference"},
    ),
    "megatron-migration-deepspeed-nvidia": ScenePreset(
        name="megatron-migration-deepspeed-nvidia",
        mode="training",
        chip_type="nvidia",
        chip_vendor_sdk="cuda",
        target_framework="megatron-core",
        source_framework="deepspeed",
        default_precision="bf16",
        network_topology="single_node",
        constraints={"is_training", "is_migration"},
    ),
    # Future RL extension — one line config
    # "megatron-rl-nvidia": ScenePreset(
    #     name="megatron-rl-nvidia",
    #     mode="training",
    #     chip_type="nvidia", chip_vendor_sdk="cuda",
    #     target_framework="megatron-core", source_framework="",
    #     default_precision="bf16",
    #     network_topology="multi_node_ib",
    #     constraints={"is_training", "is_rl", "requires_multi_node"},
    # ),
}
