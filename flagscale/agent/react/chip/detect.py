"""Chip auto-detection — probes the runtime environment to identify hardware."""

from __future__ import annotations

import os
import subprocess
import logging

from flagscale.agent.react.chip.base import ChipCapability
from flagscale.agent.react.chip.registry import get_chip

logger = logging.getLogger(__name__)


def detect_chip() -> ChipCapability | None:
    """Detect the current chip from the runtime environment.

    Detection order:
    1. FLAGSCALE_CHIP_VENDOR env var (explicit override)
    2. NVIDIA GPU detection (nvidia-smi)
    3. Future: Tianshu, Ascend, ROCm, etc.

    Returns:
        ChipCapability instance or None if detection fails.
    """
    # 1. Explicit override via env var
    vendor_override = os.environ.get("FLAGSCALE_CHIP_VENDOR", "").strip().lower()
    chip_type_override = os.environ.get("FLAGSCALE_CHIP_TYPE", "").strip()
    if vendor_override:
        chip = get_chip(vendor_override, chip_type_override)
        if chip:
            logger.info("Chip detected via env override: %s/%s",
                        vendor_override, chip_type_override or "default")
            return chip

    # 2. NVIDIA detection
    chip = _detect_nvidia()
    if chip:
        return chip

    # 3. Future: Add domestic chip detection here
    # Example for Tianshu:
    # if os.environ.get("TIANSHU_HOME"):
    #     return get_chip("tianshu")
    #
    # Example for Ascend:
    # if os.environ.get("ASCEND_HOME"):
    #     return get_chip("ascend")

    logger.debug("No chip detected from environment")
    return None


def _detect_nvidia() -> ChipCapability | None:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0].upper()
            # Map GPU name to chip_type
            if "H100" in gpu_name:
                chip_type = "H100"
            elif "A100" in gpu_name:
                chip_type = "A100"
            else:
                chip_type = "A100"  # Default to A100 capabilities for unknown NVIDIA
            chip = get_chip("nvidia", chip_type)
            if chip:
                logger.info("NVIDIA GPU detected: %s → %s", gpu_name, chip_type)
            return chip
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None
