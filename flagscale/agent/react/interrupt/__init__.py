"""Interrupt layer — A类 gate (横切行为约束).

Interrupts are self-contained objects that own their state and make
independent decisions. They are NOT callable functions on the agent.

Key design:
- Each Interrupt owns its counters/flags internally (no agent._xxx scatter)
- Activation is controlled by `activate_on` constraints matching ScenePreset
- Interventions are returned as data, agent decides what to action
"""

from __future__ import annotations

from .base import Interrupt, Intervention, Observation

__all__ = ["Interrupt", "Intervention", "Observation"]
