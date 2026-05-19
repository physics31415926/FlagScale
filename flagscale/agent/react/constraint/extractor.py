"""Constraint extractor — LLM-based extraction from skill prose.

Reads a skill's SKILL.md content and extracts hard constraints that can be
enforced at the tool-call level. Uses Judge.classify("extract_constraints").
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from flagscale.agent.react.constraint import Constraint, ConstraintTrigger

logger = logging.getLogger(__name__)


def extract_constraints(
    skill_content: str,
    classify_fn: Callable[[str, dict], Any],
    skill_name: str = "unknown",
) -> list[Constraint]:
    """Extract hard constraints from skill prose via LLM.

    Args:
        skill_content: Raw SKILL.md text (including frontmatter).
        classify_fn: Judge.classify or equivalent callable.
        skill_name: Skill identifier for constraint IDs.

    Returns:
        List of compiled Constraint objects. Empty list on failure.
    """
    if not skill_content.strip():
        return []

    try:
        raw = classify_fn("extract_constraints", {"skill_content": skill_content})
    except Exception as e:
        logger.warning("extract_constraints LLM call failed for skill=%s: %s", skill_name, e)
        return []

    if not isinstance(raw, list):
        logger.warning("extract_constraints returned non-list for skill=%s: %s", skill_name, type(raw))
        return []

    constraints = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            c = _compile_one(item, skill_name, i)
            if c is not None:
                constraints.append(c)
        except Exception as e:
            logger.debug("Skipping malformed constraint %d in skill=%s: %s", i, skill_name, e)

    logger.info("Extracted %d constraints from skill=%s", len(constraints), skill_name)
    return constraints


def _compile_one(item: dict, skill_name: str, index: int) -> Constraint | None:
    """Compile a single raw dict from LLM into a Constraint object.

    Expected LLM output format per item:
    {
        "description": "Never delete experiment output directories",
        "tool_names": ["shell"],
        "keywords": ["rm", "rmdir", "shutil.rmtree"],
        "severity": "error",
        "prompt": "Does this command delete an experiment output directory?",
        "correction": "Do not delete experiment directories. Use archive instead.",
        "check_phase": "pre"
    }
    """
    description = item.get("description", "").strip()
    if not description:
        return None

    # Build trigger
    tool_names = set(item.get("tool_names", []) or [])
    keywords = list(item.get("keywords", []) or [])
    trigger = ConstraintTrigger(tool_names=tool_names, keywords=keywords)

    # Build constraint
    constraint_id = f"{skill_name}_{index}"
    severity = item.get("severity", "error")
    if severity not in ("error", "warning"):
        severity = "error"

    check_phase = item.get("check_phase", "pre")
    if check_phase not in ("pre", "post"):
        check_phase = "pre"

    return Constraint(
        id=constraint_id,
        description=description,
        trigger=trigger,
        severity=severity,
        prompt=item.get("prompt", f"Does this tool call violate: {description}"),
        correction=item.get("correction", f"Constraint violated: {description}"),
        check_phase=check_phase,
    )
