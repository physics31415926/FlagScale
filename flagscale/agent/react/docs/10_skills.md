# Skill System

## Overview

Skills are domain-specific knowledge packages that guide the agent through complex tasks. Each skill is a `SKILL.md` file containing structured instructions, phase guidance, and domain rules.

## Skill Discovery

`SkillManager` scans the `skills/` directory for subdirectories containing `SKILL.md`:

```
flagscale/agent/react/skills/
├── model-porter/SKILL.md     # Model porting methodology
├── data-prep/SKILL.md        # Data pipeline preparation
├── train-config/SKILL.md     # Training config generation
├── train-run/SKILL.md        # Training execution
├── train-monitor/SKILL.md    # Training monitoring
├── env-setup/SKILL.md        # Environment setup
├── topo-detect/SKILL.md      # Hardware topology detection
├── reproduce/SKILL.md        # Baseline reproduction
└── precision-alignment/SKILL.md  # Loss curve alignment
```

## Skill Loading

1. **Auto-detection**: `JudgesMixin._skill_judge()` matches user intent to available skills via keyword matching
2. **Manual loading**: User can explicitly load via `load_skill` tool
3. **Injection**: Skill content is wrapped in XML tags and injected into the system prompt

## Key Skill Features

### model-porter
- **Analysis 0**: Mandatory model structure enumeration before any code
- Component checklist tracking
- Diagnostic print strategy at module boundaries

### data-prep
- **Critical prerequisite**: Parallelism strategy documentation (TP/PP/DP/EP/CP/SP)
- Covers MoE routing, context parallelism, packed samples
- Must document all 6 parallel dimensions before writing data code

## Skill Integration Points

| Component | Integration |
|-----------|------------|
| `prompt.py` | Injects skill content into system prompt |
| `gates.py` | Phase ordering checks reference active skill |
| `skill_lifecycle.py` | Manages phase transitions within skill |
| `agent.py` | Auto-loads skills based on user intent |
