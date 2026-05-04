# Workspace Layout & Storage Management — Summary

## Overview

Standardized directory layout and storage management for FlagScale projects. Covers shared storage detection, fixed paths, experiment isolation, disk space pre-checks, and artifact deduplication.

## Standard Directory Layout

All artifacts go under `<root>` (detected storage root):

```
<root>/
├── models/          # HuggingFace checkpoints, tokenizers
├── datasets/        # Training/eval data
├── experiments/     # Training outputs (one subdir per experiment)
├── checkpoints/     # Converted Megatron checkpoints
├── logs/            # Aggregated logs
└── conda_envs/      # Conda environments (conda create --prefix)
```

## Storage Root Detection (Step 1)

Priority order:
1. **Shared storage mount** (NFS, Lustre, GPFS, Ceph, BeeGFS) — required for multi-node
2. **Largest persistent volume** — if no shared storage
3. **`/workspace`** — last resort fallback

Detection commands:
```bash
df -hT 2>/dev/null
mount | grep -iE 'type (nfs|lustre|gpfs|ceph|fuse\.ceph|beegfs|panfs|cifs)'
```

Always confirm with user before proceeding. Record choice in workspace_state.

## Key Rules

### Experiment Isolation
- Each experiment gets its own directory: `<root>/experiments/<exp_name>/`
- **NEVER overwrite** an existing experiment directory — create a new one with a suffix
- Naming: `<model>_<parallelism>_<task>_<version>` (e.g., `qwen3_tp8_pretrain_v1`)

### Model Downloads
- Use `huggingface_hub.snapshot_download(repo_id, local_dir=<root>/models/<name>)`
- Always check if model already exists before downloading
- Record path in memory after download

### Conda Environments
- Use `--prefix <root>/conda_envs/<name>` for shared storage visibility
- Activate with full path: `conda run --prefix <path> <command>`

## Disk Space Pre-checks

Before any large operation:
```bash
df -h <target_directory>
```

Estimates:
- **Checkpoint size** ≈ `param_count × 2 bytes` (BF16)
- **Total checkpoint storage** ≈ `ckpt_size × (total_steps / save_interval)`
- Warn if free space < 1.5× estimated size

## Artifact Discovery

Before creating or downloading anything:
1. Check memory for previously recorded paths
2. Check standard paths under `<root>/`
3. Check common alternatives (`~/.cache/huggingface/hub/`, `/tmp/`)
4. Only download/create what's actually missing
5. Record new paths in memory after creation

## When to Load Full Skill

Load the full `workspace-layout` skill when:
- Setting up a new project workspace from scratch
- Debugging multi-node shared storage issues
- Understanding detailed artifact deduplication rules
- Configuring custom storage layouts

This summary covers the essentials. For detailed detection scripts and edge cases, load the full skill.
