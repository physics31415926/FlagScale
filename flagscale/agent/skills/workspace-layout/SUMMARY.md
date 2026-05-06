# Workspace-Layout — Summary

Standardized directory layout and storage management for FlagScale projects.

**Load when**: downloading models/data, creating conda environments, organizing experiment outputs, or before any operation that creates large files.

Workspace root = FlagScale's parent directory. Fixed paths: `models/`, `datasets/`, `envs/`, `outputs/`. Conda envs use `--prefix <root>/envs/<name>`. Includes disk space pre-checks and shared storage detection for multi-node.
