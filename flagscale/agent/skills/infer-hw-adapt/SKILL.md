---
name: infer-hw-adapt
description: Adapt and fix vllm-plugin-FL for specific hardware backends after plugin version upgrades. Covers progressive testing,
  patch creation (test → diagnose → patch → re-test loop), patch review, and PR submission. Assumes environment is already set up
  via infer-env-setup skill.
keywords:
- inference
- adapt
- hardware
- upgrade
- plugin
- vllm
- test
- fix
- patch
- 推理
- 适配
- 硬件适配
- 版本升级
- 推理修复
- FlagGems
- triton
parameters:
- name: hardware_backend
  description: Target hardware backend (e.g., metax, ascend, musa)
  default: metax
- name: ssh_host
  description: SSH host alias or connection string for the target machine (e.g., metax_c550, user@host). Ask user if not provided.
- name: vllm_version
  description: Target vLLM version being adapted to (e.g., 0.20.2)
- name: model_name
  description: Test model name (e.g., Qwen3.5-35B-A3B)
- name: tp_size
  description: Tensor parallel size for test model
  default: "2"
requires:
- infer-env-setup
- workspace-layout
suggests:
- debug-strategy
- ops-discipline
constraints:
- id: test_progression_order
  description: Tests must run in order — unit → functional → offline inference → serving
  trigger:
    tools:
    - shell
    keywords:
    - pytest
    - python examples/
    - vllm serve
  prompt: Check if the agent is skipping test stages (e.g., jumping to e2e without passing unit tests first)
  correction: 'Follow test progression: unit tests first, then functional, then offline inference, then serving. Fix issues at each stage before proceeding.'
- id: no_modify_vllm_source
  description: Never modify vLLM source code — all fixes go through plugin patches
  trigger:
    tools:
    - edit_file
    - write_file
    keywords:
    - site-packages/vllm
    - vllm/
  prompt: Check if the agent is modifying vLLM source instead of creating a plugin patch
  correction: Do not modify vLLM source. Create patches in the plugin's patches/ directory instead.
- id: triton_failure_diagnosis
  description: When Triton compilation fails, diagnose before patching
  trigger:
    keywords:
    - PassManager::run failed
    - triton
    - compilation
    - CompilationError
  prompt: Check if the agent is blindly patching a Triton failure without identifying the specific kernel and checking FlagGems alternatives
  correction: 'Diagnosis flow: identify kernel from traceback → check FlagGems for alternative → create targeted patch → file feedback for missing ops.'
- id: patch_review_before_pr
  description: After all tests pass, review every change for necessity before creating PR
  trigger:
    tools:
    - shell
    keywords:
    - git add
    - git commit
    - git push
  prompt: Check if the agent is committing changes without first reviewing whether each patch is minimal and necessary
  correction: 'Before committing: run git diff, verify each change is necessary, remove debug leftovers, ensure patches are hardware-gated, and confirm TODO comments exist for workarounds.'
- id: no_git_add_all
  description: Never use git add . before squashing — stage only the files you intend to include
  trigger:
    tools:
    - shell
    keywords:
    - git add .
    - git add -A
    - git add --all
  prompt: Check if the agent is staging all files at once, which may include debug tools or unintended changes
  correction: 'Before squashing: remove debug tools (git rm tools/probe_*.py tools/test_*.py), then use git reset --soft origin/main and git add only the files that belong in the PR. Never include probe scripts, temporary test utilities, or debug prints.'
- id: model_path_must_exist
  description: Verify model weights exist on disk before launching any inference
  trigger:
    tools:
    - shell
    keywords:
    - vllm serve
    - python examples/
    - offline_inference
    - MODEL_PATH
  prompt: Check if the agent is launching inference without first verifying the model path exists and contains valid weights
  correction: 'Before inference: run `ls <model_path>/config.json` (or equivalent) to confirm model weights are present. If missing, download or mount them first.'
- id: realtime_log_monitoring
  description: When running tests or inference, stream logs in real-time and save to file
  trigger:
    tools:
    - shell
    keywords:
    - pytest
    - python examples/
    - vllm serve
    - torchrun
  prompt: Check if the agent is running tests/inference without streaming and persisting output to a log file
  correction: 'Stream logs in real-time and persist to file: use `2>&1 | tee /workspace/adapt-logs/<stage>_$(date +%Y%m%d_%H%M%S).log` or tmux with `tmux capture-pane`. When diagnosing failures later, read the saved log file instead of re-running the command.'
- id: sensitive_content_check
  description: Before committing, check for passwords, tokens, keys, IPs, or proxy credentials
  trigger:
    tools:
    - shell
    keywords:
    - git add
    - git commit
    - git push
  prompt: Check if the agent is committing code that may contain sensitive content (passwords, tokens, SSH keys, hardcoded IPs, proxy credentials)
  correction: 'Before committing, run: git diff --cached | grep -iE "(password|token|secret|pem|private_key|proxy|172\.|192\.168|10\.)" — if anything matches, remove it before committing.'
soft_constraints:
- id: sync_code_before_test
  description: Remind to sync local changes to container before running tests
  trigger:
    tools:
    - shell
    keywords:
    - pytest
    - python examples/
    - vllm serve
  suggestion: 'If you edited code locally since the last sync, push changes to the container before testing. For editable installs, file changes are picked up automatically if the workspace volume is shared. If editing on a separate machine, use git push+pull to sync.'
context_injection:
  always:
  - Critical Rules
  - Test Progression
  - Execution Logs
  by_tool:
    shell:
    - Running Tests
    - Patch Strategy
    edit_file:
    - Patch Strategy
    write_file:
    - Patch Strategy
---
# Hardware Adaptation after Plugin Upgrade

Adapt and fix vllm-plugin-FL for specific hardware backends after each plugin version upgrade.

## When to Use This Skill

Every time vllm-plugin-FL upgrades its base vLLM version (e.g., 0.19 → 0.20), hardware-specific code paths may break because:
- vLLM internal APIs change (worker, model_runner, ops dispatch)
- New Triton kernels are introduced that the hardware's Triton backend doesn't support
- FlagGems op coverage may lag behind new vLLM requirements
- Plugin patch points may shift or become invalid

This skill covers the **adaptation and testing cycle** for one hardware backend per invocation. Environment setup (SSH, container, installation) is handled by `infer-env-setup`.

## Prerequisites

Before starting adaptation, ensure the environment is ready (via `infer-env-setup`):
- SSH connection confirmed
- Docker container running with correct image, device mounts, and workspace volume
- vLLM (CPU-only), vllm-plugin-FL (editable), and FlagGems installed
- All imports verified (`import vllm`, `import vllm_plugin_fl`, `import flag_gems`)

If any of these are not ready, run `infer-env-setup` first.

## Critical Rules

1. **Test in order**: unit → functional → offline inference → serving. Fix each stage before proceeding.
2. **Never modify vLLM source** — all hardware adaptations go through plugin patches.
3. **Stream and persist logs** — use `2>&1 | tee /workspace/adapt-logs/<stage>.log`; diagnose from log files, don't re-run commands.
4. **After tests pass, review all changes** — remove anything unnecessary before PR.
5. **One patch per failure** — fix one issue, re-test, then move to the next.
6. **Patches are hardware-gated** — use `if current_platform.is_<backend>()` or equivalent.
7. **Every workaround has a TODO** — state when it can be removed.
8. **Sync code before testing** — if editing locally, push changes to container before running tests.
9. **Check device occupancy before tests** — use the backend's monitoring tool to confirm devices are free.
10. **Use tmux for long-running commands** — SSH sessions will timeout otherwise.
11. **Workspace orientation first** — run Stage 0 probe before any test. Record plugin path, branch, vLLM version, and model path in memory. Never guess paths.
12. **Read once, memorize** — after reading any source file, record key findings with `memory_write`. Check `memory_read` before re-opening a file. Never read the same file twice unless it was modified.
13. **Squash before PR** — squash all adaptation commits into one clean commit with a comprehensive message. Remove debug tools before squashing.
14. **Batch independent tool calls** — when multiple shell commands, file reads, or memory operations are independent, execute them in one response to reduce round-trips. Example: read 3 patch files → one response with 3 `read_file` calls, not 3 separate responses.

---

## Running Tests

### Progress Tracking

Maintain a stage tracker throughout the adaptation process. After each stage, record the result:

```
=== Adaptation Progress ===
Hardware: <hardware_backend>
vLLM version: <version>
Plugin branch: adapt/<backend>-vllm-<version>
FlagGems commit: <hash>
Docker image: <image_tag>

[x] Stage 1: Unit Tests — PASS (timestamp)
[x] Stage 2: Functional Tests — PASS (timestamp)
[ ] Stage 3: Offline Inference — IN PROGRESS
[ ] Stage 4: Serving Test — PENDING
[ ] Stage 5: Patch Review — PENDING
[ ] Stage 6: PR Preparation — PENDING
```

Update this tracker as you progress. If resuming after a break, check the tracker to determine current state.

### Success Criteria

Adaptation is **complete** when ALL of the following are true:
1. Stage 1–4 all PASS (no skipped failures)
2. Stage 5 patch review confirms minimum necessary diff
3. Stage 6 PR is created with all version info recorded

### Step 0: Workspace Orientation (MANDATORY before any test)

Before running any test, collect all environment facts in **one batched probe**. Do NOT skip this step even if you think you know the paths.

```bash
# Run all of these in ONE shell call (batch them):
ssh <ssh_host> "docker exec <container> bash -c '
  echo \"=== plugin workspace ===\"
  ls /workspace/adapt-*/vllm-plugin-FL/vllm_fl/__init__.py 2>/dev/null || \
    ls /workspace/vllm-plugin-FL/vllm_fl/__init__.py 2>/dev/null || echo NOT FOUND
  echo \"=== plugin git branch ===\"
  git -C /workspace/adapt-*/vllm-plugin-FL branch --show-current 2>/dev/null || \
    git -C /workspace/vllm-plugin-FL branch --show-current 2>/dev/null
  echo \"=== vllm version ===\"
  python3 -c \"import vllm; print(vllm.__version__)\"
  echo \"=== plugin installed ===\"
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\"
  echo \"=== adapt-logs dir ===\"
  ls /workspace/adapt-logs/ 2>/dev/null || echo NOT FOUND
  echo \"=== models available ===\"
  ls /workspace/models/ 2>/dev/null || echo NOT FOUND
'"
```

**Record these findings immediately** with `memory_write`:
- `infer_plugin_workspace`: absolute path to the plugin repo inside the container
- `infer_plugin_branch`: current git branch
- `infer_vllm_version`: installed vLLM version
- `infer_models_path`: path to available models

**Use these recorded paths for ALL subsequent commands.** Never guess or reconstruct paths from memory.

## Test Progression

Run tests in strict order. Fix all failures at each stage before proceeding to the next.

### Stage 0: Workspace Orientation (MANDATORY)

**Before any test, record all paths to memory to avoid path confusion:**

```bash
ssh <ssh_host> "docker exec <container> bash -c '
  echo \"=== plugin workspace ===\" &&
  find /workspace -name \"vllm_fl\" -type d 2>/dev/null | head -5 &&
  echo \"=== plugin git info ===\" &&
  for dir in \$(find /workspace -name \".git\" -type d 2>/dev/null | grep vllm-plugin-FL); do
    repo=\$(dirname \$dir)
    echo \"Repo: \$repo\"
    git -C \$repo branch --show-current
    git -C \$repo log -1 --oneline
  done &&
  echo \"=== vllm version ===\" &&
  python3 -c \"import vllm; print(vllm.__version__)\" &&
  echo \"=== plugin installed ===\" &&
  python3 -c \"import vllm_fl; print(vllm_fl.__file__)\" &&
  echo \"=== adapt-logs ===\" &&
  ls -lh /workspace/adapt-logs/ 2>/dev/null | tail -10 &&
  echo \"=== models ===\" &&
  ls -lh /workspace/models/ 2>/dev/null | head -10
'"
```

**Immediately record to memory:**
```
memory_write('<backend>_plugin_workspace', '<discovered_path>')
memory_write('<backend>_plugin_branch', '<branch_name>')
memory_write('<backend>_vllm_version', '<version>')
memory_write('<backend>_model_path', '/workspace/models/<model_name>')
memory_write('<backend>_log_dir', '/workspace/adapt-logs')
```

**Never guess paths. Always read from memory or re-probe.**

### Stage 1: Unit Tests

```bash
cd /workspace/adapt/<backend>-vllm-<version>/vllm-plugin-FL
VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v 2>&1 | tee /workspace/adapt-logs/unit_$(date +%Y%m%d_%H%M%S).log
```

Purpose: verify import compatibility, API surface, basic plugin registration.

Monitor with `duration=120` (2 min), `process_pattern="pytest"`. Unit tests complete in under 60s on most backends; if pytest dies the monitor returns immediately rather than waiting out the full timeout.

### Stage 2: Functional Tests

```bash
VLLM_PLUGINS=fl pytest tests/functional_tests/ -x -v 2>&1 | tee /workspace/adapt-logs/functional_$(date +%Y%m%d_%H%M%S).log
```

Purpose: verify operator correctness, kernel dispatch, dtype handling.

Monitor with `duration=300` (5 min), `process_pattern="pytest"`, `fail_pattern="FAILED|ERROR|hang|timeout"`. Functional tests can hang indefinitely on graph capture failures — the `process_pattern` ensures the monitor returns if pytest dies silently. If a test hangs beyond 5 min, kill it and diagnose the specific test with `-k <test_name>`.

### Stage 3: Offline Inference

```bash
VLLM_PLUGINS=fl MODEL_PATH=/workspace/models/<model> TP_SIZE=2 \
  python examples/<model>_offline_inference.py 2>&1 | tee /workspace/adapt-logs/offline_$(date +%Y%m%d_%H%M%S).log
```

Purpose: full model execution without serving overhead. Validates model loading, forward pass, sampling.

Monitor with `duration=600` (10 min), `process_pattern="python"`, `success_pattern="Prompt.*Output:|Generated text:"`. Model loading takes 2–4 min on first run; the `success_pattern` returns as soon as output is generated rather than waiting the full timeout.

### Stage 4: Serving Test

```bash
VLLM_PLUGINS=fl vllm serve /workspace/models/<model> \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --trust-remote-code 2>&1 | tee /workspace/adapt-logs/serving_$(date +%Y%m%d_%H%M%S).log

# In another session, test the API:
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "prompt": "Hello", "max_tokens": 32}'
```

Purpose: production readiness — OpenAI-compatible API serving.

Monitor with `duration=600` (10 min), `process_pattern="vllm.*serve"`, `success_pattern="Uvicorn running|Application startup complete"`. The server is ready when Uvicorn reports startup; the monitor returns immediately rather than waiting the full timeout.

### Stage 5: Patch Review (after all tests pass)

Once all 4 test stages pass, review every modification made during the process.

**Step 1: Pull latest code to local workspace first**

```bash
# On local machine — ensure local branch is up to date with what was tested
git fetch origin
git checkout <branch>
git pull origin <branch>
```

**Step 2: Review the diff**

```bash
git diff origin/main --stat   # overview of changed files
git diff origin/main          # full diff
```

For each change, ask:
1. **Is this patch necessary?** — Could the issue be solved by configuration, environment variable, or upstream fix instead?
2. **Is the scope minimal?** — Does the patch touch only what's needed, or does it include debug leftovers, commented-out code, or unrelated formatting?
3. **Is there a TODO for removal?** — Every workaround patch must have a clear condition for when it can be removed (e.g., "remove when FlagGems adds `topk` for <backend> backend").
4. **Does it break other backends?** — Ensure patches are gated by hardware detection (`if current_platform.is_<backend>()` or equivalent), not applied unconditionally.

**Step 3: Remove non-essential files**

Debug tools, probe scripts, and temporary test files added during adaptation must NOT go into the PR. Remove them before committing:

```bash
git rm tools/probe_<backend>.py tools/test_*.py 2>/dev/null || true
```

Remove any changes that fail these checks. The goal is the **minimum diff** that makes the target hardware work.

#### Stage 6: PR Preparation

After patch review, organize changes into a clean PR.

**Step 1: Squash commits into one clean commit**

```bash
# Count commits ahead of main
git log origin/main..HEAD --oneline | wc -l

# If more than 1 commit, squash them
git reset --soft origin/main
git commit -m "fix(<backend>): adapt for vLLM <version>

- Fix 1: <what> (<why>)
- Fix 2: <what> (<why>)
...

Tested on <hardware> (<GPU count>x <GPU model>):
- Unit tests: PASS (349/349)
- Functional tests: PASS (23/23)
- Offline inference: PASS (<model>)
- Serving: PASS (OpenAI API)
"
```

**Why squash?** Hardware adaptation PRs are feature-complete units. The intermediate "fix attempt 1", "fix attempt 2", "revert", "try again" commits are noise. One commit = one logical change = easier to review, revert, or cherry-pick.

**Step 2: Push to fork**

```bash
git push -f origin <branch>  # -f because we squashed
```

**Step 3: Create PR**

```bash
gh pr create --base main --head <branch> \
  --title "fix(<backend>): adapt for vLLM <version>" \
  --body-file PR_DESCRIPTION.md
```

**PR_DESCRIPTION.md template:**

```markdown
## Summary

Adapt vllm-plugin-FL for <backend> hardware on vLLM <version>.

## Changes

1. **<file>**: <what> — <why>
2. **<file>**: <what> — <why>
...

## Test Results

Hardware: <GPU count>x <GPU model> (<driver version>)

| Stage | Result | Notes |
|-------|--------|-------|
| Unit tests | ✅ PASS | 349/349 |
| Functional tests | ✅ PASS | 23/23 |
| Offline inference | ✅ PASS | <model>, TP=<N> |
| Serving | ✅ PASS | OpenAI API `/v1/completions` |

## FlagGems Needed Ops

The following ops currently fall back to PyTorch and need native <backend> implementation in FlagGems:

- `op_name_1` (used in <layer>)
- `op_name_2` (used in <layer>)
...

---

> 🤖 This PR was generated by [Kiro](https://kiro.dev) (FlagScale Agent). All code changes were verified by running the full test suite on real <backend> hardware.
```

---

## Patch Strategy

When a test stage fails, follow the **test → diagnose → patch → re-test** loop:

```
┌─────────────┐
│  Run Test   │
└──────┬──────┘
       │ FAIL
       ▼
┌─────────────┐
│  Diagnose   │  ← read traceback, identify root cause
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Create Patch│  ← minimal fix in plugin, NOT in vLLM source
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Re-test    │  ← re-run the SAME stage that failed
└──────┬──────┘
       │ PASS → proceed to next stage
       │ FAIL → back to Diagnose (max 2 retries, then escalate)
       ▼
```

### Diagnosis Flow

1. **Read the full traceback** — identify the failing file, function, and line
2. **Classify the error**:
   - Triton compilation failure (`PassManager::run failed`) → kernel incompatibility
   - ImportError → missing dependency or wrong version
   - Shape/dtype mismatch → API change in new vLLM version
   - NCCL/communication error → distributed setup issue
3. **Check if FlagGems has an alternative** in the target backend (`flag_gems/_<backend>/`)
4. **If FlagGems has it**: ensure the dispatch is wired correctly in the plugin
5. **If FlagGems doesn't have it**: create a PyTorch fallback patch

### Patch Rules

- **One patch per failure** — fix one issue, re-test, then move to the next
- **Patches go in the plugin** (e.g., `<backend>/patches/`), NEVER in vLLM source
- **Gate by hardware** — use `if current_platform.is_<backend>()` or equivalent
- **Include a TODO** — every workaround must state when it can be removed
- **Record feedback** — if FlagGems is missing an op, note it for the team
- **Read once, memorize** — after reading any source file, immediately record key findings with `memory_write`. Check `memory_read` before opening a file again. Never re-read the same file unless it was modified. This prevents the common failure mode of reading `__init__.py` or `git log` 10+ times in one session.
- **No debug tools in PR** — probe scripts, test utilities, and temporary tools added during adaptation must be removed before the PR. Use `git rm tools/probe_*.py tools/test_*.py` before squashing.

### Patch File Structure

Patches live in the plugin's backend-specific directory:

```python
# <backend>/patches/<kernel_name>.py
"""
Patch: redirect <kernel_name> to PyTorch fallback.
Reason: <backend> Triton backend does not support <specific_op>.
TODO: Remove when FlagGems adds native _<backend> implementation.
"""

def patched_function(...):
    # PyTorch-based fallback implementation
    ...
```

### Triton Failure Diagnosis (common case)

1. Error shows `PassManager::run failed` in worker process
2. Look for `triton/backends/<backend>/compiler.py` in traceback → confirms backend Triton compilation issue
3. Identify the specific kernel from the call stack
4. Check if FlagGems has an alternative implementation
5. If not, create a patch that redirects to PyTorch fallback
6. File feedback to FlagGems team for native implementation

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `VLLM_PLUGINS=fl` | Activate the FL plugin | Required for all tests |
| `VLLM_TARGET_DEVICE=empty` | CPU-only vLLM install | Only during pip install |
| `MODEL_PATH` | Model weights location | `/workspace/models/Qwen3.5-35B-A3B` |
| `TP_SIZE` | Tensor parallel size | `2` |
| `PP_SIZE` | Pipeline parallel size | `1` |

---

## Error Handling

### Common Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `PassManager::run failed` | Triton kernel incompatible with backend | Create PyTorch fallback patch |
| `ImportError: flag_gems` | FlagGems not installed or wrong branch | Reinstall from correct branch |
| `CUDA out of memory` | Model too large for TP config | Increase TP size or use smaller model |
| `Connection refused` on curl | Server not ready or crashed | Check server logs, wait for "Started server" |
| DNS resolution failure | Docker bridge networking | Use `--network host` |

### Log Locations

- vLLM server logs: stdout/stderr of the serve command
- Plugin logs: check `VLLM_LOGGING_LEVEL=DEBUG` for plugin dispatch info
- Container logs: `docker logs vllm-plugin-test`
- Adaptation logs: `/workspace/adapt-logs/` (persisted via workspace volume)

---

## PR Checklist

Before submitting the adaptation PR, verify **all** items:

### Tests
- [ ] Unit tests pass: `VLLM_PLUGINS=fl pytest tests/unit_tests/ -x -v`
- [ ] Functional tests pass: `VLLM_PLUGINS=fl pytest tests/functional/ -x -v`
- [ ] Offline inference runs successfully with test model
- [ ] Serving endpoint responds correctly (`/v1/completions` or `/v1/chat/completions`)

### Code Quality
- [ ] No vLLM source modifications — all changes are in the plugin
- [ ] Every patch has a `TODO: Remove when ...` comment
- [ ] No debug prints, temporary hacks, or commented-out code left behind
- [ ] Patches are gated by platform check (`if current_platform.is_<backend>()`)
- [ ] `git diff main` reviewed — only necessary changes remain

### Sensitive Content
- [ ] No passwords, tokens, or API keys in code or comments
- [ ] No SSH config, private keys, or `.pem` file paths committed
- [ ] No hardcoded IP addresses or internal hostnames (use `<ssh_host>` placeholders)
- [ ] No proxy credentials in git config or environment variables
- [ ] Run `git diff main | grep -iE '(password|token|secret|pem|private_key|proxy)'` — should return nothing

### Commits & PR
- [ ] All commits squashed into one clean commit (see Stage 6 for squash instructions)
- [ ] Commit message describes all changes with *what* and *why*
- [ ] Commit message includes test results summary
- [ ] Branch name follows convention: `adapt/<backend>-vllm-<version>`
- [ ] PR description includes:
  - Target backend and vLLM version
  - Reason for each patch (what broke and why)
  - Test results summary (which stages pass, key metrics)
  - FlagGems needed-ops list (ops that currently fall back to PyTorch and need native implementation)
  - Agent attribution footer (see PR template in Stage 6)

### Evidence
- [ ] Test pass logs saved in `/workspace/adapt-logs/` (unit, functional, offline, serving)
- [ ] Any FlagGems missing-op feedback recorded (for upstream report)

---

## Related Skills

- `infer-env-setup` — environment setup (SSH, container, installation)
- `debug-strategy` — systematic debugging when tests fail repeatedly
- `ops-discipline` — shell safety and environment awareness
- `workspace-layout` — shared storage paths for models and artifacts
