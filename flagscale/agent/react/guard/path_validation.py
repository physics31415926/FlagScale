"""PathValidationGuard — warns before shell commands reference non-existent paths.

Fires a non-blocking inject when a shell command contains a path-like token
that does not exist on the local filesystem. This catches the common failure
mode where the agent guesses a path (e.g. confusing two similarly-named
directories) and wastes a round-trip on a guaranteed FileNotFoundError.

Design decisions:
- Non-blocking (inject, not block): the agent may be constructing a path
  that will be created by the command itself (mkdir, touch, etc.), or the
  path may exist on a remote host. We warn, not refuse.
- Only fires for paths that look "absolute or explicit" (start with / or
  a Windows drive letter, or contain os.sep). Relative single-word tokens
  like "python" or "ls" are skipped.
- Skips creation commands (mkdir, touch, cp, mv, scp, rsync, tee, >)
  because those are expected to reference non-existent targets.
- Fires at most once per unique missing path per session to avoid spam.
- Priority 12 — runs after SafetyGuard (10) but before LoopDetect (20).
"""

from __future__ import annotations

import logging
import os
import re

from flagscale.agent.react.guard import Guard, GuardContext, GuardVerdict

logger = logging.getLogger(__name__)

# Commands that intentionally target non-existent paths (create/copy/move)
_CREATION_VERBS = re.compile(
    r'^\s*(mkdir|touch|cp\s|mv\s|scp\s|rsync\s|tee\s|install\s|-o\s)',
    re.IGNORECASE,
)

# Redirect operators that create files
_REDIRECT_CREATE = re.compile(r'(?<![<>])>\s*\S')

# Patterns that look like filesystem paths worth checking:
# - Unix absolute: /foo/bar
# - Windows absolute: C:\foo or C:/foo
# - Explicit relative with separator: ./foo or ../foo or foo/bar/baz
_PATH_RE = re.compile(
    r'(?<!\w)'                          # not preceded by word char
    r'('
    r'(?:[A-Za-z]:[/\\][^\s\'";&|>]+)'  # Windows: C:\... or C:/...
    r'|'
    r'(?:/[^\s\'";&|><]+)'              # Unix absolute: /foo/bar
    r'|'
    r'(?:\.{1,2}/[^\s\'";&|><]+)'       # Explicit relative: ./foo or ../foo
    r')'
)

# Tokens to skip even if they match the path regex
_SKIP_SUFFIXES = ('.py', '.sh', '.bash', '.zsh')  # script names passed as args are ok


class PathValidationGuard(Guard):
    """Warns (non-blocking) when a shell command references a non-existent path."""

    name = "path_validation"
    priority = 12
    activate_on_tools = {"shell"}

    def __init__(self):
        self._warned_paths: set[str] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        cmd = ctx.tool_args.get("command", "")
        if not cmd:
            return None

        # Skip commands that are expected to create new paths
        if _CREATION_VERBS.match(cmd) or _REDIRECT_CREATE.search(cmd):
            return None

        missing = self._find_missing_paths(cmd)
        if not missing:
            return None

        # Filter out already-warned paths to avoid spam
        new_missing = [p for p in missing if p not in self._warned_paths]
        if not new_missing:
            return None

        self._warned_paths.update(new_missing)

        paths_str = ", ".join(f"`{p}`" for p in new_missing[:3])
        return GuardVerdict.inject(
            f"[PathCheck] The following path(s) do not exist on the local filesystem: "
            f"{paths_str}. "
            "Verify the path before running — check memory for recorded paths or "
            "use `shell` with `ls`/`dir` to probe the directory first.",
            reason="missing_path_in_command",
        )

    def _find_missing_paths(self, cmd: str) -> list[str]:
        """Extract path-like tokens from cmd and return those that don't exist."""
        candidates = _PATH_RE.findall(cmd)
        missing = []
        seen = set()
        for raw in candidates:
            # Strip trailing punctuation / quotes that may have been captured
            path = raw.rstrip("'\";,)")
            if not path or path in seen:
                continue
            seen.add(path)

            # Skip very short tokens — likely flags, not paths
            if len(path) < 4:
                continue

            # Skip if it looks like a URL (http/https/ftp scheme or protocol-relative //)
            if (path.startswith("http://") or path.startswith("https://")
                    or path.startswith("ftp://") or path.startswith("//")):
                continue

            # Determine if this is an absolute path on the current platform
            # Also accept Unix-style absolute paths (starting with /) on any platform
            is_abs = os.path.isabs(path) or path.startswith("/")
            is_explicit_rel = path.startswith("./") or path.startswith("../")

            if not is_abs and not is_explicit_rel:
                continue

            if not os.path.exists(path):
                missing.append(path)

        return missing

    def reset_turn(self):
        pass  # Warned-path set persists across turns to avoid re-warning
