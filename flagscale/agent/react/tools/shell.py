"""Shell command tool with safety checks and user confirmation."""

import os
import re
import subprocess

from flagscale.agent.react.tools.base import Tool

FATAL_PATTERNS = [
    r"rm\s+-[^\s]*r[^\s]*f\s+/\s*$",
    r"rm\s+-[^\s]*f[^\s]*r\s+/\s*$",
    r"rm\s+-rf\s+/(?:\s|$)",
    r"mkfs\.",
    r"dd\s+if=",
    r":\(\)\{\s*:\|:&\s*\};:",
    r">\s*/dev/sd[a-z]",
    r"chmod\s+-R\s+777\s+/\s*$",
]

CONFIRM_PATTERNS = [
    r"\brm\s+",
    r"\bkill\b",
    r"\bkillall\b",
    r"\bpkill\b",
    r"\breboot\b",
    r"\bshutdown\b",
    r"\bsystemctl\s+(stop|restart|disable)",
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard",
    r"\bgit\s+clean\s+-[^\s]*f",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bmv\s+/",
    r"\bcp\s+.*\s+/",
    r"\bpip\s+install\b",
    r"\bpip\s+uninstall\b",
    r"\bconda\s+install\b",
    r"\bconda\s+remove\b",
    r"\bapt\s+(install|remove|purge)",
    r"\byum\s+(install|remove|erase)",
    r"\bcurl\s+.*\|\s*(ba)?sh",
    r"\bwget\s+.*\|\s*(ba)?sh",
]

_FATAL_RE = re.compile("|".join(FATAL_PATTERNS))
_CONFIRM_RE = re.compile("|".join(CONFIRM_PATTERNS))


def _default_confirm(command: str) -> bool:
    """Ask user to confirm a potentially risky command."""
    print(f"\n\033[33m⚠  Risky command:\033[0m {command}")
    try:
        answer = input("\033[33m   Allow? [y/N]: \033[0m").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command and return its output (stdout + stderr)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, timeout: int = 120, check_dangerous: bool = True,
                 confirm_fn=None, require_confirm: bool = True, env: dict = None):
        self._timeout = timeout
        self._check_dangerous = check_dangerous
        self._require_confirm = require_confirm
        self._confirm_fn = confirm_fn or _default_confirm
        self._env = env or {}

    def execute(self, **kwargs) -> str:
        command = kwargs["command"]

        if self._check_dangerous and _FATAL_RE.search(command):
            return f"FATAL: Refused to execute potentially dangerous command: {command}"

        if self._require_confirm and _CONFIRM_RE.search(command):
            if not self._confirm_fn(command):
                return "DENIED: User declined to execute this command."

        try:
            run_env = {**os.environ, **self._env} if self._env else None
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=run_env,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr
            if not output:
                output = "(no output)"
            if result.returncode != 0:
                hint = _network_error_hint(output, self._env)
                if hint:
                    output += hint
            return output
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {self._timeout} seconds."
        except Exception as e:
            return f"ERROR: {e}"


_NETWORK_ERROR_PATTERNS = re.compile(
    r"Could not resolve host|Connection refused|Connection timed out|"
    r"Network is unreachable|No route to host|"
    r"Failed to connect|Connection reset by peer|"
    r"unable to access|SSL connection timeout|"
    r"Failed to establish a new connection|"
    r"Temporary failure in name resolution",
    re.IGNORECASE,
)

_PROXY_HINT = (
    "\n\n💡 Network error detected and no proxy configured. "
    "Set proxy in ~/.flagscale/agent.yaml:\n"
    "  shell_env:\n"
    '    HTTP_PROXY: "http://host:port"\n'
    '    HTTPS_PROXY: "http://host:port"\n'
    "Then use /reload to apply."
)


def _network_error_hint(output: str, env: dict) -> str | None:
    if not _NETWORK_ERROR_PATTERNS.search(output):
        return None
    proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"}
    if proxy_keys & set(env):
        return None
    return _PROXY_HINT
