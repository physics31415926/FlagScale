"""Display utilities for agent interactive output."""

import os
import re
import sys
import threading
import time


def _use_color():
    return os.environ.get("NO_COLOR") is None and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code, text):
    if _use_color():
        return f"\033[{code}m{text}\033[0m"
    return text


def dim(text):
    return _c("2", text)


def green(text):
    return _c("32", text)


def yellow(text):
    return _c("33", text)


def cyan(text):
    return _c("36", text)


def red(text):
    return _c("31", text)


def bold(text):
    return _c("1", text)


def magenta(text):
    return _c("35", text)


def blue(text):
    return _c("34", text)


def _fmt_tokens(n):
    if n is None:
        return "?"
    if n >= 100000:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n:,}"
    return str(n)


# ── Tool icons ──────────────────────────────────────────────────────────

_TOOL_ICONS = {
    "shell": "⚡",
    "write_file": "📝",
    "read_file": "📖",
    "edit_file": "✏️",
    "web_fetch": "🌐",
    "web_search": "🔍",
    "memory_write": "💾",
    "memory_read": "🧠",
    "plan_create": "📋",
    "plan_update": "📋",
    "plan_status": "📋",
    "find_latest_log": "📄",
}


def _tool_icon(name):
    return _TOOL_ICONS.get(name, "⚙️")


# ── Spinner for long-running tools ──────────────────────────────────────

class _Spinner:
    """Inline spinner that updates on the same line."""
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, prefix=""):
        self._prefix = prefix
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not _use_color():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        t0 = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - t0
            frame = self._FRAMES[i % len(self._FRAMES)]
            line = f"\r  {dim(frame)} {dim(self._prefix)} {dim(f'{elapsed:.0f}s')}"
            sys.stdout.write(line)
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        # Clear the spinner line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


_active_spinner = None


# ── Banner ──────────────────────────────────────────────────────────────

def banner(provider, model, mode=None, extra_lines=None):
    from flagscale.agent import __version__
    title = f"FlagScale Agent v{__version__}"
    mode_str = f" | Mode: {mode}" if mode else ""
    info = f"Provider: {provider} | Model: {model}{mode_str}"
    cmds = "Commands: /skill  /file  /plan  /save  /load  /export  /cache  /memory  /mode  /reload  /quit"
    lines = [info, cmds]
    if extra_lines:
        lines.extend(extra_lines)
    width = max(len(title), *(len(l) for l in lines)) + 4
    print(cyan(f"╭─ {title} {'─' * (width - len(title) - 3)}╮"))
    for l in lines:
        print(cyan(f"│  {l}{' ' * (width - len(l) - 2)}│"))
    print(cyan(f"╰{'─' * width}╯"))
    print()


# ── Thinking ────────────────────────────────────────────────────────────

def thinking():
    print(dim("⏳ Thinking..."), end="", flush=True)


def thinking_clear():
    print("\r\033[K", end="", flush=True)


# ── LLM done ────────────────────────────────────────────────────────────

def llm_done(elapsed, input_tokens=None, output_tokens=None, cost_str=None):
    parts = [green("✓"), f"{elapsed:.1f}s"]
    if input_tokens is not None:
        parts.append(f"↑{_fmt_tokens(input_tokens)}")
    if output_tokens is not None:
        parts.append(f"↓{_fmt_tokens(output_tokens)}")
    if cost_str:
        parts.append(cost_str)
    print(dim(" | ".join(parts)))


# ── Tool start / done (compact single-line) ─────────────────────────────

# Track whether tool_start printed a newline (spinner) or stayed on same line
_tool_inline = False


def tool_start(name, args_summary=""):
    """Show tool invocation and start spinner for long-running tools."""
    global _active_spinner, _tool_inline
    icon = _tool_icon(name)
    if name == "shell":
        label = f"  {icon} {args_summary}" if args_summary else f"  {icon} {name}"
    else:
        label = f"  {icon} {name}"
        if args_summary:
            label += f" {args_summary}"
    print(dim(label), end="", flush=True)
    # Start spinner for potentially long tools (shell)
    if name == "shell":
        _active_spinner = _Spinner()
        print()  # newline before spinner
        _active_spinner.start()
        _tool_inline = False
    else:
        # Non-shell tools: stay on same line, tool_done will append
        _tool_inline = True


def tool_done(name, elapsed, detail="", error=False):
    """Show tool completion — inline if fast, new line if spinner was used."""
    global _active_spinner, _tool_inline
    if _active_spinner:
        _active_spinner.stop()
        _active_spinner = None
    if error:
        status = red(f"✖ {elapsed:.1f}s")
    elif elapsed > 5:
        status = yellow(f"✓ {elapsed:.1f}s")
    else:
        status = dim(f"✓ {elapsed:.1f}s")

    if _tool_inline:
        # Append to same line as tool_start
        print(f" {status}")
    else:
        # New line (after spinner cleared)
        line = f"    {status}"
        if detail:
            line += f" {dim(detail)}"
        print(line)
    _tool_inline = False


# ── Turn / session summary ──────────────────────────────────────────────

def turn_summary(turn_num, elapsed, input_tokens, output_tokens, cost_str=None):
    parts = [f"Turn {turn_num}", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)}"]
    if cost_str:
        parts.append(cost_str)
    print(dim(f"── {' | '.join(parts)} ──"))
    print()


def session_summary(turns, elapsed, input_tokens, output_tokens, cost_str=None):
    print()
    parts = [f"Session: {turns} turns", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)}"]
    if cost_str:
        parts.append(cost_str)
    print(dim(" | ".join(parts)))


# ── Budget ──────────────────────────────────────────────────────────────

def budget_warning(cost_str):
    print(yellow(f"  ⚠  Budget warning: {cost_str}"))


def budget_exceeded(cost_str):
    print(red(f"  ✖  Budget exceeded: {cost_str}. Stopping."))


# ── File / session ──────────────────────────────────────────────────────

def file_injected(path, chars):
    print(dim(f"  📎 {path} ({chars:,} chars)"))


def session_saved(path):
    print(green(f"  ✓ Session saved: {path}"))


def session_loaded(path, turns):
    print(green(f"  ✓ Session loaded: {path} ({turns} user turns)"))


def session_list(sessions):
    if not sessions:
        print(dim("  No saved sessions."))
        return
    import datetime
    for s in sessions[:10]:
        ts = datetime.datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
        print(dim(f"    {s['id']}  {ts}  ({s['turns']} turns)"))
        print(dim(f"      {s['path']}"))


# ── Skill / plan ────────────────────────────────────────────────────────

def skill_auto_loaded(name):
    print(magenta(f"  🔧 Auto-loaded skill: {name}"))


def plan_created(title, step_count):
    print(green(f"  📋 Plan created: {title} ({step_count} steps)"))


def plan_step_updated(step_id, title, status):
    icons = {"done": green("✓"), "doing": yellow("→"), "skipped": dim("-"), "blocked": red("!")}
    icon = icons.get(status, " ")
    print(f"    [{icon}] Step {step_id}: {title}")


def plan_completed(title):
    print(green(f"  📋 Plan completed: {title}"))


def plan_abandoned(title):
    print(yellow(f"  📋 Plan abandoned: {title}"))


def plan_summary(text):
    for line in text.split("\n"):
        if line.startswith("Plan:"):
            print(cyan(line))
        elif line.strip().startswith("[✓]"):
            print(dim(line))
        elif line.strip().startswith("[→]"):
            print(yellow(line))
        elif line.startswith("Progress:"):
            print(dim(line))
        else:
            print(line)


def complexity_hint():
    print(magenta("  📋 Complex task detected — suggesting plan creation."))


# ── Autosave ────────────────────────────────────────────────────────────

def autosave_found(turn_count, user_turns, last_user_msg, timestamp):
    import datetime
    ts = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    print(yellow("╭─ Unfinished session detected ──────────────────╮"))
    print(yellow(f"│  Time: {ts}"))
    print(yellow(f"│  {turn_count} turns, {user_turns} user messages"))
    if last_user_msg:
        preview = last_user_msg[:60] + ("..." if len(last_user_msg) > 60 else "")
        print(yellow(f"│  Last message: {preview}"))
    print(yellow("╰─────────────────────────────────────────────────╯"))


def autosave_resumed(turn_count):
    print(green(f"  ✓ Resumed previous session ({turn_count} turns). You can continue."))
    print()


def interrupted():
    print(yellow("\n  ⚠  Interrupted. Back to prompt."))


# ── Markdown rendering ──────────────────────────────────────────────────

def render_markdown(text):
    """Render markdown with basic syntax highlighting for terminal output."""
    if not _use_color():
        return text

    lines = text.split("\n")
    output = []
    in_code_block = False
    code_lang = ""

    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                code_lang = line[3:].strip()
                output.append(dim(f"┌─ {code_lang}" if code_lang else "┌─"))
            else:
                output.append(dim("└─"))
                code_lang = ""
            continue

        if in_code_block:
            output.append(f"  {_c('36', line)}")
            continue

        if line.startswith("# "):
            output.append(bold(line))
        elif line.startswith("## "):
            output.append(bold(line))
        elif line.startswith("### "):
            output.append(bold(line))
        elif line.startswith("- ") or line.startswith("* "):
            output.append(f"  {line}")
        elif re.match(r"^\d+\.\s", line):
            output.append(f"  {line}")
        else:
            line = re.sub(r"`([^`]+)`", lambda m: _c("36", m.group(1)), line)
            line = re.sub(r"\*\*([^*]+)\*\*", lambda m: _c("1", m.group(1)), line)
            output.append(line)

    return "\n".join(output)
