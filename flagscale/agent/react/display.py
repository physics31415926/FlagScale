"""Display utilities for agent interactive output."""

import os
import re
import sys
import threading
import time


# ── Thread-safe stdout ─────────────────────────────────────────────────

_stdout_lock = threading.Lock()


def _print(*args, **kwargs):
    """Thread-safe print."""
    with _stdout_lock:
        print(*args, **kwargs)


def _write(text):
    """Thread-safe sys.stdout.write + flush."""
    with _stdout_lock:
        sys.stdout.write(text)
        sys.stdout.flush()


def _use_color():
    return os.environ.get("NO_COLOR") is None and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code, text):
    if _use_color():
        return f"\033[{code}m{text}\033[0m"
    return text


def _c256(n, text):
    """256-color foreground."""
    if _use_color():
        return f"\033[38;5;{n}m{text}\033[0m"
    return text


def dim(text):
    return _c256(245, text)


def green(text):
    return _c256(114, text)


def yellow(text):
    return _c256(214, text)


def cyan(text):
    return _c256(80, text)


def red(text):
    return _c256(203, text)


def bold(text):
    return _c("1", text)


def magenta(text):
    return _c256(141, text)


def blue(text):
    return _c256(117, text)


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
            with _stdout_lock:
                sys.stdout.write(line)
                sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        with _stdout_lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


_active_spinner = None


def _stop_all_spinners():
    global _active_spinner, _parallel_display
    if _active_spinner:
        _active_spinner.stop()
        _active_spinner = None
    if _parallel_display:
        _parallel_display.finish()
        _parallel_display = None


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
    _print(cyan(f"╭─ {title} {'─' * (width - len(title) - 3)}╮"))
    for l in lines:
        _print(cyan(f"│  {l}{' ' * (width - len(l) - 2)}│"))
    _print(cyan(f"╰{'─' * width}╯"))
    _print()


# ── Thinking ────────────────────────────────────────────────────────────

def thinking():
    _write(dim("⏳ Thinking..."))


def thinking_clear():
    _write("\r\033[K")


# ── LLM done ────────────────────────────────────────────────────────────

def llm_done(elapsed, input_tokens=None, output_tokens=None):
    parts = [green("✓"), f"{elapsed:.1f}s"]
    if input_tokens is not None:
        parts.append(f"↑{_fmt_tokens(input_tokens)}")
    if output_tokens is not None:
        parts.append(f"↓{_fmt_tokens(output_tokens)}")
    _print(dim(" | ".join(parts)))


# ── Tool start / done (compact single-line) ─────────────────────────────

# Track whether tool_start printed a newline (spinner) or stayed on same line
_tool_inline = False


def tool_start(name, args_summary=""):
    """Show tool invocation and start spinner for long-running tools."""
    global _active_spinner, _tool_inline
    icon = _tool_icon(name)
    label = f"  {icon} {name}"
    if args_summary:
        label += f" {args_summary}"
    _print(dim(label), end="", flush=True)
    if name == "shell":
        _active_spinner = _Spinner()
        _print()
        _active_spinner.start()
        _tool_inline = False
    else:
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
        suffix = f" {status}"
        if detail:
            suffix += f" {dim(detail)}"
        _print(suffix)
    else:
        line = f"    {status}"
        if detail:
            line += f" {dim(detail)}"
        _print(line)
    _tool_inline = False


# ── Parallel tool display ──────────────────────────────────────────────

class _ParallelDisplay:
    """In-place updating display for parallel tool execution."""
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, tool_summaries):
        """tool_summaries: list of (name, args_summary) tuples."""
        self._tools = tool_summaries
        self._n = len(tool_summaries)
        self._results = {}  # index -> (elapsed, error)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = 0
        self._extra_lines = 0  # lines printed below display area

    def start(self):
        if not _use_color() or self._n == 0:
            for name, args in self._tools:
                icon = _tool_icon(name)
                label = f"  {icon} {name}"
                if args:
                    label += f" {args}"
                _print(dim(label))
            return
        # Print initial lines with pending indicator
        with _stdout_lock:
            for name, args in self._tools:
                icon = _tool_icon(name)
                label = f"{icon} {name}"
                if args:
                    label += f" {args}"
                frame = self._FRAMES[0]
                sys.stdout.write(f"  {dim(label)} {dim(frame)}\n")
            sys.stdout.flush()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def add_extra_lines(self, count):
        """Track lines printed below the display area by external code."""
        with self._lock:
            self._extra_lines += count

    def mark_done(self, index, elapsed, error=False, detail=""):
        with self._lock:
            self._results[index] = (elapsed, error, detail)

    def _animate(self):
        while not self._stop.is_set():
            self._redraw()
            self._frame += 1
            self._stop.wait(0.1)

    def _redraw(self):
        with self._lock:
            results = dict(self._results)
            extra = self._extra_lines
        with _stdout_lock:
            total_up = self._n + extra
            if total_up > 0:
                sys.stdout.write(f"\033[{total_up}A")
            for i in range(self._n):
                name, args = self._tools[i]
                icon = _tool_icon(name)
                label = f"{icon} {name}"
                if args:
                    label += f" {args}"
                if i in results:
                    elapsed, error, detail = results[i]
                    if error:
                        status = red(f"✖ {elapsed:.1f}s")
                    elif elapsed > 5:
                        status = yellow(f"✓ {elapsed:.1f}s")
                    else:
                        status = dim(f"✓ {elapsed:.1f}s")
                    line = f"  {label} {status}"
                    if detail:
                        line += f" {dim(detail)}"
                else:
                    frame = self._FRAMES[self._frame % len(self._FRAMES)]
                    line = f"  {dim(label)} {dim(frame)}"
                sys.stdout.write(f"\r\033[K{line}\n")
            if extra > 0:
                sys.stdout.write(f"\033[{extra}B")
            sys.stdout.flush()

    def finish(self):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        if _use_color() and self._n > 0:
            self._redraw()


_parallel_display = None


def parallel_tools_start(tool_summaries):
    """Print all tool names and start in-place updating display.

    tool_summaries: list of (name, args_summary) tuples
    Returns: _ParallelDisplay instance
    """
    global _parallel_display
    _parallel_display = _ParallelDisplay(tool_summaries)
    _parallel_display.start()
    return _parallel_display


def parallel_tool_update(index, elapsed, error=False, detail=""):
    """Update a specific tool line with completion status."""
    global _parallel_display
    if _parallel_display:
        _parallel_display.mark_done(index, elapsed, error, detail)


def parallel_extra_lines(count):
    """Notify parallel display that external code printed extra lines below it."""
    if _parallel_display and not _parallel_display._stop.is_set():
        _parallel_display.add_extra_lines(count)


def parallel_tools_finish():
    """Stop parallel display and do final redraw."""
    global _parallel_display
    if _parallel_display:
        _parallel_display.finish()
        _parallel_display = None


# ── Turn / session summary ──────────────────────────────────────────────

def turn_summary(turn_num, elapsed, input_tokens, output_tokens):
    parts = [f"Turn {turn_num}", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)}"]
    _print(dim(f"── {' | '.join(parts)} ──"))
    _print()


def session_summary(turns, elapsed, input_tokens, output_tokens):
    _print()
    parts = [f"Session: {turns} turns", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)}"]
    _print(dim(" | ".join(parts)))


# ── File / session ──────────────────────────────────────────────────────

def file_injected(path, chars):
    _print(dim(f"  📎 {path} ({chars:,} chars)"))


def session_saved(path):
    _print(green(f"  ✓ Session saved: {path}"))


def session_loaded(path, turns):
    _print(green(f"  ✓ Session loaded: {path} ({turns} user turns)"))


def session_list(sessions):
    if not sessions:
        _print(dim("  No saved sessions."))
        return
    import datetime
    for s in sessions[:10]:
        ts = datetime.datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
        _print(dim(f"    {s['id']}  {ts}  ({s['turns']} turns)"))
        _print(dim(f"      {s['path']}"))


# ── Skill / plan ────────────────────────────────────────────────────────

def skill_auto_loaded(name):
    _print(magenta(f"  🔧 Auto-loaded skill: {name}"))


def plan_created(title, step_count):
    _print(green(f"  📋 Plan created: {title} ({step_count} steps)"))


def plan_step_updated(step_id, title, status):
    icons = {"done": green("✓"), "doing": yellow("→"), "skipped": dim("-"), "blocked": red("!")}
    icon = icons.get(status, " ")
    _print(f"    [{icon}] Step {step_id}: {title}")


def plan_completed(title):
    _print(green(f"  📋 Plan completed: {title}"))


def plan_abandoned(title):
    _print(yellow(f"  📋 Plan abandoned: {title}"))


def plan_summary(text):
    for line in text.split("\n"):
        if line.startswith("Plan:"):
            _print(cyan(line))
        elif line.strip().startswith("[✓]"):
            _print(dim(line))
        elif line.strip().startswith("[→]"):
            _print(yellow(line))
        elif line.startswith("Progress:"):
            _print(dim(line))
        else:
            _print(line)


def complexity_hint():
    _print(magenta("  📋 Complex task detected — suggesting plan creation."))


# ── Autosave ────────────────────────────────────────────────────────────

def autosave_found(turn_count, user_turns, last_user_msg, timestamp):
    import datetime
    ts = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    _print(yellow("╭─ Unfinished session detected ──────────────────╮"))
    _print(yellow(f"│  Time: {ts}"))
    _print(yellow(f"│  {turn_count} turns, {user_turns} user messages"))
    if last_user_msg:
        preview = last_user_msg[:60] + ("..." if len(last_user_msg) > 60 else "")
        _print(yellow(f"│  Last message: {preview}"))
    _print(yellow("╰─────────────────────────────────────────────────╯"))


def autosave_resumed(turn_count):
    _print(green(f"  ✓ Resumed previous session ({turn_count} turns). You can continue."))
    _print()


def interrupted():
    _stop_all_spinners()
    _print(yellow("\n  ⚠  Interrupted. Back to prompt."))


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
