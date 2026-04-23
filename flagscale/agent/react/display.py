"""Display utilities for agent interactive output."""

import os
import re
import sys


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


def _fmt_tokens(n):
    if n >= 1000:
        return f"{n:,}"
    return str(n)


def banner(provider, model, extra_lines=None):
    title = "FlagScale Agent"
    info = f"Provider: {provider} | Model: {model}"
    cmds = "Commands: /skill  /file  /save  /load  /export  /reload  /quit"
    lines = [info, cmds]
    if extra_lines:
        lines.extend(extra_lines)
    width = max(len(title), *(len(l) for l in lines)) + 4
    print(cyan(f"╭─ {title} {'─' * (width - len(title) - 3)}╮"))
    for l in lines:
        print(cyan(f"│  {l}{' ' * (width - len(l) - 2)}│"))
    print(cyan(f"╰{'─' * width}╯"))
    print()


def thinking():
    print(dim("⏳ Thinking..."), end="", flush=True)


def thinking_clear():
    print("\r\033[K", end="", flush=True)


def llm_done(elapsed, input_tokens=None, output_tokens=None, cost_str=None):
    parts = [green("✓"), f"{elapsed:.1f}s"]
    if input_tokens is not None:
        parts.append(f"↑{_fmt_tokens(input_tokens)}")
    if output_tokens is not None:
        parts.append(f"↓{_fmt_tokens(output_tokens)}")
    if cost_str:
        parts.append(cost_str)
    print(dim(" | ".join(parts)))


def tool_start(name, args_summary=""):
    label = f"⚡ {name}"
    if args_summary:
        label += f"({args_summary})"
    print(yellow(label))


def tool_done(name, elapsed):
    print(dim(f"  ✓ {name} ({elapsed:.1f}s)"))


def turn_summary(turn_num, elapsed, input_tokens, output_tokens, cost_str=None):
    parts = [f"Turn {turn_num}", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)} tokens"]
    if cost_str:
        parts.append(cost_str)
    print(dim(f"── {' | '.join(parts)} ──"))
    print()


def session_summary(turns, elapsed, input_tokens, output_tokens, cost_str=None):
    print()
    parts = [f"Session: {turns} turns", f"{elapsed:.1f}s",
             f"↑{_fmt_tokens(input_tokens)} ↓{_fmt_tokens(output_tokens)} tokens"]
    if cost_str:
        parts.append(cost_str)
    print(dim(" | ".join(parts)))


def budget_warning(cost_str):
    print(yellow(f"⚠  Budget warning: {cost_str}"))


def budget_exceeded(cost_str):
    print(red(f"✖  Budget exceeded: {cost_str}. Stopping."))


def file_injected(path, chars):
    print(dim(f"📎 Injected {path} ({chars:,} chars)"))


def session_saved(path):
    print(green(f"✓ Session saved: {path}"))


def session_loaded(path, turns):
    print(green(f"✓ Session loaded: {path} ({turns} user turns)"))


def session_list(sessions):
    if not sessions:
        print(dim("No saved sessions."))
        return
    import datetime
    for s in sessions[:10]:
        ts = datetime.datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
        print(dim(f"  {s['id']}  {ts}  ({s['turns']} turns)"))
        print(dim(f"    {s['path']}"))


def skill_auto_loaded(name):
    print(magenta(f"🔧 Auto-loaded skill: {name}"))


def interrupted():
    print(yellow("\n⚠  Interrupted. Back to prompt."))


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
