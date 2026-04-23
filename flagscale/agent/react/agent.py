"""ReAct agent — the core loop."""

import json
import logging
import os
import sys
import time
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory

from flagscale.agent.react import display
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.cost import CostTracker
from flagscale.agent.react.history import HistoryManager
from flagscale.agent.react.logger import setup_logging
from flagscale.agent.react.providers import get_provider
from flagscale.agent.react.retry import retry_with_backoff
from flagscale.agent.react.session import save_session, load_session, list_sessions
from flagscale.agent.react.skills import SkillManager
from flagscale.agent.react.tools import ToolRegistry
from flagscale.agent.react.tools.edit_file import EditFileTool
from flagscale.agent.react.tools.load_skill import LoadSkillTool
from flagscale.agent.react.tools.read_file import ReadFileTool
from flagscale.agent.react.tools.shell import ShellTool
from flagscale.agent.react.tools.write_file import WriteFileTool
from flagscale.agent.react.tools.web_fetch import WebFetchTool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FlagScale Agent, an AI assistant specialized in large model training with the FlagScale framework.

You have access to the following tools:
- read_file: Read the contents of a file
- write_file: Create or overwrite a file
- edit_file: Edit a file by replacing an exact string match
- shell: Execute a shell command
- web_fetch: Fetch a URL and extract its text content (for reading docs, GitHub pages, error references)
- load_skill: Load a skill's detailed instructions

You also have the following skills that can be loaded for specialized tasks:
{skills}

Skills are pre-built knowledge modules. When a user asks what skills you have, or what you can do, always list ALL available skills above with their descriptions. To activate a skill, use the load_skill tool with the skill name — this gives you detailed domain-specific instructions.

Working directory: {cwd}

When working on tasks:
1. Think step by step. Break complex tasks into smaller steps.
2. Use tools one at a time and reason about each result before proceeding.
3. If a skill is relevant to the user's request, load it first to get specialized instructions.
4. Be specific and actionable in your responses.
5. When you discover something unexpected or potentially wrong (e.g., unreasonable config values, missing files, resource conflicts), STOP and ask the user to confirm before proceeding. Many users are not experts — proactively flag issues and explain what you found, why it might be a problem, and suggest a fix. Do NOT silently work around problems or wait multiple iterations to report them.

Shell command guidelines:
- NEVER use `find /` or search from root. Always search from the working directory or a specific known path.
- Prefer `grep -r` or `find .` scoped to the project directory.
- Use `ls`, `tree`, or `find . -maxdepth 2` to explore directory structure first.
- For code search, prefer `grep -rn "pattern" . --include="*.py"` over broad find+xargs.
- Avoid commands that take more than 30 seconds. If a command might be slow, add limits (e.g., `head`, `| head -50`, `-maxdepth`)."""

SKILL_KEYWORDS = {}


def _build_skill_keywords(skills):
    mapping = {}
    for s in skills:
        name = s["name"]
        desc = (s.get("description", "") + " " + name).lower()
        words = set(desc.replace("-", " ").replace("_", " ").split())
        for w in words:
            if len(w) > 3:
                mapping.setdefault(w, set()).add(name)
        mapping.setdefault(name.lower(), set()).add(name)
        for kw in s.get("keywords", []):
            mapping.setdefault(str(kw).lower(), set()).add(name)
    return mapping


class ReactAgent:
    """A ReAct agent with streaming, history management, and parallel tool execution."""

    def __init__(self, config: AgentConfig):
        setup_logging()
        self.config = config
        self.skill_manager = SkillManager(config.skill_dirs)
        self.tool_registry = ToolRegistry()

        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(WriteFileTool())
        self.tool_registry.register(EditFileTool())
        self.tool_registry.register(
            ShellTool(
                timeout=config.shell_timeout,
                check_dangerous=config.dangerous_commands_check,
                require_confirm=config.confirm_commands,
                env=config.shell_env,
            )
        )
        self.tool_registry.register(LoadSkillTool(self.skill_manager))
        self.tool_registry.register(WebFetchTool(proxies=self._build_proxies()))
        self._load_plugin_tools()

        if not config.api_key:
            raise ValueError(
                "API key not found. Set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )
        self.provider = get_provider(config.provider, config.model, config.api_key, config.base_url)

        self.history = HistoryManager(max_context_tokens=config.max_context_tokens)
        self._refresh_system_prompt()

        self._turn_count = 0
        self._session_start = time.time()
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        self._cost_tracker = CostTracker(config.model, config.max_cost)
        self._loaded_skills = set()
        self._interrupted = False

    # ── Plugin tools (P2-8) ──────────────────────────────────────────────

    def _load_plugin_tools(self):
        dirs = self.config.plugin_tool_dirs
        if not dirs:
            return
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(d, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        spec = json.load(f)
                    tool = _PluginShellTool(spec)
                    self.tool_registry.register(tool)
                    logger.info("Loaded plugin tool: %s from %s", tool.name, path)
                except Exception as e:
                    logger.warning("Failed to load plugin tool %s: %s", path, e)

    # ── System prompt ────────────────────────────────────────────────────

    def _refresh_system_prompt(self):
        skills = self.skill_manager.list_skills()
        skills_text = (
            "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
            if skills else "(no skills available)"
        )
        prompt = SYSTEM_PROMPT.format(skills=skills_text, cwd=os.getcwd())

        global SKILL_KEYWORDS
        SKILL_KEYWORDS = _build_skill_keywords(skills)

        msgs = self.history.messages
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": prompt}
        else:
            self.history._messages.insert(0, {"role": "system", "content": prompt})
        self._system_prompt = prompt

    def _build_proxies(self):
        env = self.config.shell_env
        http = env.get("HTTP_PROXY") or env.get("http_proxy")
        https = env.get("HTTPS_PROXY") or env.get("https_proxy")
        if not http and not https:
            return None
        proxies = {}
        if http:
            proxies["http"] = http
        if https:
            proxies["https"] = https
        return proxies

    def _reload_config(self):
        if self.config.reload():
            shell_tool = self.tool_registry.get("shell")
            shell_tool._env = self.config.shell_env
            web_fetch_tool = self.tool_registry.get("web_fetch")
            web_fetch_tool._proxies = self._build_proxies()

    # ── Main entry ───────────────────────────────────────────────────────

    def run(self, single_shot_query=None):
        if single_shot_query:
            self._run_single_shot(single_shot_query)
            return

        display.banner(self.config.provider, self.config.model)

        history_file = os.path.join(os.path.expanduser("~"), ".flagscale", "input_history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        completer = WordCompleter(
            ["/quit", "/reload", "/skill", "/file", "/save", "/load", "/export"],
            sentence=True,
        )
        session = PromptSession(
            history=FileHistory(history_file),
            completer=completer,
        )

        while True:
            try:
                user_input = session.prompt("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._exit()
                break

            if not user_input:
                continue

            cmd = user_input.split()[0] if user_input.startswith("/") else None
            if cmd == "/quit":
                self._exit()
                break
            elif cmd == "/reload":
                self._reload_config()
                self._refresh_system_prompt()
                print("Config and skills reloaded.")
                continue
            elif cmd == "/skill":
                self._handle_skill_command(user_input)
                continue
            elif cmd == "/file":
                self._handle_file_command(user_input)
                continue
            elif cmd == "/save":
                self._handle_save_command(user_input)
                continue
            elif cmd == "/load":
                self._handle_load_command(user_input)
                continue
            elif cmd == "/export":
                self._handle_export_command(user_input)
                continue

            if self.config.auto_skill:
                self._auto_load_skills(user_input)

            self.history.append({"role": "user", "content": user_input})
            self._react_loop()

    def _run_single_shot(self, query):
        if self.config.auto_skill:
            self._auto_load_skills(query)
        self.history.append({"role": "user", "content": query})
        self._react_loop()

    def _exit(self):
        session_elapsed = time.time() - self._session_start
        cost_str = self._cost_tracker.format_cost()
        display.session_summary(
            self._turn_count, session_elapsed,
            self._session_input_tokens, self._session_output_tokens,
            cost_str,
        )
        print("Bye!")

    # ── ReAct loop ───────────────────────────────────────────────────────

    def _react_loop(self):
        schemas = self.tool_registry.to_schemas(self.provider.schema_format)
        self._turn_count += 1
        self._interrupted = False
        turn_start = time.time()
        turn_input_tokens = 0
        turn_output_tokens = 0

        for iteration in range(self.config.max_iterations):
            if self._interrupted:
                break

            if self._cost_tracker.budget_exceeded():
                display.budget_exceeded(self._cost_tracker.format_cost())
                break

            display.thinking()
            t0 = time.time()
            messages = self.history.get_messages()

            try:
                response, usage = self._call_llm_stream(messages, schemas)
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                break
            except Exception as e:
                display.thinking_clear()
                print(display.red(f"✖ LLM error: {e}"))
                logger.exception("LLM call failed")
                break

            elapsed = time.time() - t0

            input_tok = usage.get("input_tokens") or 0
            output_tok = usage.get("output_tokens") or 0
            if input_tok:
                turn_input_tokens += input_tok
                self._session_input_tokens += input_tok
            if output_tok:
                turn_output_tokens += output_tok
                self._session_output_tokens += output_tok
            self._cost_tracker.add(input_tok, output_tok)

            cost_str = self._cost_tracker.format_cost()
            display.llm_done(elapsed, input_tok, output_tok, cost_str)

            if self._cost_tracker.budget_warning():
                display.budget_warning(cost_str)

            logger.info("LLM call #%d: %.1fs", iteration + 1, elapsed)

            self.history.append(self.provider.format_assistant_message(response))

            if not response["tool_calls"]:
                break

            try:
                results = self._execute_tools(response["tool_calls"])
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                break

            tool_results = [
                self.provider.format_tool_result(tc["id"], result)
                for tc, result in zip(response["tool_calls"], results)
            ]
            self._append_tool_results(tool_results)
        else:
            if not self._interrupted:
                print("\n[warning] Reached maximum iterations, stopping.\n")

        turn_elapsed = time.time() - turn_start
        cost_str = self._cost_tracker.format_cost()
        display.turn_summary(self._turn_count, turn_elapsed, turn_input_tokens, turn_output_tokens, cost_str)

    # ── LLM streaming with error recovery (P0-3) ────────────────────────

    def _call_llm_stream(self, messages, schemas):
        content_parts = []
        tool_calls = []
        current_tool = None
        usage = {}

        stream = retry_with_backoff(
            lambda: self.provider.chat_stream(messages, schemas),
            max_retries=3,
        )

        display.thinking_clear()

        try:
            for event in stream:
                if event["type"] == "text":
                    text = event["content"]
                    if display._use_color():
                        text = display.render_markdown(text) if "\n```" in text else text
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    content_parts.append(event["content"])
                elif event["type"] == "tool_start":
                    current_tool = {
                        "id": event["id"],
                        "name": event["name"],
                        "arguments_json": "",
                    }
                    tool_calls.append(current_tool)
                elif event["type"] == "tool_delta":
                    if current_tool:
                        current_tool["arguments_json"] += event["arguments_delta"]
                elif event["type"] == "usage":
                    usage = {
                        "input_tokens": event.get("input_tokens"),
                        "output_tokens": event.get("output_tokens"),
                    }
                elif event["type"] == "done":
                    break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning("Stream interrupted: %s", e)
            if not content_parts and not tool_calls:
                raise

        if content_parts:
            print()

        parsed_tool_calls = None
        if tool_calls:
            parsed_tool_calls = []
            for tc in tool_calls:
                try:
                    arguments = json.loads(tc["arguments_json"]) if tc["arguments_json"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                parsed_tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": arguments})

        return {"content": "".join(content_parts) or None, "tool_calls": parsed_tool_calls}, usage

    # ── Tool execution ───────────────────────────────────────────────────

    def _execute_tools(self, tool_calls):
        if len(tool_calls) == 1:
            return [self._execute_tool(tool_calls[0])]

        results = [None] * len(tool_calls)
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
            futures = {
                pool.submit(self._execute_tool, tc): i for i, tc in enumerate(tool_calls)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def _execute_tool(self, tool_call):
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        def _fmt_arg(k, v):
            s = str(v)
            if len(s) > 80:
                s = s[:77] + "..."
            if isinstance(v, str):
                return f'{k}="{s}"'
            return f'{k}={s}'

        args_summary = ", ".join(
            _fmt_arg(k, v) for k, v in list(arguments.items())[:3]
        )
        display.tool_start(tool_name, args_summary)

        t0 = time.time()
        try:
            result = self.tool_registry.execute(tool_name, **arguments)
        except Exception as e:
            result = f"ERROR: {e}"
        elapsed = time.time() - t0

        logger.info("Tool %s: %.1fs, result %d chars", tool_name, elapsed, len(result))
        display.tool_done(tool_name, elapsed)
        return result

    def _append_tool_results(self, tool_results):
        """Append tool results, merging into one message for Anthropic compatibility."""
        if not tool_results:
            return
        if len(tool_results) == 1:
            self.history.append(tool_results[0])
            return
        first = tool_results[0]
        if first.get("role") == "user" and isinstance(first.get("content"), list):
            merged_content = []
            for tr in tool_results:
                merged_content.extend(tr["content"])
            self.history.append({"role": "user", "content": merged_content})
        else:
            for tr in tool_results:
                self.history.append(tr)

    # ── Auto skill loading (P3-12) ───────────────────────────────────────

    def _auto_load_skills(self, user_input):
        text = user_input.lower()
        words = set(text.replace("-", " ").replace("_", " ").split())
        candidates = set()
        for kw, names in SKILL_KEYWORDS.items():
            if kw in words or kw in text:
                candidates.update(names)

        for skill_name in candidates - self._loaded_skills:
            try:
                content = self.skill_manager.load(skill_name)
                tool_call_id = f"auto_{uuid.uuid4().hex[:8]}"
                fake_response = {
                    "content": None,
                    "tool_calls": [{"id": tool_call_id, "name": "load_skill", "arguments": {"name": skill_name}}],
                }
                self.history.append(self.provider.format_assistant_message(fake_response))
                self.history.append(self.provider.format_tool_result(tool_call_id, content))
                self._loaded_skills.add(skill_name)
                display.skill_auto_loaded(skill_name)
            except Exception:
                pass

    # ── Commands ─────────────────────────────────────────────────────────

    def _handle_skill_command(self, user_input):
        skill_name = user_input[len("/skill"):].strip()
        if not skill_name:
            print("Usage: /skill <name>")
            return

        try:
            content = self.skill_manager.load(skill_name)
        except FileNotFoundError as e:
            print(f"Skill not found: {e}")
            return

        tool_call_id = f"skill_{uuid.uuid4().hex[:8]}"
        fake_response = {
            "content": None,
            "tool_calls": [{"id": tool_call_id, "name": "load_skill", "arguments": {"name": skill_name}}],
        }
        self.history.append(self.provider.format_assistant_message(fake_response))
        self.history.append(self.provider.format_tool_result(tool_call_id, content))
        self._loaded_skills.add(skill_name)

        self.history.append({
            "role": "user",
            "content": f"I've loaded the '{skill_name}' skill. Please acknowledge and tell me how you can help with it.",
        })
        self._react_loop()

    def _handle_file_command(self, user_input):
        path = user_input[len("/file"):].strip()
        if not path:
            print("Usage: /file <path>")
            return
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        display.file_injected(path, len(content))
        self.history.append({
            "role": "user",
            "content": f"[File: {path}]\n```\n{content}\n```",
        })

    def _handle_save_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        sid = parts[1].strip() if len(parts) > 1 else None
        msgs = [m for m in self.history.messages if m.get("role") != "system"]
        metadata = {
            "provider": self.config.provider,
            "model": self.config.model,
            "turns": self._turn_count,
        }
        path = save_session(msgs, self.config.session_dir, sid, metadata)
        display.session_saved(path)

    def _handle_load_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else None

        if not target:
            sessions = list_sessions(self.config.session_dir)
            display.session_list(sessions)
            return

        path = target
        if not os.path.isfile(path):
            d = self.config.session_dir
            if not d:
                from pathlib import Path
                d = os.path.join(Path.home(), ".flagscale", "sessions")
            candidate = os.path.join(d, f"{target}.json")
            if os.path.isfile(candidate):
                path = candidate
            else:
                print(f"Session not found: {target}")
                return

        try:
            data = load_session(path)
        except Exception as e:
            print(f"Error loading session: {e}")
            return

        msgs = data.get("messages", [])
        self.history._messages = [self.history.messages[0]] if self.history.messages and self.history.messages[0].get("role") == "system" else []
        self.history._messages.extend(msgs)
        user_turns = len([m for m in msgs if m.get("role") == "user"])
        display.session_loaded(path, user_turns)

    def _handle_export_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        path = parts[1].strip() if len(parts) > 1 else f"session_{int(time.time())}.md"
        path = os.path.expanduser(path)

        lines = [f"# FlagScale Agent Session Export\n"]
        lines.append(f"Provider: {self.config.provider} | Model: {self.config.model}\n")
        lines.append(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")

        for msg in self.history.messages:
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                parts_text = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts_text.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            parts_text.append(f"[Tool: {block.get('name', '')}]")
                        elif block.get("type") == "tool_result":
                            inner = block.get("content", "")
                            if len(inner) > 200:
                                inner = inner[:200] + "..."
                            parts_text.append(f"[Result: {inner}]")
                content = "\n".join(parts_text)

            if role == "user":
                lines.append(f"\n## User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"\n## Assistant\n\n{content}\n")
            elif role == "tool":
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"\n> Tool result: {content}\n")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(display.green(f"✓ Exported to {path}"))
        except Exception as e:
            print(f"Error exporting: {e}")


class _PluginShellTool:
    """A tool loaded from a JSON spec that wraps a shell command template."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.description = spec.get("description", "")
        self.parameters = spec.get("parameters", {"type": "object", "properties": {}})
        self.max_result_size = spec.get("max_result_size", 50000)
        self._command_template = spec.get("command", "")
        self._timeout = spec.get("timeout", 120)

    def execute(self, **kwargs):
        import subprocess
        cmd = self._command_template
        for k, v in kwargs.items():
            cmd = cmd.replace(f"{{{k}}}", str(v))
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self._timeout)
            output = (result.stdout or "") + (result.stderr or "")
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {self._timeout}s."
        except Exception as e:
            return f"ERROR: {e}"

    def to_openai_schema(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def to_anthropic_schema(self):
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}