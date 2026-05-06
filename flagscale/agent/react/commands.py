"""Slash command handlers for ReactAgent."""

import json
import os
import time
import uuid

import yaml
from pathlib import Path

from flagscale.agent.react import display
from flagscale.agent.react.session import (
    save_conversation, load_conversation, list_sessions, get_session_dir,
    find_resumable_sessions, mark_completed,
)

import logging

logger = logging.getLogger(__name__)


def _is_tool_result_msg(msg):
    """Check if a message is a tool result (not a real user turn)."""
    content = msg.get("content", "")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


class CommandsMixin:
    """Mixin providing slash command handlers."""

    def _handle_memory_command(self, user_input):
        parts = user_input.split()
        if len(parts) < 2:
            print("Usage: /memory list | /memory clear [type] | /memory delete <key>")
            return
        sub = parts[1]
        if sub == "list":
            entries = self.session_memory.list_entries()
            if not entries:
                print("No memory entries.")
                return
            for e in entries:
                key = e.get("key", "?")
                mem_type = e.get("type", "?")
                content = e.get("content", "")
                if len(content) > 120:
                    content = content[:117] + "..."
                print(f"  [{mem_type}] \033[1m{key}\033[0m")
                print(f"          {content}")
        elif sub == "clear":
            if len(parts) >= 3:
                mem_type = parts[2]
                count = self.session_memory.clear_by_type(mem_type)
                print(f"Cleared {count} '{mem_type}' memory entries.")
            else:
                count = self.session_memory.clear()
                print(f"Cleared {count} memory entries.")
        elif sub == "delete":
            if len(parts) < 3:
                print("Usage: /memory delete <key>")
                return
            key = parts[2]
            if self.session_memory.delete(key):
                print(f"Deleted memory '{key}'.")
            else:
                print(f"No memory '{key}' found.")
        else:
            print("Usage: /memory list | /memory clear | /memory delete <key>")

    def _handle_mode_command(self, user_input):
        parts = user_input.split()
        if len(parts) < 2:
            print(f"Current mode: {self.config.mode}")
            print("Usage: /mode confirm | /mode auto")
            print("  confirm — risky commands require user confirmation (default)")
            print("  auto    — fully autonomous: no confirmations, auto-continues between turns")
            return
        new_mode = parts[1].lower()
        if new_mode not in ("confirm", "auto"):
            print(f"Unknown mode '{new_mode}'. Available: confirm, auto")
            return
        self.config.mode = new_mode
        if new_mode == "auto":
            self.config.confirm_commands = False
            shell_tool = self.tool_registry.get("shell")
            if shell_tool:
                shell_tool._require_confirm = False
            print(f"Mode: auto — fully autonomous (max {self.config.max_auto_turns} auto turns, Ctrl+C to stop).")
        else:
            self.config.confirm_commands = True
            shell_tool = self.tool_registry.get("shell")
            if shell_tool:
                shell_tool._require_confirm = True
            print("Mode: confirm — risky commands will require confirmation.")

    def _handle_plan_command(self, user_input):
        parts = user_input.split()
        sub = parts[1] if len(parts) >= 2 else "status"
        if sub == "status" or sub == "show":
            text = self.task_plan.summary()
            display.plan_summary(text)
        elif sub == "list":
            plans = self.task_plan.list_plans()
            if not plans:
                print("No plans.")
                return
            for p in plans:
                status_str = p["status"]
                done = p["done"]
                total = p["total"]
                print(f"  {p['id']}  {p['title']}  [{status_str}]  {done}/{total} steps")
        elif sub == "abandon":
            try:
                plan = self.task_plan.abandon(reason="user requested via /plan abandon")
                display.plan_abandoned(plan["title"])
            except ValueError as e:
                print(f"  {e}")
        elif sub == "clear":
            count = self.task_plan.clear_completed()
            print(f"Cleared {count} completed/abandoned plans.")
        else:
            print("Usage: /plan [status|list|abandon|clear]")

    def _handle_skill_command(self, user_input):
        skill_name = user_input[len("/skill"):].strip()
        if not skill_name:
            print("Usage: /skill <name>")
            return

        try:
            content = self.skill_manager.load(skill_name)
            content = self._maybe_summarize_skill(skill_name, content)
        except FileNotFoundError as e:
            print(f"Skill not found: {e}")
            return

        tool_call_id = f"skill_{uuid.uuid4().hex[:8]}"
        fake_response = {
            "content": None,
            "tool_calls": [{"id": tool_call_id, "name": "load_skill", "arguments": {"name": skill_name}}],
        }
        self.history.append(self.provider.format_assistant_message(fake_response))
        self.history.append(self.provider.format_tool_result(
            tool_call_id, f"[Skill '{skill_name}' loaded — content available in system context]"))
        self._loaded_skills.add(skill_name)
        self._active_skill_content[skill_name] = content
        self._skill_load_iterations[skill_name] = self._total_iterations
        self._refresh_system_prompt()

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
        msgs = [m for m in self.history.full_log if m.get("role") != "system"]
        metadata = {
            "provider": self.config.provider,
            "model": self.config.model,
            "turns": self._turn_count,
        }
        path = save_conversation(
            self._session_dir, sid or self._session_id, msgs,
            loaded_skills=list(self._active_skill_content.keys()),
            metadata=metadata,
        )
        display.session_saved(path)

    def _handle_load_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else None

        if not target:
            sessions = list_sessions()
            display.session_list(sessions)
            return

        path = target
        if not os.path.isfile(path):
            candidate_dir = get_session_dir(target)
            candidate = os.path.join(candidate_dir, "conversation.json")
            if os.path.isfile(candidate):
                path = candidate
            else:
                print(f"Session not found: {target}")
                return

        try:
            data = load_conversation(path if os.path.isdir(path) else os.path.dirname(path))
        except Exception as e:
            print(f"Error loading session: {e}")
            return

        msgs = data.get("messages", [])
        self.history._messages = [self.history.messages[0]] if self.history.messages and self.history.messages[0].get("role") == "system" else []
        self.history._messages.extend(msgs)
        self.history._full_log = list(msgs)
        user_turns = len([m for m in msgs if m.get("role") == "user"])
        display.session_loaded(path, user_turns)

    def _handle_export_command(self, user_input):
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            path = os.path.expanduser(parts[1].strip())
        else:
            d = self._session_dir
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"session_{self._session_id}.md")

        lines = [f"# FlagScale Agent Session Export\n"]
        lines.append(f"Provider: {self.config.provider} | Model: {self.config.model}")
        lines.append(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")

        messages = self.history.full_log
        turn_num = 0

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            content = msg.get("content", "")

            if role == "user" and not _is_tool_result_msg(msg):
                turn_num += 1
                lines.append(f"\n---\n\n## Turn {turn_num}\n")

            if isinstance(content, list):
                parts_text = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            parts_text.append(block.get("text", ""))
                        elif btype == "tool_use":
                            name = block.get("name", "")
                            inp = block.get("input", {})
                            inp_str = json.dumps(inp, ensure_ascii=False, indent=2)
                            parts_text.append(f"[Tool: {name}]\n```json\n{inp_str}\n```")
                        elif btype == "tool_result":
                            inner = block.get("content", "")
                            parts_text.append(f"[Result]\n```\n{inner}\n```")
                    elif isinstance(block, str):
                        parts_text.append(block)
                content = "\n\n".join(parts_text)

            if role == "user":
                if _is_tool_result_msg(msg):
                    lines.append(f"\n**Tool Result:**\n\n{content}\n")
                else:
                    lines.append(f"\n### User\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"\n### Assistant\n\n{content}\n")
            elif role == "tool":
                lines.append(f"\n**Tool Result:**\n\n```\n{content}\n```\n")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(display.green(f"✓ Exported to {path} ({len(messages)} messages)"))
        except Exception as e:
            print(f"Error exporting: {e}")

    def _handle_resume_command(self, user_input):
        parts = user_input.split()
        target_id = parts[1].strip() if len(parts) > 1 else None

        all_sessions = list_sessions(self._sessions_root)

        if not target_id:
            # Show all sessions (exclude current)
            others = [s for s in all_sessions if s.get("session_id") != self._session_id]
            if not others:
                print("No other sessions found.")
                return
            import datetime
            print("Sessions:")
            for s in others:
                ts = datetime.datetime.fromtimestamp(s.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
                status = "active" if not s.get("completed", True) else "done"
                turns = s.get("turns", s.get("user_turns", "?"))
                print(f"  {s['session_id']}  [{status}]  ({ts}, {turns} turns)")
            print(f"\nUsage: /resume <session_id>")
            return

        # Find matching session — support prefix match across all sessions
        match = None
        for s in all_sessions:
            if s["session_id"] == target_id or s["session_id"].startswith(target_id):
                match = s
                break

        if not match:
            # Try direct directory lookup
            candidate_dir = get_session_dir(target_id, self._sessions_root)
            if os.path.isdir(candidate_dir):
                conv = load_conversation(candidate_dir)
                if conv:
                    match = {"session_id": target_id, "session_dir": candidate_dir}
                else:
                    print(f"Session '{target_id}' found but has no conversation data.")
                    return
            else:
                print(f"Session '{target_id}' not found.")
                return

        if "session_dir" not in match:
            match["session_dir"] = get_session_dir(match["session_id"], self._sessions_root)

        old_dir = match["session_dir"]
        conv = load_conversation(old_dir)
        if not conv:
            print(f"Failed to load conversation from {old_dir}")
            return

        msgs = conv.get("messages", [])
        loaded_skills = conv.get("loaded_skills", [])
        self.history._messages = [self.history.messages[0]] if self.history.messages and self.history.messages[0].get("role") == "system" else []
        self.history._messages.extend(msgs)
        self.history._full_log = list(msgs)
        for skill_name in loaded_skills:
            try:
                content = self.skill_manager.load(skill_name)
                if content:
                    self._active_skill_content[skill_name] = content
                    self._loaded_skills.add(skill_name)
            except Exception:
                pass
        mark_completed(old_dir)
        # Clean up the empty session dir created at startup
        old_startup_dir = self._session_dir
        self._session_dir = old_dir
        self._session_id = match["session_id"]
        if old_startup_dir != old_dir and os.path.isdir(old_startup_dir):
            try:
                if not os.listdir(old_startup_dir):
                    os.rmdir(old_startup_dir)
            except OSError:
                pass

        # Repoint all session-dir-dependent state
        self.task_plan._dir = os.path.join(old_dir, "plans")
        self._experiment_manager._dir = os.path.join(old_dir, "experiments")
        plan_create_tool = self.tool_registry.get("plan_create")
        if plan_create_tool:
            plan_create_tool._session_id = match["session_id"]
        memory_write_tool = self.tool_registry.get("memory_write")
        if memory_write_tool:
            memory_write_tool._session_id = match["session_id"]

        self._refresh_system_prompt()
        display.session_resumed(match["session_id"])

    def _handle_compact_command(self, user_input):
        """Manually trigger context compaction."""
        compacted = self.history.force_compact(target_ratio=0.60)
        if compacted:
            est = sum(1 for _ in self.history._messages)
            print(display.green(f"✓ Compacted ({est} messages remaining)"))
        else:
            print("Context is already within target size, no compaction needed.")
