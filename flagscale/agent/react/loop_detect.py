"""Loop detection and tool-call deduplication mixin."""


class LoopDetectMixin:
    """Detects repeated tool calls and caches results to avoid redundant work."""

    _LOOP_DETECTION_WINDOW = 10
    _LOOP_DETECTION_THRESHOLD = 3
    _AUTOSAVE_INTERVAL = 5

    def _get_tool_call_key(self, tool_name, arguments):
        """Generate a hashable key for a tool call.

        Keys must capture the FULL semantics of the call — two calls that would
        produce different results must have different keys.  Never truncate or
        hash content; use the full values so that legitimate retries with
        different new_string / content are not falsely flagged as loops.
        """
        if tool_name == "shell":
            return (tool_name, arguments.get("command", ""))
        elif tool_name == "read_file":
            path = arguments.get("path", "")
            start_line = arguments.get("start_line", "")
            end_line = arguments.get("end_line", "")
            return (tool_name, path, str(start_line), str(end_line))
        elif tool_name == "edit_file":
            path = arguments.get("path", "") or arguments.get("file_path", "")
            old_str = arguments.get("old_string", "")
            new_str = arguments.get("new_string", "")
            return (tool_name, path, old_str, new_str)
        elif tool_name == "write_file":
            path = arguments.get("path", "") or arguments.get("file_path", "")
            content = arguments.get("content", "")
            return (tool_name, path, content)
        elif tool_name == "load_skill":
            return (tool_name, arguments.get("name", ""))
        elif tool_name == "memory_write":
            return (tool_name, arguments.get("key", ""), arguments.get("content", ""))
        else:
            parts = []
            for k, v in sorted(arguments.items()):
                parts.append(f"{k}={v}")
            return (tool_name, "|".join(parts))

    def _check_loop_detection(self, tool_name, arguments):
        """Detect repeated identical tool calls that indicate the agent is stuck.

        Only triggers when the EXACT same call (same tool + same full arguments)
        appears >= threshold times in the recent window.  The blocked call itself
        is NOT counted — otherwise a single block poisons the window and prevents
        the agent from ever retrying with different args.
        """
        key = self._get_tool_call_key(tool_name, arguments)

        count = self._recent_tool_calls.count(key)
        if count >= self._LOOP_DETECTION_THRESHOLD:
            # Do NOT append this call — it's blocked, shouldn't pollute the window
            return (
                f"\n\n⚠️ [LOOP DETECTION] You've called {tool_name} with the same arguments "
                f"{count} times in the last {self._LOOP_DETECTION_WINDOW} tool calls. "
                f"This suggests you're stuck in a loop.\n"
                f"STOP and take a different approach:\n"
                f"1. Diagnose WHY the previous attempts failed\n"
                f"2. Try a fundamentally different strategy\n"
                f"3. If blocked, write what you know to workspace and ask the user\n"
            )

        # Only record after confirming it's not a loop — keeps window clean
        self._recent_tool_calls.append(key)
        if len(self._recent_tool_calls) > self._LOOP_DETECTION_WINDOW:
            self._recent_tool_calls = self._recent_tool_calls[-self._LOOP_DETECTION_WINDOW:]

        return ""

    def _check_duplicate_read(self, tool_name, arguments):
        """Detect duplicate tool calls — within-turn cache hit or cross-compaction re-read."""
        if tool_name == "read_file":
            path = arguments.get("path", "")
            if not path:
                return None
            start = arguments.get("start_line", "")
            end = arguments.get("end_line", "")
            key = ("read_file", path, str(start), str(end))
            if key in self._tool_call_cache:
                return self._tool_call_cache[key]
            # Cross-iteration dedup: if same file+range was read before, return short hint
            if path in self._files_read_this_session and key not in self._tool_call_cache:
                if not start and not end:
                    # Full-file re-read — return a nudge instead of blocking
                    return (
                        f"[ALREADY READ] You already read {path} in full this session. "
                        "Use memory_write to save key findings instead of re-reading. "
                        "If you need a specific section, use start_line/end_line."
                    )
                return None
        elif tool_name == "memory_write":
            mem_key = arguments.get("key", "")
            if not mem_key:
                return None
            key = ("memory_write", mem_key)
            if key in self._tool_call_cache:
                return self._tool_call_cache[key]
        else:
            return None
        return None

    def _cache_tool_result(self, tool_name, arguments, result):
        """Cache tool results within a turn to avoid redundant calls."""
        if tool_name == "read_file" and "ERROR" not in result[:20]:
            path = arguments.get("path", "")
            if path:
                start = arguments.get("start_line", "")
                end = arguments.get("end_line", "")
                self._tool_call_cache[("read_file", path, str(start), str(end))] = result
        elif tool_name == "memory_write" and "ERROR" not in result[:20]:
            mem_key = arguments.get("key", "")
            if mem_key:
                self._tool_call_cache[("memory_write", mem_key)] = result
