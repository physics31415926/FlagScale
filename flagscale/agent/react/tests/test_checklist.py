"""Tests for checklist module — all rule evaluation via LLM classify (no regex).

Rules are evaluated in a single batched LLM call per tool observation.
"""

from types import SimpleNamespace

from flagscale.agent.react.judge import Judge, JudgeBudget
from flagscale.agent.react.checklist.base import ChecklistItem, ChecklistEngine, Checklist


class MockProvider:
    """Returns controlled JSON responses in sequence."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"][:500])
        resp = self.responses.pop(0) if self.responses else "{}"
        return {"content": resp}


def _make_obs(tool_name="", tool_args=None, tool_result=None,
              phase_name="exec", classify_fn=None):
    return SimpleNamespace(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        phase_name=phase_name,
        classify_fn=classify_fn,
    )


# ── ChecklistItem ──────────────────────────────────────────────────────


class TestChecklistItem:
    def test_defaults(self):
        item = ChecklistItem(id="test", description="Test check")
        assert item.id == "test"
        assert item.phases == {"*"}
        assert item.severity == "warning"
        assert item.max_reminders == 3
        assert item.prompt == ""

    def test_phases_default_wildcard(self):
        item = ChecklistItem(id="test", description="Test")
        assert "*" in item.phases


# ── ChecklistEngine.evaluate_batch ───────────────────────────────────────


class TestChecklistEngine:
    def test_batch_single_violation(self):
        """One item violated → returns its id."""
        provider = MockProvider(responses=[
            '{"violations": [{"id": "no_todos", "reason": "contains TODO"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="no_todos",
            description="No TODO placeholders",
            prompt="DETECT if the file content contains TODO placeholders.",
            reminder="Remove TODOs",
        )
        obs = _make_obs("write_file",
            {"path": "/tmp/test.py", "content": "# TODO: implement"},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch([item], obs)
        assert violations == ["no_todos"]
        assert len(provider.calls) == 1

    def test_batch_no_violation(self):
        """No violations → returns empty list."""
        provider = MockProvider(responses=['{"violations": []}'])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="no_todos",
            description="No TODO placeholders",
            prompt="DETECT if the file content contains TODO placeholders.",
            reminder="Remove TODOs",
        )
        obs = _make_obs("write_file",
            {"path": "/tmp/test.py", "content": "clean"},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch([item], obs)
        assert violations == []
        assert len(provider.calls) == 1

    def test_batch_multi_item_one_violation(self):
        """Multiple items, only one violated → returns that one ID."""
        provider = MockProvider(responses=[
            '{"violations": [{"id": "bad_cmd", "reason": "dangerous"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        items = [
            ChecklistItem(id="bad_cmd", description="Dangerous command",
                          prompt="DETECT dangerous commands.",
                          reminder="Don't"),
            ChecklistItem(id="no_todos", description="No TODOs",
                          prompt="DETECT TODOs.", reminder="Remove"),
        ]
        obs = _make_obs("shell",
            {"command": "rm -rf /"},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch(items, obs)
        assert violations == ["bad_cmd"]
        assert len(provider.calls) == 1

    def test_batch_trigger_on_filter(self):
        """Items with mismatched trigger_on are excluded from batch."""
        provider = MockProvider(responses=['{"violations": []}'])
        judge = Judge(provider)
        engine = ChecklistEngine()
        items = [
            ChecklistItem(id="shell_only", description="Shell check",
                          trigger_on={"tool": "shell"},
                          prompt="DETECT errors.", reminder="Fix"),
            ChecklistItem(id="write_only", description="Write check",
                          trigger_on={"tool": "write_file"},
                          prompt="DETECT bad writes.", reminder="Fix"),
        ]
        obs = _make_obs("shell",
            {"command": "ls"},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch(items, obs)
        assert violations == []
        assert len(provider.calls) == 1
        # Verify only shell_only was sent to LLM
        call_content = provider.calls[0]
        assert "shell_only" in call_content
        assert "write_only" not in call_content

    def test_batch_no_classify_fn(self):
        """No judge available → returns empty list."""
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="no_todos", description="No TODOs",
            prompt="DETECT TODOs.", reminder="Remove TODOs",
        )
        obs = _make_obs("write_file",
            {"path": "/tmp/test.py", "content": "# TODO"})
        violations = engine.evaluate_batch([item], obs)
        assert violations == []

    def test_batch_no_items_with_prompt(self):
        """All items have empty prompt → no LLM call, returns empty."""
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        engine = ChecklistEngine()
        items = [
            ChecklistItem(id="a", description="A", prompt="",
                          reminder="Should not fire"),
            ChecklistItem(id="b", description="B", prompt="",
                          reminder="Should not fire"),
        ]
        obs = _make_obs("shell",
            {"command": "anything"},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch(items, obs)
        assert violations == []
        assert len(provider.calls) == 0

    def test_batch_facts_injected_into_context(self):
        """Auto-detected facts are passed to the LLM in batch mode."""
        provider = MockProvider(responses=[
            '{"violations": [{"id": "shared_storage", "reason": "no --prefix"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        engine.facts = {"shared_storage": ["/share/project"]}
        item = ChecklistItem(
            id="shared_storage",
            description="Use --prefix on shared storage",
            trigger_on={"tool": "shell"},
            prompt="DETECT if conda create uses -n instead of --prefix on shared storage.",
            reminder="Use --prefix",
        )
        obs = _make_obs("shell",
            {"command": "conda create -n test python=3.12 -y"},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch([item], obs)
        assert violations == ["shared_storage"]
        assert len(provider.calls) == 1

    def test_batch_empty_candidates(self):
        """No items at all → no LLM call."""
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        engine = ChecklistEngine()
        violations = engine.evaluate_batch([], _make_obs("shell",
            {"command": "ls"}, classify_fn=judge.classify))
        assert violations == []
        assert len(provider.calls) == 0


# ── Checklist.check() (batched via evaluate_batch) ───────────────────────


class TestChecklist:
    def test_evaluates_matching_phase(self):
        provider = MockProvider(responses=[
            '{"violations": [{"id": "no_todos", "reason": "contains TODO"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="no_todos",
            description="No TODOs",
            phases={"write"},
            prompt="DETECT TODOs.",
            reminder="Remove TODOs before committing",
        )
        checklist = Checklist(engine=engine, items=[item])
        obs = _make_obs("write_file",
            {"path": "/tmp/test.py", "content": "# TODO: fix"},
            phase_name="write",
            classify_fn=judge.classify)
        reminders = checklist.check(obs)
        assert len(reminders) == 2  # summary + detail
        assert "no_todos" in reminders[0].message
        assert "Remove TODOs" in reminders[1].message
        assert len(provider.calls) == 1

    def test_skips_non_matching_phase(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="no_todos",
            description="No TODOs",
            phases={"write"},
            prompt="DETECT TODOs.",
            reminder="Remove TODOs",
        )
        checklist = Checklist(engine=engine, items=[item])
        obs = _make_obs("write_file",
            {"path": "/tmp/test.py", "content": "# TODO"},
            phase_name="analysis",
            classify_fn=judge.classify)
        reminders = checklist.check(obs)
        assert len(reminders) == 0
        assert len(provider.calls) == 0  # No LLM call — skipped by phase

    def test_wildcard_phase_matches_all(self):
        provider = MockProvider(responses=[
            '{"violations": [{"id": "always_check", "reason": "match"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="always_check",
            description="Always applies",
            phases={"*"},
            prompt="DETECT TODOs.",
            reminder="Remove TODOs",
        )
        checklist = Checklist(engine=engine, items=[item])
        obs = _make_obs("read_file",
            {"path": "/tmp/test.py", "content": "# TODO"},
            phase_name="analysis",
            classify_fn=judge.classify)
        reminders = checklist.check(obs)
        assert len(reminders) == 2  # summary + detail

    def test_max_reminders_threshold(self):
        provider = MockProvider(responses=[
            '{"violations": [{"id": "no_todos", "reason": "match"}]}',
            '{"violations": [{"id": "no_todos", "reason": "match"}]}',
            '{"violations": [{"id": "no_todos", "reason": "match"}]}',
            '{"violations": [{"id": "no_todos", "reason": "match"}]}',
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="no_todos",
            description="No TODOs",
            phases={"*"},
            prompt="DETECT TODOs.",
            reminder="Remove TODOs",
            max_reminders=3,
        )
        checklist = Checklist(engine=engine, items=[item])

        # First call: new violation → alerts generated
        obs = _make_obs("write_file",
            {"path": "/tmp/test.py", "content": "# TODO"},
            classify_fn=judge.classify)
        reminders = checklist.check(obs)
        assert len(reminders) == 2  # summary + detail
        msg = " ".join(r.message for r in reminders)
        assert "no_todos" in msg

        # Subsequent calls: same violation, not new → no alerts
        for _ in range(3):
            obs = _make_obs("write_file",
                {"path": "/tmp/test.py", "content": "# TODO"},
                classify_fn=judge.classify)
            reminders = checklist.check(obs)
            assert len(reminders) == 0

    def test_no_match_no_reminders(self):
        provider = MockProvider(responses=['{"violations": []}'])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="no_todos",
            description="No TODOs",
            phases={"*"},
            prompt="DETECT TODOs.",
            reminder="Remove TODOs",
        )
        checklist = Checklist(engine=engine, items=[item])
        obs = _make_obs("write_file",
            {"path": "/tmp/test.py", "content": "clean code"},
            classify_fn=judge.classify)
        reminders = checklist.check(obs)
        assert len(reminders) == 0

    def test_batch_multi_item_single_llm_call(self):
        """Multiple matching items → single LLM call for all."""
        provider = MockProvider(responses=[
            '{"violations": [{"id": "rule_a", "reason": "found"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        items = [
            ChecklistItem(id="rule_a", description="A",
                          phases={"*"}, prompt="Detect A.",
                          reminder="Fix A"),
            ChecklistItem(id="rule_b", description="B",
                          phases={"*"}, prompt="Detect B.",
                          reminder="Fix B"),
        ]
        checklist = Checklist(engine=engine, items=items)
        obs = _make_obs("shell",
            {"command": "bad"},
            classify_fn=judge.classify)
        reminders = checklist.check(obs)
        assert len(reminders) == 2  # summary + detail
        assert "rule_a" in reminders[0].message
        assert "rule_b" not in " ".join(r.message for r in reminders)
        assert len(provider.calls) == 1  # Single LLM call

    def test_violation_id_not_in_items_ignored(self):
        """LLM returns a violation ID not in our items → silently skip."""
        provider = MockProvider(responses=[
            '{"violations": [{"id": "nonexistent", "reason": "??"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        items = [
            ChecklistItem(id="real_rule", description="Real",
                          phases={"*"}, prompt="Detect.",
                          reminder="Fix"),
        ]
        checklist = Checklist(engine=engine, items=items)
        obs = _make_obs("shell",
            {"command": "x"},
            classify_fn=judge.classify)
        reminders = checklist.check(obs)
        assert len(reminders) == 0  # ID not found → ignored


# ── Checklist.from_skill_constraints ────────────────────────────────────


class TestChecklistFromSkill:
    def test_builds_from_constraints(self):
        skill_meta = {
            "constraints": [
                {
                    "id": "no_secrets",
                    "description": "No API keys in code",
                    "phases": ["write"],
                    "prompt": "DETECT API keys.",
                    "reminder": "Use env vars for secrets",
                    "severity": "error",
                    "max_reminders": 5,
                }
            ]
        }
        checklist = Checklist.from_skill_constraints(
            engine=ChecklistEngine(), skill_meta=skill_meta)
        assert len(checklist._items) == 1
        item = checklist._items[0]
        assert item.id == "no_secrets"
        assert item.severity == "error"
        assert item.max_reminders == 5
        assert "write" in item.phases
        assert item.prompt == "DETECT API keys."

    def test_empty_constraints(self):
        checklist = Checklist.from_skill_constraints(
            engine=ChecklistEngine(), skill_meta={})
        assert len(checklist._items) == 0

    def test_multiple_constraints(self):
        skill_meta = {
            "constraints": [
                {"id": "a", "description": "A", "prompt": "Detect A.", "reminder": "fix a"},
                {"id": "b", "description": "B", "prompt": "Detect B.", "reminder": "fix b"},
            ]
        }
        checklist = Checklist.from_skill_constraints(
            engine=ChecklistEngine(), skill_meta=skill_meta)
        assert len(checklist._items) == 2


# ── env_protect_pytorch checklist rule (batch) ──────────────────────────


class TestEnvProtectPyTorch:
    def test_flagscale_pip_install_triggers_rule(self):
        """pip install -e '.[cuda-train]' (without --no-deps) triggers env protection."""
        provider = MockProvider(responses=[
            '{"violations": [{"id": "env_protect_pytorch", "reason": "no --no-deps"}]}'
        ])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="env_protect_pytorch",
            description="Prevent pip install from upgrading pinned PyTorch",
            phases={"*"},
            trigger_on={"tool": "shell"},
            prompt=(
                "DETECT if this shell command is a pip install of FlagScale or its requirements "
                "WITHOUT --no-deps."
            ),
            reminder="[ENV PROTECTION] This pip install command may upgrade PyTorch...",
            severity="error",
            max_reminders=5,
        )
        obs = _make_obs("shell",
            {"command": "pip install -e \".[cuda-train]\""},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch([item], obs)
        assert violations == ["env_protect_pytorch"]

    def test_no_deps_install_is_allowed(self):
        """pip install --no-deps -e . should NOT trigger."""
        provider = MockProvider(responses=['{"violations": []}'])
        judge = Judge(provider)
        engine = ChecklistEngine()
        item = ChecklistItem(
            id="env_protect_pytorch",
            description="Prevent pip install from upgrading pinned PyTorch",
            phases={"*"},
            trigger_on={"tool": "shell"},
            prompt=(
                "DETECT if this shell command is a pip install of FlagScale or its requirements "
                "WITHOUT --no-deps."
            ),
            reminder="[ENV PROTECTION] This pip install command may upgrade PyTorch...",
            severity="error",
            max_reminders=5,
        )
        obs = _make_obs("shell",
            {"command": "pip install --no-deps -e ."},
            classify_fn=judge.classify)
        violations = engine.evaluate_batch([item], obs)
        assert violations == []
