"""Tests for the extracted gate framework."""

import pytest

from flagscale.agent.react.gates import Gate, GateRunner, ProgressGate


class TestProgressGate:
    def test_productive_tool_resets(self):
        gate = ProgressGate()
        gate.consecutive_reads = 15
        gate.reads_since_last_new_file = 10
        gate.triggers = 2

        state = {"porting_mode": False, "has_plan": False, "files_read_count": 5}
        msg, block = gate.check("write_file", {}, state)
        assert msg == ""
        assert not block
        assert gate.consecutive_reads == 0
        assert gate.reads_since_last_new_file == 0

    def test_normal_threshold_soft_warning(self):
        gate = ProgressGate()
        gate.reads_since_last_new_file = 12
        gate.last_unique_file_count = 5

        state = {"porting_mode": False, "has_plan": False,
                 "files_read_count": 5, "consecutive_train_failures": 0}
        msg, block = gate.check("read_file", {"path": "/a.py"}, state)
        assert "PROGRESS NOTE" in msg
        assert not block

    def test_normal_threshold_hard_block(self):
        gate = ProgressGate()
        gate.reads_since_last_new_file = 20
        gate.last_unique_file_count = 5

        state = {"porting_mode": False, "has_plan": False,
                 "files_read_count": 5, "consecutive_train_failures": 0}
        msg, block = gate.check("read_file", {"path": "/a.py"}, state)
        assert "PROGRESS BLOCK" in msg
        assert block

    def test_porting_mode_higher_threshold(self):
        gate = ProgressGate()
        gate.reads_since_last_new_file = 20
        gate.last_unique_file_count = 5

        state = {"porting_mode": True, "has_plan": True,
                 "files_read_count": 5, "consecutive_train_failures": 0}
        msg, block = gate.check("read_file", {"path": "/a.py"}, state)
        # In porting mode, threshold is 30, so 21 reads should NOT trigger
        assert msg == ""
        assert not block

    def test_porting_mode_triggers_at_30(self):
        gate = ProgressGate()
        gate.reads_since_last_new_file = 30
        gate.last_unique_file_count = 5

        state = {"porting_mode": True, "has_plan": True,
                 "files_read_count": 5, "consecutive_train_failures": 0}
        msg, block = gate.check("read_file", {"path": "/a.py"}, state)
        assert "PROGRESS NOTE" in msg
        assert not block

    def test_debugging_mode_threshold(self):
        gate = ProgressGate()
        gate.reads_since_last_new_file = 15
        gate.last_unique_file_count = 5

        state = {"porting_mode": False, "has_plan": False,
                 "files_read_count": 5, "consecutive_train_failures": 2}
        msg, block = gate.check("read_file", {"path": "/a.py"}, state)
        # Debugging threshold is 18, so 16 should NOT trigger
        assert msg == ""
        assert not block

    def test_new_file_discovery_resets_staleness(self):
        gate = ProgressGate()
        gate.reads_since_last_new_file = 11
        gate.last_unique_file_count = 5

        state = {"porting_mode": False, "has_plan": False,
                 "files_read_count": 6, "consecutive_train_failures": 0}
        msg, block = gate.check("read_file", {"path": "/new.py"}, state)
        assert msg == ""
        assert gate.reads_since_last_new_file == 0

    def test_hard_cap_safety_net(self):
        gate = ProgressGate()
        gate.consecutive_reads = 40
        gate.triggers = 0
        gate.last_unique_file_count = 40

        state = {"porting_mode": False, "has_plan": True,
                 "files_read_count": 40, "consecutive_train_failures": 0}
        msg, block = gate.check("read_file", {"path": "/a.py"}, state)
        assert "CHECKPOINT SUGGESTION" in msg
        assert not block

    def test_porting_hard_cap_is_60(self):
        gate = ProgressGate()
        gate.consecutive_reads = 45
        gate.triggers = 0
        gate.last_unique_file_count = 45

        state = {"porting_mode": True, "has_plan": True,
                 "files_read_count": 45, "consecutive_train_failures": 0}
        msg, block = gate.check("read_file", {"path": "/a.py"}, state)
        # Porting hard cap is 60, so 46 should NOT trigger
        assert msg == ""
        assert not block


class TestGateRunner:
    def test_dedup_same_warning(self):
        runner = GateRunner()

        class AlwaysWarnGate(Gate):
            name = "always_warn"
            def check(self, tool_name, arguments, state):
                return "same warning", False

        runner.register(AlwaysWarnGate())

        state = {}
        msg1, _ = runner.check_all("read_file", {}, state)
        assert msg1 == "same warning"

        msg2, _ = runner.check_all("read_file", {}, state)
        assert msg2 is None  # Deduplicated

    def test_reset_clears_dedup(self):
        runner = GateRunner()

        class AlwaysWarnGate(Gate):
            name = "always_warn"
            def check(self, tool_name, arguments, state):
                return "same warning", False

        runner.register(AlwaysWarnGate())

        state = {}
        runner.check_all("read_file", {}, state)
        runner.reset_all()
        msg, _ = runner.check_all("read_file", {}, state)
        assert msg == "same warning"  # After reset, warning fires again

    def test_hard_block_stops_execution(self):
        runner = GateRunner()

        class BlockGate(Gate):
            name = "blocker"
            def check(self, tool_name, arguments, state):
                return "BLOCKED", True

        class WarnGate(Gate):
            name = "warner"
            def check(self, tool_name, arguments, state):
                return "warning", False

        runner.register(WarnGate())
        runner.register(BlockGate())

        msg, block = runner.check_all("read_file", {}, state={})
        assert block
        assert "BLOCKED" in msg
