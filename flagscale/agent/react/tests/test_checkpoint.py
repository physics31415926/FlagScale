"""Tests for checkpoint capture and auto-persistence (checkpoint.py)."""

import pytest
from unittest.mock import MagicMock, patch

from flagscale.agent.react.checkpoint import CheckpointMixin


class FakeAgent(CheckpointMixin):
    """Minimal stub providing state CheckpointMixin expects."""

    def __init__(self):
        self._experiment_manager = MagicMock()
        self._files_written_this_session = set()
        self._porting_mode = False

    def _is_training_launch(self, cmd):
        return "flagscale train" in cmd or "torchrun" in cmd


class TestExtractErrorSummary:

    def setup_method(self):
        self.agent = FakeAgent()

    def test_finds_error_line(self):
        result = "Loading model...\nRuntimeError: CUDA out of memory\nProcess exited"
        summary = self.agent._extract_error_summary(result)
        assert "RuntimeError" in summary
        assert "CUDA out of memory" in summary

    def test_finds_traceback_line(self):
        result = "step 1\nstep 2\nTraceback (most recent call last):\n  File x.py"
        summary = self.agent._extract_error_summary(result)
        assert "Traceback" in summary

    def test_truncates_long_line(self):
        long_error = "RuntimeError: " + "x" * 300
        result = f"ok\n{long_error}\nend"
        summary = self.agent._extract_error_summary(result)
        assert len(summary) <= 200
        assert summary.endswith("...")

    def test_fallback_to_first_nonempty_line(self):
        result = "some output without error keywords"
        summary = self.agent._extract_error_summary(result)
        assert summary == "some output without error keywords"

    def test_empty_result(self):
        summary = self.agent._extract_error_summary("")
        assert summary == "Unknown error"

    def test_only_whitespace(self):
        summary = self.agent._extract_error_summary("\n\n  \n")
        assert summary == "Unknown error"


class TestCheckHydraCacheStale:

    def setup_method(self):
        self.agent = FakeAgent()
        self.agent._experiment_manager.get_current_experiment.return_value = "qwen3_exp"

    def test_no_warning_without_config_edits(self):
        self.agent._files_written_this_session = set()
        result = self.agent._check_hydra_cache_stale("flagscale train outputs/qwen3_exp")
        assert result == ""

    def test_warning_with_yaml_edit(self):
        self.agent._files_written_this_session = {"examples/qwen3/conf/train.yaml"}
        result = self.agent._check_hydra_cache_stale("flagscale train outputs/qwen3_exp")
        assert "HYDRA CACHE" in result
        assert "train.yaml" in result
        assert "outputs/qwen3_exp" in result

    def test_warning_with_yml_config(self):
        self.agent._files_written_this_session = {"config/model.yml"}
        result = self.agent._check_hydra_cache_stale("flagscale train outputs/my_run")
        assert "HYDRA CACHE" in result

    def test_no_warning_for_non_config_yaml(self):
        self.agent._files_written_this_session = {"README.md", "src/model.py"}
        result = self.agent._check_hydra_cache_stale("flagscale train outputs/exp1")
        assert result == ""

    def test_uses_experiment_name_as_fallback(self):
        self.agent._files_written_this_session = {"examples/conf/train.yaml"}
        result = self.agent._check_hydra_cache_stale("flagscale train --some-flag")
        assert "outputs/qwen3_exp" in result

    def test_no_warning_without_output_dir_or_experiment(self):
        self.agent._experiment_manager.get_current_experiment.return_value = None
        self.agent._files_written_this_session = {"examples/conf/train.yaml"}
        result = self.agent._check_hydra_cache_stale("flagscale train")
        assert result == ""


class TestCheckpointTrainingFailure:

    def setup_method(self):
        self.agent = FakeAgent()

    def test_records_failure_to_experiment(self):
        self.agent._experiment_manager.get_current_experiment.return_value = "exp1"
        result = "RuntimeError: CUDA OOM\nProcess killed"
        warning = self.agent._checkpoint_training_failure("flagscale train", result)
        self.agent._experiment_manager.update_last_attempt.assert_called_once()
        call_args = self.agent._experiment_manager.update_last_attempt.call_args[0]
        assert call_args[0] == "exp1"
        assert "FAILED:" in call_args[1]
        assert warning == ""

    def test_no_experiment_returns_empty(self):
        self.agent._experiment_manager.get_current_experiment.return_value = None
        warning = self.agent._checkpoint_training_failure("cmd", "error")
        assert warning == ""
        self.agent._experiment_manager.update_last_attempt.assert_not_called()

    def test_update_failure_returns_warning(self):
        self.agent._experiment_manager.get_current_experiment.return_value = "exp1"
        self.agent._experiment_manager.update_last_attempt.side_effect = RuntimeError("disk full")
        warning = self.agent._checkpoint_training_failure("cmd", "RuntimeError: x")
        assert "Experiment update failed" in warning


class TestCheckpointNewError:

    def setup_method(self):
        self.agent = FakeAgent()

    def test_first_error_recorded(self):
        result = self.agent._checkpoint_new_error("oom_cuda0", "CUDA OOM on device 0")
        assert result == ""
        assert "oom_cuda0" in self.agent._seen_errors

    def test_duplicate_error_ignored(self):
        self.agent._checkpoint_new_error("oom_cuda0", "first")
        result = self.agent._checkpoint_new_error("oom_cuda0", "second")
        assert result == ""

    def test_different_errors_both_recorded(self):
        self.agent._checkpoint_new_error("oom", "oom error")
        self.agent._checkpoint_new_error("nccl", "nccl timeout")
        assert "oom" in self.agent._seen_errors
        assert "nccl" in self.agent._seen_errors


class TestAutoPersistOnEvent:

    def setup_method(self):
        self.agent = FakeAgent()
        self.agent._experiment_manager.get_current_experiment.return_value = "exp1"

    def test_training_launch_success(self):
        self.agent._auto_persist_on_event(
            "shell", {"command": "flagscale train --config x"},
            "iteration 1, loss 10.5", False
        )
        self.agent._experiment_manager.update_last_attempt.assert_called_once()
        call_args = self.agent._experiment_manager.update_last_attempt.call_args[0]
        assert "SUCCESS" in call_args[1]

    def test_training_launch_failure(self):
        self.agent._auto_persist_on_event(
            "shell", {"command": "flagscale train --config x"},
            "Traceback:\nRuntimeError: bad config", True
        )
        self.agent._experiment_manager.update_last_attempt.assert_called_once()
        call_args = self.agent._experiment_manager.update_last_attempt.call_args[0]
        assert "FAIL" in call_args[1]

    def test_non_training_shell_ignored(self):
        self.agent._auto_persist_on_event("shell", {"command": "ls -la"}, "output", False)
        self.agent._experiment_manager.update_last_attempt.assert_not_called()

    def test_write_file_in_porting_mode(self):
        self.agent._porting_mode = True
        self.agent._auto_persist_on_event(
            "write_file", {"path": "/src/model/attention.py"}, "OK", False
        )
        # No crash — _auto_record_code_change is a no-op but should be called

    def test_exception_swallowed(self):
        self.agent._experiment_manager.get_current_experiment.side_effect = RuntimeError("boom")
        # Should not raise
        self.agent._auto_persist_on_event("shell", {"command": "flagscale train"}, "error", True)
