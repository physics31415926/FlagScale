"""Tests for session persistence."""

import json
import os

import pytest

from flagscale.agent.react.session import save_session, load_session, list_sessions


class TestSession:
    def test_save_and_load(self, tmp_path):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        path = save_session(msgs, session_dir=str(tmp_path), session_id="test1")
        assert os.path.isfile(path)

        data = load_session(path)
        assert data["id"] == "test1"
        assert len(data["messages"]) == 3
        assert data["messages"][1]["content"] == "Hello"

    def test_save_with_metadata(self, tmp_path):
        msgs = [{"role": "user", "content": "test"}]
        path = save_session(msgs, session_dir=str(tmp_path), metadata={"model": "gpt-4o"})
        data = load_session(path)
        assert data["metadata"]["model"] == "gpt-4o"

    def test_list_sessions(self, tmp_path):
        save_session([{"role": "user", "content": "a"}], session_dir=str(tmp_path), session_id="s1")
        save_session([{"role": "user", "content": "b"}], session_dir=str(tmp_path), session_id="s2")
        sessions = list_sessions(session_dir=str(tmp_path))
        assert len(sessions) == 2
        ids = {s["id"] for s in sessions}
        assert "s1" in ids
        assert "s2" in ids

    def test_list_sessions_empty(self, tmp_path):
        sessions = list_sessions(session_dir=str(tmp_path))
        assert sessions == []

    def test_list_sessions_nonexistent_dir(self):
        sessions = list_sessions(session_dir="/nonexistent/path/xyz")
        assert sessions == []

    def test_list_sessions_counts_user_turns(self, tmp_path):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        save_session(msgs, session_dir=str(tmp_path), session_id="multi")
        sessions = list_sessions(session_dir=str(tmp_path))
        assert sessions[0]["turns"] == 2

    def test_auto_session_id(self, tmp_path):
        path = save_session([{"role": "user", "content": "x"}], session_dir=str(tmp_path))
        data = load_session(path)
        assert data["id"].startswith("session_")
