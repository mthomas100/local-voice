"""Pure-Python parts of the bridge: argv building, event mapping, the heredoc helper."""
from pathlib import Path

from pi_rpc_bridge import (ConfirmRequest, MessageEnd, Notify, SpaceConfig, Status, TextDelta, ThinkingDelta, ToolCallStarted,
                           ToolEnd, ToolUpdate, atlas_capture_command, heredoc_command, interrupted_note, to_event)


def test_argv_allowlists_and_session():
    cfg = SpaceConfig(name="home", root=Path("/tmp"), tools=["read", "ls"], skills=["/s/a"], extensions=["/e/x.ts"],
                      append_system_prompt=["be brief"], session_dir=None, session_name="voice-home")
    a = cfg.argv()
    assert a[:3] == ["pi", "--mode", "rpc"]
    assert ["--tools", "read,ls"] == a[a.index("--tools"):a.index("--tools") + 2]
    assert "--no-skills" in a and a[a.index("--skill") + 1] == "/s/a"
    assert "--no-extensions" in a and a[a.index("--extension") + 1] == "/e/x.ts"
    assert "--no-session" in a and "--approve" in a and a[a.index("--name") + 1] == "voice-home"
    assert ["--thinking", "off"] == a[a.index("--thinking"):a.index("--thinking") + 2]


def test_argv_continue_only_when_a_session_exists(tmp_path):
    cfg = SpaceConfig(name="s", root=tmp_path, session_dir=tmp_path / "sess")
    assert "--continue" not in cfg.argv()
    (tmp_path / "sess").mkdir()
    (tmp_path / "sess" / "x.jsonl").write_text("{}\n")
    assert "--continue" in cfg.argv()


def test_child_env_scrubs_foreign_session_identity(monkeypatch):
    monkeypatch.setenv("KB_SESSION", "claude-code:abc")
    monkeypatch.setenv("CLAUDE_ENV_FILE", "/tmp/x")
    env = SpaceConfig(name="s", root=Path("/tmp"), env={"FOO": "1"}).child_env()
    assert "KB_SESSION" not in env and "CLAUDE_ENV_FILE" not in env
    assert env["PI_OFFLINE"] == "1" and env["FOO"] == "1"


def test_event_mapping():
    assert isinstance(to_event({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hi", "contentIndex": 0}}), TextDelta)
    assert isinstance(to_event({"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "delta": "hm"}}), ThinkingDelta)
    tc = to_event({"type": "message_update", "assistantMessageEvent": {"type": "toolcall_start", "id": "c1", "toolName": "bash"}})
    assert isinstance(tc, ToolCallStarted) and tc.name == "bash"
    up = to_event({"type": "tool_execution_update", "toolCallId": "c1", "toolName": "bash",
                   "partialResult": {"content": [{"type": "text", "text": "tick 1\n"}]}})
    assert isinstance(up, ToolUpdate) and up.text == "tick 1\n"
    end = to_event({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash", "isError": True,
                    "result": {"content": [{"type": "text", "text": "Command aborted"}]}})
    assert isinstance(end, ToolEnd) and not end.ok
    me = to_event({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "a"}],
                                                      "stopReason": "aborted", "errorMessage": "Request was aborted"}})
    assert isinstance(me, MessageEnd) and me.stop_reason == "aborted" and me.text == "a"
    cr = to_event({"type": "extension_ui_request", "id": "u1", "method": "confirm", "title": "T", "message": "M", "timeout": 2500})
    assert isinstance(cr, ConfirmRequest) and cr.timeout_ms == 2500
    assert isinstance(to_event({"type": "extension_ui_request", "id": "u2", "method": "notify", "message": "x", "notifyType": "warning"}), Notify)
    st = to_event({"type": "extension_ui_request", "id": "u3", "method": "setStatus", "statusKey": "hold"})
    assert isinstance(st, Status) and st.text is None


def test_heredoc_marker_never_collides():
    cmd = heredoc_command("cat", "a\nATLAS_END\nrm -rf ~")
    first, *rest = cmd.split("\n")
    marker = first.split("<<'")[1].rstrip("'")
    assert marker != "ATLAS_END" and rest[-1] == marker and rest.count(marker) == 1
    assert atlas_capture_command("hi").startswith("python3 atlas.py capture --via voice <<'ATLAS_END'\nhi\n")
    assert "--to journal/ideas.md" in atlas_capture_command("idea", to="journal/ideas.md")


def test_interrupted_note():
    assert "heard only" in interrupted_note("Once upon")
    assert "before the user heard any" in interrupted_note("  ")
