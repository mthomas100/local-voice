"""Approvals end to end (PROTOCOL.md "Approvals", 2026-10-05): lvclient, the same core as the two apps, against the
mock's approval scenarios over a real WebSocket.

lvclient keeps the questions as the apps do and logs each card as they would draw it (`approval` lines: shown,
answered, closed), so these tests check what the card shows, what goes back on the wire, and when the card closes,
with no app launched. The mock logs what it received (`approval-answer`). Planted ground truth: each scenario's
summary, action and choices are known, and so is the answer each test gives.
"""

from __future__ import annotations

import re

ASK = "press; say-wait:tone:440:0.5; release; wait:approval-shown"
# Each question is also asked aloud (its own reply, q<n>) before the answer (r<n>): two end_of_turns per question.
DONE = "wait:end_of_turn; wait:end_of_turn; wait:sent:played_ms; wait:sent:played_ms"


def approvals(c, phase: str | None = None) -> list[dict]:
    return [r for r in c.records if r["event"] == "approval" and (phase is None or r["phase"] == phase)]


def replies(c) -> list[str]:
    return [msg["delta"] for msg in c.recv("reply_text")]


def index(c, event: str, t: str) -> int:
    """Position of the first `sent`/`recv` record of message type t in the client's log."""
    return next(i for i, r in enumerate(c.records) if r["event"] == event and r["msg"]["t"] == t)


def test_write_with_a_preview_done(mock, client):
    m = mock("--approval", "write")
    c = client(m.url, f"{ASK}; choose:allow_once; {DONE}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr

    (shown,) = approvals(c, "shown")
    assert shown["headline"] == "Create a new file plan.txt in your notes folder with the text below."
    assert shown["effect"] == "create" and shown["facts"] == ["write", "home space", "act mode"]
    assert shown["exact"] == [{"kind": "path", "label": "New file", "text": "/Users/owner/notes/plan.txt"}]
    text = "Plan for Saturday\n\n- walk before breakfast\n- call Mum at 11\n"
    assert shown["preview"] == {"kind": "text", "label": "Text to be written", "chars": len(text), "cut_marker": None}
    assert [(ch["id"], ch["label"], ch["role"]) for ch in shown["choices"]] == [
        ("allow_once", "Do it", "primary"), ("deny", "Don't", "deny")]
    assert shown["timeout_ms"] == 120000 and shown["legacy"] is False

    (sent,) = c.sent("confirm_response")
    assert sent == {"t": "confirm_response", "id": "c1", "confirmed": True, "choice": "allow_once"}
    (a,) = m.events("approval-answer")
    assert (a["by"], a["matches"], a["offered"], a["consistent"], a["outcome"]) == ("button", True, True, True,
                                                                                  "allowed")
    assert index(c, "sent", "confirm_response") < index(c, "recv", "tool"), "nothing ran before the answer"
    assert replies(c)[-1] == "Done. plan.txt is in your notes folder."
    (answered,) = approvals(c, "answered")
    assert answered["choice"] == "allow_once" and answered["confirmed"] is True and not approvals(c, "closed")


def test_edit_with_a_diff_refused(mock, client):
    m = mock("--approval", "edit")
    c = client(m.url, f"{ASK}; choose:deny; {DONE}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (shown,) = approvals(c, "shown")
    assert shown["exact"][0]["label"] == "File to change" and shown["effect"] == "modify"
    assert shown["preview"]["kind"] == "diff" and shown["preview"]["label"] == "Changes"
    assert c.sent("confirm_response") == [{"t": "confirm_response", "id": "c1", "confirmed": False, "choice": "deny"}]
    (a,) = m.events("approval-answer")
    assert a["outcome"] == "denied" and a["consistent"] is True
    assert not c.recv("tool"), "refused: nothing ran"
    assert replies(c)[-1] == "OK, I left todo.md as it was."


def test_shell_command(mock, client):
    m = mock("--approval", "bash")
    c = client(m.url, f"{ASK}; confirm:yes; {DONE}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (shown,) = approvals(c, "shown")
    assert shown["exact"] == [{"kind": "command", "label": "Command", "text": "ls -lt ~/Downloads | head -20"}]
    assert shown["cwd"] == "/Users/owner" and shown["effect"] == "run" and shown["preview"] is None
    assert c.sent("confirm_response")[0]["choice"] == "allow_once"
    assert m.events("approval-answer")[0]["outcome"] == "allowed"


def test_allow_for_the_rest_of_the_session(mock, client):
    m = mock("--approval", "session")
    again = "press; say-wait:tone:440:0.5; release; wait:end_of_turn; wait:sent:played_ms"
    c = client(m.url, f"{ASK}; choose:allow_session; {DONE}; {again}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (shown,) = approvals(c, "shown")
    assert [(ch["id"], ch["role"]) for ch in shown["choices"]] == [
        ("allow_once", "primary"), ("allow_session", "secondary"), ("deny", "deny")]
    assert shown["choices"][1]["label"] == "Allow this for the rest of the session"
    assert shown["spoken_hint"] == "Or say “yes”, “yes, for this session” or “no”."
    assert c.sent("confirm_response") == [
        {"t": "confirm_response", "id": "c1", "confirmed": True, "choice": "allow_session"}]
    # The same kind of action again: the server does not ask, so no second card.
    (skipped,) = m.events("approval-skipped")
    assert skipped["tool"] == "kb" and len(c.recv("confirm_request")) == 1
    assert replies(c)[-1] == "Done. The page is in your knowledge base."


def test_answered_aloud_closes_the_card(mock, client):
    m = mock("--approval", "cancel")
    voice = "sleep:900; press; say-wait:tone:440:0.4; release; wait:approval-closed"
    c = client(m.url, f"{ASK}; {voice}; {DONE}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (closed,) = approvals(c, "closed")
    assert closed["why"] == "answered" and closed["by"] == "server"
    assert not c.sent("confirm_response"), "the card closed; nothing to send"
    assert m.events("approval-answer")[0]["by"] == "voice" and m.events("answer-turn")
    assert replies(c)[-1] == "The newest is a PDF from this morning."


def test_no_answer_times_out(mock, client):
    m = mock("--approval", "timeout")
    c = client(m.url, f"{ASK}; wait:approval-closed:8000; {DONE}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (shown,), (closed,) = approvals(c, "shown"), approvals(c, "closed")
    assert shown["timeout_ms"] == 3000
    assert closed["why"] == "timeout" and closed["by"] == "server"
    assert 2.9 <= closed["t"] - shown["t"] <= 3.6, "closed when its time ran out"
    assert not c.sent("confirm_response"), "silence is no: nothing is sent"
    assert m.events("approval-answer")[0]["by"] == "silence"
    assert replies(c)[-1] == "I didn't create plan.txt, because there was no answer."


def test_old_style_request(mock, client):
    """An old-style question, as early servers asked it: title and message only."""
    m = mock("--approval", "legacy")
    c = client(m.url, f"{ASK}; confirm:yes; {DONE}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (shown,) = approvals(c, "shown")
    assert shown["legacy"] is True and shown["headline"] == "May I change your knowledge base?"
    assert shown["exact"] == [{"kind": "details", "label": "Details", "text": "kb new Issue tea-brewing-notes "
                               "--title Tea brewing notes"}]
    assert [ch["label"] for ch in shown["choices"]] == ["Do it", "Don't"] and shown["choices_offered"] is False
    assert c.sent("confirm_response") == [{"t": "confirm_response", "id": "c1", "confirmed": True}], "no choice"
    (a,) = m.events("approval-answer")
    assert a["choice"] is None and a["outcome"] == "allowed"


def test_cut_preview_marker(mock, client):
    m = mock("--approval", "long")
    c = client(m.url, f"{ASK}; confirm:no; {DONE}")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    preview = approvals(c, "shown")[0]["preview"]
    assert re.fullmatch(r"… \[cut: [\d,]+ more characters\]", preview["cut_marker"] or ""), preview
    assert preview["chars"] > 4000


def test_a_new_turn_overtakes_the_question(mock, client):
    m = mock("--approval", "write,none")
    c = client(m.url, f"{ASK}; wait:sent:played_ms; text:never mind; wait:approval-closed; wait:end_of_turn; "
                      "wait:end_of_turn; wait:sent:played_ms")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (closed,) = approvals(c, "closed")
    assert closed["why"] == "overtaken" and closed["by"] == "server"
    assert not c.sent("confirm_response") and m.events("approval-answer")[0]["by"] == "overtaken"
    assert replies(c)[-1] == "you typed: never mind"


def test_a_choice_not_on_the_card_is_never_sent(mock, client):
    m = mock("--approval", "write")
    c = client(m.url, f"{ASK}; choose:allow_session", timeout=20)
    assert c.proc.returncode == 1
    assert "no allow_session on its card" in c.summary["failure"]
    assert not c.sent("confirm_response") and not m.received("confirm_response")


def test_a_dropped_connection_closes_the_card(mock, client):
    m = mock("--approval", "write", "--drop-after-welcome-s", "2.0")
    c = client(m.url, f"{ASK}; wait:approval-closed:6000")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (closed,) = approvals(c, "closed")
    assert closed["why"] == "connection lost" and closed["by"] == "client"
    assert not c.sent("confirm_response")


def test_the_apps_approval_session_on_lvclient(mock, client):
    """The apps' approval session (test_apps.py, run with the built apps) on lvclient: the same script and
    the same checks, with no app launched."""
    from test_apps import APPROVAL_PLAN, APPROVAL_SCRIPT, check_approvals
    m = mock("--approval", APPROVAL_PLAN)
    c = client(m.url, APPROVAL_SCRIPT, timeout=120)
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    check_approvals(m, c.records)
