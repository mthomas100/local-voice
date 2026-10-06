"""The status view's data end to end: lvclient fetches the mock's /v1/status when something happened (the connection
came up, a turn ended, the space changed, a switch went unanswered), never on a timer, and switches space and mode
with the server's answer, a refusal, or no answer at all (the M1 orchestrator ignores both messages).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from test_apps import DASHBOARD, check_dashboard

PTT_TURN = "press; say-wait:tone:440:1.0; release; wait:end_of_turn; wait:sent:played_ms"


def get_status(m) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{m.port}/v1/status", timeout=5) as r:
        assert r.headers["Content-Type"] == "application/json"
        return json.loads(r.read())


def statuses(c) -> list[dict]:
    return c.events("status")


def outcomes(c) -> list[tuple[str, str, str]]:
    return [(r["phase"], r["kind"], r["name"]) for r in c.events("switch") if r["phase"] != "requested"]


def test_status_served_per_protocol(mock):
    """PROTOCOL.md's fields, and the orchestrator's `spaces`, `turns` and `uptime_s`."""
    s = get_status(mock())
    for key in ("v", "state", "space", "mode", "tier", "model", "hold", "tool", "last_turn", "clients"):
        assert key in s, key
    assert s["v"] == 1 and s["space"] == "home" and s["hold"] == {"phase": "open", "why": ""} and s["tool"] is None
    assert set(s["last_turn"]) >= {"eos_to_first_audio_ms", "stt_ms", "llm_ttft_ms", "tts_first_audio_ms"}
    assert sorted(s["spaces"]) == ["atlas", "home", "kb", "voice"]
    assert s["spaces"]["atlas"]["description"] == "your atlas journal"


def test_status_follows_the_session(mock, client):
    """Fetched on connect and after the turn; the turn's latency and count show; nothing polls in between."""
    m = mock("--tool")
    c = client(m.url, "connect; wait:ready; wait:status:5000; " + PTT_TURN + "; wait:status:5000; sleep:3000")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    got = statuses(c)
    first, last = got[0], got[-1]
    assert first["reason"] in ("connected", "space-changed") and first["status"]["space"] == "home"
    assert first["status"]["model"] == "mock/qwen38" and len(first["status"]["spaces"]) == 4
    assert last["reason"] == "turn-ended"
    turn = last["status"]["last_turn"]
    assert turn["eos_to_first_audio_ms"] > 0 and turn["tools"] == ["read"] and last["status"]["turns"] == 1
    assert last["status"]["tool"] is None, "the tool ended before the answer"
    clients = last["status"]["clients"]
    assert [(x["device"], x["client"]) for x in clients] == [("e2e", "test")]
    # About 6 s connected: one fetch for connect (welcome and space share it) and one for the turn's end.
    requests = m.events("status-request")
    assert len(requests) == 2, f"{len(requests)} status requests; the client must not poll"


def test_tool_shows_in_status_while_it_runs(mock, client):
    m = mock("--tool", "--tool-seconds", "1.5")
    proc, _ = client(m.url, PTT_TURN, wait=False)
    seen = None
    deadline = time.time() + 15
    while time.time() < deadline and seen is None:
        s = get_status(m)
        if s["tool"]:
            seen = s
        time.sleep(0.05)
    proc.wait(30)
    assert seen and seen["tool"]["name"] == "read" and seen["tool"]["label"] == "reading your journal"
    assert seen["state"] == "thinking"
    assert get_status(m)["tool"] is None


def test_switch_space_and_mode(mock, client):
    """Switches the server answers with a `space` message, and two it refuses with an `error`."""
    m = mock()
    c = client(m.url, "connect; wait:ready; wait:status:5000; space:atlas; wait:switch:switched; mode:act; wait:switch:switched; "
                      "next:status:5000; space:nowhere; wait:switch:refused; mode:sideways; wait:switch:refused; sleep:300")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    assert [msg["name"] for msg in m.received("space")] == ["atlas", "nowhere"]
    assert [msg["name"] for msg in m.received("mode")] == ["act", "sideways"]
    assert outcomes(c) == [("switched", "space", "atlas"), ("switched", "mode", "act"),
                           ("refused", "space", "nowhere"), ("refused", "mode", "sideways")]
    refusals = [r for r in c.events("switch") if r["phase"] == "refused"]
    assert refusals[0]["reason"] == "there is no space called nowhere"
    # The status fetched after the switches shows the new space with what only the status knows.
    after = [r["status"] for r in statuses(c) if r["status"]["space"] == "atlas" and r["status"]["mode"] == "act"]
    assert after and after[-1]["tier"] == "trusted" and after[-1]["model"] == "mock/qwen38"
    assert [r["reason"] for r in statuses(c)] == ["connected", "space-changed"], "one fetch for both switches"
    s = get_status(m)
    assert (s["space"], s["mode"], s["tier"]) == ("atlas", "act", "trusted")


def test_switch_without_an_answer(mock, client):
    """The M1 orchestrator ignores `space` and `mode`: after 5 s the switch counts as unanswered, and a status fetch
    shows where the server really is."""
    m = mock("--no-switch")
    c = client(m.url, "connect; wait:ready; space:atlas; wait:switch:no-answer:8000; next:status:3000")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    assert outcomes(c) == [("no-answer", "space", "atlas")]
    requested = c.events("switch")[0]["t"]
    unanswered = [r for r in c.events("switch") if r["phase"] == "no-answer"][0]["t"]
    assert 5.0 <= unanswered - requested <= 5.6
    last = statuses(c)[-1]
    assert last["reason"] == "switch-unanswered" and last["status"]["space"] == "home"
    assert m.events("switch-ignored")


def test_status_unavailable(mock, client):
    m = mock("--no-status")
    c = client(m.url, "connect; wait:ready; wait:status-failed:5000")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (failed,) = c.events("status_failed")
    assert failed["error"] == "the server has no /v1/status (404)" and failed["reason"] in ("connected", "space-changed")


def test_the_apps_dashboard_script_on_lvclient(mock, client):
    """The script both apps run for their dashboard (test_apps.py), on lvclient: the same core and script runner,
    checked now, while the apps themselves wait for a window in which they may be launched."""
    m = mock()
    c = client(m.url, DASHBOARD)
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    check_dashboard(c.records)
