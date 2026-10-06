"""The two apps themselves, unattended, against the mock server (definition of done M2.2).

The Mac menu-bar app runs from its build folder; the iPhone app runs in the iOS simulator. Both use their test mode:
a synthetic microphone (no permission prompt), output muted, a session script from launch arguments, and a JSONL
event log. Nothing touches the user's own Option+Space: the Mac run registers an unused combination to prove the Carbon
registration works, and the iPhone runs never ask for the microphone.

Run with `../build.sh apps-e2e` (builds both apps first). Marked `apps`; plain `pytest` skips them.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import time

import pytest

from conftest import APPLE, read_jsonl

pytestmark = pytest.mark.apps

PRODUCTS = APPLE / "build" / "DerivedData" / "Build" / "Products"
MAC_APP = PRODUCTS / "Debug" / "LocalVoice.app"
IOS_APP = PRODUCTS / "Debug-iphonesimulator" / "LocalVoice.app"
BUNDLE = "local.voice.iphone"
SIMULATOR = os.environ.get("LV_SIMULATOR", "iPhone 18 Pro")
TURN = "press; say-wait:tone:440:1.0; release; wait:end_of_turn; wait:sent:played_ms"
# The dashboard follows the server: its status on connect, a space and a mode switch, a turn, the status after it.
DASHBOARD = ("connect; wait:ready; wait:status:5000; space:atlas; wait:switch:switched; mode:act; "
             "wait:switch:switched; wait:status:space-changed:5000; " + TURN + "; wait:status:turn-ended:5000")


def events(records, name):
    return [r for r in records if r.get("event") == name]


def check_turn(mock, records, device):
    hello = mock.received("hello")[0]
    assert hello["device"] == device and hello["mic"] in ("ptt", "vad")
    turn = mock.events("turn")[0]
    assert turn["bad_sizes"] == [], "640-byte frames"
    assert abs(turn["speech_hz"] - 440) < 5 and abs(turn["speech_seconds"] - 1.0) <= 0.08
    played = {r["reply_id"]: r for r in mock.events("played")}
    assert abs(played["r1"]["ms"] - 1200) <= 60
    done = events(records, "script-done")
    assert done and done[-1]["ok"] is True, "the in-app script finished"
    assert any(r["event"] == "recv" and r["msg"]["t"] == "audio_start" for r in records)
    assert any(r["event"] == "playback" and r["kind"] == "finished" for r in records)


# Approvals (PROTOCOL.md, 2026-10-05): every kind of question in one session, each answered as a person would, by each
# kind of button, aloud and by silence. The mock asks one per turn in this order; the second "session" turn is allowed
# for the session, so it asks nothing (and so would an old-style kb new after it: the allowance covers the same tool
# and command prefix, PROTOCOL.md, hence legacy comes first). Each card is pictured while it is up: the Mac draws its
# panel (<name>.<label>.png), the iPhone harness takes a simulator screenshot (<name>-<label>.png).
APPROVAL_PLAN = "write,edit,legacy,session,session,long,cancel,timeout"
APPROVAL_LABELS = ["write", "edit", "legacy", "session", "long", "cancel", "timeout"]
TWO = "wait:end_of_turn; wait:end_of_turn"  # each question is asked aloud (q<n>), then answered (r<n>)


def ask(label: str) -> str:
    return f"press; say-wait:tone:440:0.6; release; wait:approval-shown; sleep:700; snapshot:{label}; sleep:800"


APPROVAL_SCRIPT = "; ".join([
    "connect; wait:ready",
    ask("write"), "choose:allow_once", TWO,
    ask("edit"), "choose:deny", TWO,
    ask("legacy"), "confirm:yes", TWO,
    ask("session"), "choose:allow_session", TWO,
    "press; say-wait:tone:440:0.6; release; wait:end_of_turn",
    ask("long"), "confirm:no", TWO,
    ask("cancel"), "press; say-wait:tone:440:0.4; release; wait:approval-closed", TWO,
    ask("timeout"), "wait:approval-closed:8000", TWO,
    "sleep:1500",
])


def check_approvals(mock, records):
    """What the app's cards showed, what it sent, and how each closed, against what the mock asked and received."""
    approvals = [r for r in records if r.get("event") == "approval"]
    shown = [r for r in approvals if r["phase"] == "shown"]
    assert [r["id"] for r in shown] == [f"c{i}" for i in range(1, 8)], "seven cards, the session's repeat not asked"
    assert [r["effect"] for r in shown] == ["create", "modify", None, "create", "create", "run", "create"]
    assert [len(r["choices"]) for r in shown] == [2, 2, 2, 3, 2, 2, 2]
    assert re.fullmatch(r"… \[cut: [\d,]+ more characters\]", shown[4]["preview"]["cut_marker"] or "")
    assert shown[1]["preview"]["kind"] == "diff" and shown[2]["legacy"] is True
    answered = [(r["id"], r["choice"]) for r in approvals if r["phase"] == "answered"]
    assert answered == [("c1", "allow_once"), ("c2", "deny"), ("c3", "allow_once"), ("c4", "allow_session"),
                        ("c5", "deny")]
    closed = [(r["id"], r["why"], r["by"]) for r in approvals if r["phase"] == "closed"]
    assert closed == [("c6", "answered", "server"), ("c7", "timeout", "server")]
    sent = [r["msg"] for r in records if r.get("event") == "sent" and r["msg"]["t"] == "confirm_response"]
    assert [(m["id"], m["confirmed"], m.get("choice")) for m in sent] == [
        ("c1", True, "allow_once"), ("c2", False, "deny"), ("c3", True, None), ("c4", True, "allow_session"),
        ("c5", False, "deny")], "an old-style question gets confirmed alone"
    outcomes = [(r["id"], r["by"], r["outcome"]) for r in mock.events("approval-answer")]
    assert outcomes == [("c1", "button", "allowed"), ("c2", "button", "denied"), ("c3", "button", "allowed"),
                        ("c4", "button", "allowed"), ("c5", "button", "denied"), ("c6", "voice", "allowed"),
                        ("c7", "silence", "denied")]
    assert all(r["consistent"] for r in mock.events("approval-answer") if r["by"] == "button")
    assert len(mock.events("approval-skipped")) == 1
    finished = [r for r in records if r.get("event") in ("script-done", "summary")]  # an app's, or lvclient's
    assert finished and finished[-1]["ok"] is True, "the script finished"


def check_dashboard(records):
    statuses = events(records, "status")
    assert statuses and statuses[0]["status"]["space"] == "home", "the status came on connect"
    switched = [(r["kind"], r["name"]) for r in events(records, "switch") if r["phase"] == "switched"]
    assert switched == [("space", "atlas"), ("mode", "act")]
    last = [r for r in statuses if r["reason"] == "turn-ended"][-1]["status"]
    assert (last["space"], last["mode"], last["tier"]) == ("atlas", "act", "trusted")
    assert last["last_turn"]["eos_to_first_audio_ms"] > 0, "the turn's latency reached the dashboard"


# MARK: Mac


@pytest.mark.skipif(not MAC_APP.exists(), reason="build the Mac app first (./build.sh mac)")
def test_mac_app_completes_a_turn(mock, run_dir):
    m = mock()
    log = run_dir / "mac-app.jsonl"
    snapshot = run_dir / "mac.png"
    argv = [str(MAC_APP / "Contents" / "MacOS" / "LocalVoice"),
            "-LVServerURL", m.url, "-LVDevice", "mac-e2e", "-LVAudio", "synthetic", "-LVOutputVolume", "0",
            "-LVScript", TURN, "-LVEventLog", str(log), "-LVQuitAfterScript", "YES", "-LVShowPanel", "YES",
            # A combination nobody uses, so a dictation app's Option+Space (such as Handy) is never touched.
            "-LVRegisterHotKey", "YES", "-LVHotKey", "control+option+shift+command+f19",
            "-LVSnapshot", str(snapshot)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    (run_dir / "mac-app.out").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = read_jsonl(log)
    hotkey = events(records, "hotkey")[0]
    assert hotkey["registered"] is True and hotkey["status"] == 0, "Carbon RegisterEventHotKey works"
    engine = [r for r in events(records, "engine") if r["status"] == "running"]
    assert engine and "synthetic microphone" in engine[0]["detail"] and "(muted)" in engine[0]["detail"]
    check_turn(m, records, "mac-e2e")
    assert snapshot.exists() and snapshot.stat().st_size > 1000, "the panel rendered"


@pytest.mark.skipif(not MAC_APP.exists(), reason="build the Mac app first (./build.sh mac)")
def test_mac_app_dashboard_follows_the_server(mock, run_dir):
    m = mock()
    log = run_dir / "mac-dashboard.jsonl"
    snapshot = run_dir / "mac-dashboard.png"
    argv = [str(MAC_APP / "Contents" / "MacOS" / "LocalVoice"),
            "-LVServerURL", m.url, "-LVDevice", "mac-e2e", "-LVAudio", "synthetic", "-LVOutputVolume", "0",
            "-LVScript", DASHBOARD, "-LVEventLog", str(log), "-LVQuitAfterScript", "YES", "-LVShowPanel", "YES",
            "-LVRegisterHotKey", "NO", "-LVSnapshot", str(snapshot)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    (run_dir / "mac-dashboard.out").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = read_jsonl(log)
    check_dashboard(records)
    check_turn(m, records, "mac-e2e")
    card = run_dir / "mac-dashboard.dashboard.png"
    assert snapshot.exists() and card.exists() and card.stat().st_size > 1000, "the panel and the card rendered"


@pytest.mark.skipif(not MAC_APP.exists(), reason="build the Mac app first (./build.sh mac)")
def test_mac_app_approval_cards(mock, run_dir):
    m = mock("--approval", APPROVAL_PLAN)
    log = run_dir / "mac-approvals.jsonl"
    snapshot = run_dir / "mac-approvals.png"
    argv = [str(MAC_APP / "Contents" / "MacOS" / "LocalVoice"),
            "-LVServerURL", m.url, "-LVDevice", "mac-e2e", "-LVAudio", "synthetic", "-LVOutputVolume", "0",
            "-LVScript", APPROVAL_SCRIPT, "-LVEventLog", str(log), "-LVQuitAfterScript", "YES", "-LVShowPanel", "YES",
            "-LVRegisterHotKey", "NO", "-LVSnapshot", str(snapshot)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    (run_dir / "mac-approvals.out").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    check_approvals(m, read_jsonl(log))
    for label in APPROVAL_LABELS:
        png = run_dir / f"mac-approvals.{label}.png"
        assert png.exists() and png.stat().st_size > 10_000, f"the panel with the {label} card"


@pytest.mark.skipif(not MAC_APP.exists(), reason="build the Mac app first (./build.sh mac)")
def test_mac_app_closes_an_old_style_question_at_its_time(mock, run_dir):
    """A server from before Approvals sends no confirm_cancel: the card closes 3 s after its time, and sends nothing."""
    m = mock("--approval", "legacy", "--approval-timeout-ms", "2000")
    log = run_dir / "mac-expiry.jsonl"
    script = ("connect; wait:ready; press; say-wait:tone:440:0.6; release; wait:approval-shown; "
              "wait:approval-closed:9000; " + TWO + "; sleep:1500")
    argv = [str(MAC_APP / "Contents" / "MacOS" / "LocalVoice"),
            "-LVServerURL", m.url, "-LVDevice", "mac-e2e", "-LVAudio", "synthetic", "-LVOutputVolume", "0",
            "-LVScript", script, "-LVEventLog", str(log), "-LVQuitAfterScript", "YES", "-LVRegisterHotKey", "NO"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    (run_dir / "mac-expiry.out").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = read_jsonl(log)
    (shown,) = [r for r in records if r.get("event") == "approval" and r["phase"] == "shown"]
    (closed,) = [r for r in records if r.get("event") == "approval" and r["phase"] == "closed"]
    assert closed["why"] == "no answer in time" and closed["by"] == "client"
    assert 4.7 <= closed["t"] - shown["t"] <= 5.6, "2 s of time, then 3 s for a cancel that never comes"
    assert not [r for r in records if r.get("event") == "sent" and r["msg"]["t"] == "confirm_response"]
    assert not m.sent("confirm_cancel") and m.events("approval-answer")[0]["by"] == "silence"


# MARK: iPhone (simulator)


def simctl(*args, check=True, timeout=120) -> str:
    return subprocess.run(["xcrun", "simctl", *args], capture_output=True, text=True, check=check,
                          timeout=timeout).stdout


@pytest.fixture(scope="module")
def simulator():
    devices = json.loads(simctl("list", "devices", "available", "-j"))["devices"]
    udid = next((d["udid"] for runtime, ds in devices.items() if "iOS" in runtime for d in ds
                 if d["name"] == SIMULATOR), None)
    assert udid, f"no available {SIMULATOR} simulator"
    simctl("boot", udid, check=False)
    simctl("bootstatus", udid, "-b", timeout=300)
    simctl("install", udid, str(IOS_APP))
    yield udid


def run_ios(udid, run_dir, name, url, script, extra=(), wait_s=60):
    """Launches the app with a script, waits for it to finish, screenshots it, and returns the event log. A script
    step snapshot:<label> is a screenshot too (<name>-<label>.png), taken while the script sleeps after it."""
    log_rel = f"e2e/{name}.jsonl"
    container = pathlib.Path(simctl("get_app_container", udid, BUNDLE, "data").strip())
    device_log = container / "Documents" / log_rel
    if device_log.exists():
        device_log.unlink()
    simctl("launch", "--terminate-running-process", udid, BUNDLE,
           "-LVServerURL", url, "-LVDevice", f"iphone-{name}", "-LVAudio", "synthetic", "-LVOutputVolume", "0",
           "-LVScript", script, "-LVEventLog", log_rel, "-LVQuitAfterScript", "NO", *extra)
    deadline = time.time() + wait_s
    records = []
    pictured = set()
    while time.time() < deadline:
        records = read_jsonl(device_log) if device_log.exists() else []
        for r in events(records, "step"):
            label = r["step"].removeprefix("snapshot:")
            if r["step"].startswith("snapshot:") and label not in pictured:
                pictured.add(label)
                simctl("io", udid, "screenshot", str(run_dir / f"{name}-{label}.png"), check=False)
        if events(records, "script-done") or events(records, "script-failed"):
            break
        time.sleep(0.25)
    time.sleep(1.0)
    simctl("io", udid, "screenshot", str(run_dir / f"{name}.png"), check=False)
    shutil.copy(device_log, run_dir / f"{name}.jsonl") if device_log.exists() else None
    simctl("terminate", udid, BUNDLE, check=False)
    return read_jsonl(run_dir / f"{name}.jsonl")


@pytest.mark.skipif(not IOS_APP.exists(), reason="build the iPhone app first (./build.sh ios)")
def test_iphone_app_push_to_talk_turn_with_live_activity(mock, run_dir, simulator):
    m = mock()
    records = run_ios(simulator, run_dir, "ios-ptt", m.url, TURN, extra=["-LVTestLiveActivity", "YES"])
    check_turn(m, records, "iphone-ios-ptt")
    live = events(records, "live-activity")
    assert live and live[0]["result"] == "started", live
    assert any(r["result"] == "updated" for r in live), "the Live Activity followed the session state"


@pytest.mark.skipif(not IOS_APP.exists(), reason="build the iPhone app first (./build.sh ios)")
def test_iphone_action_button_intent_starts_hands_free(mock, run_dir, simulator):
    """What the Action button runs: StartTalkingIntent.perform() in the app, then an open-microphone turn."""
    m = mock()
    script = "wait:hands-free-on; wait:ready; sleep:300; say-wait:tone:440:1.0; wait:end_of_turn; wait:sent:played_ms"
    records = run_ios(simulator, run_dir, "ios-intent", m.url, script, extra=["-LVTestIntent", "YES"])
    assert m.received("hello")[-1]["mic"] == "vad", "hands-free is an open-microphone session"
    check_turn(m, records, "iphone-ios-intent")


@pytest.mark.skipif(not IOS_APP.exists(), reason="build the iPhone app first (./build.sh ios)")
def test_iphone_callkit_call_mode_request(mock, run_dir, simulator):
    """CallKit in the simulator: the request is made and its outcome logged (the full call path needs a device)."""
    m = mock()
    records = run_ios(simulator, run_dir, "ios-callkit", m.url, "connect; wait:ready; sleep:4000",
                      extra=["-LVTestCallKit", "YES"])
    calls = events(records, "callkit")
    assert calls, "the CallKit request ran and reported"
    (run_dir / "callkit-outcome.txt").write_text(json.dumps(calls, indent=1))


@pytest.mark.skipif(not IOS_APP.exists(), reason="build the iPhone app first (./build.sh ios)")
def test_iphone_dashboard_follows_the_server(mock, run_dir, simulator):
    """The status sheet open (-LVShowStatus) while the app switches space and mode and completes a turn."""
    m = mock()
    records = run_ios(simulator, run_dir, "ios-dashboard", m.url, DASHBOARD, extra=["-LVShowStatus", "YES"])
    check_dashboard(records)
    check_turn(m, records, "iphone-ios-dashboard")


@pytest.mark.skipif(not IOS_APP.exists(), reason="build the iPhone app first (./build.sh ios)")
def test_iphone_app_approval_sheet(mock, run_dir, simulator):
    """Every kind of question on the iPhone's sheet, with the Live Activity saying a question waits."""
    m = mock("--approval", APPROVAL_PLAN)
    records = run_ios(simulator, run_dir, "ios-approvals", m.url, APPROVAL_SCRIPT,
                      extra=["-LVTestLiveActivity", "YES"], wait_s=150)
    check_approvals(m, records)
    for label in APPROVAL_LABELS:
        assert (run_dir / f"ios-approvals-{label}.png").exists(), f"a screenshot of the {label} sheet"
    waiting = [r for r in events(records, "live-activity")
               if r.get("result") == "updated" and r.get("status") == "Waiting for your answer"]
    assert waiting and all(r["question"] for r in waiting), "the Live Activity said a question waits, and which"
