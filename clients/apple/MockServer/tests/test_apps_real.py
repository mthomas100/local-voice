"""The two apps against the real orchestrator (M2.2, its second half): the Mac menu-bar app as a real process and the
iPhone app in the iOS simulator, each completing a push-to-talk turn with a spoken question through its synthetic
microphone, and each showing the server's answer to a switch on its dashboard.

Markers `apps` and `real`; run with `../build.sh apps-real` (LV_REAL_URL set). A real turn runs models
on the GPU, so the rules of test_real_server.py apply: nothing runs unless measure/bench/gpu_clear.sh passes when the
module starts, and each test stops if a GPU job or a hold appears. `LV_REAL_URL=mock` dry-runs both against a scripted
mock that ignores switches, as the M1 orchestrator does.

The switches ask for a space and a mode that cannot exist. A real name could move the server for later turns: an
`atlas` space in M3 may be rooted in a real journal, and `act` mode gives the agent write tools. The M1 server
ignores both messages (the apps show "no answer" after 5 s); M3 answers with an `error` (the apps show "refused").

Machine ground truth only: the question is a `say -v Samantha` recording, its transcript is scored against the known
words and the reply must contain the one-word answer. Evidence: MockServer/runs/<stamp>/<test>/ (the app's JSONL log,
the Mac panel and dashboard snapshots, the iPhone screenshot) and MockServer/runs/<stamp>/apps-real-results.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

import pytest

from conftest import RUNS, read_jsonl
from test_apps import BUNDLE, IOS_APP, MAC_APP, events, run_ios, simctl, simulator  # noqa: F401 (fixture)
from test_real_server import (DRY, FRANCE, REAL, check_answer, check_played_to_the_end, gpu_line, gpu_log,
                              gpu_still_ours, interrupts_outside_replies, measure, releases, render)

pytestmark = [pytest.mark.apps, pytest.mark.real,
              pytest.mark.skipif(not REAL, reason="set LV_REAL_URL to the server's ws:// URL, or to mock for a dry run")]

NOWHERE_SPACE = "apple-e2e-nowhere"
NOWHERE_MODE = "apple-e2e-sideways"
RESULTS: list[dict] = []


def record(test: str, **data):
    RESULTS.append({"test": test, "target": "mock (dry run)" if DRY else REAL, "at": time.strftime("%H:%M:%S"), **data})
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "apps-real-results.json").write_text(json.dumps(RESULTS, indent=1))


def script(switch: str, clip: str) -> str:
    """Connect, read the status, ask for a switch that cannot happen and wait out its 5 s, then one push-to-talk turn
    and the status the app fetches after it."""
    return (f"connect; wait:ready; wait:status:5000; {switch}; sleep:6000; sleep:500; press; say-wait:file:{clip}; "
            "release; wait:end_of_turn:90000; wait:playback-finished:30000; wait:status:turn-ended:5000")


@pytest.fixture(scope="module")
def question():
    folder = RUNS / "real-speech"
    folder.mkdir(parents=True, exist_ok=True)
    return render(FRANCE, folder)


@pytest.fixture(scope="module")
def gpu_go() -> str:
    if DRY:
        return "dry run: the mock runs no models"
    code, line = gpu_line()
    gpu_log(f"start (apps): {line}")
    if code != 0:
        pytest.skip(f"GPU not clear, nothing run: {line}")
    return line


@pytest.fixture
def url(gpu_go, mock, run_dir) -> str:
    if DRY:
        turns = run_dir / "turns.json"
        turns.write_text(json.dumps([{"transcript": FRANCE, "reply": "Paris is the capital of France."}]))
        return mock("--reply-voice", "say", "--turns", str(turns), "--think-ms", "300", "--no-switch").url
    why = gpu_still_ours()
    if why:
        pytest.skip(f"stopping: {why}")
    return REAL


def check_app_turn(records, name: str, said: str, switch_kind: str) -> dict:
    """The turn as the app saw it, its dashboard before and after, and the switch's outcome."""
    done = events(records, "script-done")
    assert done and done[-1]["ok"] is True, f"the in-app script finished: {events(records, 'script-failed')}"
    welcome = next(r["msg"] for r in records if r["event"] == "recv" and r["msg"]["t"] == "welcome")
    release = releases(records)[0]
    row = measure(records, said=said, since=release)
    # The app writes its log on the main actor as it consumes events, up to about 110 ms after they happen (the latency
    # event and `playback started`, emitted together, were logged 49 ms apart in a dry run on 2026-10-05), so the
    # times between its log lines are not latencies; the core's own `latency` event, on the core's clock, is.
    row.pop("audio_start_t"), row.pop("playback_t")
    statuses = events(records, "status")
    first, after = statuses[0]["status"], [r for r in statuses if r["reason"] == "turn-ended"][-1]["status"]
    outcomes = [r["phase"] for r in events(records, "switch") if r["phase"] != "requested" and r["kind"] == switch_kind]
    server = after.get("last_turn") or {}
    shown = next((r for r in records if r["event"] == "latency" and r.get("reply") == row["reply_id"]), {})
    result = {**row, "welcome": welcome, "switch": {"kind": switch_kind, "outcomes": outcomes},
              "client_release_to_first_sound_ms": shown.get("release_to_playback_ms"),
              "status_on_connect": {k: first.get(k) for k in ("space", "mode", "tier", "model", "hold")},
              "status_after_turn": {k: after.get(k) for k in ("space", "mode", "tier", "model", "hold", "last_turn")},
              "client_stop_to_first_sound_ms": row["stop_to_playback_ms"],
              "server_eos_to_first_audio_ms": server.get("eos_to_first_audio_ms")}
    record(name, **result)
    check_answer(row, "paris")
    check_played_to_the_end(row)
    assert not interrupts_outside_replies(records)
    assert first.get("space") == welcome["space"], "the dashboard's first status names the session's space"
    assert outcomes and outcomes[0] in ("no-answer", "refused"), outcomes
    assert after.get("space") == welcome["space"], "the impossible switch moved nothing"
    assert server.get("eos_to_first_audio_ms", 0) > 0, "the turn's latency reached the dashboard"
    # The dashboard's "from release to sound here" is the wait the person has: from the release, which comes one wait
    # for the last capture chunk (30-150 ms) before `stop`, the moment the server's own number starts from.
    rel, stop = shown.get("release_to_playback_ms"), shown.get("stop_to_playback_ms")
    assert rel is not None and stop is not None and 30 <= rel - stop <= 160, shown
    return result


@pytest.mark.skipif(not MAC_APP.exists(), reason="build the Mac app first (./build.sh mac)")
def test_mac_app_turn_against_the_real_server(url, run_dir, question):
    log, snapshot = run_dir / "mac-real.jsonl", run_dir / "mac-real.png"
    argv = [str(MAC_APP / "Contents" / "MacOS" / "LocalVoice"),
            "-LVServerURL", url, "-LVDevice", "mac-apps-real", "-LVAudio", "synthetic", "-LVOutputVolume", "0",
            "-LVScript", script(f"space:{NOWHERE_SPACE}", str(question.path)), "-LVEventLog", str(log),
            "-LVQuitAfterScript", "YES", "-LVShowPanel", "YES", "-LVRegisterHotKey", "NO", "-LVSnapshot", str(snapshot)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    (run_dir / "mac-real.out").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    check_app_turn(read_jsonl(log), "Mac app, push-to-talk", FRANCE, "space")
    assert snapshot.exists() and (run_dir / "mac-real.dashboard.png").exists(), "the panel and the card rendered"


@pytest.mark.skipif(not IOS_APP.exists(), reason="build the iPhone app first (./build.sh ios)")
def test_iphone_app_turn_against_the_real_server(url, run_dir, question, simulator):  # noqa: F811
    # The app reads the clip from its own container: the simulator's sandbox is not the place to test host paths.
    container = simctl("get_app_container", simulator, BUNDLE, "data").strip()
    clip = f"{container}/Documents/e2e/{question.path.name}"
    subprocess.run(["mkdir", "-p", f"{container}/Documents/e2e"], check=True)
    shutil.copy(question.path, clip)
    records = run_ios(simulator, run_dir, "ios-real", url, script(f"mode:{NOWHERE_MODE}", clip),
                      extra=["-LVShowStatus", "YES", "-LVTestLiveActivity", "YES"], wait_s=180)
    check_app_turn(records, "iPhone app (simulator), push-to-talk", FRANCE, "mode")
    live = events(records, "live-activity")
    assert live and live[0]["result"] == "started" and any(r["result"] == "updated" for r in live), live
