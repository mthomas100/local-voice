"""The turn-log reader, the idle gates and the outbox, with no model and no Pi."""
from __future__ import annotations

import http.server
import json
import threading
from datetime import datetime, timedelta

import pytest
import synth
from conftest import fake_gpu_clear, gpu_calls, set_gpu

from local_voice_brain import gates, outbox, turnlog

NOW = datetime.fromisoformat("2026-10-05T06:30:00-07:00")


def test_read_day_skips_bad_lines_ignores_other_types_and_dedupes(tmp_path):
    synth.write_day(tmp_path, None)
    turns, problems = turnlog.read_day(tmp_path, synth.DAY)
    assert len(turns) == 9                                   # 9 turns; the duplicate collapsed; session_start ignored
    assert len(problems) == 1 and "not JSON" in problems[0]
    assert [t.t_start for t in turns] == sorted(t.t_start for t in turns)
    b1 = next(t for t in turns if t.session.endswith("b2") and t.turn == 1)
    assert b1.interrupted and b1.spoken_text == "The VoiceChat 11B benchmark ran for sixty-four seconds"
    c2 = next(t for t in turns if t.session.endswith("c3") and t.turn == 2)
    assert c2.captured_text == synth.MUSING_2_SAVED and c2.user_text == synth.MUSING_2_SAID
    assert turnlog.days_available(tmp_path) == [synth.DAY]


@pytest.mark.parametrize("patch,why", [
    ({"t_start": "2026-10-04T09:00:00"}, "offset"), ({"v": 2}, "version"), ({"turn": "3"}, "integer"),
    ({"user_text": None}, "missing user_text"), ({"space": 5}, "space"),
])
def test_parse_record_refuses_what_it_cannot_trust(patch, why):
    rec = synth.turn("s", 1, "2026-10-04T09:00:00-07:00", None, "home", "u", "r")
    with pytest.raises(ValueError, match=why):
        turnlog.parse_record({**rec, **patch})


def test_last_activity_and_quiet_gate(tmp_path):
    synth.write_day(tmp_path, None)
    assert turnlog.last_activity(tmp_path).isoformat().startswith("2026-10-04T21:07:30")
    assert gates.quiet(tmp_path, NOW, 20).ok
    synth.write_next_day_turn(tmp_path, "06:20")
    g = gates.quiet(tmp_path, NOW, 20)
    assert not g.ok and "9 min ago" in g.detail      # it ended at 06:20:09
    assert gates.quiet(tmp_path / "empty", NOW, 20).ok


def test_gpu_clear_gate_runs_the_script_and_treats_missing_as_busy(tmp_path):
    assert not gates.gpu_clear(tmp_path / "nope.sh").ok
    s = fake_gpu_clear(tmp_path / "gpu.sh", ok=True)
    g = gates.gpu_clear(s)
    assert g.ok and g.detail.endswith("CLEAR")
    set_gpu(s, False)
    g = gates.gpu_clear(s)
    assert not g.ok and g.detail.endswith("BUSY") and gpu_calls(s) == 2


def test_wait_gpu_clear_polls_until_clear(tmp_path):
    s = fake_gpu_clear(tmp_path / "gpu.sh", ok=False)
    naps = []

    def sleep(sec):
        naps.append(sec)
        if len(naps) == 2:
            set_gpu(s, True)

    g = gates.wait_gpu_clear(s, wait_s=60, poll_s=5, sleep=sleep)
    assert g.ok and naps == [5, 5] and gpu_calls(s) == 3
    set_gpu(s, False)
    assert not gates.wait_gpu_clear(s, wait_s=0, poll_s=5, sleep=sleep).ok


def test_user_idle_reads_ioreg():
    sample = '    | | |   "HIDIdleTime" = 1200000000000\n'
    assert gates.hid_idle_seconds(sample) == 1200.0
    assert gates.user_idle(15, sample).ok and not gates.user_idle(30, sample).ok
    assert gates.user_idle(0).ok
    assert not gates.user_idle(15, "nothing").ok
    assert gates.hid_idle_seconds() is not None              # the real ioreg answers (read only)


@pytest.fixture
def status_server():
    state = {"body": b'{"v": 1, "state": "idle"}'}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1/status", state
    srv.shutdown()
    srv.server_close()


def test_orchestrator_gate(status_server):
    url, state = status_server
    assert gates.orchestrator(url, ["thinking", "speaking"]).ok
    state["body"] = json.dumps({"state": "speaking"}).encode()
    assert not gates.orchestrator(url, ["thinking", "speaking"]).ok
    state["body"] = b"<html>"
    assert not gates.orchestrator(url, ["thinking"]).ok
    assert gates.orchestrator("http://127.0.0.1:9/v1/status", ["thinking"]).ok      # not running is fine


def test_outbox_queue_pending_expiry_and_delivery(tmp_path):
    p = outbox.queue(tmp_path, day="2026-10-04", text="From Sunday: hello.", now=NOW, expiry_days=2,
                     cites=[{"session": "s", "turn": 1}], kb_digest="x.md")
    outbox.queue(tmp_path, day="2026-10-03", text="older", now=NOW - timedelta(days=1), expiry_days=2)
    (tmp_path / "outbox" / "junk.json").write_text("{")
    got = outbox.pending(tmp_path, NOW)
    assert [g.day for g in got] == ["2026-10-03", "2026-10-04"] and got[1].data["kb_digest"] == "x.md"
    assert [g.day for g in outbox.pending(tmp_path, NOW + timedelta(days=1, hours=1))] == ["2026-10-04"]
    dest = outbox.mark_delivered(p, NOW)
    assert not p.exists() and json.loads(dest.read_text())["delivered"].startswith("2026-10-05T06:30")
    assert [g.day for g in outbox.pending(tmp_path, NOW)] == ["2026-10-03"]
    with pytest.raises(ValueError):
        outbox.queue(tmp_path, day="d", text="t", now=datetime(2026, 1, 1), expiry_days=1)


def test_outbox_loads_by_path_without_this_package(tmp_path):
    """TURN_LOG.md promises the orchestrator can load outbox.py by path; do exactly that in a fresh interpreter."""
    import subprocess
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "local_voice_brain" / "outbox.py"
    code = (f"import importlib.util, datetime\n"
            f"spec = importlib.util.spec_from_file_location('anything', {str(src)!r})\n"
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"now = datetime.datetime.now().astimezone()\n"
            f"p = m.queue({str(tmp_path)!r}, day='2026-10-04', text='hi', now=now, expiry_days=1)\n"
            f"print([d.text for d in m.pending({str(tmp_path)!r}, now)])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "['hi']"
