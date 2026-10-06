"""Protocol v1 end to end: the Swift client core (lvclient) against the mock server, over a real WebSocket.

Ground truth is planted: the synthetic microphone speaks tones of known pitch and length, the mock answers with a
tone of known pitch and length, and both sides log what they saw. Nothing here needs a microphone, a speaker, a
model or a permission prompt.
"""

from __future__ import annotations

import array
import math
import subprocess
import time
import wave

import pytest

PTT_TURN = "press; say-wait:tone:440:1.0; release; wait:end_of_turn; wait:sent:played_ms"


def wav_hz(path) -> tuple[float, float]:
    """Pitch and audible seconds of a recorded mono 16-bit WAV."""
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
    loud = [i for i, x in enumerate(a) if abs(x) > 2000]
    if not loud:
        return 0.0, 0.0
    s = a[loud[0] : loud[-1]]
    crossings = [i for i in range(1, len(s)) if s[i - 1] < 0 <= s[i]]
    hz = (len(crossings) - 1) / ((crossings[-1] - crossings[0]) / rate) if len(crossings) > 2 else 0.0
    return hz, (loud[-1] - loud[0]) / rate


def turns(mock) -> list[dict]:
    return mock.events("turn")


def played(mock) -> dict[str, dict]:
    return {r["reply_id"]: r for r in mock.events("played")}


def test_push_to_talk_turn(mock, client):
    m = mock()
    c = client(m.url, PTT_TURN)
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr

    hello = m.received("hello")[0]
    assert hello == {"t": "hello", "v": 1, "client": "test", "device": "e2e", "mic": "ptt"}
    order = [msg["t"] for msg in m.received()]
    assert order[:2] == ["hello", "start"], "start is the first thing after hello"
    assert "stop" in order

    (turn,) = turns(m)
    assert turn["bad_sizes"] == [], "every binary frame is 640-1280 bytes"
    assert abs(turn["speech_hz"] - 440) < 5, "the 16 kHz capture conversion keeps pitch"
    assert abs(turn["speech_seconds"] - 1.0) <= 0.06, "and length"
    assert turn["first_speech_frame"] <= 1, "pre-connect buffering lost nothing at the start"
    assert turn["zero_tail_ms"] >= 100, "about 100 ms of silence before stop"

    p = played(m)["r1"]
    assert abs(p["ms"] - 1200) <= 40 and p["sent_ms"] == 1200

    s = c.summary
    assert s["ok"] and abs(s["played_ms"]["r1"] - 1200) <= 40
    assert abs(s["audible_s"] - 1.2) <= 0.06
    lat = s["latency"][0]
    assert lat["stop_to_audio_start_ms"] is not None and lat["stop_to_playback_ms"] >= lat["stop_to_audio_start_ms"]
    hz, seconds = wav_hz(c.wav)
    assert abs(hz - 523.25) < 4, "the reply tone comes out at its pitch (24 kHz in, 48 kHz out)"
    assert abs(seconds - 1.2) <= 0.06


def test_device_engine_muted(mock, client):
    """The same turn on LiveAudioIO: the real output device with the volume at zero, .dataPlayedBack completions."""
    m = mock()
    c = client(m.url, PTT_TURN, audio="device-muted")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    engine = [r for r in c.events("engine") if r["status"] == "running"]
    assert engine and "(muted)" in engine[0]["detail"]
    assert abs(played(m)["r1"]["ms"] - 1200) <= 60


def test_open_mic_turn_and_mic_gate(mock, client):
    """The gate silences the microphone for 600 ms after playback drains; a control run without it hears it all."""
    script = ("handsfree-on; wait:ready; sleep:300; say-wait:tone:440:1.0; wait:end_of_turn; "
              "wait:playback-finished; say-wait:tone:880:1.5; sleep:1500")
    gated = mock()
    c = client(gated.url, script, mic="vad", name="gated")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    control = mock()
    c2 = client(control.url, script, mic="vad", name="control", extra=["--gate-ms", "0"])
    assert c2.proc.returncode == 0, c2.proc.stdout + c2.proc.stderr

    t_gated, t_control = turns(gated), turns(control)
    assert len(t_gated) >= 2 and len(t_control) >= 2
    assert abs(t_gated[0]["speech_hz"] - 440) < 5 and abs(t_gated[0]["speech_seconds"] - 1.0) <= 0.1
    assert abs(t_control[1]["speech_seconds"] - 1.5) <= 0.1, "control: the whole second tone arrives"
    lost = t_control[1]["speech_seconds"] - t_gated[1]["speech_seconds"]
    assert 0.5 <= lost <= 0.7, f"the gate silenced {lost:.2f} s of the second tone (600 ms configured)"
    assert any(r["armed"] for r in c.events("gate")) and any(not r["armed"] for r in c.events("gate"))
    assert played(gated)["r1"]["ms"] >= 1150, "played_ms at the end of an open-mic reply too"


def test_server_barge_in(mock, client):
    """Speech during a reply: the server interrupts, the client flushes at once and reports what was heard."""
    m = mock("--reply-seconds", "5")
    script = ("handsfree-on; wait:ready; sleep:300; say-wait:tone:440:1.0; wait:playback-started; sleep:1000; "
              "say:tone:660:1.0; wait:interrupt; wait:sent:played_ms; sleep:300")
    c = client(m.url, script, mic="vad")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr

    (barge,) = m.events("barge-in")
    p = played(m)["r1"]
    assert p["ms"] < 5000
    assert 900 <= p["ms"] <= 1500, f"heard {p['ms']} ms: about the 1 s before the user spoke"
    assert p["ms"] <= barge["sent_ms"] + 50, "never more than was sent"
    assert not m.received("interrupt"), "the client does not echo a server interrupt"
    tails = [t for t in c.summary["flush_tails"] if t["why"] == "server interrupt"]
    assert tails and tails[0]["tail_ms"] < 30, f"output silent {tails} ms after the interrupt arrived"


def test_push_to_talk_barge_in(mock, client):
    """Pressing talk over a reply: flush, interrupt, played_ms, start; then a normal second turn."""
    m = mock("--reply-seconds", "4")
    script = ("press; say-wait:tone:440:0.8; release; wait:playback-started; sleep:1000; "
              "press; say-wait:tone:660:0.6; release; wait:end_of_turn; wait:sent:played_ms; wait:sent:played_ms")
    c = client(m.url, script)
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr

    order = [msg["t"] for msg in m.received() if msg["t"] in ("start", "stop", "interrupt", "played_ms")]
    assert order == ["start", "stop", "interrupt", "played_ms", "start", "stop", "played_ms"]
    assert m.received("interrupt")[0]["reply_id"] == "r1"
    assert not m.sent("interrupt"), "the server does not interrupt a reply the client already stopped"
    p = played(m)
    assert 900 <= p["r1"]["ms"] <= 1250, p["r1"]
    assert abs(p["r2"]["ms"] - 4000) <= 60
    assert abs(turns(m)[1]["speech_hz"] - 660) < 5
    tails = [t for t in c.summary["flush_tails"] if t["why"] == "client interrupt"]
    assert tails and tails[0]["tail_ms"] < 30


def test_pre_connect_buffering(mock, client):
    """Talking before the server is ready: start and the first words wait in order, nothing is lost."""
    m = mock("--slow-welcome-ms", "700")
    c = client(m.url, PTT_TURN)
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (turn,) = turns(m)
    assert turn["first_speech_frame"] <= 1
    assert abs(turn["speech_seconds"] - 1.0) <= 0.06
    assert [msg["t"] for msg in m.received()][:2] == ["hello", "start"]


@pytest.mark.parametrize("flag,code", [("--reject", 4403), ("--protocol-error-after-hello", 1002)])
def test_permanent_close_codes_stop_retrying(mock, client, flag, code):
    m = mock(flag)
    c = client(m.url, f"connect; wait:stopped:{code}:5000; sleep:2000")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    assert len(m.events("connect")) == 1, "no reconnect"
    stopped = [r for r in c.events("connection") if r["state"] == "stopped"]
    assert stopped and stopped[0]["code"] == code


def test_takeover_4409(mock, client):
    m = mock()
    first, first_run = client(m.url, "connect; wait:ready; wait:stopped:4409:10000; sleep:1500",
                              device="same", name="first", wait=False)
    time.sleep(1.5)
    second = client(m.url, "connect; wait:ready; sleep:2000", device="same", name="second")
    first.wait(30)
    assert first.returncode == 0, first.stdout.read() + first.stderr.read()
    assert second.proc.returncode == 0
    closes = m.events("close")
    assert [(r["conn"], r["code"]) for r in closes] == [(1, 4409)]
    assert len(m.events("connect")) == 2, "the replaced client does not fight back"


def test_reconnect_after_a_dropped_connection(mock, client):
    m = mock("--drop-after-welcome-s", "0.8")
    c = client(m.url, "connect; wait:ready; wait:closed:1006:5000; wait:ready:10000; " + PTT_TURN)
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    hellos = m.received("hello")
    assert len(hellos) == 2 and hellos[0] == hellos[1], "same device, same hello"
    assert turns(m)[0]["conn"] == 2
    waiting = [r for r in c.events("connection") if r["state"] == "waiting"]
    assert waiting and waiting[0]["code"] == 1006


def test_acknowledgement_and_answer_back_to_back(mock, client):
    m = mock("--ack", "--tool")
    c = client(m.url, PTT_TURN + "; wait:sent:played_ms")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    p = played(m)
    assert abs(p["a1"]["ms"] - 300) <= 40 and abs(p["r2"]["ms"] - 1200) <= 40
    tools = [msg["phase"] for msg in c.recv("tool")]
    assert tools == ["start", "end"]


def test_held_gpu(mock, client):
    m = mock("--hold", "--hold-object")
    c = client(m.url, PTT_TURN + "; wait:sent:played_ms")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    welcome = c.recv("welcome")[0]
    assert welcome["hold"] == {"phase": "held", "why": "mock render"}
    phases = [msg["phase"] for msg in c.recv("hold")]
    assert phases[:2] == ["held", "open"]
    assert "held" in [msg["v"] for msg in c.recv("state")]
    assert set(played(m)) == {"n1", "r2"}, "the busy notice, then the answer once the hold ended"


def test_typed_turn(mock, client):
    """Typed words are the user's turn. Like the orchestrator (a user message, no TranscriptionFrame), the mock sends
    no `transcript` back for them: the apps show typed words themselves."""
    m = mock()
    c = client(m.url, "connect; wait:ready; text:what is on my calendar; wait:end_of_turn")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    assert m.received("text")[0]["text"] == "what is on my calendar"
    assert not c.recv("transcript")
    assert c.recv("reply_text")[0]["delta"] == "you typed: what is on my calendar"


def test_speech_fixture_from_say(mock, client, run_dir):
    """Real speech in (a `say` recording, 22.05 kHz AIFF, loaded and resampled twice), a `say` voice out."""
    aiff = run_dir / "question.aiff"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), "what did I write in my journal yesterday"], check=True)
    m = mock("--reply-voice", "say")
    c = client(m.url, f"press; say-wait:file:{aiff}; release; wait:end_of_turn; wait:sent:played_ms")
    assert c.proc.returncode == 0, c.proc.stdout + c.proc.stderr
    (turn,) = turns(m)
    assert turn["bad_sizes"] == []
    assert 1.0 <= turn["speech_seconds"] <= 3.5
    assert c.summary["audible_s"] > 1.0


@pytest.mark.slow
def test_keepalive(mock, client):
    m = mock()
    c = client(m.url, "connect; wait:ready; sleep:31500", timeout=60)
    assert c.proc.returncode == 0
    pings = m.received("ping")
    assert len(pings) >= 2 and [p["n"] for p in pings][:2] == [1, 2]
    assert len(c.recv("pong")) >= 2
    assert not m.events("close"), "a quiet session stays open"
