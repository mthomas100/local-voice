"""The session recorder (local_voice/recorder.py), model-free: where each track puts what arrives, on one clock; that a
recording stops at max_minutes and that nothing in it can break a session; and the whole orchestrator recording a
conversation (tests/harness.py: the stub LLM, a fake recogniser, a noise voice, and the echo bench's client whose
microphone hears its own speaker 150 ms late at -20 dB), checked against what the client itself played and heard."""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from local_voice.recorder import MIC_RATE, PLAY_RATE, SessionRecorder, attach_recorder, unique_folder

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH / "tools"))


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"), w.getframerate()


def frame(value: int, n: int) -> bytes:
    return np.full(n, value, dtype="<i2").tobytes()


def events(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]


def recorder(tmp_path: Path, clock: Clock, **kw) -> SessionRecorder:
    return SessionRecorder(tmp_path / "rec" / "s-test", session_id="s-test", meta={"client": "test"},
                           max_s=kw.pop("max_s", 600.0), clock=clock, wall=lambda: 1_800_000_000.0, **kw)


def test_both_tracks_sit_on_the_recordings_clock(tmp_path):
    """Microphone frames end where they arrived and go back to back; frames more than 0.25 s late mean frames were
    missing, and silence fills to them; a burst after a stall is appended. Playback chunks sit where they were sent, or
    right after the chunk before (Pipecat's WebSocket output sends one chunk ahead), with silence between replies and
    one "sent" event per stretch. Events carry the track clock, time.monotonic() and the wall clock."""
    clk = Clock()
    rec = recorder(tmp_path, clk).start()
    t0 = rec.t0
    for i in range(5):                          # 0.00-0.10 s: five 20 ms frames, each arriving as it ends
        clk.t = t0 + 0.02 * (i + 1)
        rec.mic(frame(100 + i, 320), at=clk.t)
    rec.mic(frame(200, 320), at=t0 + 1.0)      # nothing for 0.88 s: silence, then 0.98-1.00 s
    for v in (300, 301, 302):                   # three frames arriving together at 1.5 s
        rec.mic(frame(v, 320), at=t0 + 1.5)
    for k, at in enumerate((0.5, 0.5, 0.54, 0.58)):   # a reply: the first two chunks leave together
        rec.playback(frame(1000 + k, 960), 24000, at=t0 + at)
    rec.playback(frame(2000, 960), 24000, at=t0 + 2.0)  # the next reply
    rec.event("bot", at=t0 + 0.5, on=True)
    clk.t = t0 + 2.5
    rec.close()
    rec.join()
    mic, mr = read_wav(rec.folder / "mic.wav")
    play, pr = read_wav(rec.folder / "playback.wav")
    assert (mr, pr) == (MIC_RATE, PLAY_RATE)
    at = lambda s: int(round(s * MIC_RATE))                                            # noqa: E731
    assert [int(mic[at(0.02 * i)]) for i in range(5)] == [100, 101, 102, 103, 104]
    assert not mic[at(0.10):at(0.98)].any() and mic[at(0.98)] == 200 and mic[at(0.999)] == 200
    assert [int(mic[at(x)]) for x in (1.48, 1.50, 1.52)] == [300, 301, 302] and len(mic) == at(1.54)
    pat = lambda s: int(round(s * PLAY_RATE))                                         # noqa: E731
    assert [int(play[pat(x)]) for x in (0.5, 0.54, 0.58, 0.62)] == [1000, 1001, 1002, 1003]
    assert not play[pat(0.66):pat(2.0)].any() and play[pat(2.0)] == 2000
    ev = events(rec.folder)
    bot = next(e for e in ev if e["ev"] == "bot")
    assert bot["t"] == 0.5 and bot["mono"] == round(t0 + 0.5, 4) and bot["wall"] == 1_800_000_000.5
    sent = [(e["from_s"], e["to_s"]) for e in ev if e["ev"] == "sent"]
    assert sent == [(0.5, 0.66), (2.0, 2.04)]
    meta = json.loads((rec.folder / "meta.json").read_text())
    st = meta["stats"]
    assert (st["mic_frames"], st["mic_gaps"], st["play_chunks"], st["dropped_items"], st["overlap_samples"]) == \
        (9, 2, 5, 0, 0)
    assert st["mic_ahead_max_s"] == pytest.approx(0.04) and meta["stop_reason"] == "the session ended"
    assert meta["client"] == "test" and meta["t0_mono"] == t0 and meta["versions"]["pipecat-ai"] == "1.12.0"


def test_it_stops_at_max_minutes_and_never_raises_into_the_session(tmp_path, caplog):
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        clk = Clock()
        rec = recorder(tmp_path, clk, max_s=1.0).start()
        rec.mic(frame(5, 320), at=rec.t0 + 0.5)
        rec.mic(frame(6, 320), at=rec.t0 + 1.2)          # past max_minutes: stops, says so once
        rec.mic(frame(7, 320), at=rec.t0 + 1.3)
        rec.event("bot", at=rec.t0 + 1.4, on=True)
        rec.join()
        assert not rec.active and rec.stop_reason.startswith("max_minutes")
        assert sum("record.max_minutes" in m for m in lines) == 1
        mic = read_wav(rec.folder / "mic.wav")[0]
        assert len(mic) == 8000 and (mic[7680:] == 5).all() and not mic[:7680].any() and not events(rec.folder)
        # a hook given garbage turns the recording off and returns; it never raises
        bad = SessionRecorder(tmp_path / "rec" / "s-bad", session_id="s-bad", meta={}, max_s=60.0).start()
        bad.mic(object())                                   # not bytes
        assert not bad.active and bad.failed and "mic" in bad.failed
        bad.playback(frame(1, 960))                         # a no-op now
        bad.join()
        assert any("recording s-bad: stopped" in m for m in lines)
        # a folder that cannot be made: no recording, a log line, and the session is not told
        (tmp_path / "file").write_text("x")
        session = SimpleNamespace(id="s-x", client="test", device="d", mic="vad", protocol_v1=True, extra={})
        cfg = SimpleNamespace(record={"enabled": True, "dir": tmp_path / "file", "max_minutes": 1.0}, raw={}, path="")
        assert attach_recorder(session, transport=SimpleNamespace(), cfg=cfg) is None
        assert any("recording s-x: not recording" in m for m in lines) and "recorder" not in session.extra
        cfg.record["enabled"] = False
        assert attach_recorder(session, transport=None, cfg=cfg) is None
    finally:
        logger.remove(sink)


def test_a_full_queue_drops_items_instead_of_waiting_and_keeps_the_clock(tmp_path):
    clk = Clock()
    rec = recorder(tmp_path, clk, queue_items=3)           # the writer not started yet: nothing drains the queue
    for i in range(6):
        rec.mic(frame(10 + i, 320), at=rec.t0 + 0.02 * (i + 1))
    assert rec.stats["dropped_items"] == 3 and rec.stats["mic_frames"] == 6
    rec.start()
    for _ in range(200):                                    # the writer takes the three queued
        if rec._q.empty():
            break
        time.sleep(0.01)
    rec.mic(frame(99, 320), at=rec.t0 + 0.14)               # still where the clock says, after the hole
    rec.close()
    rec.join()
    mic, _ = read_wav(rec.folder / "mic.wav")
    assert [int(mic[320 * i]) for i in range(7)] == [10, 11, 12, 0, 0, 0, 99]


def test_a_resumed_session_records_into_a_new_folder(tmp_path):
    (tmp_path / "s-1").mkdir()
    assert unique_folder(tmp_path, "s-1") == tmp_path / "s-1-2"
    assert unique_folder(tmp_path, "s-2") == tmp_path / "s-2"


def test_short_holes_in_a_stream_are_made_up_within_20_ms(tmp_path):
    """A client that pauses 60 ms between two sends (the test client between speak() and silence()), then streams on:
    within a second every frame is that late, and the track moves up by it; jitter alone moves nothing."""
    clk = Clock()
    rec = recorder(tmp_path, clk).start()
    t = rec.t0 + 0.02
    jitter = [0.0, 0.004, 0.001, 0.007, 0.002]
    for i in range(100):                                     # 2 s steady, jittered arrivals
        rec.mic(frame(1, 320), at=t + 0.02 * i + jitter[i % 5])
    t += 2.0 + 0.06                                          # a 60 ms pause, then 2 s more
    for i in range(100):
        rec.mic(frame(2, 320), at=t + 0.02 * i + jitter[i % 5])
    rec.close()
    rec.join()
    mic, _ = read_wav(rec.folder / "mic.wav")
    second = int(np.flatnonzero(mic == 2)[0]) / MIC_RATE
    ends = np.flatnonzero(mic == 2)[-1] / MIC_RATE
    assert rec.stats["mic_resyncs"] == 1 and rec.stats["mic_resync_s"] == pytest.approx(0.06, abs=0.001)
    assert second == pytest.approx(2.0, abs=0.001)           # placed back to back, then moved up within a second ...
    assert ends == pytest.approx(4.06 - 1 / MIC_RATE, abs=0.001)      # ... so its last frame ends where it arrived


# ---------------------------------------------------------------------------------------------- end to end

STORY = ["The lighthouse stood at the end of a long stone pier.", "His cat followed him all the way up the stairs.",
         "Ships far out at sea saw the light and kept away."]


@pytest.mark.needs_pi
async def test_the_orchestrator_records_aligned_tracks_that_the_report_reads(tmp_path):
    """A conversation recorded through the whole orchestrator (protocol v1, the guard on and the hold `all`, so every
    hook fires). The client is the echo bench's EchoClient: its microphone streams without a break and hears its own
    simulated speaker 150 ms late at -20 dB (below the energy VAD, so the reply plays through), plus a planted
    request. The voice is tests/recording_fakes.NoiseSynthesizer (a different sound per sentence). Checked against
    what the client itself sent and played: the request where the client planted it in mic.wav, the reply where the
    client began playing it in playback.wav, the echo 150 ms after the reply (session_report's GCC-PHAT), the clips
    exactly each sentence's audio, and the events of every kind."""
    import asyncio

    import echo_bench as bench
    import session_report as sr
    from echo_mixer import EchoSettings
    from harness import rig
    from local_voice.engines.base import float_to_pcm16
    from recording_fakes import NoiseSynthesizer

    voice = {"impl": "recording_fakes:NoiseSynthesizer", "seconds_per_char": 0.03, "lead_s": 0.08, "level": 0.1}
    ov = {"record": {"enabled": True, "dir": str(tmp_path / "recordings"), "max_minutes": 5},
          "echo": {"guard": True, "hold_for_words": "all"},
          "tts": {"adapter": "noise", "adapters": {"noise": voice}}}
    t = np.arange(int(0.8 * MIC_RATE)) / MIC_RATE
    request = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    async with rig(tmp_path, stt_text="tell me a story", overrides=ov) as r:
        r.stub.script([{"text": " ".join(STORY), "delay_ms": 5}])
        c = bench.EchoClient(r.url, device="rec-test", echo=EchoSettings(level_db=-20.0, delay_ms=150.0,
                                                                         noise_dbfs=-60.0))
        await c.connect()
        await c.start_mic()
        await asyncio.sleep(0.5)
        planted = c.plant(request, label="request", kind="speech", text="tell me a story")
        reply = await c.wait_reply(since=0.0, deadline=c.now() + 20)
        assert reply is not None and await c.wait_until(lambda: c.over(reply), c.now() + 30)
        await asyncio.sleep(1.0)                           # the echo runs 150 ms behind the speaker
        rec = r.orch.clients["rec-test"].session.extra["recorder"]
        await c.stop_mic()
        await c.close()
        for _ in range(100):
            if not rec.active:
                break
            await asyncio.sleep(0.05)
        rec.join(5.0)
    folder = rec.folder
    assert folder.parent == tmp_path / "recordings" and folder.name.startswith("s-")
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["stop_reason"] == "the session ended" and meta["client"] == "test" and meta["protocol_v1"]
    assert meta["playback_tap"] == "on_audio_sent" and meta["config"]["record"]["enabled"]
    mic, _ = read_wav(folder / "mic.wav")
    play, _ = read_wav(folder / "playback.wav")
    # the request where the client put it: client mic sample k is at c.t0 + c.mic_t0 + k / 16000
    want = planted.start + round((c.t0 + c.mic_t0 - rec.t0) * MIC_RATE)
    x = mic.astype(np.float32) / 32768
    got = want - 800 + int(np.argmax(np.correlate(x[want - 800:want + 800 + len(request)], request, mode="valid")))
    assert abs(got - want) <= 0.01 * MIC_RATE, (got, want)
    # the reply where the client began to play it, and its echo 150 ms after it
    a = sr.analyse(folder)
    first = min(cl["t_play_s"] for cl in a["clips"])
    client_first = c.t0 + reply.segments[0][0] - rec.t0
    assert abs(first - client_first) <= 0.01, (first, client_first)
    (echo,) = [e for e in a["live_echo"] if e.get("found")]
    print(f"\nrecorded: request {1000 * (got - want) / MIC_RATE:+.1f} ms from where the client put it, reply "
          f"{1000 * (first - client_first):+.1f} ms from where it began to play, echo {echo['delay_ms']} ms at "
          f"{echo['level_db']} dB (z {echo['z']})")
    assert abs(echo["delay_ms"] - 150) <= 10 and abs(echo["level_db"] + 20) <= 2, a["live_echo"]
    assert not a["replays"]
    # one clip per sentence, each exactly the audio the voice made for it (the first trimmed of its lead silence)
    synth = NoiseSynthesizer(voice)
    assert [cl["text"] for cl in a["clips"]] == STORY and all(cl["complete"] for cl in a["clips"])
    for i, cl in enumerate(a["clips"]):
        clip, _ = sr.read_wav(folder / "clips" / cl["clip"])
        made = np.frombuffer(float_to_pcm16(synth.audio(STORY[i])), dtype="<i2")
        assert np.array_equal(clip, made[len(made) - len(clip):] if i == 0 else made), i
        assert i > 0 or 0 < len(made) - len(clip) <= int(0.08 * 24000)
    # every kind of event, on the same clock
    kinds = {e["ev"] for e in (json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines())}
    assert {"vad", "turn", "bot", "stt", "guard", "decision", "tts", "sent", "v1", "played_ms"} <= kinds, kinds
    assert a["counts"]["tts"] == 3 and (folder / "report.md").exists()
    assert len(play) / 24000 >= first + sum(len(synth.audio(s)) for s in STORY) / 24000 - 0.1


async def test_the_browser_page_is_recorded_too(tmp_path):
    """The browser entry (SmallWebRTC, static/index.html in a separate headless Chromium, tests/test_browser_page.py's
    set-up): its output transport is Pipecat's own, tapped at write_audio_frame, so the reply it sent is in
    playback.wav; the fake microphone's tone (the WAV Chromium plays as its microphone, 0.5-1.7 s) is in mic.wav, as
    the page delivered it: after Chrome's echo canceller, noise suppression and gain control (index.html asks for all
    three), which took the tone from -3 to -21 dBFS within its 1.2 s (2026-10-05)."""
    import asyncio
    import shutil

    from harness import free_port, rig
    from test_browser_page import CHROMIUM, one_utterance_wav, open_page

    if not CHROMIUM.exists() or not shutil.which("pi"):
        pytest.skip("needs the cached Playwright Chromium 1243 and pi")
    from playwright.async_api import async_playwright

    bport = free_port()
    ov = {"server": {"browser": {"enabled": True, "port": bport}},
          "record": {"enabled": True, "dir": str(tmp_path / "recordings"), "max_minutes": 5}}
    async with rig(tmp_path, stt_script=["what is in the folder"], stt_text="", overrides=ov) as r:
        r.stub.script([{"text": "There are two notes in it."}])
        async with async_playwright() as p:
            browser, page = await open_page(p, bport, one_utterance_wav(tmp_path / "mic.wav"))
            await page.wait_for_function("document.body.innerText.includes('There are two notes in it.')",
                                         timeout=20000)
            await asyncio.sleep(1.5)
            rec = next(iter(r.orch.browser_sessions.values())).extra["recorder"]
            await browser.close()
    # aiortc may take a while to notice a closed headless browser; the server's shutdown closes the connection, the
    # session's worker cleans up its observers, and that ends the recording
    for _ in range(200):
        if not rec.active:
            break
        await asyncio.sleep(0.05)
    rec.join(5.0)
    meta = json.loads((rec.folder / "meta.json").read_text())
    assert meta["stop_reason"] == "the session ended", meta.get("stop_reason")
    assert meta["client"] == "browser" and meta["playback_tap"] == "write_audio_frame" and not meta["protocol_v1"]
    mic, _ = read_wav(rec.folder / "mic.wav")
    play, _ = read_wav(rec.folder / "playback.wav")
    loud = lambda a, rate: np.flatnonzero(np.abs(a.astype(np.int32)) > 330) / rate    # noqa: E731  (-40 dBFS)
    tone, said = loud(mic, MIC_RATE), loud(play, PLAY_RATE)
    assert tone.size and 1.1 <= tone[-1] - tone[0] <= 1.3, (tone[0], tone[-1])   # the 1.2 s tone, in one piece
    assert meta["stats"]["mic_gaps"] == 0 and meta["stats"]["mic_resyncs"] == 0  # WebRTC streams without a break
    assert said.size and said[0] > tone[-1]                                # the reply, after the question
    ev = [json.loads(line) for line in (rec.folder / "events.jsonl").read_text().splitlines()]
    assert {"stt", "bot", "tts", "sent"} <= {e["ev"] for e in ev}
    stt = next(e for e in ev if e["ev"] == "stt" and e["final"])
    assert stt["text"] == "what is in the folder" and tone[-1] < stt["t"] < said[0]
