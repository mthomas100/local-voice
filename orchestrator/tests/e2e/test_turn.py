"""M1 definitions of done 2-4 with real models:

2. a protocol v1 client streams `say` speech at real-time pace and gets the right final transcript and a spoken reply;
   end of speech to first audio is logged per turn (target <= 1.5 s warm, goal <= 0.9 s);
3. a tool turn ("what does my knowledge base say about the hold gate?") makes Pi call a tool, an acknowledgement is
   spoken while it runs, and the result is spoken;
4. barge-in: speech during a reply produces `interrupt` within 300 ms of speech start, `played_ms` truncates the
   context, and the next turn works.

Machine ground truth only (no human listening): the text `say` spoke is known, so the transcript is scored by word
error rate; the reply audio is transcribed back by the same STT adapter and compared with the reply text.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pytest

from e2e_fixtures import RUN_DIR, gpu_still_ours, kb_clone, orch, run_dir  # noqa: F401 - fixtures
from local_voice.client import V1Client, say_pcm, speech_end_s


def first_audible_s(pcm: bytes, threshold: float = 0.01) -> float:
    a = np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
    loud = np.flatnonzero(a > threshold)
    return loud[0] / 16000 if loud.size else 0.0

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio(loop_scope="module")]

RESULTS: list[dict] = []


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(r), len(h)]) / max(1, len(r))


@dataclass
class Turn:
    said: str
    transcript: str = ""
    transcript_wer: float | None = None
    reply_text: str = ""
    reply_audio_s: float = 0.0
    eos_to_first_audio_ms: float | None = None
    eos_to_first_audible_ms: float | None = None
    eos_to_transcript_ms: float | None = None
    total_ms: float | None = None
    tools: list[str] = field(default_factory=list)
    round_trip: str = ""
    round_trip_wer: float | None = None
    server_line: dict | None = None
    load1: float | None = None          # 1-minute load average when the turn ended (contention marker)
    hold_phase: str = ""
    at: str = ""                        # wall-clock time of the turn, for the run's time window
    wav: str = ""                       # the reply audio as received, under the run dir's audio/


def record(name: str, **data):
    RESULTS.append({"test": name, **data})
    (RUN_DIR / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))


_SAID: dict[str, bytes] = {}


def save_wav(name: str, pcm24: bytes) -> str:
    """The reply audio (24 kHz mono PCM16) as a WAV in the run dir, for listening and for tools/speech_gaps.py."""
    import wave
    d = RUN_DIR / "audio"
    d.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    path = d / f"{len(list(d.glob('*.wav'))):02d}-{slug}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm24)
    return str(path.relative_to(RUN_DIR))


def said(text: str) -> bytes:
    """`say` rendered once, before any timing: rendering blocks the event loop for a second or two, which inflated
    the first run's barge-in figure (2,536 ms measured from before the rendering, 2026-10-05)."""
    if text not in _SAID:
        _SAID[text] = say_pcm(text)
    return _SAID[text]


async def connect(orch, device: str) -> V1Client:
    """A client that, like a real microphone, has been sending room silence before the person speaks."""
    c = V1Client(orch.e2e_url, device=device)
    await c.connect()
    await c.silence(1.0)
    return c


async def ask(c: V1Client, text: str, *, wait_s: float = 90.0) -> Turn:
    """Speak `text` (say -> 16 kHz), keep the mic open with silence, wait for the reply's end_of_turn."""
    t = Turn(said=text)
    pcm = said(text)
    start = c.now()
    await c.stream(pcm)
    eos = start + speech_end_s(pcm)
    tail = asyncio.create_task(c.silence(wait_s))
    try:
        tr = await c.wait_for(lambda m: m.get("t") == "transcript" and m.get("final"), timeout=30, since=start)
        t.eos_to_transcript_ms = round(1000 * (next(at for at, m in c.messages if m is tr) - eos))
        # the turn ends with one end_of_turn: the acknowledgement and the answer share the turn's reply id
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and bool(c.replies.get(m.get("reply_id"))
                                                                          and c.replies[m["reply_id"]].bytes),
                         timeout=wait_s, since=eos)
        replies = [r for r in c.replies.values() if r.started_at is not None and r.started_at > eos]
        assert replies, "no spoken reply"
        first = min(replies, key=lambda r: r.started_at)
        # a pause inside the utterance can split it into two VAD segments, each with its own final transcript
        finals = [m["text"] for at, m in c.messages
                  if start <= at <= first.started_at and m.get("t") == "transcript" and m.get("final")]
        t.transcript = " ".join(finals)
        t.transcript_wer = round(wer(text, t.transcript), 3)
        t.eos_to_first_audio_ms = round(1000 * (first.first_audio_at - eos)) if first.first_audio_at else None
        loud = first.first_loud_at or first.first_audio_at
        t.eos_to_first_audible_ms = round(1000 * (loud - eos)) if loud else None
        t.reply_text = "".join("".join(r.text) for r in sorted(replies, key=lambda r: r.started_at))
        t.reply_audio_s = round(sum(r.audio_s for r in replies), 2)
        t.tools = [m["name"] for at, m in c.messages if at > eos and m.get("t") == "tool" and m.get("phase") == "start"]
        last_end = max((r.ended_at or 0) for r in replies)
        t.total_ms = round(1000 * (last_end - eos)) if last_end else None
        t.wav = save_wav(text, b"".join(bytes(r.pcm) for r in sorted(replies, key=lambda r: r.started_at)))
        t.load1 = round(os.getloadavg()[0], 2)
        t.at = time.strftime("%H:%M:%S")
    finally:
        tail.cancel()
    return t


async def round_trip(orch, pcm24: bytes) -> str:
    """Transcribe the reply audio with the loaded STT adapter (24 kHz -> 16 kHz): is it intelligible speech?"""
    from scipy.signal import resample_poly  # noqa: PLC0415 - scipy comes with mlx-audio

    a = np.frombuffer(pcm24, dtype="<i2").astype(np.float32) / 32768.0
    b = resample_poly(a, 2, 3).astype(np.float32)
    return await orch.runtime.transcribe((np.clip(b, -1, 1) * 32767).astype("<i2").tobytes())


def check_gpu():
    why = gpu_still_ours()
    if why:
        pytest.skip(f"stopping: {why}")


async def test_spoken_turns_and_latency(orch):
    """DoD 2. One warm-up turn (it may include loading qwen38), then five measured turns."""
    check_gpu()
    questions = ["What is the capital of France?", "Say hello in Spanish.", "What color is the sky on a clear day?",
                 "Name a fruit that is yellow.", "What is the opposite of cold?", "Which animal says moo?"]
    for q in questions:
        said(q)
    c = await connect(orch, "e2e-turns")
    turns = []
    try:
        for i, q in enumerate(questions):
            check_gpu()
            t = await ask(c, q)
            t.hold_phase = orch.hold.state.phase
            sess = orch.clients.get("e2e-turns")
            if sess and sess.session.latency.lines:
                t.server_line = sess.session.latency.lines[-1]
            rep = [r for r in c.replies.values() if r.started_at and r.bytes and "".join(r.text).strip()]
            if rep:
                t.round_trip = await round_trip(orch, bytes(rep[-1].pcm))
                t.round_trip_wer = round(wer("".join(rep[-1].text), t.round_trip), 3)
            turns.append(t)
            record("turn", warm=i > 0, **asdict(t))
            await c.silence(1.0)
    finally:
        await c.close()
    warm = [t for t in turns[1:] if t.eos_to_first_audio_ms is not None]
    lat = sorted(t.eos_to_first_audio_ms for t in warm)
    audible = sorted(t.eos_to_first_audible_ms for t in warm)
    record("turn-summary", n=len(warm), first_audio_median_ms=statistics.median(lat), first_audio_max_ms=max(lat),
           first_audible_median_ms=statistics.median(audible),
           transcript_wer_median=statistics.median(t.transcript_wer for t in turns))
    for t in turns:
        assert t.transcript_wer is not None and t.transcript_wer <= 0.25, (t.said, t.transcript)
        assert t.reply_audio_s > 0.3 and t.reply_text.strip(), t
    assert statistics.median(lat) <= 1500, f"warm end of speech to first audio, median {statistics.median(lat)} ms"


async def test_tool_turn_acknowledged_and_answered(orch):
    """DoD 3."""
    check_gpu()
    said("What does my knowledge base say about the hold gate?")
    c = await connect(orch, "e2e-tool")
    try:
        t = await ask(c, "What does my knowledge base say about the hold gate?", wait_s=120)
    finally:
        await c.close()
    how = (orch.clients["e2e-tool"].session.agent.last_reply or {}).get("ack")
    record("tool-turn", **asdict(t), ack=how)
    assert "kb" in t.tools or t.tools, f"no tool was called: {t}"
    # the canned acknowledgement, or the model's own words said before its tool call (then no canned one follows)
    ack = orch.cfg.acks.get(t.tools[0], orch.cfg.acks["default"]) if how == "canned" else ""
    assert how in ("canned", "model") and t.reply_text.startswith(ack), \
        f"the acknowledgement was not spoken first ({how}): {t.reply_text[:80]!r}"
    answer = t.reply_text[len(ack):].strip()
    assert len(answer.split()) >= 4, f"no spoken answer after the acknowledgement: {t.reply_text!r}"


async def test_barge_in(orch):
    """DoD 4, as the person hears it: the reply's audio stops within 300 ms of the start of their speech. It is held
    back at the first frame Silero scores as speech (bargein.py), and `interrupt` follows when the VAD confirms (the
    client then flushes and reports played_ms); both times are recorded. Measured from the start of the clip, which
    is stricter than the speech onset (the first sample above -40 dBFS, also recorded)."""
    check_gpu()
    # no comma in the request: Smart Turn called "...a lighthouse keeper and his cat," complete at the comma's pause
    # in the first run and the turn split in two (a turn-taking finding of its own)
    story = said("Tell me a long story about a lighthouse keeper and his cat with lots of detail.")
    said("Stop. What is the capital of Italy?")
    c = await connect(orch, "e2e-barge")
    try:
        start = c.now()
        await c.stream(story)
        quiet = asyncio.create_task(c.silence(60))
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=60, since=start)
        while True:
            r = c.current
            if r is not None and r.first_audio_at is not None and c.now() - r.first_audio_at > 2.0:
                break
            await asyncio.sleep(0.02)
        quiet.cancel()
        interrupted = r
        speech_start = c.now()
        followup = asyncio.create_task(ask(c, "Stop. What is the capital of Italy?"))
        intr = await c.wait_for(lambda m: m.get("t") == "interrupt", timeout=10, since=speech_start)
        intr_at = next(at for at, m in c.messages if m is intr)
        intr_ms = round(1000 * (intr_at - speech_start))
        t = await followup
        sess = orch.clients.get("e2e-barge")       # gone from the server's table once the connection closes
    finally:
        await c.close()
    silent = interrupted.silent_from(speech_start)
    stop_ms = round(1000 * (min(silent, intr_at) - speech_start)) if silent is not None else intr_ms
    onset_ms = round(1000 * first_audible_s(said("Stop. What is the capital of Italy?")))
    pauses = [(e.kind, None if e.held_ms is None else round(e.held_ms)) for e in sess.session.reply_pause.events] \
        if sess and sess.session.reply_pause else None
    log = (orch.state_dir / "pi-logs" / "home.log").read_text()
    notes = re.findall(r'You were interrupted\. The user heard only this much of your last reply: \\"(.*?)\\"\)', log)
    record("barge-in", audio_stop_after_speech_ms=stop_ms, interrupt_after_speech_ms=intr_ms, speech_onset_ms=onset_ms,
           pause_events=pauses, played_ms=interrupted.played_ms, reply_audio_sent_s=round(interrupted.audio_s, 2),
           heard_note=notes[-1] if notes else None, followup=asdict(t))
    assert intr["reply_id"] == interrupted.id
    assert stop_ms <= 300, f"the reply kept playing {stop_ms} ms after speech start (interrupt at {intr_ms} ms)"
    assert notes, "the next prompt did not carry what the user heard"
    heard = notes[-1]
    sent_words = len("".join(interrupted.text).split())
    assert 0 < len(heard.split()) < max(sent_words, 1) + 1
    assert "rome" in t.reply_text.lower(), t.reply_text


def ds4_lines(since: str, pattern: str) -> list[str]:
    """ds4-server's own log lines at or after HH:MM:SS today that contain `pattern` (llama-swap's upstream log,
    read only)."""
    import subprocess

    try:
        text = subprocess.run(["curl", "-s", "-m", "3", "http://127.0.0.1:8091/logs/stream/upstream"],
                              capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return []
    out = []
    for line in text.splitlines():
        m = re.match(r"\d{4} (\d\d:\d\d:\d\d) ds4-server: (.*)", line)
        if m and m.group(1) >= since and pattern in m.group(2):
            out.append(line)
    return out


async def test_speaking_again_while_the_reply_is_written_keeps_the_llm_state(orch):
    """Brief step 4: the person speaks while the reply is still being written (here 0.3 s after a typed question,
    before any of the reply is heard). The run is not aborted but finishes silently in context (agent.py), so the
    next request extends ds4's state: no `live kv cache miss` in ds4's log, and the next turn starts at the usual
    speed instead of after a replay (9.9 s in the 13:08 run). Over 12 conversational replies generation ended
    0.2-0.84 s after the reply's audio started (test_reply_timing.py), so a barge-in in a reply's first second lands
    here too. The question is typed so the collision does not depend on Smart Turn: spoken, "What does a heat pump
    do?" was judged incomplete twice in a row (14:03 and 14:07) and held open until its 3 s fallback."""
    check_gpu()
    said("Does it work in winter too?")
    c = await connect(orch, "e2e-early")
    try:
        # a typed turn first, so ds4's cold miss on a fresh run's first request is not counted (its log has 1 s marks)
        at = c.now()
        await c.send({"t": "text", "text": "Hi."})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=60, since=at)
        await c.silence(1.1)
        since = time.strftime("%H:%M:%S")
        await c.send({"t": "text", "text": "What does a heat pump do?"})
        await c.silence(0.3)
        agent = orch.clients["e2e-early"].session.agent
        # a follow-up the model answers from what it knows: "And how much does one cost?" sent it searching the kb
        # and then the rig's docs for a way to search the web for 90 s (a persona finding, 2026-10-05)
        t = await ask(c, "Does it work in winter too?")
        silent = dict(agent.silent_runs)
    finally:
        await c.close()
    misses = ds4_lines(since, "live kv cache miss")
    log = (orch.state_dir / "pi-logs" / "home.log").read_text()
    notes = re.findall(r'\(You were interrupted[^)]*\)', log)
    record("speaking-again", silent_runs=silent, ds4_kv_misses=misses, heard_note=notes[-1] if notes else None,
           followup=asdict(t))
    assert silent == {"settled": 1, "aborted": 0}, silent
    assert not misses, misses
    assert t.eos_to_first_audio_ms is not None and t.eos_to_first_audio_ms < 3000, t


ESC50 = Path(__file__).resolve().parents[3] / "state" / "bargein" / "esc50" / "audio"   # tools/fetch_esc50.sh


def control_clip(name: str, ref: bytes) -> bytes | None:
    """An ESC-50 recording (Piczak 2015, CC BY-NC 3.0) at 16 kHz, cut to where it sounds and scaled so its loudest
    100 ms is as loud as `ref`'s: a sound as loud as the person's voice at the microphone, as tools/bargein_bench.py
    scales its controls."""
    import soundfile as sf
    from scipy.signal import resample_poly

    f = ESC50 / name
    if not f.exists():
        return None
    x, sr = sf.read(str(f), dtype="float32", always_2d=False)
    x = resample_poly(x, 160, 441).astype(np.float32) if sr == 44100 else x

    def loudest_db(a: np.ndarray) -> float:
        c = np.convolve(a.astype(np.float64) ** 2, np.ones(1600) / 1600, mode="valid")
        return 10 * np.log10(max(c.max(), 1e-12))
    r = np.frombuffer(ref, dtype="<i2").astype(np.float32) / 32768
    g = min(10 ** ((loudest_db(r) - loudest_db(x)) / 20), 0.99 / max(np.abs(x).max(), 1e-6))
    x = x * g
    loud = np.flatnonzero(np.abs(x) > 0.01)
    x = x[max(0, loud[0] - 800):loud[-1] + 800]
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


LONG_REPLY = ("The lighthouse stood at the end of a long stone pier, and every evening the keeper climbed its spiral "
              "stairs to light the lamp. His cat followed him all the way up, step by step, and sat by the window "
              "while the beam swept across the dark water. Ships far out at sea saw the light and knew where the rocks "
              "were. In the morning the keeper polished the glass, wound the clockwork, and wrote the weather in his "
              "logbook before he slept.")


async def test_planted_controls_over_a_reply(orch):
    """False barge-ins, with planted controls over a playing reply (tools/bargein_bench.py chose them): a cough
    Silero scores as speech for a frame or so but never confirms, keyboard typing it never scores as speech, then a
    backchannel. The reply is spoken by the orchestrator itself (as an acknowledgement or a digest is), so its length
    does not depend on the model. The cough and the typing must leave the reply playing (the cough may pause it for a
    moment). "Mm-hm." is speech to any voice detector, so it is expected to interrupt, as it did before the pause;
    that is recorded, not asserted."""
    from pipecat.frames.frames import TTSSpeakFrame

    check_gpu()
    ref = said("Stop. What is the capital of Italy?")
    controls = [("cough", control_clip("1-30830-A-24.wav", ref)), ("typing", control_clip("1-137-A-32.wav", ref)),
                ("mm-hm", said("Mm-hm."))]
    if any(clip is None for _, clip in controls):
        pytest.skip("ESC-50 controls not fetched (tools/fetch_esc50.sh)")
    c = await connect(orch, "e2e-controls")
    out: dict[str, dict] = {}
    try:
        sess = orch.clients["e2e-controls"].session
        quiet = asyncio.create_task(c.silence(120))
        start = c.now()
        await sess.queue(TTSSpeakFrame(LONG_REPLY, append_to_context=False))
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=30, since=start)
        while True:
            r = c.current
            if r is not None and r.first_audio_at is not None and c.now() - r.first_audio_at > 2.0:
                break
            await asyncio.sleep(0.02)
        for name, clip in controls:
            assert r.ended_at is None and r.interrupted_at is None, f"the reply was over before the {name}"
            quiet.cancel()
            at = c.now()
            await c.stream(clip)
            quiet = asyncio.create_task(c.silence(120))
            await asyncio.sleep(2.5)
            t0 = c.t0 + at
            evs = [(e.kind, None if e.held_ms is None else round(e.held_ms)) for e in sess.reply_pause.events
                   if t0 <= e.at <= t0 + 3.0]
            gaps = [round(1000 * (b[0] - a[1])) for a, b in zip(r.segments, r.segments[1:]) if a[1] >= at - 0.05]
            out[name] = {"interrupted": bool(c.of_type("interrupt", since=at)), "pause_events": evs,
                         "clip_s": round(len(clip) / 32000, 2), "gaps_ms": gaps[:3]}
            if out[name]["interrupted"]:
                break
        quiet.cancel()
    finally:
        await c.close()
    record("planted-controls", reply_id=r.id, reply_audio_s=round(r.audio_s, 2), controls=out)
    assert not out["cough"]["interrupted"], out
    assert not out["typing"]["interrupted"], out
    assert out["typing"]["pause_events"] == [], out
