"""Model-free tests of the planted-echo bench (tools/echo_mixer.py, tools/echo_bench.py; 2026-10-05).

Every property the bench's numbers rest on is measured here on synthetic signals instead of trusted: the echo's delay
to the sample, its level to a tenth of a dB, the resampler's gain, timing and stopband, the room's decay and energy,
the noise floor's level and slope, the simulated player's timeline, and frame-by-frame rendering equal to rendering at
once. Then the plan against the brief, the config a scratch server reads, the routed stub, the metrics and the report
on canned streams, and two ends to end: an EchoClient against a fake protocol v1 server (the echo arrives on the wire
at its delay and level), and scenarios against the model-free orchestrator (tests/harness.py: an echo at -10 dB makes
the energy VAD barge in with nothing planted, the same reply with no echo or at -30 dB plays to its end).

Nothing here loads a model, calls `say` or a real LLM, or touches the GPU: the clips are tones and noise.
"""
from __future__ import annotations

import asyncio
import http.client
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH / "tools"))

import echo_bench as bench  # noqa: E402
from echo_mixer import (  # noqa: E402
    FRAME,
    MIC_RATE,
    SPK_RATE,
    EchoMixer,
    EchoSettings,
    PinkNoise,
    db,
    decay_rt60,
    float_to_pcm16,
    pcm16_to_float,
    played,
    resample_24k_to_16k,
    rms,
    room_response,
)
from local_voice.client import Reply  # noqa: E402
from local_voice.config import DEFAULT_CONFIG, load_config  # noqa: E402
from local_voice.pi_rpc import interrupted_note  # noqa: E402
from local_voice.scratch import make_scratch  # noqa: E402


def sine(f: float, seconds: float, rate: int, amp: float = 0.5) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * f * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


def noise(seconds: float, rate: int, level: float = 0.1, seed: int = 7) -> np.ndarray:
    return (level * np.random.default_rng(seed).standard_normal(int(seconds * rate))).astype(np.float32)


def fixed_speaker(x24: np.ndarray, at: int = 0):
    """A speaker that played `x24` from speaker sample `at` on, and nothing else."""
    def speaker(j0: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        a, b = max(j0, at), min(j0 + n, at + len(x24))
        if a < b:
            out[a - j0:b - j0] = x24[a - at:b - at]
        return out
    return speaker


def lag_of(x: np.ndarray, y: np.ndarray) -> int:
    """Where y sits in x: the peak of their cross-correlation (FFT)."""
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n)), n)
    return int(np.argmax(c[:len(x)]))


# ------------------------------------------------------------------------------------------------ the mixer


def test_resampling_keeps_level_and_time_and_removes_what_would_alias():
    for f in (300.0, 1000.0, 5000.0):
        y = resample_24k_to_16k(sine(f, 1.0, SPK_RATE))
        ref = sine(f, 1.0, MIC_RATE)
        mid = slice(500, 15500)                      # away from the clip's edges
        assert len(y) == MIC_RATE
        assert abs(db(rms(y[mid]) / rms(ref[mid]))) < 0.05, f
        assert np.abs(y[mid] - ref[mid]).max() < 2e-3, f     # same phase: no delay
    for f in (9000.0, 10000.0, 11000.0):             # would fold to 7, 6 and 5 kHz at 16 kHz
        y = resample_24k_to_16k(sine(f, 1.0, SPK_RATE))
        assert db(rms(y[500:15500]) / rms(sine(f, 1.0, SPK_RATE))) < -60, f
    for m in (100, 333, 1001):                       # speaker sample 3m is mic sample 2m
        x = np.zeros(3600, np.float32)
        x[3 * m] = 1.0
        assert int(np.argmax(resample_24k_to_16k(x))) == 2 * m


def test_the_echo_is_resampled_delayed_and_scaled_as_set():
    x24 = noise(2.0, SPK_RATE, 0.1)
    y16 = resample_24k_to_16k(x24)
    at = 4800                                        # the speaker played it from 0.2 s
    for level, delay_ms in ((-30.0, 150.0), (-20.0, 150.0), (-10.0, 150.0), (-20.0, 300.0)):
        m = EchoMixer(EchoSettings(level_db=level, delay_ms=delay_ms, noise_dbfs=None), fixed_speaker(x24, at))
        mic = m.render(0, int(3.0 * MIC_RATE))
        want = at * 2 // 3 + int(delay_ms * MIC_RATE / 1000)
        lag = lag_of(mic, y16)
        assert lag == want, (level, delay_ms, lag, want)
        assert abs(db(rms(mic[lag:lag + len(y16)]) / rms(y16)) - level) < 0.05, (level, delay_ms)
        assert np.all(mic[:want - 64] == 0)          # nothing before the echo but the (absent) noise floor
    with pytest.raises(ValueError):                  # an echo needed before its frame could be sent
        EchoMixer(EchoSettings(level_db=-20.0, delay_ms=10.0), fixed_speaker(x24))


def test_the_room_decays_as_specified_keeps_a_broadband_level_and_is_fixed_by_its_seed():
    t = np.arange(4800) / 16000
    # the measurement itself, on a pure envelope (cut at RT60 like the room, which bends Schroeder's curve by 0.03%)
    assert abs(decay_rt60(10.0 ** (-3.0 * t / 0.3)) - 0.3) < 1e-3
    h = room_response()
    assert abs(decay_rt60(h) - 0.3) < 0.03
    assert abs(float(np.sum(h.astype(np.float64) ** 2)) - 1.0) < 1e-5
    assert np.array_equal(h, room_response()) and not np.array_equal(h, room_response(seed=1))
    x24 = noise(2.0, SPK_RATE, 0.1)
    dry = EchoMixer(EchoSettings(level_db=-20.0, noise_dbfs=None), fixed_speaker(x24))
    wet = EchoMixer(EchoSettings(level_db=-20.0, reverb=True, noise_dbfs=None), fixed_speaker(x24))
    a, b = int(0.4 * MIC_RATE), int(1.9 * MIC_RATE)
    d, w = dry.render(0, int(3 * MIC_RATE)), wet.render(0, int(3 * MIC_RATE))
    assert abs(db(rms(w[a:b]) / rms(d[a:b]))) < 0.5      # unit energy: a broadband echo keeps its level
    end = 2400 + 2 * len(x24) // 3                   # the dry echo stops here (plus the resampler's 2 ms)
    assert rms(d[end + 40:end + 2000]) == 0.0
    tail = w[end + 40:end + 4790]                    # ... and the room rings on for RT60, then nothing
    assert db(rms(tail[:800]) / rms(w[a:b])) > -25 and rms(w[end + 4800 + 64:]) == 0.0


def test_the_pink_floor_is_minus_60_dbfs_and_falls_3_db_per_octave():
    p = PinkNoise(-60.0)
    z = p.take(0, 10 * MIC_RATE)
    assert abs(db(rms(z)) + 60.0) < 0.1
    seg = 4096
    win = np.hanning(seg)
    psd = np.mean([np.abs(np.fft.rfft(z[i:i + seg] * win)) ** 2 for i in range(0, len(z) - seg, seg // 2)], axis=0)
    f = np.fft.rfftfreq(seg, 1 / MIC_RATE)
    sel = (f >= 100) & (f <= 6000)
    slope = np.polyfit(np.log10(f[sel]), np.log10(psd[sel]), 1)[0]
    assert abs(slope + 1.0) < 0.1, slope             # 1/f: -10 dB a decade, -3 dB an octave
    assert np.array_equal(p.take(40000, 3000), z[40000:43000])       # any stretch on its own equals the long render


def test_rendering_frame_by_frame_equals_rendering_at_once():
    x24 = noise(3.0, SPK_RATE, 0.1)
    s = EchoSettings(level_db=-10.0, delay_ms=150.0, reverb=True, noise_dbfs=-60.0)
    whole, framed = EchoMixer(s, fixed_speaker(x24, 2400)), EchoMixer(s, fixed_speaker(x24, 2400))
    clip = sine(440.0, 0.5, MIC_RATE, 0.3)
    for m in (whole, framed):
        p = m.plant(clip, 7 * FRAME + 5, label="request", kind="speech", text="hello")
    n = 150
    a = whole.render(0, n * FRAME)
    b = b"".join(framed.frame(i) for i in range(n))
    assert float_to_pcm16(a) == b                     # bit for bit
    assert rms(a[FRAME * 20:]) > 0 and np.any(a[: 7 * FRAME])   # the echo, the clip and the noise floor are in it
    assert (p.start, p.onset) == (7 * FRAME + 5, 7 * FRAME + 5 + int(np.flatnonzero(np.abs(clip) > 0.01)[0]))
    assert p.end == 7 * FRAME + 5 + int(np.flatnonzero(np.abs(clip) > 0.01)[-1]) + 1


def test_played_follows_the_simulated_player_its_gaps_and_its_flush():
    """local_voice.client.Reply's own timeline: a frame plays from its arrival or right after the one before; a paused
    send (bargein.py) leaves a gap; an interrupt flushes the rest."""
    r = Reply("r1")
    chunk = 960                                       # 40 ms at 24 kHz, the server's chunk
    arrivals = [10.50, 10.52, 10.55, 10.92, 10.93]    # three back to back, a 0.3 s pause, two more
    for k, at in enumerate(arrivals):
        pcm = float_to_pcm16(np.full(chunk, 0.1 * (k + 1), np.float32))
        r.pcm += pcm
        r.bytes += len(pcm)
        r.arrived(at, len(pcm))
    assert [[round(s, 3), round(e, 3)] for s, e in r.segments] == [[10.5, 10.62], [10.92, 11.0]]
    t0 = 10.0
    out = played([r], t0, 0, int(1.2 * SPK_RATE))
    at = lambda t: out[int(round((t - t0) * SPK_RATE))]     # noqa: E731
    assert abs(at(10.51) - 0.1) < 1e-3 and abs(at(10.55) - 0.2) < 1e-3 and abs(at(10.60) - 0.3) < 1e-3
    assert at(10.70) == 0.0 and abs(at(10.93) - 0.4) < 1e-3 and abs(at(10.99) - 0.5) < 1e-3
    assert np.count_nonzero(out) == 5 * chunk
    r.interrupted_at = 10.95                           # the client flushed its player here
    out = played([r], t0, 0, int(1.2 * SPK_RATE))
    assert np.count_nonzero(out) == 3 * chunk + int(round(0.03 * SPK_RATE))
    assert at(10.94) != 0.0 and at(10.97) == 0.0
    assert not played([r], None, 0, 100).any()        # no microphone yet: no timeline


def test_the_replay_is_the_last_seconds_the_client_played():
    r = Reply("r1")
    pcm = float_to_pcm16(noise(10.0, SPK_RATE, 0.1, seed=3))
    r.pcm += pcm
    r.bytes += len(pcm)
    r.played_ms = 10000.0                              # end_of_turn: all of it played
    full = pcm16_to_float(pcm)
    assert np.array_equal(bench.replay_clip(r, 8.0), resample_24k_to_16k(full[-8 * SPK_RATE:]))
    r.played_ms = 5000.0                               # cut after 5 s: only what was heard can come back
    clip = bench.replay_clip(r, 8.0)
    assert len(clip) == 5 * MIC_RATE and np.array_equal(clip, resample_24k_to_16k(full[:5 * SPK_RATE]))


def test_the_read_aloud_sentence_is_picked_from_the_reply():
    assert bench.pick_sentence(bench.LONG_REPLY) == bench.LONG_REPLY.split(". ")[0] + "."      # 24 words
    assert bench.pick_sentence("Sure. Rome is the capital of Italy and its largest city, by far. Want more?") == \
        "Rome is the capital of Italy and its largest city, by far."
    assert bench.pick_sentence("Hi. Yes, okay.") == "Yes, okay." and bench.pick_sentence("") == ""


# ------------------------------------------------------------------------------------------------ the plan


def test_the_plan_is_the_briefs():
    by = {s.name: s for s in bench.SCENARIOS}
    live = {(s.level_db, s.delay_ms, s.reverb) for s in bench.SCENARIOS if s.kind == "live"}
    assert live == {(lv, 150.0, rv) for lv in (-30.0, -20.0, -10.0) for rv in (False, True)} | {(-20.0, 300.0, True)}
    b = [s for s in bench.SCENARIOS if s.kind == "bargein"]
    assert len(b) == 1 and b[0].bargein_after_s == 2.0 and b[0].level_db == -20.0
    assert bench.BARGEIN == "Stop. What is the capital of Italy?"
    rp = by["replay"]
    assert (rp.level_db, rp.replay_after_s, rp.replay_s, rp.replay_db) == (None, 6.0, 8.0, 0.0)
    ra = by["read-aloud"]
    assert ra.level_db is None and bench.READ_ALOUD in bench.LONG_REPLY and bench.READER_VOICE != bench.PERSON_VOICE
    assert by["control"].level_db is None
    assert (bench.CONFIGS["off"], bench.CONFIGS["on"]) == ({"guard": False, "hold_for_words": False},
                                                           {"guard": True, "hold_for_words": "all"})
    assert bench.DEFAULT_CONFIGS == ("off", "on")                  # production's apps setting, `guard`, is opt-in
    assert bench.CONFIGS["guard"] == {"guard": True, "hold_for_words": False}
    pairs = bench.plan()
    assert [c for c, _ in pairs] == ["off"] * len(bench.SCENARIOS) + ["on"] * len(bench.SCENARIOS)
    e = bench.estimate(pairs)
    assert 12 <= e["expected_min"] <= 25 and e["worst_case_min"] > e["expected_min"]
    assert bench.spoken_s(bench.LONG_REPLY, bench.TTS_WPS) > 30      # a long reply: 84 words, about 32 s
    with pytest.raises(ValueError):
        bench.plan(only=["no-such"])
    with pytest.raises(ValueError):
        bench.plan(configs=["maybe"])
    out = subprocess.run([sys.executable, str(ORCH / "tools/echo_bench.py"), "plan"], capture_output=True, text=True,
                         check=True).stdout
    assert "22 scenario runs over 2 server starts" in out and "min of GPU time" in out


def git_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    for name, text in files.items():
        (path / name).write_text(text)
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", *cmd], cwd=path, check=True, capture_output=True)
    return path


def test_the_scenario_config_reaches_a_server_started_with_scratch(tmp_path):
    """./run.sh --scratch DIR rewrites DIR/config.yaml at every start (make_scratch); load_config merges
    DIR/config.local.yaml, which the bench writes, over it. Both configs, each after a fresh make_scratch."""
    kb = git_repo(tmp_path / "src-kb", {"AGENTS.md": "kb\n"})
    sp = tmp_path / "src-spaces.yaml"
    sp.write_text(yaml.safe_dump({"version": 1, "defaults": {"model": "local/qwen38"}, "spaces": {"home": {
        "root": None, "description": "your Mac", "triggers": ["home"], "skills": [], "tools": ["read"]}}}))
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["agent"]["spaces_file"] = str(sp)
    raw["agent"]["persona_file"] = str(ORCH / "persona.md")
    raw["pi"]["extensions"] = {k: str((ORCH / v).resolve()) if not str(v).startswith("~") else v
                               for k, v in raw["pi"]["extensions"].items()}
    src = tmp_path / "src-config.yaml"
    src.write_text(yaml.safe_dump(raw))
    real = tmp_path / "models.json"
    real.write_text(json.dumps({"providers": {"local": {"baseUrl": "http://127.0.0.1:8090/v1",
                                                        "api": "openai-completions", "models": [{"id": "qwen38"}]}}}))
    dest = tmp_path / "scratch"
    models = bench.stub_models(real, dest / "stub-models.json", "http://127.0.0.1:7861/v1")
    assert json.loads(models.read_text())["providers"]["local"]["baseUrl"] == "http://127.0.0.1:7861/v1"
    assert json.loads(real.read_text())["providers"]["local"]["baseUrl"] == "http://127.0.0.1:8090/v1"   # only read
    for config in ("on", "guard", "off"):
        bench.write_local_config(dest, config, models=models)
        cfg = load_config(make_scratch(dest, config=src, kb_source=kb, port=8771, browser_port=7861))
        assert cfg.port == 8771 and cfg.hosts == ["127.0.0.1"] and cfg.browser_enabled is False
        assert cfg.pi_models_source == models
        if config == "on":
            assert cfg.echo["guard"] is True and cfg.echo["hold_for_words"] == "all"
        elif config == "guard":
            assert cfg.echo["guard"] is True and cfg.echo["hold_for_words"] == "off"
        else:                                         # both off: config.py gives {} (Pipecat's own turn start)
            assert cfg.echo == {}
        assert bench.server_echo(dest) == cfg.echo


def sse_text(data: str) -> str:
    out = []
    for line in data.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            for ch in json.loads(line[6:]).get("choices") or []:
                out.append((ch.get("delta") or {}).get("content") or "")
    return "".join(out)


def test_the_routed_stub_answers_by_the_persons_own_words(tmp_path):
    assert bench.route_reply(bench.REQUEST)["text"] == bench.LONG_REPLY
    # the orchestrator's note quotes the agent; only the person's words (the last paragraph) choose
    assert bench.route_reply(interrupted_note("The capital of Italy is Rome.") + "what was that")["text"] == \
        bench.DEFAULT_REPLY
    stub = bench.start_stub(0, tmp_path / "stub")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", stub.port, timeout=10)

        def ask(prompt: str) -> str:
            body = json.dumps({"model": "qwen38", "stream": True, "messages": [
                {"role": "system", "content": "persona"}, {"role": "user", "content": prompt}]})
            conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
            return sse_text(conn.getresponse().read().decode())

        assert ask(bench.REQUEST) == bench.LONG_REPLY
        # the same connection again (Pi keeps it alive): the stub re-reads a body it already read, then carries on
        assert ask(interrupted_note(bench.LONG_REPLY[:60]) + bench.BARGEIN) == bench.BARGEIN_REPLY
        assert ask([{"type": "text", "text": "And then?"}]) == bench.DEFAULT_REPLY
        assert len(stub.requests()) == 3
    finally:
        stub.stop()


# ------------------------------------------------------------------------------------------------ metrics


WALL0 = 1_791_000_000.0


def iso(t: float) -> str:
    return datetime.fromtimestamp(WALL0 + t).astimezone().isoformat(timespec="milliseconds")


def log_line(t: float, msg: str, where: str = "local_voice.echo_guard:process_frame:216") -> tuple[float, str]:
    stamp = datetime.fromtimestamp(WALL0 + t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return WALL0 + t, f"{stamp} | INFO     | {where} - {msg}"


def reply(rid: str, started: float, segments: list, text: str, *, ended: float | None = None,
          interrupted: float | None = None) -> dict:
    played = sum(e - s for s, e in segments) if interrupted is None else sum(
        max(0.0, min(e, interrupted) - s) for s, e in segments)
    return {"id": rid, "started_at": started, "first_audio_at": segments[0][0], "first_loud_at": segments[0][0],
            "ended_at": ended, "interrupted_at": interrupted, "played_ms": round(1000 * played),
            "audio_s": round(sum(e - s for s, e in segments), 3), "segments": segments, "text": text}


def planted(label: str, kind: str, text: str, onset: float, end: float) -> dict:
    return {"label": label, "kind": kind, "text": text, "gain_db": 0.0, "start_t": onset - 0.05, "onset_t": onset,
            "end_t": end, "clip_s": end - onset + 0.1}


def turn(n: int, t: float, text: str, *, interrupted: bool = False) -> dict:
    return {"v": 1, "type": "turn", "session": "s-1", "turn": n, "t_start": iso(t), "t_end": iso(t + 3),
            "user_text": text, "reply_text": "", "interrupted": interrupted}


def run_of(replies: list, planted_: list, messages: list, main: str | None, name: str = "x") -> dict:
    return {"config": "off", "scenario": {"name": name}, "echo": "-", "device": "d", "session": "s-1", "wall0": WALL0,
            "mic_t0": 0.0, "t_begin": 0.0, "t_end": 60.0, "frames_sent": 3000, "messages": messages,
            "replies": replies, "planted": planted_, "main_reply": main, "notes": []}


REQ = planted("request", "speech", bench.REQUEST, 1.05, 5.0)
FIRST_SENTENCE = bench.LONG_REPLY.split(". ")[0] + "."


def test_metrics_live_echo_that_barges_in():
    r1 = reply("r1", 6.0, [[6.05, 6.6]], FIRST_SENTENCE, interrupted=6.6)
    r2 = reply("r2", 8.5, [[8.55, 18.0]], bench.DEFAULT_REPLY, ended=18.1)
    msgs = [[5.6, {"t": "transcript", "final": True, "text": bench.REQUEST}],
            [6.6, {"t": "interrupt", "reply_id": "r1"}],
            [7.8, {"t": "transcript", "final": True, "text": "The lighthouse stood at the end of a long"}]]
    turns = [turn(1, 1.2, bench.REQUEST), turn(2, 6.4, "The lighthouse stood at the end of a long", interrupted=True)]
    log = [log_line(6.3, "barge-in d: speech cue, reply paused", "local_voice.bargein:log:256"),
           log_line(6.6, "barge-in d: confirmed after 300 ms", "local_voice.bargein:log:259"),
           log_line(6.4, "LLMUserAggregator#0: User started speaking (strategy: VADUserTurnStartStrategy#0)",
                    "pipecat.processors.aggregators.llm_response_universal:_on_user_turn_started:1346"),
           log_line(99.0, "barge-in d: speech cue, reply paused", "local_voice.bargein:log:256")]    # after the window
    m = bench.metrics(run_of([r1, r2], [REQ], msgs, "r1"), turns, log)
    assert [u["kind"] for u in m["user_turns"]] == ["speech:request", "echo"]
    assert (m["n_user_turns"], m["n_echo_turns"], m["false_barge_ins"], m["replies_cut"]) == (2, 1, 1, 1)
    assert m["interrupts"] == [{"t": 6.6, "reply_id": "r1", "cause": "false"}]
    assert m["main_reply"]["complete"] is False and m["main_reply"]["played_s"] == 0.55
    assert m["request"] == {"onset_t": 1.05, "turn": True, "text": bench.REQUEST, "wer": 0.0, "answered": True}
    assert (m["log"]["paused"], m["log"]["confirmed"], m["log"]["user_started"], m["log"]["dropped"]) == (1, 1, 1, 0)
    assert len(m["log_lines"]) == 3


def test_metrics_true_barge_in_over_echo_with_the_guard():
    bi = planted("barge-in", "speech", bench.BARGEIN, 10.0, 12.2)
    r1 = reply("r1", 6.0, [[6.05, 9.0], [9.4, 10.07]], bench.LONG_REPLY, interrupted=10.35)
    r2 = reply("r2", 13.5, [[13.55, 15.0]], bench.BARGEIN_REPLY, ended=15.1)
    msgs = [[10.35, {"t": "interrupt", "reply_id": "r1"}]]
    turns = [turn(1, 1.2, bench.REQUEST), turn(2, 10.2, "Stop. What is the capital of Italy?")]
    log = [log_line(7.0, "EchoFilter#0: echo of the agent's own speech (6/6 words of 'The lighthouse'): dropped 'x'"),
           log_line(8.0, "EchoFilter#0: echo of the agent's own speech (5/5 words of 'The lighthouse'): dropped 'y'"),
           # turn_start.py's own wording (53c27a4): a hold, its resume on echo, a start on the person's words, and a
           # start while idle, which was never held
           log_line(7.6, "BusyHoldStartStrategy#0: hold (speech while the agent is speaking) after holding 0.00 s",
                    "local_voice.turn_start:_record:343"),
           log_line(8.0, "BusyHoldStartStrategy#0: resume (the agent's own speech heard back) after holding 0.41 s "
                    "final 'x'", "local_voice.turn_start:_record:343"),
           log_line(10.3, "BusyHoldStartStrategy#0: start (the person's own words) after holding 0.30 s interim 'Stop'",
                    "local_voice.turn_start:_record:343"),
           log_line(0.9, "BusyHoldStartStrategy#0: start (the agent is idle) after holding 0.00 s",
                    "local_voice.turn_start:_record:343"),
           log_line(9.0, "barge-in d: speech cue, reply paused", "local_voice.bargein:log:256"),
           log_line(9.4, "barge-in d: resumed after 400 ms", "local_voice.bargein:log:259"),
           log_line(10.05, "barge-in d: speech cue, reply paused", "local_voice.bargein:log:256")]
    m = bench.metrics(run_of([r1, r2], [REQ, bi], msgs, "r1"), turns, log)
    assert m["false_barge_ins"] == 0 and m["interrupts"][0]["cause"] == "speech:barge-in"
    b = m["barge-in"]
    assert (b["interrupt_ms"], b["audio_stop_ms"], b["wer"], b["answered"], b["reply_playing"]) == (350, 70, 0.0, True,
                                                                                                    True)
    assert m["main_reply"]["pauses"] == 1 and m["main_reply"]["paused_ms"] == 400
    assert (m["log"]["dropped"], m["log"]["turn_start"], m["log"]["paused"], m["log"]["resumed"]) == (2, 4, 2, 1)
    assert (m["log"]["held"], m["log"]["held_resumed"], m["log"]["held_started"], m["log"]["held_timed_out"]) == \
        (1, 1, 1, 0)


def test_metrics_echo_of_a_replys_tail_counts_even_as_a_fragment():
    """Live echo lags the speaker by its delay, so the echo of a reply's last words starts after the server's "Bot
    stopped speaking", when the agent is idle (seen model-free in the harness with the guard on, 2026-10-05:
    the VAD start 137 ms after it, never held, passed). One or two words of it are echo by the guard's fragment rule."""
    r1 = reply("r1", 6.0, [[6.05, 38.0]], bench.LONG_REPLY, ended=38.1)
    r1["sentences"] = [s if s.endswith(".") else s + "." for s in bench.LONG_REPLY.split(". ")]
    turns = [turn(1, 1.2, bench.REQUEST), turn(2, 38.3, "slept."), turn(3, 40.9, "before he slept"),
             turn(4, 45.0, "Why?"), turn(5, 20.0, "keeper")]
    m = bench.metrics(run_of([r1], [REQ], [], "r1"), turns)
    kinds = {u["text"]: (u["kind"], u["matched"], u["during_reply"], u["after_reply_s"]) for u in m["user_turns"]}
    assert kinds["slept."] == ("echo", "fragment", False, 0.3)
    assert kinds["before he slept"][:2] == ("echo", "3/3") and kinds["before he slept"][3] == 2.9
    assert kinds["Why?"][0] == "other" and kinds["keeper"][:3] == ("echo", "fragment", True)
    assert (m["n_echo_turns"], m["n_tail_echo_turns"]) == (3, 1)        # only "slept." is within 2 s of the end
    row = {"config": "on", "scenario": {"name": "live-20-dry"}, "echo": "-20 dB, 150 ms, dry", "metrics": m}
    md = bench.report_md([row])
    assert "| 3/5 (1 at a reply's end) |" in md and "turn 2 at 38.3 s 0.3 s after a reply, echo (fragment)" in md


def test_metrics_replay_cut_and_read_aloud():
    rp = planted("replay", "echo", bench.LONG_REPLY, 44.0, 52.0)
    r1 = reply("r1", 6.0, [[6.05, 38.0]], bench.LONG_REPLY, ended=38.1)
    r2 = reply("r2", 46.0, [[46.05, 47.5]], bench.DEFAULT_REPLY, interrupted=47.5)
    r3 = reply("r3", 50.0, [[50.05, 59.0]], bench.DEFAULT_REPLY, ended=59.1)
    turns = [turn(1, 1.2, bench.REQUEST),
             turn(2, 44.3, "In the morning the keeper polished the glass wound the clockwork"),
             turn(3, 47.4, "and wrote the weather in his logbook before he slept")]
    m = bench.metrics(run_of([r1, r2, r3], [REQ, rp], [[47.5, {"t": "interrupt", "reply_id": "r2"}]], "r1"), turns)
    assert m["replay"] == {"onset_t": 44.0, "end_t": 52.0, "turns": 2, "echo_turns": 2, "replies_after": 2,
                           "replies_cut": 1, "interrupts": 1}
    assert m["interrupts"][0]["cause"] == "replay" and m["false_barge_ins"] == 1 and m["main_reply"]["complete"]
    ra = planted("read-aloud", "speech", bench.READ_ALOUD, 40.0, 46.0)
    r2 = reply("r2", 47.5, [[47.55, 57.0]], bench.DEFAULT_REPLY, ended=57.1)
    heard = ("His cat followed him all the way up step by step and sat by the window while the beam swept across the "
             "dark water.")                            # the recogniser's punctuation, not the reply's
    m = bench.metrics(run_of([r1, r2], [REQ, ra], [], "r1"), [turn(1, 1.2, bench.REQUEST), turn(2, 40.2, heard)])
    assert [u["kind"] for u in m["user_turns"]] == ["speech:request", "speech:read-aloud"] and m["n_echo_turns"] == 0
    assert m["read-aloud"]["turn"] and m["read-aloud"]["answered"] and m["read-aloud"]["wer"] == 0.0


def test_the_report_puts_off_and_on_side_by_side(tmp_path):
    r1 = reply("r1", 6.0, [[6.05, 6.6]], FIRST_SENTENCE, interrupted=6.6)
    off = bench.metrics(run_of([r1], [REQ], [[6.6, {"t": "interrupt", "reply_id": "r1"}]], "r1"),
                        [turn(1, 1.2, bench.REQUEST), turn(2, 6.4, "The lighthouse stood at the end of a long")])
    r1 = reply("r1", 6.0, [[6.05, 9.0], [9.5, 38.0]], bench.LONG_REPLY, ended=38.1)
    on = bench.metrics(run_of([r1], [REQ], [], "r1"), [turn(1, 1.2, bench.REQUEST)])
    rows = [{"config": "off", "scenario": {"name": "live-10-dry"}, "echo": "-10 dB, 150 ms, dry", "metrics": off},
            {"config": "on", "scenario": {"name": "live-10-dry"}, "echo": "-10 dB, 150 ms, dry", "metrics": on}]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "run.json").write_text(json.dumps({"stamp": "20261005-200000", "llm": "stub",
                                                   "server_echo": {"off": {}, "on": {"guard": True}}}))
    md = bench.write_report(tmp_path).read_text()
    assert ("| live-10-dry | -10 dB, 150 ms, dry | 1/2 → 0/1 | 1 → 0 | no (0 pauses, 0 ms) → yes (1 pauses, 500 ms) |"
            in md)
    assert "the server read {\"guard\": true}" in md and "turn 2 at 6.4 s during a reply, echo" in md
    out = subprocess.run([sys.executable, str(ORCH / "tools/echo_bench.py"), "report", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip().endswith("report.md")


# ------------------------------------------------------------------------------------------------ the server's life

FAKE_RUN_SH = """#!{python}
# a stand-in for run.sh: /v1/status on --port, a record of how it was stopped, or an early exit (FAKE_EXIT)
import argparse, http.server, json, os, signal, sys
ap = argparse.ArgumentParser()
for k in ("--scratch", "--port", "--browser-port"):
    ap.add_argument(k)
a = ap.parse_args()
if os.environ.get("FAKE_EXIT"):
    sys.exit(int(os.environ["FAKE_EXIT"]))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({{"v": 1, "state": "idle", "argv": sys.argv[1:]}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def log_message(self, *x):
        pass
def stop(*_):
    open(os.path.join(a.scratch, "stopped-by"), "w").write("SIGINT")
    os._exit(0)
signal.signal(signal.SIGINT, stop)
http.server.HTTPServer(("127.0.0.1", int(a.port)), H).serve_forever()
"""


def test_the_scratch_server_is_waited_for_then_stopped_with_sigint(tmp_path, monkeypatch):
    from harness import free_port

    fake = tmp_path / "run.sh"
    fake.write_text(FAKE_RUN_SH.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setattr(bench, "RUN_SH", fake)
    port = free_port()
    s = bench.ScratchServer(tmp_path, tmp_path / "server.log", port=port, browser_port=free_port())
    status = s.start(timeout_s=20)
    assert status["state"] == "idle" and status["argv"][:4] == ["--scratch", str(tmp_path), "--port", str(port)]
    with pytest.raises(RuntimeError, match="already listens"):
        bench.ScratchServer(tmp_path, tmp_path / "other.log", port=port).start(timeout_s=5)
    s.stop()
    assert (tmp_path / "stopped-by").read_text() == "SIGINT" and not bench.port_open(port)
    monkeypatch.setenv("FAKE_EXIT", "75")                  # run.sh's answer while the hold gate is held
    with pytest.raises(RuntimeError, match="75: the hold gate"):
        bench.ScratchServer(tmp_path, tmp_path / "held.log", port=free_port()).start(timeout_s=20)


def test_run_refuses_to_start_unless_gpu_clear_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "render_clips", lambda **kw: pytest.fail("rendered clips without the GPU check"))
    code = bench.main(["run", "--gpu-clear", "false", "--out", str(tmp_path / "run"), "--scratch", str(tmp_path)])
    assert code == 75 and not (tmp_path / "run" / "results.jsonl").exists()
    assert "refusing to start" in (tmp_path / "run" / "bench.log").read_text()


# ------------------------------------------------------------------------------------------------ end to end


async def test_an_echo_client_puts_the_echo_on_the_wire_at_its_delay_and_level():
    """A fake protocol v1 server records the microphone stream and sends one reply (1.5 s of -20 dBFS white noise, all
    at once so the player plays it as one stretch). In what the server received, the reply comes back exactly
    150 ms after the client's player started it, 10 dB down, with nothing else in the stream, and every frame the
    client rendered arrived, at a microphone's pace."""
    from websockets.asyncio.server import serve

    x24 = noise(1.5, SPK_RATE, 0.1, seed=11)
    pcm = float_to_pcm16(x24)
    got = bytearray()
    first_at: list[float] = []

    async def handler(ws):
        await ws.recv()                                    # hello
        await ws.send(json.dumps({"t": "welcome", "v": 1, "session": "s-fake", "state": "listening"}))
        async for data in ws:
            if not isinstance(data, bytes):
                continue
            if not got:
                first_at.append(asyncio.get_running_loop().time())
            got.extend(data)
            if len(got) >= MIC_RATE and len(got) - len(data) < MIC_RATE:      # after 0.5 s of microphone
                await ws.send(json.dumps({"t": "audio_start", "reply_id": "r1", "rate": SPK_RATE}))
                for i in range(0, len(pcm), 1920):
                    await ws.send(pcm[i:i + 1920])
                await ws.send(json.dumps({"t": "audio_end", "reply_id": "r1"}))
                await ws.send(json.dumps({"t": "end_of_turn", "reply_id": "r1"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        c = bench.EchoClient(f"ws://127.0.0.1:{port}/v1/voice", device="fake",
                             echo=EchoSettings(level_db=-10.0, delay_ms=150.0, noise_dbfs=None))
        await c.connect()
        await c.start_mic()
        r = await c.wait_reply(since=0.0, deadline=c.now() + 10)
        assert r is not None and await c.wait_until(lambda: c.over(r), c.now() + 10)
        await asyncio.sleep(0.5)                           # the echo runs 150 ms behind the player
        elapsed = asyncio.get_running_loop().time() - first_at[0]
        await c.stop_mic()
        await c.close()
    assert len(r.segments) == 1 and c.mic_error is None
    x = pcm16_to_float(got)
    y = resample_24k_to_16k(x24)
    want = (r.segments[0][0] - c.mic_t0) * MIC_RATE + 2400
    lag = lag_of(x, y)
    assert abs(lag - want) <= 1, (lag, want)
    assert abs(db(rms(x[lag:lag + len(y)]) / rms(y)) + 10.0) < 0.1
    assert not x[:lag - 64].any() and not x[lag + len(y) + 64:].any()
    assert len(got) // (2 * FRAME) in (c.frames_rendered - 1, c.frames_rendered)
    assert abs(len(x) / MIC_RATE - elapsed) < 0.1          # sent at a microphone's pace, not faster


@pytest.mark.needs_pi
async def test_scenarios_end_to_end_against_the_model_free_orchestrator(tmp_path):
    """The bench's own scenario runner and metrics against the whole orchestrator built from fakes (tests/harness.py:
    the energy VAD, the fake recogniser and voice, a real Pi on the routed stub). The fake voice is a 220 Hz tone at
    0.2: its echo at -10 dB (RMS 0.045) is over the energy VAD's 0.02 and barges in with nothing planted, at -30 dB
    (0.0045) it is not, and with no echo the reply plays to its end. The echo config is set off explicitly, so this
    does not depend on config.yaml's default."""
    from harness import rig

    story = " ".join(f"Sentence number {i} is here." for i in range(1, 9))     # 8 x 0.78 s of the fake voice
    stub = bench.start_stub(0, tmp_path / "stub",
                            route=lambda p: {"text": story if "story" in p.split("\n\n")[-1] else "Okay then.",
                                             "delay_ms": 5})
    t = np.arange(int(0.8 * MIC_RATE)) / MIC_RATE
    tone = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    clips = bench.Clips(request=tone, bargein=tone, read_aloud=tone,
                        texts={"request": "tell me a story", "barge-in": "stop", "read-aloud": "his cat"})
    out = {}
    async with rig(tmp_path, stt_text="tell me a story", stub=stub,
                   overrides={"echo": {"guard": False, "hold_for_words": False}}) as r:
        for sc in (bench.Scenario("control", "control", max_s=20.0),
                   bench.Scenario("live-30-dry", "live", -30.0, max_s=20.0),
                   bench.Scenario("live-10-dry", "live", -10.0, max_s=8.0)):
            run = await bench.run_scenario(r.url, sc, clips, config="off", device=f"t-{sc.name}", quiet_s=1.0,
                                           settle_s=1.2, lead_s=0.5, log=lambda s: None)
            out[sc.name] = bench.result_row(run, tmp_path / "turns", tmp_path / "state" / "logs")["metrics"]
    for name in ("control", "live-30-dry"):
        m = out[name]
        assert m["interrupts"] == [] and m["main_reply"]["complete"], (name, m["interrupts"], m["main_reply"])
        assert m["main_reply"]["text"].replace(" ", "") == story.replace(" ", "")
        assert m["request"]["turn"] and m["request"]["answered"] and not m["notes"], (name, m["notes"])
    m = out["live-10-dry"]
    assert m["false_barge_ins"] >= 1 and m["main_reply"]["complete"] is False, (m["interrupts"], m["main_reply"])
    assert m["interrupts"][0]["reply_id"] == m["main_reply"]["id"]


@pytest.mark.needs_pi
async def test_run_attached_to_a_server_writes_results_header_and_report(tmp_path, monkeypatch):
    """`echo_bench.py run --attach` the way the GPU run will use it, against the model-free orchestrator: the GPU
    check (`true` here), the scenario, the turn log read from the scratch layout (<dir>/turns: the harness writes its
    turn log there too), results.jsonl, run.json and report.md. In a thread, since `run` has its own event loop."""
    from harness import rig

    story = " ".join(f"Sentence number {i} is here." for i in range(1, 5))
    stub = bench.start_stub(0, tmp_path / "stub",
                            route=lambda p: {"text": story if "story" in p.split("\n\n")[-1] else "Okay then."})
    t = np.arange(int(0.8 * MIC_RATE)) / MIC_RATE
    tone = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    clips = bench.Clips(request=tone, bargein=tone, read_aloud=None,
                        texts={"request": "tell me a story", "barge-in": "stop", "read-aloud": ""})
    monkeypatch.setattr(bench, "render_clips", lambda **kw: clips)
    out = tmp_path / "run"
    async with rig(tmp_path, stt_text="tell me a story", stub=stub,
                   overrides={"echo": {"guard": False, "hold_for_words": False}}) as r:
        port = int(r.url.split(":")[2].split("/")[0])
        code = await asyncio.to_thread(bench.main, [
            "run", "--attach", "--configs", "off", "--only", "control", "--llm", "real", "--gpu-clear", "true",
            "--port", str(port), "--scratch", str(tmp_path), "--out", str(out)])
    assert code == 0
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [(r["config"], r["scenario"]["name"]) for r in rows] == [("off", "control")]
    m = rows[0]["metrics"]
    assert m["request"]["turn"] and m["main_reply"]["complete"] and m["interrupts"] == [], m
    header = json.loads((out / "run.json").read_text())
    assert header["llm"] == "real" and header["estimate"]["scenarios"] == 1 and "(attached)" in header["server"]
    assert "| control | none | 0/1 | 0 | yes (0 pauses, 0 ms) |" in (out / "report.md").read_text()
