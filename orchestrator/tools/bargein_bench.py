"""Barge-in bench (CPU only; no GPU, no model call): how fast each detector notices a person interrupting a reply,
and how often it fires on planted controls that must not interrupt one. Machine ground truth only (a project rule:
no human listening or labelling).

Cases, each streamed in 20 ms frames like the protocol v1 client, after 2 s of what the microphone carries while a
reply plays (`--backgrounds`: digital silence, as the e2e client sends, and a quiet room, pink noise at -60 dBFS):
- interruptions: `say` phrases in several voices; onset = the first sample above -40 dBFS, the convention the e2e
  tests use for the end of speech. The e2e test measures from the start of its clip, about 13 ms earlier for "Stop.".
- controls that must not interrupt: real recordings from ESC-50 (Piczak 2015, CC BY-NC 3.0; fetched by
  tools/fetch_esc50.sh) of coughs, sneezes, breathing, laughter, knocks, typing, clicks, claps,
  footsteps, sipping, a clock, a can, a dog, a cat, a vacuum cleaner and rain, each scaled so its loudest 100 ms is
  as loud as the median interruption's (a sound as loud as your voice at the microphone); synthetic white, pink and
  brown noise at -50/-40/-30 dBFS; and backchannels said by `say` ("Mm-hm.", "Yeah." ...), which are speech, so any
  voice detector fires on them and only the words can tell them apart.

Detectors:
- silero:<confidence>/<start_secs>/<min_volume>: Pipecat 1.12's SileroVADAnalyzer state machine (the model, its
  5-second state reset and its smoothed loudness are the pipeline's own; per-frame scores are computed once per case
  and the state machine replayed for each setting, checked against the analyzer itself for the current setting).
  It fires on QUIET -> SPEAKING, which is when the pipeline broadcasts the interruption and the client gets `interrupt`.
- pause:<pre-trigger>: stop sending the reply's audio at an early cue and send `interrupt` only when the current
  Silero setting confirms; resume if it has not confirmed within `--pause-window-s`. Reported as the time to the pause
  (what the person hears, plus the client's small playout buffer), how many controls cause a pause, and how long.

Usage:  .venv/bin/python tools/bargein_bench.py ../state/bargein/<stamp>
Writes rows.jsonl (one row per case, background and detector) and summary.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH))

RATE = 16000
CHUNK = 320                     # 20 ms input frames, as the protocol v1 clients send them
LEAD_S = 2.0                    # reply-time microphone before the case starts
TAIL_S = 0.6
ONSET = 0.01                    # -40 dBFS

PHRASES = ["Stop.", "Stop. What is the capital of Italy?", "Stop, stop.", "Sorry, what?", "Shush.", "So what?",
           "Wait.", "No.", "Hold on.", "Hang on a second.", "Actually, no.", "Hey.", "Excuse me.", "Okay, stop.",
           "That's enough.", "Pause.", "But why?", "Never mind.", "Can I ask something?", "Let me stop you there.",
           "What?", "Thanks, that's all."]
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Ralph"]
BACKCHANNELS = ["Mm-hm.", "Uh-huh.", "Yeah.", "Right.", "Okay.", "Sure.", "I see.", "Yep.", "Oh.", "Hmm."]
SETTINGS = [(0.7, 0.15, 0.5), (0.7, 0.1, 0.5), (0.7, 0.064, 0.5), (0.6, 0.1, 0.5), (0.5, 0.15, 0.5),
            (0.7, 0.2, 0.6)]
CURRENT = (0.7, 0.15, 0.5)      # config.yaml turn.vad on 2026-10-05


@dataclass
class Case:
    name: str
    kind: str                   # interruption | backchannel | control:<category> | noise
    audio: np.ndarray           # float32 mono 16 kHz
    onset_s: float = 0.0        # within `audio`


@dataclass
class Trace:
    """Per 512-sample Silero frame: when the pipeline would have it (end of the input chunk that completed it, in
    seconds from the case start), its confidence and smoothed volume; per 20 ms chunk: time and RMS in dBFS."""
    t: list[float] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    vol: list[float] = field(default_factory=list)
    chunk_t: list[float] = field(default_factory=list)
    chunk_db: list[float] = field(default_factory=list)


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def loudest_100ms_db(x: np.ndarray) -> float:
    n = RATE // 10
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    c = np.convolve(x.astype(np.float64) ** 2, np.ones(n) / n, mode="valid")
    return 10 * np.log10(max(c.max(), 1e-12))


def onset_s(x: np.ndarray) -> float:
    idx = np.flatnonzero(np.abs(x) > ONSET)
    return idx[0] / RATE if idx.size else 0.0


def say_clip(text: str, voice: str) -> np.ndarray:
    from local_voice.client import say_pcm
    return np.frombuffer(say_pcm(text, voice=voice), "<i2").astype(np.float32) / 32768


def colored(kind: str, n: int, rng) -> np.ndarray:
    w = rng.normal(0, 1, n)
    if kind == "white":
        x = w
    else:
        spec = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1 / RATE)
        f[0] = f[1]
        spec /= np.sqrt(f) if kind == "pink" else f
        x = np.fft.irfft(spec, n)
    return (x / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def background(kind: str, n: int, rng) -> np.ndarray:
    if kind == "silence":
        return np.zeros(n, np.float32)
    if kind == "room":      # a quiet room: pink noise at -60 dBFS RMS
        return colored("pink", n, rng) * 10 ** (-60 / 20)
    raise ValueError(kind)


def build_cases(esc_dir: Path, rng) -> list[Case]:
    from scipy.signal import resample_poly
    import soundfile as sf

    cases: list[Case] = []
    for v in VOICES:
        for p in PHRASES:
            a = say_clip(p, v)
            cases.append(Case(f"{v}: {p}", "interruption", a, onset_s(a)))
    speech_db = statistics.median(loudest_100ms_db(c.audio) for c in cases)
    for v in VOICES:
        for p in BACKCHANNELS:
            a = say_clip(p, v)
            cases.append(Case(f"{v}: {p}", "backchannel", a, onset_s(a)))
    meta = {}
    csv = esc_dir / "esc50.csv"
    if csv.exists():
        for line in csv.read_text().splitlines()[1:]:
            f, _fold, _t, cat, *_ = line.split(",")
            meta[f] = cat
    for f in sorted((esc_dir / "audio").glob("*.wav")):
        x, sr = sf.read(str(f), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = resample_poly(x, 160, 441).astype(np.float32) if sr == 44100 else x
        if np.abs(x).max() < 1e-4:
            continue
        g = 10 ** ((speech_db - loudest_100ms_db(x)) / 20)
        g = min(g, 0.99 / max(np.abs(x).max(), 1e-6))      # never clip: a click scaled to speech RMS would
        cases.append(Case(f"esc50 {f.name}", f"control:{meta.get(f.name, '?')}", (x * g).astype(np.float32)))
    for kind in ("white", "pink", "brown"):
        for db in (-50, -40, -30):
            cases.append(Case(f"{kind} noise {db} dBFS", "noise", colored(kind, 2 * RATE, rng) * 10 ** (db / 20)))
    return cases


async def trace(case: Case, bg: str, rng) -> Trace:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    n_lead, n_tail = int(LEAD_S * RATE), int(TAIL_S * RATE)
    x = background(bg, n_lead + len(case.audio) + n_tail, rng)
    x[n_lead:n_lead + len(case.audio)] += case.audio
    data = pcm16(x)
    vad = SileroVADAnalyzer(sample_rate=RATE, params=VADParams())   # scores do not depend on the params
    vad.set_sample_rate(RATE)
    nb = vad._vad_frames_num_bytes
    tr, buf = Trace(), b""
    for i in range(0, len(data), CHUNK * 2):
        chunk = data[i:i + CHUNK * 2]
        t_end = (i + len(chunk)) / (2 * RATE) - LEAD_S
        a = np.frombuffer(chunk, "<i2").astype(np.float32) / 32768
        tr.chunk_t.append(t_end)
        tr.chunk_db.append(10 * np.log10(max(float(np.mean(a.astype(np.float64) ** 2)), 1e-12)))
        buf += chunk
        while len(buf) >= nb:
            fr, buf = buf[:nb], buf[nb:]
            conf = float(np.asarray(vad.voice_confidence(fr)).ravel()[0])
            vol = vad._get_smoothed_volume(fr)
            vad._prev_volume = vol
            tr.t.append(t_end)
            tr.conf.append(conf)
            tr.vol.append(float(vol))
    return tr


def silero_fires(tr: Trace, conf: float, start_secs: float, min_volume: float, stop_secs: float = 0.2) -> list[float]:
    """Pipecat 1.12 VADAnalyzer._run_analyzer, replayed: times of QUIET/STOPPING -> SPEAKING after the lead."""
    frame_s = 512 / RATE
    start_n, stop_n = round(start_secs / frame_s), round(stop_secs / frame_s)
    state, starting, stopping, fires = "QUIET", 0, 0, []
    # frames completed by the same input chunk are evaluated together, then the start/stop counters are checked
    i = 0
    while i < len(tr.t):
        j = i
        while j < len(tr.t) and tr.t[j] == tr.t[i]:
            speaking = tr.conf[j] >= conf and tr.vol[j] >= min_volume
            if speaking:
                if state == "QUIET":
                    state, starting = "STARTING", 1
                elif state == "STARTING":
                    starting += 1
                elif state == "STOPPING":
                    state, stopping = "SPEAKING", 0
            else:
                if state == "STARTING":
                    state, starting = "QUIET", 0
                elif state == "SPEAKING":
                    state, stopping = "STOPPING", 1
                elif state == "STOPPING":
                    stopping += 1
            j += 1
        if state == "STARTING" and starting >= start_n:
            state, starting = "SPEAKING", 0
            if tr.t[i] > 0:
                fires.append(tr.t[i])
        if state == "STOPPING" and stopping >= stop_n:
            state, stopping = "QUIET", 0
        i = j
    return fires


def energy_cue(tr: Trace, rise_db: float = 15.0, floor_min_db: float = -45.0, frames: int = 2) -> list[float]:
    """Times (after the lead) where the 20 ms RMS rose rise_db over a tracked noise floor and above floor_min_db
    for `frames` chunks in a row. The floor follows quiet chunks fast and loud ones slowly."""
    floor, run, cues, armed = None, 0, [], True
    for t, db in zip(tr.chunk_t, tr.chunk_db):
        floor = db if floor is None else (min(db, floor + 0.05) if db < floor + rise_db else floor + 0.05)
        loud = db > max(floor + rise_db, floor_min_db)
        run = run + 1 if loud else 0
        if run >= frames and armed:
            if t > 0:
                cues.append(t)
            armed = False
        if run == 0:
            armed = True
    return cues


def silero_weak_cue(tr: Trace, conf: float, frames: int) -> list[float]:
    run, cues, armed = 0, [], True
    for t, c in zip(tr.t, tr.conf):
        run = run + 1 if c >= conf else 0
        if run >= frames and armed:
            if t > 0:
                cues.append(t)
            armed = False
        if run == 0:
            armed = True
    return cues


async def verify_replay(cases: list[Case], traces: dict, n: int = 12) -> int:
    """The replay must match the real analyzer for the current setting (fire times within one input chunk)."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams, VADState

    bad = 0
    for c in cases[:n]:
        rng = np.random.default_rng(11)
        n_lead, n_tail = int(LEAD_S * RATE), int(TAIL_S * RATE)
        x = background("silence", n_lead + len(c.audio) + n_tail, rng)
        x[n_lead:n_lead + len(c.audio)] += c.audio
        data = pcm16(x)
        vad = SileroVADAnalyzer(sample_rate=RATE, params=VADParams(confidence=CURRENT[0], start_secs=CURRENT[1],
                                                                   stop_secs=0.2, min_volume=CURRENT[2]))
        vad.set_sample_rate(RATE)
        prev, real = VADState.QUIET, []
        for i in range(0, len(data), CHUNK * 2):
            st = await vad.analyze_audio(data[i:i + CHUNK * 2])
            t_end = (i + CHUNK * 2) / (2 * RATE) - LEAD_S
            if st == VADState.SPEAKING and prev != VADState.SPEAKING and t_end > 0:
                real.append(round(t_end, 3))
            prev = st
        replay = [round(t, 3) for t in silero_fires(traces[(c.name, "silence")], *CURRENT)]
        if real != replay:
            bad += 1
            print(f"replay mismatch {c.name}: real {real} replay {replay}", flush=True)
    return bad


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--esc50", type=Path, default=ORCH.parent / "state" / "bargein" / "esc50")
    ap.add_argument("--backgrounds", default="silence,room")
    ap.add_argument("--pause-window-s", type=float, default=0.4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    t0 = time.monotonic()
    cases = build_cases(args.esc50, rng)
    print(f"{len(cases)} cases built in {time.monotonic() - t0:.0f} s", flush=True)
    traces = {}
    for bg in args.backgrounds.split(","):
        for c in cases:
            traces[(c.name, bg)] = await trace(c, bg, np.random.default_rng(11))
    print(f"traced in {time.monotonic() - t0:.0f} s", flush=True)
    mismatches = await verify_replay([c for c in cases if c.kind == "interruption"], traces)

    detectors = {f"silero:{c}/{s}/{v}": (lambda tr, c=c, s=s, v=v: silero_fires(tr, c, s, v)) for c, s, v in SETTINGS}
    cues = {"energy+15dB/2": lambda tr: energy_cue(tr), "energy+10dB/2": lambda tr: energy_cue(tr, rise_db=10.0),
            "silero>=0.15/2": lambda tr: silero_weak_cue(tr, 0.15, 2), "silero>=0.5/1": lambda tr: silero_weak_cue(tr, 0.5, 1)}
    rows = []
    for (name, bg), tr in traces.items():
        c = next(x for x in cases if x.name == name)
        start = c.onset_s if c.kind in ("interruption", "backchannel") else 0.0
        end = len(c.audio) / RATE + 0.3     # a fire up to 0.3 s after the sound ended still belongs to it
        for d, fn in detectors.items():
            fires = [t for t in fn(tr) if t <= end]
            rows.append({"case": name, "kind": c.kind, "bg": bg, "detector": d, "fired": bool(fires),
                         "latency_ms": round(1000 * (fires[0] - start)) if fires else None})
        confirm = [t for t in silero_fires(tr, *CURRENT) if t <= end]
        for cname, fn in cues.items():
            cue = [t for t in fn(tr) if t <= end]
            stops = [*cue[:1], *confirm[:1]]
            if not stops:
                rows.append({"case": name, "kind": c.kind, "bg": bg, "detector": f"pause:{cname}", "fired": False,
                             "latency_ms": None, "pause_ms": 0, "confirmed": False})
                continue
            first = min(stops)
            confirmed = [t for t in confirm if first <= t <= first + args.pause_window_s]
            # a pause the current Silero setting does not confirm lasts until the window runs out, then the reply resumes
            pause_ms = round(1000 * ((confirmed[0] if confirmed else first + args.pause_window_s) - first))
            rows.append({"case": name, "kind": c.kind, "bg": bg, "detector": f"pause:{cname}", "fired": True,
                         "latency_ms": round(1000 * (first - start)), "confirmed": bool(confirmed),
                         "pause_ms": pause_ms})
    with (args.out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    lines = [f"# Barge-in bench {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"{sum(c.kind == 'interruption' for c in cases)} interruptions ({len(PHRASES)} phrases x {len(VOICES)} voices), "
             f"{sum(c.kind == 'backchannel' for c in cases)} backchannels, "
             f"{sum(c.kind.startswith('control') for c in cases)} ESC-50 controls, {sum(c.kind == 'noise' for c in cases)} noises; "
             f"backgrounds {args.backgrounds}; replay vs analyzer mismatches: {mismatches}.", ""]
    kinds = sorted({c.kind for c in cases if c.kind.startswith("control")})
    for bg in args.backgrounds.split(","):
        lines += [f"## Background: {bg}", "",
                  "| detector | interruption: median / p90 / max ms, <=300 ms | backchannel fired | controls fired | noise fired | controls by kind (fired/n) |",
                  "|---|---|---|---|---|---|"]
        for d in [*detectors, *(f"pause:{k}" for k in cues)]:
            rs = [r for r in rows if r["bg"] == bg and r["detector"] == d]
            lat = [r["latency_ms"] for r in rs if r["kind"] == "interruption" and r["latency_ms"] is not None]
            miss = sum(1 for r in rs if r["kind"] == "interruption" and not r["fired"])
            def frac(k):
                sel = [r for r in rs if r["kind"] == k]
                return f"{sum(r['fired'] for r in sel)}/{len(sel)}"
            ctl = [r for r in rs if r["kind"].startswith("control")]
            by = ", ".join(f"{k.split(':')[1]} {frac(k)}" for k in kinds if any(r['fired'] for r in rs if r['kind'] == k))
            within = sum(x <= 300 for x in lat)
            lines.append(f"| {d} | {statistics.median(lat):.0f} / {pct(lat, 90)} / {max(lat)}, {within}/{len(lat)}"
                         f"{f', missed {miss}' if miss else ''} | {frac('backchannel')} | "
                         f"{sum(r['fired'] for r in ctl)}/{len(ctl)} | {frac('noise')} | {by} |")
            if d.startswith("pause:"):
                ps = [r for r in ctl if r["fired"]]
                if ps:
                    lines.append(f"|  ↳ on controls: {len(ps)} pauses, {sum(r['confirmed'] for r in ps)} of them "
                                 f"confirmed into an interrupt, the rest resumed after a median "
                                 f"{statistics.median(r['pause_ms'] for r in ps if not r['confirmed']) if any(not r['confirmed'] for r in ps) else 0:.0f} ms | | | | | |")
        lines.append("")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
