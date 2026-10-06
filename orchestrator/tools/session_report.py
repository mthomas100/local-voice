"""What one recorded session says about the echo and the voice, by machine, on the CPU, with no model (2026-10-05).

    .venv/bin/python tools/session_report.py ../state/recordings/<session id> [--turn-log FILE ...] [--no-clips]

The input is a folder local_voice/recorder.py wrote (mic.wav, playback.wav, events.jsonl, meta.json, all on one clock).
Why: in an early live test about a quarter of the voice turns were the agent's own words, the last ~10 s of one reply
heard again 6-59 s after the replies ended, two of them cutting a reply; nothing showed what played them (the person
may have read the lines aloud), and the voice laughed, swung in pitch and slurred, but no audio was kept. A recorded
session (`./run.sh --record`) answers that; this reads it four ways, and nobody has to listen (a project rule):

(a) echo turns: tools/echo_turns.py's detector on this session's turns (the turn log; the sentences each TTS
    generation said and when the bot was speaking come from events.jsonl): user turns whose words are mostly the
    agent's own recent words.
(b) live echo, per reply: where the reply's own audio comes back in the microphone and how loud. GCC-PHAT (the
    cross-spectrum whitened, so the peak is a delay rather than the loudest band) between playback.wav at 16 kHz and
    mic.wav, lags 0-1 s, 100 Hz-7 kHz; the delay is the peak (refined between samples), "found" needs the peak to
    stand ECHO_Z_MIN robust standard deviations above the rest; the level is the least-squares gain of the delayed
    playback in the mic at that lag, in dB relative to the playback (the direct path: a room's reverberant tail is not
    counted), and its share of the microphone's energy over the reply. The delay is the round trip as the server
    sees it (the recorder's clock): downlink, the client's playout, the room, the microphone, the uplink.
(c) replays: microphone windows (2.4 s, every 0.26 s, with sound in most of their frames) whose audio matches
    playback from 2 s to 5 min earlier, by a coarse spectral fingerprint (Haitsma and Kalker's: 32 bits per frame, the
    signs of the energy differences between 33 bands from 300 Hz to 3 kHz and frame to frame; insensitive to level
    and to a smooth colouring of the sound), on 512 ms frames every 128 ms. A window matches when under REPLAY_BER of
    its bits differ from the best playback window; runs of matching windows at one lag are one replay, with its
    source time and lag (to about 0.1 s). Measured on planted signals (tests/test_session_report.py, 2026-10-05): a
    replay scores 0.12 dry and 0.15-0.19 through a synthetic room (RT60 0.3-0.5 s, at -10 and -20 dB), the same
    words read aloud in another voice 0.40-0.41 and other speech 0.40: with 32 ms frames a room's tail smeared the
    frame-to-frame differences (0.36) to within reach of the 0.45 null. Reading a line aloud matches the words, not
    the audio: the person's voice is other sound,
    so an echo turn with no replay under it was said anew (read aloud, or the person's own words that happen to
    repeat the agent's), while one with a replay under it was the agent's own audio played again.
(d) voice clips: every TTS generation located in playback.wav by its fingerprint (the generation's own first 0.1 s,
    recorded by services/tts.py: the bytes the output sent, so the match is exact) and cut from where it starts to
    where the next one starts or the sending stopped, with its text, in tools/tts_quality_bench.py's layout
    (<dir>/clips/session/<id>-0.wav and .json, run.json): the bench's own stages take that folder as a run,
    `tts_quality_bench.py asr|score|report --run-dir <dir>/clips` (ASR on the GPU, the scorers CPU torch: later).

Writes report.md and report.json (every number) in the recording's folder.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

TOOLS = Path(__file__).resolve().parent
ORCH = TOOLS.parent
for _p in (str(ORCH), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import echo_turns  # noqa: E402
from echo_mixer import resample_24k_to_16k  # noqa: E402

MIC_RATE, PLAY_RATE = 16000, 24000
# (b) live echo
ECHO_MAX_LAG_S = 1.0
ECHO_BAND_HZ = (100.0, 7000.0)
ECHO_Z_MIN = 8.0                 # unrelated sound: the largest of 16,001 lags sits about 4-5 robust SDs up
ECHO_MIN_S = 0.3                 # replies shorter than this are not measured
# (c) replays
FP_WIN, FP_HOP = 8192, 2048      # 512 ms frames every 128 ms at 16 kHz (module docstring: why not 32 ms)
FP_EDGES = np.geomspace(300.0, 3000.0, 34)   # 33 bands, 32 bits
REPLAY_WIN = 16                  # frames per window: 2.4 s
REPLAY_STEP = 2                  # a window every 0.26 s
REPLAY_MIN_LAG_S, REPLAY_MAX_LAG_S = 2.0, 300.0
REPLAY_BER = 0.30                # a window matches below this bit error rate
ACTIVE_SHARE = 0.75              # a window needs sound in this share of its frames (mic and playback)
ACTIVE_DBFS = -50.0              # a frame has sound above this RMS (and, in the mic, 10 dB over its floor)
# (d) clips
CLIP_KIND, CLIP_SETTING = "session", "session"


# ------------------------------------------------------------------------------------------------ reading

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: not mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy(), w.getframerate()


@dataclass
class Recording:
    dir: Path
    meta: dict[str, Any]
    events: list[dict[str, Any]]
    mic: np.ndarray                      # float32, 16 kHz
    play_pcm: np.ndarray                 # int16, 24 kHz: exactly what was sent
    _play16: np.ndarray | None = field(default=None, repr=False)

    @property
    def play16(self) -> np.ndarray:
        """playback.wav at 16 kHz on the same clock (echo_mixer's resampler: sample 3m at 24 kHz is sample 2m)."""
        if self._play16 is None:
            self._play16 = resample_24k_to_16k(self.play_pcm.astype(np.float32) / 32768.0)
        return self._play16

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("ev") == kind]

    @property
    def wall0(self) -> float:
        return float(self.meta.get("wall0") or 0.0)


def load(folder: Path) -> Recording:
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines() if line.strip()]
    events.sort(key=lambda e: e.get("t", 0.0))
    mic, mr = read_wav(folder / "mic.wav")
    play, pr = read_wav(folder / "playback.wav")
    if (mr, pr) != (MIC_RATE, PLAY_RATE):
        raise ValueError(f"{folder}: rates {mr}/{pr}, expected {MIC_RATE}/{PLAY_RATE}")
    return Recording(folder, meta, events, mic.astype(np.float32) / 32768.0, play)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def db(v: float) -> float:
    return 20.0 * float(np.log10(max(v, 1e-12)))


# ---------------------------------------------------------------------------------------- (b) live echo

def gcc_phat(x: np.ndarray, y: np.ndarray, max_lag: int, *, rate: int = MIC_RATE,
             band: tuple[float, float] = ECHO_BAND_HZ) -> tuple[float, float, np.ndarray]:
    """Where `x` sits in `y` at lags 0..max_lag (y[n + lag] ~ x[n]): (the peak, refined by a parabola through it and
    its neighbours; its robust z: (peak - median) / (1.4826 MAD); the correlation over the lags)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    g = np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    keep = (f >= band[0]) & (f <= band[1])
    mag = np.abs(g)
    g = np.where(keep, g / np.maximum(mag, 1e-20), 0.0)
    r = np.fft.irfft(g, n)[: max_lag + 1]
    k = int(np.argmax(r))
    lag = float(k)
    if 0 < k < len(r) - 1:
        a, b, c = r[k - 1], r[k], r[k + 1]
        den = a - 2 * b + c
        if den < 0:
            lag += 0.5 * (a - c) / den
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826
    return float(lag), (float(r[k]) - med) / mad if mad > 0 else 0.0, r


def echo_of(x: np.ndarray, y: np.ndarray, *, max_lag_s: float = ECHO_MAX_LAG_S, rate: int = MIC_RATE) -> dict:
    """The echo of playback `x` in microphone `y` (y starts where x starts and runs max_lag_s longer)."""
    max_lag = int(max_lag_s * rate)
    lag, z, _ = gcc_phat(x, y, max_lag, rate=rate)
    k = int(round(lag))
    yy = y[k:k + len(x)]
    xx = x[:len(yy)]
    ex = float(np.dot(xx, xx))
    g = float(np.dot(xx, yy)) / ex if ex > 0 else 0.0
    ey = float(np.dot(yy, yy))
    return {"found": bool(z >= ECHO_Z_MIN and g > 0), "delay_ms": round(1000.0 * lag / rate, 1), "z": round(z, 1),
            "level_db": round(db(abs(g)), 1) if g else None,
            "mic_share_db": round(10 * np.log10(max(g * g * ex / ey, 1e-12)), 1) if ey > 0 and g else None}


def reply_spans(rec: Recording) -> list[tuple[float, float]]:
    """(start, end) of each stretch the bot spoke, on the recording's clock: Pipecat's bot started/stopped events, or,
    without them, the stretches the output sent joined across gaps under 1 s."""
    spans, on = [], None
    for e in rec.of("bot"):
        if e.get("on") and on is None:
            on = e["t"]
        elif not e.get("on") and on is not None:
            spans.append((on, e["t"]))
            on = None
    if on is not None:
        spans.append((on, len(rec.play_pcm) / PLAY_RATE))
    if spans:
        return spans
    for e in rec.of("sent"):
        if spans and e["from_s"] - spans[-1][1] < 1.0:
            spans[-1] = (spans[-1][0], e["to_s"])
        else:
            spans.append((e["from_s"], e["to_s"]))
    return spans


def live_echo(rec: Recording) -> list[dict]:
    out = []
    p16, mic = rec.play16, rec.mic
    for i, (a_s, b_s) in enumerate(reply_spans(rec), 1):
        a, b = int(a_s * MIC_RATE), min(int(b_s * MIC_RATE), len(p16))
        x = p16[a:b]
        row: dict[str, Any] = {"reply": i, "from_s": round(a_s, 3), "to_s": round(b_s, 3)}
        if b - a < ECHO_MIN_S * MIC_RATE or rms(x) < 1e-4 or a >= len(mic):
            out.append({**row, "found": False, "why": "too short or silent, or no microphone then"})
            continue
        y = mic[a:b + int(ECHO_MAX_LAG_S * MIC_RATE)]
        if len(y) < len(x):
            y = np.concatenate([y, np.zeros(len(x) - len(y), np.float32)])
        out.append({**row, **echo_of(x, y)})
    return out


# ---------------------------------------------------------------------------------------------- (c) replays

def band_energies(x: np.ndarray, rate: int = MIC_RATE) -> tuple[np.ndarray, np.ndarray]:
    """(frames x 33 band energies, each frame's RMS in dBFS) for FP_WIN-sample Hann frames every FP_HOP samples."""
    x = np.asarray(x, np.float32)
    n = 1 + (len(x) - FP_WIN) // FP_HOP if len(x) >= FP_WIN else 0
    if n <= 0:
        return np.zeros((0, len(FP_EDGES) - 1)), np.zeros(0)
    win = np.hanning(FP_WIN).astype(np.float32)
    f = np.fft.rfftfreq(FP_WIN, 1.0 / rate)
    lo = np.searchsorted(f, FP_EDGES[:-1])
    hi = np.searchsorted(f, FP_EDGES[1:])
    E = np.zeros((n, len(lo)))
    level = np.zeros(n)
    block = max(1, (1 << 23) // FP_WIN)            # frames per block: a 30 min track at once would be gigabytes
    for s in range(0, n, block):
        m = min(n, s + block) - s
        idx = (s + np.arange(m))[:, None] * FP_HOP + np.arange(FP_WIN)[None, :]
        fr = x[idx]
        level[s:s + m] = 10 * np.log10(np.maximum(np.mean(fr.astype(np.float64) ** 2, axis=1), 1e-20))
        p = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
        c = np.concatenate([np.zeros((m, 1)), np.cumsum(p, axis=1)], axis=1)
        E[s:s + m] = c[:, hi] - c[:, lo]
    return E, level


def fingerprints(E: np.ndarray) -> np.ndarray:
    """One uint32 per frame (frame 0's is 0): bit m is the sign of (E[n,m] - E[n,m+1]) - (E[n-1,m] - E[n-1,m+1])."""
    out = np.zeros(len(E), np.uint32)
    if len(E) < 2:
        return out
    d = E[:, :-1] - E[:, 1:]
    bits = (d[1:] - d[:-1]) > 0
    out[1:] = (bits.astype(np.uint64) << np.arange(bits.shape[1], dtype=np.uint64)).sum(axis=1).astype(np.uint32)
    return out


def _windows_active(active: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(active.astype(np.int32))])
    return c[w:] - c[:-w] >= ACTIVE_SHARE * w


def find_replays(mic: np.ndarray, play16: np.ndarray, *, min_lag_s: float = REPLAY_MIN_LAG_S,
                 max_lag_s: float = REPLAY_MAX_LAG_S, max_ber: float = REPLAY_BER) -> list[dict]:
    """Microphone windows whose fingerprint matches playback from min_lag_s to max_lag_s earlier (module docstring),
    joined into replays."""
    Em, lm = band_energies(mic)
    Ep, lp = band_energies(play16)
    fm, fp = fingerprints(Em), fingerprints(Ep)
    W = REPLAY_WIN
    if len(fm) < W + 1 or len(fp) < W + 1:
        return []
    floor = float(np.percentile(lm, 10)) if len(lm) else -120.0
    mic_ok = _windows_active((lm > ACTIVE_DBFS) & (lm > floor + 10.0), W)
    play_ok = _windows_active(lp > ACTIVE_DBFS, W)
    sec = FP_HOP / MIC_RATE
    lo_lag, hi_lag = int(np.ceil(min_lag_s / sec)), int(max_lag_s / sec)
    hits = []
    for s in range(1, len(fm) - W, REPLAY_STEP):
        if not mic_ok[s]:
            continue
        lo, hi = max(1, s - hi_lag), min(s - lo_lag, len(fp) - W)
        if hi <= lo:
            continue
        acc = np.zeros(hi - lo, np.int64)
        for k in range(W):
            acc += np.bitwise_count(fp[lo + k:hi + k] ^ fm[s + k])
        ber = acc / (32.0 * W)
        ber[~play_ok[lo:hi]] = 1.0
        o = int(np.argmin(ber))
        if ber[o] < max_ber:
            hits.append((s, lo + o, float(ber[o])))
    replays: list[dict] = []
    for s, o, b in hits:
        lag = s - o
        last = replays[-1] if replays else None
        if last and s - last["_s"] <= 2 * REPLAY_STEP and abs(lag - last["_lag"]) <= 1:
            last.update(_s=s, mic_to_s=round((s + W) * sec, 2), source_to_s=round((o + W) * sec, 2),
                        ber=round(min(last["ber"], b), 3), windows=last["windows"] + 1)
            last["_lags"].append(lag)
        else:
            replays.append({"_s": s, "_lag": lag, "_lags": [lag], "mic_from_s": round(s * sec, 2),
                            "mic_to_s": round((s + W) * sec, 2), "source_from_s": round(o * sec, 2),
                            "source_to_s": round((o + W) * sec, 2), "ber": round(b, 3), "windows": 1})
    for r in replays:
        lag = int(round(float(np.median(r.pop("_lags")))))
        r["lag_s"] = round(lag * sec, 2)
        del r["_s"], r["_lag"]
        # the extent: frames whose own bits agree at this lag (5-frame median), not the first window's start
        a, b = int(round(r["mic_from_s"] / sec)), int(round(r["mic_to_s"] / sec))
        k = np.arange(max(a, lag + 1), min(b, len(fp) + lag, len(fm)))
        if len(k) >= 5:
            err = np.bitwise_count(fm[k] ^ fp[k - lag]) / 32.0
            sm = np.array([np.median(err[max(0, i - 2):i + 3]) for i in range(len(err))])
            good = np.flatnonzero(sm < max_ber)
            if good.size:                     # frame centres, half a hop either side
                c = lambda i: i * sec + FP_WIN / MIC_RATE / 2        # noqa: E731
                lo, hi = int(k[good[0]]), int(k[good[-1]])
                r.update(mic_from_s=round(c(lo) - sec / 2, 2), mic_to_s=round(c(hi) + sec / 2, 2),
                         source_from_s=round(c(lo - lag) - sec / 2, 2), source_to_s=round(c(hi - lag) + sec / 2, 2))
    return replays


# --------------------------------------------------------------------------------------------- (a) turns

def turn_log_files(rec: Recording) -> list[Path]:
    """The turn log days this recording spans, from the config it ran with (brain.turn_log_dir)."""
    conf = rec.meta.get("config") or {}
    d = (conf.get("brain") or {}).get("turn_log_dir")
    if not d:
        return []
    base = Path(rec.meta.get("config_path") or ORCH / "config.yaml").parent
    p = Path(d).expanduser()
    p = p if p.is_absolute() else (base / p).resolve()
    start = rec.wall0
    end = start + float(rec.meta.get("duration_s") or len(rec.mic) / MIC_RATE)
    days = sorted({datetime.fromtimestamp(t).strftime("%Y-%m-%d") for t in (start, end)})
    return [p / f"{day}.jsonl" for day in days if (p / f"{day}.jsonl").exists()]


def echo_turn_rows(rec: Recording, turn_logs: list[Path], replays: list[dict]) -> list[dict]:
    sid = rec.meta.get("session")
    turns = [json.loads(line) for p in turn_logs for line in p.read_text().splitlines() if line.strip()]
    turns = sorted((t for t in turns if t.get("type") == "turn" and t.get("session") == sid
                    and t.get("input", "voice") == "voice"), key=lambda t: t["t_start"])
    if not turns:
        return []
    echo = (rec.meta.get("config") or {}).get("echo") or {}
    said = [(e["wall"], e["text"]) for e in rec.of("tts") if e.get("text")]
    spans = [(rec.wall0 + a, rec.wall0 + b) for a, b in reply_spans(rec)]
    rows = echo_turns.detect(turns, window_s=float(echo.get("window_s", 300.0)),
                             min_words=int(echo.get("min_words", 3)), min_share=float(echo.get("min_share", 0.6)),
                             said=said, speaking=spans)
    for row in rows:
        t = datetime.fromisoformat(row["t_start"]).timestamp() - rec.wall0
        row["t_s"] = round(t, 2)
        under = [r for r in replays if r["mic_from_s"] - 2.5 <= t <= r["mic_to_s"] + 0.5]
        row["replay"] = under[0] if under else None
    return rows


# ---------------------------------------------------------------------------------------------- (d) clips

def locate(play_bytes: bytes, fp: bytes, start: int, stop: int | None = None) -> int | None:
    """The first sample at or after `start` where the bytes `fp` begin in the playback, aligned to a sample."""
    pos = 2 * max(0, start)
    while True:
        i = play_bytes.find(fp, pos, None if stop is None else 2 * stop + len(fp))
        if i < 0:
            return None
        if i % 2 == 0:
            return i // 2
        pos = i + 1


def sent_stretches(rec: Recording) -> list[tuple[int, int]]:
    out = [tuple(e["samples"]) for e in rec.of("sent") if e.get("samples")]
    return sorted((int(a), int(b)) for a, b in out)


def clip_plan(rec: Recording) -> list[dict]:
    """Every TTS generation, in order, with where it starts in playback.wav and how much of it went out."""
    play = rec.play_pcm.tobytes()
    gens = sorted(rec.of("tts"), key=lambda e: (e["t"], e.get("gen", 0)))
    stretches = sent_stretches(rec)
    plan, prev = [], 0
    for e in gens:
        fp = base64.b64decode(e.get("fp") or "")
        at = int(e.get("fp_at") or 0)
        row = {"gen": e.get("gen"), "text": e.get("text", ""), "status": e.get("status"), "samples": e.get("samples"),
               "seams": e.get("seams") or [], "t_request": e["t"], "first_audio_s": e.get("first_audio_s"),
               "start": None}
        if len(fp) >= 2:
            lo = max(prev, int((e["t"] - 0.25) * PLAY_RATE) + at)
            m = locate(play, fp, lo)
            if m is None and len(fp) > 480:                # cut early: its first 10 ms may still have gone out
                m = locate(play, fp[:480], lo)
            if m is not None:
                row["start"] = m - at
                prev = m + 1
        plan.append(row)
    starts = sorted(r["start"] for r in plan if r["start"] is not None)
    for r in plan:
        if r["start"] is None:
            continue
        a, want = r["start"], int(r["samples"] or 0)
        limit = next((x for x in starts if x > a), None)          # where the next generation starts
        # the generation's sent audio: from its start, through any gap the output left (a voice slower than real
        # time, a pause gate), until all of it went out, the next one starts, or the sending stopped (interrupted)
        pieces, got = [], 0
        for lo, hi in (stretches or [(a, a + want)]):
            if hi <= a:
                continue
            lo = max(lo, a)
            if limit is not None:
                hi = min(hi, limit)
            if hi <= lo:
                break
            k = min(hi - lo, want - got)
            pieces.append((lo, lo + k))
            got += k
            if got >= want:
                break
        r["pieces"] = pieces
        r["sent_samples"] = got
        r["complete"] = got >= want
    return plan


def write_clips(rec: Recording, out: Path) -> list[dict]:
    """The clips (module docstring (d)) in the voice bench's run layout, with run.json for its report."""
    import tts_quality_bench as bench

    plan = clip_plan(rec)
    sdir = out / CLIP_SETTING
    sdir.mkdir(parents=True, exist_ok=True)
    conf = rec.meta.get("config") or {}
    tts = conf.get("tts") or {}
    adapter = tts.get("adapter", "")
    items, rows = [], []
    for r in plan:
        if r["start"] is None or r["sent_samples"] <= 0:
            continue
        cid = f"g{int(r['gen'] or len(rows) + 1):04d}"
        pcm = np.concatenate([rec.play_pcm[a:b] for a, b in r["pieces"]])
        x = pcm.astype(np.float32) / 32768.0
        n = len(x)
        seams = [int(s) for s in r["seams"] if 0 < int(s) < n]
        gen = {"text": r["text"], "start": 0, "samples": n, "seams": seams, "chunk_samples": []}
        heard = bench.onset_s(x, PLAY_RATE)
        side = {"bench_version": bench.BENCH_VERSION, "item": cid, "kind": CLIP_KIND, "r": 0, "setting": CLIP_SETTING,
                "seed": None, "seeded": False, "sample_rate": PLAY_RATE, "samples": n, "duration_s": n / PLAY_RATE,
                "text": r["text"], "n_words": len(bench.normalise(r["text"])), "n_chars": len(r["text"]),
                "generations_n": 1, "llm_chars_first": 0, "llm_chars_first_one_per_sentence": 0,
                "first_chunk_s": r["first_audio_s"], "first_out_s": None, "lead_raw_s": None,
                "trim_onset_found": None, "lead_heard_s": heard if heard is not None else n / PLAY_RATE,
                "max_lead_raw_s": None, "hit_cap": False, "seams": seams, "controls": bench.control_points([gen]),
                "gen_starts": [0], "generations": [gen],
                "session": rec.meta.get("session"), "gen": r["gen"], "status": r["status"], "complete": r["complete"],
                "t_play_s": round(r["start"] / PLAY_RATE, 4), "t_request_s": round(r["t_request"], 4),
                "generated_samples": r["samples"], "sent_pieces": r["pieces"], "source": str(rec.dir)}
        with wave.open(str(sdir / f"{cid}-0.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(PLAY_RATE)
            w.writeframes(pcm.astype("<i2").tobytes())
        (sdir / f"{cid}-0.json").write_text(json.dumps(side, indent=1))
        items.append({"id": cid, "kind": CLIP_KIND, "text": r["text"], "sentences": [r["text"]], "turn": None,
                      "logged": False})
        rows.append({**{k: r[k] for k in ("gen", "text", "status", "complete", "sent_samples")},
                     "clip": f"{CLIP_SETTING}/{cid}-0.wav", "t_play_s": side["t_play_s"]})
    settings = [{"name": CLIP_SETTING, "adapter": adapter, "impl": ((tts.get("adapters") or {}).get(adapter) or {})
                 .get("impl", ""), "settings": {k: v for k, v in ((tts.get("adapters") or {}).get(adapter) or {}).items()
                                               if k != "impl"},
                  "stream": True, "min_words": int(((tts.get("group") or {}).get("min_words")) or 0),
                  "max_sentences": 1, "override": {}}]
    now = datetime.now().isoformat(timespec="seconds")
    (out / "run.json").write_text(json.dumps({"bench_version": bench.BENCH_VERSION, "reps": 1, "trim": {},
                                              "settings": settings, "items": items, "created": now, "updated": now,
                                              "source": str(rec.dir), "session": rec.meta.get("session")}, indent=1))
    return rows


# ---------------------------------------------------------------------------------------------- the report

def analyse(folder: Path, turn_logs: list[Path] | None = None, clips: bool = True) -> dict:
    rec = load(folder)
    replays = find_replays(rec.mic, rec.play16)
    logs = turn_log_files(rec) if turn_logs is None else [Path(p) for p in turn_logs]
    out = {"session": rec.meta.get("session"), "folder": str(rec.dir), "meta": {k: v for k, v in rec.meta.items()
                                                                              if k not in ("config",)},
           "mic_s": round(len(rec.mic) / MIC_RATE, 2), "playback_s": round(len(rec.play_pcm) / PLAY_RATE, 2),
           "turn_logs": [str(p) for p in logs], "echo_turns": echo_turn_rows(rec, logs, replays),
           "live_echo": live_echo(rec), "replays": replays,
           "counts": {k: len(rec.of(k)) for k in ("vad", "turn", "bot", "stt", "guard", "decision", "interruption",
                                                  "pause", "tts", "played_ms")}}
    out["clips"] = write_clips(rec, rec.dir / "clips") if clips else []
    (rec.dir / "report.json").write_text(json.dumps(out, indent=1, default=str))
    (rec.dir / "report.md").write_text(report_md(out, rec))
    return out


def _f(v: Any, nd: int = 1) -> str:
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def report_md(a: dict, rec: Recording) -> str:
    m, conf = rec.meta, rec.meta.get("config") or {}
    echo, tts = conf.get("echo") or {}, conf.get("tts") or {}
    voice = (tts.get("adapters") or {}).get(tts.get("adapter"), {}) or {}
    st = m.get("stats") or {}
    L = [f"# Session {a['session']}", ""]
    L.append(f"{m.get('client') or 'unknown'} ({m.get('transport')}, mic {m.get('mic')}), started "
             f"{m.get('started_wall')}, {a['mic_s']:.0f} s of microphone and {a['playback_s']:.0f} s of playback "
             f"recorded; stopped: {m.get('stop_reason')}. Echo guard {echo.get('guard')}, hold_for_words "
             f"{echo.get('hold_for_words')}, tail {echo.get('tail_s')} s; voice {tts.get('adapter')} "
             f"({voice.get('model')}, {voice.get('voice')}); commit "
             f"{((m.get('versions') or {}).get('commit') or 'unknown')[:7]}. Recorder: {st.get('mic_gaps', 0)} microphone gaps ({_f(st.get('mic_gap_s'), 2)} s), "
             f"{st.get('mic_resyncs', 0)} re-syncs, {st.get('dropped_items', 0)} items dropped. Every number below is "
             f"a machine measurement (tools/session_report.py); report.json has them all.")
    L += ["", "## (a) Echo turns", ""]
    rows = a["echo_turns"]
    if not a["turn_logs"]:
        L.append("No turn log found for this session (pass --turn-log).")
    else:
        n = sum(r["echo"] for r in rows)
        L.append(f"{n} of {len(rows)} voice turns are mostly the agent's own words (tools/echo_turns.py: >= "
                 f"{echo.get('min_words', 3)} words and >= {echo.get('min_share', 0.6)} of the turn, within "
                 f"{echo.get('window_s', 300)} s). Audio: a replay found under the turn (c) means the agent's own audio "
                 "was played again; none means the words were said anew (read aloud, or the person's own words).")
        L.append("")
        for r in rows:
            if r["echo"]:
                where = "during a reply" if r.get("bot_speaking_at_start") else f"{r.get('since_bot_stopped_s')} s after a reply"
                rp = r.get("replay")
                audio = (f"replay of {rp['source_from_s']:.1f} s, {rp['lag_s']:.1f} s earlier (BER {rp['ber']:.2f})"
                         if rp else "no replay under it: said anew")
                L.append(f"- turn {r['turn']} at {r['t_s']:.1f} s ({where}; {r['matched']}/{r['heard_words']} words, "
                         f"{r['lag_s']} s after they were generated): {r['user_text']!r} — {audio}")
    L += ["", "## (b) Live echo, per reply", ""]
    L.append("GCC-PHAT of each reply's playback against the microphone, lags 0-1 s: the delay is the round trip as the "
             "server sees it; the level is the delayed playback's gain in the microphone (direct path), dB relative to "
             f"the playback; found needs the peak {ECHO_Z_MIN:g} robust SDs over the rest.")
    L.append("")
    L.append("| reply | from (s) | to (s) | found | delay (ms) | level (dB) | share of mic (dB) | z |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in a["live_echo"]:          # the delay and level of a peak that is not echo are noise: report.json has them
        ok = r.get("found")
        L.append(f"| {r['reply']} | {r['from_s']:.2f} | {r['to_s']:.2f} | {'yes' if ok else 'no'} | "
                 f"{_f(r.get('delay_ms')) if ok else '–'} | {_f(r.get('level_db')) if ok else '–'} | "
                 f"{_f(r.get('mic_share_db')) if ok else '–'} | {_f(r.get('z'))} |")
    found = [r for r in a["live_echo"] if r.get("found")]
    if found:
        L.append("")
        L.append(f"Echo found in {len(found)} of {len(a['live_echo'])} replies: delay median "
                 f"{np.median([r['delay_ms'] for r in found]):.0f} ms, level median "
                 f"{np.median([r['level_db'] for r in found]):.1f} dB.")
    L += ["", "## (c) Replays", ""]
    if not a["replays"]:
        L.append(f"No microphone window matched playback from {REPLAY_MIN_LAG_S:g} s to {REPLAY_MAX_LAG_S / 60:g} min "
                 f"earlier (fingerprint BER < {REPLAY_BER}).")
    for r in a["replays"]:
        L.append(f"- microphone {r['mic_from_s']:.1f}-{r['mic_to_s']:.1f} s is playback {r['source_from_s']:.1f}-"
                 f"{r['source_to_s']:.1f} s again, {r['lag_s']:.1f} s later (best BER {r['ber']:.2f}, "
                 f"{r['windows']} windows)")
    L += ["", "## (d) Voice clips", ""]
    clips = a["clips"]
    if clips:
        cut = sum(1 for c in clips if not c["complete"])
        L.append(f"{len(clips)} generations cut from playback.wav into clips/ ({cut} not sent whole: interrupted). "
                 "Score them later with the voice bench's stages on that folder: `.venv/bin/python "
                 f"tools/tts_quality_bench.py asr|score|report --run-dir {rec.dir / 'clips'}` (asr needs the GPU, "
                 "score CPU torch, report neither).")
        L.append("")
        for c in clips[:400]:
            L.append(f"- {c['clip']} at {c['t_play_s']:.2f} s{'' if c['complete'] else ' (cut)'}: {c['text']!r}")
    else:
        L.append("No generation was located in playback.wav (or --no-clips).")
    L += ["", "## Events", "", ", ".join(f"{k} {v}" for k, v in a["counts"].items())]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", type=Path, help="a recording folder (local_voice/recorder.py)")
    ap.add_argument("--turn-log", type=Path, action="append", help="turn log file(s) (default: from the config it "
                                                                   "ran with, the days it spans)")
    ap.add_argument("--no-clips", action="store_true", help="skip cutting the voice clips")
    a = ap.parse_args(argv)
    out = analyse(a.recording, a.turn_log, clips=not a.no_clips)
    n_echo = sum(r["echo"] for r in out["echo_turns"])
    found = [r for r in out["live_echo"] if r.get("found")]
    print(f"{out['session']}: {n_echo} echo turns of {len(out['echo_turns'])}; live echo in {len(found)} of "
          f"{len(out['live_echo'])} replies; {len(out['replays'])} replays; {len(out['clips'])} clips")
    print(f"report: {Path(out['folder']) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
