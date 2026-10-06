"""Where the time in a spoken reply goes (CPU, no models): for each WAV (24 kHz mono PCM16, as tests/e2e/test_turn.py
saves the reply audio), its length, the silence before the first and after the last speech, and every silence inside
it of at least --gap seconds. Speech is a 20 ms frame whose RMS is within --db (40) dB of the file's loudest frame.

Usage:  .venv/bin/python tools/speech_gaps.py ../state/orchestrator-e2e/<stamp>/audio/*.wav [--gap 0.35]
"""
from __future__ import annotations

import argparse
import json
import wave

import numpy as np

FRAME_S = 0.02


def gaps(path: str, min_gap: float, db: float = 40.0) -> dict:
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    n = int(FRAME_S * rate)
    frames = a[: len(a) // n * n].reshape(-1, n) if len(a) >= n else np.zeros((0, n), np.float32)
    rms = np.sqrt((frames ** 2).mean(axis=1)) if len(frames) else np.zeros(0)
    if not len(rms) or rms.max() <= 0:
        return {"file": path, "length_s": round(len(a) / rate, 2), "speech": False}
    loud = rms >= rms.max() * 10 ** (-db / 20)
    idx = np.flatnonzero(loud)
    first, last = idx[0], idx[-1]
    inner = []
    run = 0
    for k in range(first, last + 1):
        if loud[k]:
            if run * FRAME_S >= min_gap:
                inner.append(round(run * FRAME_S, 2))
            run = 0
        else:
            run += 1
    length = len(a) / rate
    return {"file": path, "length_s": round(length, 2), "lead_s": round(first * FRAME_S, 2),
            "tail_s": round(length - (last + 1) * FRAME_S, 2), "gaps": inner, "gaps_s": round(sum(inner), 2),
            "speech_span_s": round((last + 1 - first) * FRAME_S, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--gap", type=float, default=0.35)
    ap.add_argument("--db", type=float, default=40.0)
    args = ap.parse_args()
    for p in args.wavs:
        print(json.dumps(gaps(p, args.gap, args.db)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
