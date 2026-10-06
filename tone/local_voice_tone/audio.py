"""Audio in, speech segments and energy out: the pause structure of one utterance, on the CPU, with no model.

The orchestrator runs Silero VAD already and may pass its speech segments; without them an energy detector stands
in: frame RMS in dB, speech above max(noise floor + 12 dB, loudest − 35 dB), gaps shorter than a pause bridged, blips
dropped. A pause is a silence of at least 250 ms inside the speech (the conventional threshold); silence before the first word and after the last is not a pause.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIN_S, HOP_S = 0.025, 0.010


def pcm16_to_float(pcm: bytes | np.ndarray) -> np.ndarray:
    """PCM int16 little-endian mono (protocol v1's mic format) or an array, to float64 in [-1, 1)."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float64) / 32768.0
    x = np.asarray(pcm)
    if x.dtype == np.int16:
        return x.astype(np.float64) / 32768.0
    return x.astype(np.float64).reshape(-1)


def frame_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame centres (s) and RMS level (dBFS) on 25 ms windows every 10 ms."""
    n, h = int(WIN_S * sr), int(HOP_S * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    db = 20.0 * np.log10(rms + 1e-10)
    centres = (np.arange(len(db)) * h + n / 2) / sr
    return centres, db


def energy_segments(x: np.ndarray, sr: int, *, min_pause_s: float = 0.25, min_speech_s: float = 0.06
                    ) -> list[tuple[float, float]]:
    """Speech segments (start, end) in seconds from frame energy."""
    t, db = frame_db(x, sr)
    if len(db) == 0 or db.max() < -80:
        return []
    peak = np.percentile(db, 99)
    # Capped at 10 dB under the peak: with no silence in the clip the "noise floor" is speech itself, and the
    # uncapped threshold would rise above every frame (found by test_level_tracks_gain, 2026-10-05).
    thr = min(max(np.percentile(db, 10) + 12.0, peak - 35.0), peak - 10.0)
    return clean_segments(_runs(db > thr, t), min_pause_s=min_pause_s, min_speech_s=min_speech_s)


def _runs(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    out, start = [], None
    half = HOP_S / 2
    for i, on in enumerate(mask):
        if on and start is None:
            start = t[i] - half
        elif not on and start is not None:
            out.append((max(0.0, start), t[i - 1] + half))
            start = None
    if start is not None:
        out.append((max(0.0, start), t[-1] + half))
    return out


def clean_segments(segs: list[tuple[float, float]], *, min_pause_s: float = 0.25, min_speech_s: float = 0.06
                   ) -> list[tuple[float, float]]:
    """Bridge gaps shorter than a pause, then drop blips shorter than `min_speech_s`."""
    merged: list[list[float]] = []
    for a, b in sorted(segs):
        if merged and a - merged[-1][1] < min_pause_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_speech_s]


@dataclass
class Pauses:
    speech_s: float
    region_s: float          # first word to last word
    count: int
    total_s: float
    longest_s: float

    @property
    def ratio(self) -> float:
        return self.total_s / self.region_s if self.region_s > 0 else 0.0


def pauses(segs: list[tuple[float, float]]) -> Pauses:
    if not segs:
        return Pauses(0.0, 0.0, 0, 0.0, 0.0)
    gaps = [b[0] - a[1] for a, b in zip(segs, segs[1:])]
    return Pauses(speech_s=sum(b - a for a, b in segs), region_s=segs[-1][1] - segs[0][0], count=len(gaps),
                  total_s=sum(gaps), longest_s=max(gaps, default=0.0))


def speech_level_db(x: np.ndarray, sr: int, segs: list[tuple[float, float]]) -> tuple[float, float]:
    """Median and p90−p10 spread of the frame level inside speech."""
    t, db = frame_db(x, sr)
    inside = np.zeros(len(t), dtype=bool)
    for a, b in segs:
        inside |= (t >= a) & (t <= b)
    v = db[inside]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.median(v)), float(np.percentile(v, 90) - np.percentile(v, 10))
