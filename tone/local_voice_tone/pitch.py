"""Fundamental frequency, behind a config-selected adapter (swap models without code changes).

`praat` (default): Praat's autocorrelation tracker through Parselmouth, the phonetics reference; about 2 ms for a
4 s utterance on this Mac (measured 2026-10-05). `pyin`: librosa's probabilistic YIN, ISC licence, slower, and its
first call pays a numba compile. Both return F0 in Hz for voiced 10 ms frames only.

Pitch is summarised in semitones, which compare across a voice's range: the level as the median re 100 Hz, the range
as p90 − p10. Frames more than an octave and a half from the median are dropped first: autocorrelation trackers
occasionally jump an octave, and a jump would otherwise widen the range.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

FLOOR_HZ, CEILING_HZ, STEP_S = 60.0, 500.0, 0.01


def f0_praat(x: np.ndarray, sr: int) -> np.ndarray:
    import parselmouth
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=STEP_S, pitch_floor=FLOOR_HZ, pitch_ceiling=CEILING_HZ)
    f = p.selected_array["frequency"]
    return f[f > 0]


def f0_pyin(x: np.ndarray, sr: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as e:  # the optional extra: uv sync --extra pyin
        raise RuntimeError("f0 = 'pyin' needs librosa (install the tone package's `pyin` extra)") from e
    f, voiced, _ = librosa.pyin(x.astype(np.float32), fmin=FLOOR_HZ, fmax=CEILING_HZ, sr=sr,
                                frame_length=1024, hop_length=int(STEP_S * sr))
    f = f[voiced & np.isfinite(f)]
    return f.astype(np.float64)


BACKENDS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {"praat": f0_praat, "pyin": f0_pyin}


def semitones(hz: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(hz / 100.0)


def summarise(f0_hz: np.ndarray) -> tuple[float, float, float]:
    """(median semitones re 100 Hz, p90 − p10 range in semitones, voiced seconds); NaNs when too little is voiced."""
    if len(f0_hz) < 10:
        return float("nan"), float("nan"), len(f0_hz) * STEP_S
    st = semitones(f0_hz)
    med = float(np.median(st))
    st = st[np.abs(st - med) <= 18.0]
    return float(np.median(st)), float(np.percentile(st, 90) - np.percentile(st, 10)), len(f0_hz) * STEP_S
