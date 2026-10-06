"""One utterance's delivery, measured: the raw values every hint is computed from, and that get logged as they are
(keep raw feature values, not only labels, so they can be re-interpreted later)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import audio, pitch, text


@dataclass
class Features:
    duration_s: float
    speech_s: float
    region_s: float
    pause_count: int
    pause_s: float
    pause_ratio: float          # pause time inside the speech region / the region
    longest_pause_s: float
    f0_median_st: float         # semitones re 100 Hz
    f0_range_st: float          # p90 − p10, semitones
    voiced_s: float
    level_db: float             # median frame level inside speech, dBFS (device-dependent: baselines are per channel)
    level_range_db: float
    words: int
    syllables: int
    rate_sps: float             # syllables per second of speech (articulation rate)
    rate_wpm: float             # words per minute over the speech region, pauses included
    fillers: int
    repetitions: int
    cutoffs: int
    discourse: int

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else (round(v, 4) if isinstance(v, float) else v))
                for k, v in asdict(self).items()}


def extract(pcm, transcript: str, *, sample_rate: int = 16000, f0: str = "praat",
            speech_segments: list[tuple[float, float]] | None = None, min_pause_s: float = 0.25) -> Features:
    x = audio.pcm16_to_float(pcm)
    if speech_segments is not None:
        segs = audio.clean_segments([(float(a), float(b)) for a, b in speech_segments], min_pause_s=min_pause_s)
    else:
        segs = audio.energy_segments(x, sample_rate, min_pause_s=min_pause_s)
    p = audio.pauses(segs)
    if segs:
        a, b = int(segs[0][0] * sample_rate), int(segs[-1][1] * sample_rate)
        hz = pitch.BACKENDS[f0](x[a:b], sample_rate) if b - a > sample_rate * 0.1 else np.array([])
    else:
        hz = np.array([])
    med, rng, voiced = pitch.summarise(hz)
    level, level_rng = audio.speech_level_db(x, sample_rate, segs)
    tc = text.count(transcript)
    return Features(
        duration_s=len(x) / sample_rate, speech_s=p.speech_s, region_s=p.region_s, pause_count=p.count,
        pause_s=p.total_s, pause_ratio=p.ratio, longest_pause_s=p.longest_s, f0_median_st=med, f0_range_st=rng,
        voiced_s=voiced, level_db=level, level_range_db=level_rng, words=tc.words, syllables=tc.syllables,
        rate_sps=tc.syllables / p.speech_s if p.speech_s > 0 else float("nan"),
        rate_wpm=tc.words / p.region_s * 60.0 if p.region_s > 0 else float("nan"),
        fillers=tc.fillers, repetitions=tc.repetitions, cutoffs=tc.cutoffs, discourse=tc.discourse)
