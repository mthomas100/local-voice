"""From measurements to one bracketed line the LLM may read, or nothing.

The line states measured differences from the person's own usual, in fixed words (faster, slower, higher, lower,
louder, quieter, more, less), never an emotion and never a guess at one (send measurements
and deviations, not emotion names; rate-limit; never let it override the words). Only features past the threshold
appear, at most three, largest first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .baseline import Baseline
from .features import Features

PREFIX = "[delivery vs usual, automatic and uncertain: "


@dataclass(frozen=True)
class Spec:
    attr: str
    floor: float               # the smallest scale the z is divided by, in the feature's own unit
    up: str
    down: str | None           # None: only the upward direction is reported


# The z-scored features. Floors: the size of ordinary utterance-to-utterance variation within one speaker, so a
# near-constant baseline (a short history, or a synthetic voice) cannot turn a small difference into a large z.
# Calibrated 2026-10-05 on `say` speech with planted deviations (tone/scripts/calibrate.py; results in tone/README.md): the
# first guesses (0.3, 0.8, 1.0, 1.5, 0.04, 0.25) let 8 of 20 ordinary utterances past 1.5, these let 4.
SPECS: dict[str, Spec] = {
    "pace": Spec("rate_sps", 0.4, "faster", "slower"),
    "pitch": Spec("f0_median_st", 1.0, "higher", "lower"),
    "pitch range": Spec("f0_range_st", 1.5, "wider", "narrower"),
    "volume": Spec("level_db", 2.0, "louder", "quieter"),
    "pausing": Spec("pause_ratio", 0.08, "more", "less"),
    "longest pause": Spec("longest_pause_s", 0.3, "longer", None),
}
# Words a hint may contain beyond numbers and units; a test holds every emitted hint to this list.
VOCABULARY = frozenset(
    {"delivery", "vs", "usual", "automatic", "and", "uncertain", "pace", "pitch", "range", "volume", "pausing",
     "longest", "pause", "a", "fillers", "in", "words", "about", "of", "the", "time", "none", "s", "sd", "1"}
    | {w for s in SPECS.values() for w in (s.up, s.down) if w})


@dataclass
class Deviation:
    name: str
    z: float
    value: float
    usual: float
    text: str


def eligible(f: Features, *, min_speech_s: float, min_words: int) -> set[str]:
    """Which features this utterance can speak for: short or mostly unvoiced ones say little."""
    ok = set()
    if f.speech_s >= min_speech_s and f.words >= min_words:
        ok |= {"pace", "volume", "fillers"}
        if f.voiced_s >= 0.5 and not math.isnan(f.f0_median_st):
            ok |= {"pitch", "pitch range"}
        if f.region_s >= 2.0:
            ok |= {"pausing", "longest pause"}
    return ok


def _describe(name: str, spec: Spec, z: float, value: float, usual: float) -> str:
    word = spec.up if z > 0 else spec.down
    if name == "pausing":
        return f"pausing {word} ({value:.0%} of the time, usual {usual:.0%})"
    if name == "longest pause":
        return f"a {value:.1f} s pause (usual longest {usual:.1f} s)"
    return f"{name} {word} ({z:+.1f} sd)"


def deviations(f: Features, b: Baseline, ok: set[str], *, z_threshold: float, min_samples: int,
               long_pause_s: float) -> list[Deviation]:
    out = []
    for name, spec in SPECS.items():
        if name not in ok:
            continue
        value = getattr(f, spec.attr)
        z, usual, n = b.z(spec.attr, value, spec.floor)
        if n < min_samples or math.isnan(z) or abs(z) < z_threshold:
            continue
        if z < 0 and spec.down is None:
            continue
        if name == "longest pause" and value < long_pause_s:
            continue
        out.append(Deviation(name, z, value, usual, _describe(name, spec, z, value, usual)))
    if "fillers" in ok:
        rate, n = b.filler_rate()
        if n >= min_samples and f.fillers >= 2:
            expected = rate * f.words
            # Counts, not a continuous measure: a Poisson z against this many words at the usual rate.
            z = (f.fillers - expected) / math.sqrt(max(expected, 0.5))
            if z >= z_threshold:
                usual = f"usual about 1 in {round(1 / rate)}" if rate > 0 else "usual none"
                out.append(Deviation("fillers", z, f.fillers, rate,
                                     f"fillers more ({f.fillers} in {f.words} words, {usual})"))
    return sorted(out, key=lambda d: -abs(d.z))


def render(devs: list[Deviation], max_items: int, trigger_z: float = 0.0) -> str | None:
    """The line, when the largest deviation reaches `trigger_z`; it then names every feature past the listing
    threshold. Six features each past 1.5 by chance put a hint on about one ordinary turn in five (measured: 4 of 20
    held-out `say` utterances, 2026-10-05); a 2.0 trigger brought that to 1 in 20 and kept 33 of 36 planted
    everyday-size changes."""
    if not devs or max(abs(d.z) for d in devs) < trigger_z:
        return None
    return PREFIX + "; ".join(d.text for d in devs[:max_items]) + "]"


def update(f: Features, b: Baseline, ok: set[str]) -> None:
    """Add this utterance to the baseline, after it was compared (it must not move its own yardstick)."""
    for name, spec in SPECS.items():
        if name in ok:
            b.add(spec.attr, getattr(f, spec.attr))
    if "fillers" in ok:
        b.add_fillers(f.fillers, f.words)
    b.n += 1
