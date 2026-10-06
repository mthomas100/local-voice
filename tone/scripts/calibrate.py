"""Calibrate the hint thresholds on synthetic speech with planted deviations (machine ground truth, no listener).

    uv run --project tone python tone/scripts/calibrate.py [--floors current|v1] [--json OUT]

A baseline of 30 ordinary `say` utterances; 20 held-out ordinary utterances (a hint on one is a false positive); and
planted deviations of everyday size on held-out sentences (a hint must name the planted feature). Each utterance is
compared with the same baseline, which is not updated during the run, then the false-positive and detection rates
are reported for several trigger thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))

from conftest import HELD_OUT, baseline_utterances, held_out_utterances, say_pcm, stretch  # noqa: E402

from local_voice_tone import hint as hn  # noqa: E402
from local_voice_tone.baseline import Baseline  # noqa: E402
from local_voice_tone.features import extract  # noqa: E402

FLOORS = {
    # v1: first guesses, 2026-10-05 morning; kept to show what calibration changed.
    "v1": {"pace": 0.3, "pitch": 0.8, "pitch range": 1.0, "volume": 1.5, "pausing": 0.04, "longest pause": 0.25},
    # current: what hint.SPECS ships.
    "current": {name: spec.floor for name, spec in hn.SPECS.items()},
}

FILLER_TEXTS = [
    "Um, I left the charger at the, uh, office, so um I will pick it up, uh, tomorrow morning.",
    "The soup needs, um, a bit more salt, uh, but otherwise it, um, tastes really good.",
    "Uh, could you look up the, um, opening hours of the, uh, hardware store near the bridge?",
    "I moved the, um, old files into a, uh, separate folder so the, um, desktop looks cleaner.",
]


def planted():
    """(case, planted feature name, pcm, transcript): four sentences per case, everyday magnitudes."""
    out = []
    for s in HELD_OUT[:4]:
        out.append(("fast (-r 240, about +35%)", "pace", *say_pcm(s, rate=240)))
        pcm, tr = say_pcm(s)
        out.append(("slow (stretched to 0.75)", "pace", stretch(pcm, 0.75), tr))
        out.append(("higher pitch (pbas 50, about +3.6 st)", "pitch", *say_pcm(s, pbas=50)))
        out.append(("lower pitch (pbas 40, about -3.6 st)", "pitch", *say_pcm(s, pbas=40)))
        out.append(("quieter (-10 dB)", "volume", *say_pcm(s, gain_db=-10)))
        out.append(("louder (+6 dB)", "volume", *say_pcm(s, gain_db=6)))
        words = s.split()
        third = len(words) // 3
        paused = " ".join(words[:third] + ["[[slnc 700]]"] + words[third:2 * third] + ["[[slnc 700]]"]
                          + words[2 * third:])
        out.append(("pausing (two 0.7 s silences)", "pausing", *say_pcm(paused)))
        long_pause = " ".join(words[:len(words) // 2] + ["[[slnc 1800]]"] + words[len(words) // 2:])
        out.append(("one 1.8 s pause", "longest pause", *say_pcm(long_pause)))
    for t in FILLER_TEXTS:
        out.append(("four fillers", "fillers", *say_pcm(t)))
    return out


def scores(floors: dict[str, float]):
    for name, f in floors.items():
        hn.SPECS[name] = hn.Spec(hn.SPECS[name].attr, f, hn.SPECS[name].up, hn.SPECS[name].down)
    b = Baseline("calibration", 200)
    for pcm, tr in baseline_utterances():
        f = extract(pcm, tr)
        hn.update(f, b, hn.eligible(f, min_speech_s=1.0, min_words=3))

    def devs(pcm, tr):
        f = extract(pcm, tr)
        ok = hn.eligible(f, min_speech_s=1.0, min_words=3)
        return hn.deviations(f, b, ok, z_threshold=1.5, min_samples=20, long_pause_s=1.5)

    normal = [devs(p, t) for p, t in held_out_utterances()]
    plant = [(case, feature, devs(p, t)) for case, feature, p, t in planted()]
    return normal, plant


def report(normal, plant, triggers=(1.5, 2.0, 2.5, 3.0)):
    rows = []
    for trig in triggers:
        fp = sum(1 for d in normal if d and max(abs(x.z) for x in d) >= trig)
        hits = sum(1 for _, feat, d in plant if d and max(abs(x.z) for x in d) >= trig and feat in [x.name for x in d])
        rows.append({"trigger": trig, "false_positives": fp, "ordinary": len(normal), "detected": hits,
                     "planted": len(plant)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", choices=sorted(FLOORS), default="current")
    ap.add_argument("--json")
    a = ap.parse_args()
    normal, plant = scores(FLOORS[a.floors])
    print(f"floors {a.floors}: {FLOORS[a.floors]}")
    print("\nordinary held-out utterances, deviations past 1.5:")
    for i, d in enumerate(normal):
        print(f"  {i + 1:2d}  " + (", ".join(f"{x.name} {x.z:+.1f}" for x in d) or "-"))
    print("\nplanted:")
    for case, feat, d in plant:
        top = max((abs(x.z) for x in d), default=0.0)
        named = "yes" if feat in [x.name for x in d] else "NO"
        print(f"  {case:40s} planted {feat:14s} named {named:3s} max|z| {top:5.1f}  "
              + ", ".join(f"{x.name} {x.z:+.1f}" for x in d))
    rows = report(normal, plant)
    print("\ntrigger  false positives    detected")
    for r in rows:
        print(f"  {r['trigger']:.1f}    {r['false_positives']:2d} / {r['ordinary']}          {r['detected']:2d} / {r['planted']}")
    if a.json:
        Path(a.json).write_text(json.dumps({"floors": FLOORS[a.floors], "rows": rows,
                                            "normal": [[(x.name, round(x.z, 2)) for x in d] for d in normal],
                                            "planted": [(c, f, [(x.name, round(x.z, 2)) for x in d])
                                                        for c, f, d in plant]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
