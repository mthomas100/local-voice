"""The hint thresholds against synthetic speech with planted deviations: few hints on ordinary utterances, and the
planted feature named when one is due. Bounds, not exact counts, so a macOS voice update does not fail the suite;
tone/scripts/calibrate.py prints the measured counts (summarised in tone/README.md)."""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import calibrate  # noqa: E402

from local_voice_tone import hint as hn  # noqa: E402

pytestmark = pytest.mark.needs_say


@pytest.fixture(scope="module")
def scored(say_baseline):
    return calibrate.scores(calibrate.FLOORS["current"])


def test_ordinary_utterances_rarely_get_a_hint(scored):
    normal, _ = scored
    rows = {r["trigger"]: r for r in calibrate.report(*scored)}
    assert rows[2.0]["false_positives"] <= 2, rows[2.0]           # measured 1 of 20 on 2026-10-05
    assert rows[1.5]["false_positives"] >= rows[2.0]["false_positives"]
    # nothing but pace ever passes 1.5 on ordinary speech here, and never by much
    assert all(abs(d.z) < 3.0 for devs in normal for d in devs)


def test_planted_deviations_are_named(scored):
    _, plant = scored
    by_case = defaultdict(list)
    for case, feature, devs in plant:
        due = hn.render(devs, 3, 2.0)
        by_case[case].append(due is not None and feature in [d.name for d in devs])
    detected = sum(sum(v) for v in by_case.values())
    assert detected >= 30, dict(by_case)                          # measured 33 of 36 on 2026-10-05
    for case, hits in by_case.items():
        needed = 2 if case.startswith(("fast", "slow")) else len(hits)   # pace is the noisy measure (README.md, calibration)
        assert sum(hits) >= needed, (case, hits)


def test_every_line_uses_only_the_fixed_vocabulary(scored):
    """No emotion label, no guess at one: a hint is built from fixed words, numbers and units only."""
    normal, plant = scored
    lines = [hn.render(d, 3) for d in normal] + [hn.render(d, 3) for _, _, d in plant]
    lines = [x for x in lines if x]
    assert len(lines) > 30
    banned = {"stressed", "anxious", "sad", "angry", "happy", "tired", "upset", "nervous", "excited", "calm",
              "rushed", "emotional", "feeling", "mood", "frustrated"}
    for line in lines:
        assert line.startswith(hn.PREFIX) and line.endswith("]")
        words = set(re.findall(r"[a-z]+", line.lower()))
        assert words <= hn.VOCABULARY, words - hn.VOCABULARY
        assert not words & banned
