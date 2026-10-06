"""The rolling personal baseline: the last N eligible utterances' values per feature, per channel.

One speaker, so no cross-speaker model is needed (the failure that haunts emotion recognition):
each value is compared with this person's own recent values. Robust statistics (median and MAD) keep one odd
utterance from moving the baseline much. Channels keep devices apart, because a phone microphone and the Mac's do
not record the same level (and a different recogniser hears different fillers).

Stored as JSON under state/tone/, read and written under an exclusive lock so several orchestrator processes can
share it.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

VERSION = 1


def channel_file(state_dir: Path, channel: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", channel or "default")
    return state_dir / f"baseline-{safe}.json"


class Baseline:
    def __init__(self, channel: str, window: int, data: dict | None = None):
        self.channel, self.window = channel, window
        d = data or {}
        self.values: dict[str, list[float]] = {k: list(v) for k, v in (d.get("values") or {}).items()}
        self.filler_counts: list[list[int]] = [list(x) for x in d.get("fillers") or []]   # [fillers, words]
        self.n: int = int(d.get("n", 0))

    # --- statistics -------------------------------------------------------------------------------------------
    def stats(self, feature: str) -> tuple[int, float, float]:
        """(samples, median, 1.4826 × MAD) for one feature."""
        v = np.asarray(self.values.get(feature) or [], dtype=float)
        if len(v) == 0:
            return 0, math.nan, math.nan
        med = float(np.median(v))
        return len(v), med, float(1.4826 * np.median(np.abs(v - med)))

    def z(self, feature: str, value: float, floor: float) -> tuple[float, float, int]:
        """Robust z of `value` against the baseline, with a floor under the scale so a very steady baseline does not
        turn a tiny difference into a large z. Returns (z, median, samples)."""
        n, med, scale = self.stats(feature)
        if n == 0 or math.isnan(value):
            return math.nan, med, n
        return (value - med) / max(scale, floor), med, n

    def filler_rate(self) -> tuple[float, int]:
        """Fillers per word over the window, and how many utterances it is from."""
        f = sum(c[0] for c in self.filler_counts)
        w = sum(c[1] for c in self.filler_counts)
        return (f / w if w else 0.0), len(self.filler_counts)

    # --- updates ----------------------------------------------------------------------------------------------
    def add(self, feature: str, value: float) -> None:
        if value is None or math.isnan(value):
            return
        v = self.values.setdefault(feature, [])
        v.append(float(value))
        del v[: max(0, len(v) - self.window)]

    def add_fillers(self, fillers: int, words: int) -> None:
        self.filler_counts.append([int(fillers), int(words)])
        del self.filler_counts[: max(0, len(self.filler_counts) - self.window)]

    def to_json(self) -> dict:
        return {"v": VERSION, "channel": self.channel, "window": self.window, "n": self.n,
                "values": {k: [round(x, 5) for x in v] for k, v in self.values.items()},
                "fillers": self.filler_counts}


@contextmanager
def locked(path: Path, channel: str, window: int) -> Iterator[Baseline]:
    """Load the channel's baseline under an exclusive lock; it is saved atomically when the block ends cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a") as lockfh:
        fcntl.flock(lockfh, fcntl.LOCK_EX)
        try:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
            b = Baseline(channel, window, data)
            yield b
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(b.to_json()) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lockfh, fcntl.LOCK_UN)
