"""The orchestrator's hook: one call per user turn, off by default.

Modes (config `mode`):
- `off` (default): `analyze` returns None at once. Nothing is computed, read or written.
- `log`: measure, compare, log and grow the baseline; never return a hint. This is the week of measurement the design
  asks for before hints reach the model.
- `on`: as `log`, and return the hint line when one is due: past the threshold, outside the cooldown, and in the
  "show" arm of the A/B (a stable hash of session and turn puts `ab_strip_fraction` of due hints in the "strip"
  arm, where they are logged but not returned, so the conversations with and without hints can be compared).

Every analysed turn is logged with its raw values, z-scores and arm to state/tone/hints-<day>.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from . import baseline as bl
from . import features as feat
from . import hint as hn
from .pitch import BACKENDS

MODES = ("off", "log", "on")


@dataclass
class ToneConfig:
    mode: str = "off"
    f0: str = "praat"                 # pitch adapter: praat | pyin
    z_threshold: float = 1.5          # the design value: a feature is named in the hint when its z passes 1.5
    trigger_z: float = 2.0            # and a hint is due only when one passes this (hint.render: false positives)
    min_samples: int = 20             # a feature says nothing until the baseline holds this many of its values
    window: int = 200                 # the rolling baseline: the last N eligible utterances per channel
    cooldown_turns: int = 3           # after a hint, the next hint is due no sooner than this many turns later
    ab_strip_fraction: float = 0.5    # in `on` mode, the share of due hints that are logged but not shown
    min_speech_s: float = 1.0
    min_words: int = 3
    min_pause_s: float = 0.25
    long_pause_s: float = 1.5         # the longest pause is reported only from this length
    max_items: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ToneConfig":
        d = dict(d or {})
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(d) - set(known))
        if unknown:
            raise ValueError(f"tone config: unknown keys {unknown}")
        cfg = cls(**{k: type(getattr(cls, k))(v) if not isinstance(getattr(cls, k), str) else str(v)
                     for k, v in d.items()})
        if cfg.mode not in MODES:
            raise ValueError(f"tone config: mode must be one of {MODES}, got {cfg.mode!r}")
        if cfg.f0 not in BACKENDS:
            raise ValueError(f"tone config: f0 must be one of {sorted(BACKENDS)}, got {cfg.f0!r}")
        if not 0.0 <= cfg.ab_strip_fraction <= 1.0:
            raise ValueError("tone config: ab_strip_fraction must be between 0 and 1")
        return cfg


@dataclass
class ToneResult:
    hint: str | None                  # what to give the model, or None
    due: str | None                   # the line that was due before cooldown and the A/B, for the log
    arm: str                          # show | strip | cooldown | none | log
    features: dict[str, Any]
    deviations: list[dict[str, Any]] = field(default_factory=list)
    baseline_n: int = 0
    ms: float = 0.0


HINT_INSTRUCTION = (
    "Sometimes the person's message starts with a bracketed line beginning \"[delivery vs usual\". It is an automatic, "
    "uncertain measurement of how they spoke compared with their own usual: pace, pitch, volume, pauses, fillers. It "
    "is not a fact about how they feel. Never name an emotion from it and never mention it on its own account. When it "
    "matters, let it shape your manner, or ask rather than assert (for example: \"you sound a bit rushed, is now a "
    "good time?\"). The words they said always come first."
)


class ToneHook:
    def __init__(self, config: dict[str, Any] | ToneConfig | None = None, *, state_dir: str | Path | None = None):
        self.cfg = config if isinstance(config, ToneConfig) else ToneConfig.from_dict(config)
        if self.cfg.mode != "off" and state_dir is None:
            raise ValueError("tone: state_dir is required unless mode is off")
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self._last_hint_turn: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.cfg.mode != "off"

    def _arm(self, session: str, turn: int) -> str:
        h = hashlib.sha256(f"{session}:{turn}".encode()).digest()
        return "strip" if int.from_bytes(h[:8], "big") / 2**64 < self.cfg.ab_strip_fraction else "show"

    def analyze(self, pcm, transcript: str, *, session: str, turn: int, sample_rate: int = 16000,
                channel: str = "default", speech_segments: list[tuple[float, float]] | None = None,
                now: datetime | None = None) -> ToneResult | None:
        """One user turn: the utterance's PCM (int16 LE mono, the bytes the recogniser got) and its final transcript.
        Returns None when the hook is off."""
        if self.cfg.mode == "off":
            return None
        t0 = time.perf_counter()
        f = feat.extract(pcm, transcript, sample_rate=sample_rate, f0=self.cfg.f0, speech_segments=speech_segments,
                         min_pause_s=self.cfg.min_pause_s)
        ok = hn.eligible(f, min_speech_s=self.cfg.min_speech_s, min_words=self.cfg.min_words)
        with bl.locked(bl.channel_file(self.state_dir, channel), channel, self.cfg.window) as b:
            devs = hn.deviations(f, b, ok, z_threshold=self.cfg.z_threshold, min_samples=self.cfg.min_samples,
                                 long_pause_s=self.cfg.long_pause_s)
            hn.update(f, b, ok)
            n = b.n
        due = hn.render(devs, self.cfg.max_items, self.cfg.trigger_z)
        hint, arm = None, "none"
        if due:
            with self._lock:
                last = self._last_hint_turn.get(session)
                if self.cfg.mode == "log":
                    arm = "log"
                elif last is not None and turn - last < self.cfg.cooldown_turns:
                    arm = "cooldown"
                else:
                    arm = self._arm(session, turn)
                    self._last_hint_turn[session] = turn       # a stripped hint starts the cooldown too
                    hint = due if arm == "show" else None
        res = ToneResult(hint=hint, due=due, arm=arm, features=f.as_dict(),
                         deviations=[{"name": d.name, "z": round(d.z, 2), "value": round(float(d.value), 4),
                                      "usual": round(float(d.usual), 4)} for d in devs],
                         baseline_n=n, ms=(time.perf_counter() - t0) * 1000)
        self._log(res, session=session, turn=turn, channel=channel, transcript_words=f.words, now=now)
        return res

    def _log(self, r: ToneResult, *, session: str, turn: int, channel: str, transcript_words: int,
             now: datetime | None) -> None:
        now = now or datetime.now().astimezone()
        rec = {"v": 1, "t": now.isoformat(timespec="milliseconds"), "session": session, "turn": turn,
               "channel": channel, "mode": self.cfg.mode, "arm": r.arm, "hint": r.hint, "due": r.due,
               "deviations": r.deviations, "features": r.features, "baseline_n": r.baseline_n,
               "f0": self.cfg.f0, "ms": round(r.ms, 2)}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / f"hints-{now.date().isoformat()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
