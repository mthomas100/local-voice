"""One JSON line per turn: end of speech to first audio, and where the time went.

The total is Pipecat's UserBotLatencyObserver: from the user's real stop (VAD stop minus stop_secs) to the output
transport starting to send the reply's audio (note 04c §4); for a WebSocket client add its playout buffer, which the
e2e client measures on its side. Beside it: the observer's own breakdown (VAD wait, turn analyzer, transcription,
LLM TTFB, aggregation, TTS), this project's per-stage timings (STT call, prompt to first token, TTS to first audible
chunk), Smart Turn's verdict, the 1-minute load average and the hold gate's phase. Lines go to <state_dir>/latency/<date>.jsonl, and the last turn to /v1/status.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import TTFAMetricsData, TurnMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, UserBotLatencyObserver


class _Metrics(BaseObserver):
    def __init__(self):
        super().__init__(observe_every_push=False)
        self.items: list[dict[str, Any]] = []

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, MetricsFrame):
            for m in data.frame.data:
                if isinstance(m, (TTFAMetricsData, TurnMetricsData)):
                    self.items.append({"kind": type(m).__name__, **m.model_dump(mode="json")})


class TurnLatencyLog:
    def __init__(self, path: Path | None, *, session: str, stages: Callable[[], dict[str, Any]] | None = None,
                 on_turn: Callable[[dict[str, Any]], None] | None = None,
                 hold_phase: Callable[[], str] | None = None):
        self.path = path
        self.session = session
        self.stages = stages
        self.on_turn = on_turn
        self.hold_phase = hold_phase
        self.lines: list[dict[str, Any]] = []
        self.latency = UserBotLatencyObserver()
        self._metrics = _Metrics()

        @self.latency.event_handler("on_latency_breakdown")
        async def _on_breakdown(_obs, breakdown: LatencyBreakdown):
            self._write(breakdown)

    @property
    def observers(self) -> list[BaseObserver]:
        return [self.latency, self._metrics]

    def _write(self, b: LatencyBreakdown) -> None:
        line: dict[str, Any] = {
            "ts": round(time.time(), 3), "session": self.session,
            "measured_from": str(b.measured_from.value if hasattr(b.measured_from, "value") else b.measured_from),
            "eos_to_first_audio_ms": round(b.total_secs * 1000, 1),
            "user_turn_ms": round(b.user_turn_secs * 1000, 1) if b.user_turn_secs is not None else None,
            "contributions": [c.model_dump(mode="json") for c in b.contributions],
            "ttfb": [t.model_dump(mode="json") for t in b.ttfb],
            "metrics": self._metrics.items,
        }
        if self.stages:
            line["stages"] = self.stages()
        # Contention, so a slow turn can be told apart later (2026-10-05): the 1-minute load
        # average (what `sysctl -n vm.loadavg` shows first) and the hold gate's phase when the turn was measured.
        line["load1"] = round(os.getloadavg()[0], 2)
        if self.hold_phase:
            phase = self.hold_phase()
            line["hold_phase"] = phase
            line["hold_open"] = phase in ("open", "absent")
        self._metrics.items = []
        self.lines.append(line)
        if self.on_turn:
            self.on_turn(line)
        if self.path is not None:
            p = self.path / f"{time.strftime('%Y-%m-%d')}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, default=str) + "\n")
