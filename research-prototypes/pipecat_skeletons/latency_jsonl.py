"""Per-turn latency to JSONL from Pipecat 1.12.0's own observers.

Facts (pipecat-ai 1.12.0):
- UserBotLatencyObserver measures end of user speech to bot speech start: the user's real stop is
  VADUserStoppedSpeakingFrame.timestamp - stop_secs, the end is BotStartedSpeakingFrame
  (observers/user_bot_latency_observer.py:650-656, 965-1001). The output transport pushes
  BotStartedSpeakingFrame when the first TTS chunk leaves its queue, just before writing it
  (base_output.py:955-974, 852-858, 705-726): for a WebSocket client that means "sent", so add the client's
  playout buffer, and leading silence inside the audio counts as speech here (the TTFA metric below
  separates it).
- Events: on_latency_measured(observer, seconds) and on_latency_breakdown(observer, LatencyBreakdown); the
  breakdown is a pydantic model whose contributions sum to the total and name each stage (VAD wait, turn
  analyzer, transcription, LLM TTFB, text aggregation, TTS) (user_bot_latency_observer.py:342-507, 536-546).
- Service TTFB and TTFA reach it only with PipelineParams(enable_metrics=True) (pipeline/worker.py:164-196)
  and services whose can_generate_metrics() is True (frame_processor.py:487-493). TTFA carries ttfb and the
  leading silence separately (metrics/metrics.py:41-61, frame_processor.py:532-548).
- Smart Turn pushes TurnMetricsData (is_complete, probability, e2e_processing_time_ms) in a MetricsFrame
  (base_smart_turn.py:240-266, turn_analyzer_user_turn_stop_strategy.py:307-310).
- The worker adds UserBotLatencyObserver itself only when tracing is on (pipeline/worker.py:462-476); add it
  as an observer (PipelineWorker(observers=[...])).
- Push-to-talk turns have no VAD stop frame unless the client's "stop" is mapped to one (protocol_v1.py
  does), otherwise nothing is measured.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, UserBotLatencyObserver


class _MetricsCollector(BaseObserver):
    """Keeps the metrics frames seen since the last line was written."""

    def __init__(self):
        """Observe each frame once (its first push)."""
        super().__init__(observe_every_push=False)
        self.items: list[dict[str, Any]] = []

    async def on_push_frame(self, data: FramePushed):
        """Collect TTFA, Smart Turn, processing time and LLM usage."""
        if not isinstance(data.frame, MetricsFrame):
            return
        for m in data.frame.data:
            if isinstance(m, (TTFAMetricsData, TurnMetricsData, ProcessingMetricsData, LLMUsageMetricsData)):
                self.items.append({"kind": type(m).__name__, **m.model_dump(mode="json")})


class TurnLatencyLog:
    """Writes one JSON line per measured turn (or greeting) to `path`."""

    def __init__(self, path: Path, *, session: str):
        """Create the observers; pass `observers` to PipelineWorker."""
        self._path = path
        self._session = session
        self.latency = UserBotLatencyObserver()
        self._metrics = _MetricsCollector()

        @self.latency.event_handler("on_latency_breakdown")
        async def _on_breakdown(observer: UserBotLatencyObserver, breakdown: LatencyBreakdown):
            self._write(breakdown)

    @property
    def observers(self) -> list[BaseObserver]:
        """The observers to add to the worker."""
        return [self.latency, self._metrics]

    def _write(self, breakdown: LatencyBreakdown):
        line = {
            "ts": time.time(),
            "session": self._session,
            "measured_from": breakdown.measured_from,
            "eos_to_first_audio_s": round(breakdown.total_secs, 4),
            "user_turn_s": breakdown.user_turn_secs,
            "ttfb": [t.model_dump(mode="json") for t in breakdown.ttfb],
            "contributions": [c.model_dump(mode="json") for c in breakdown.contributions],
            "function_calls": [f.model_dump(mode="json") for f in breakdown.function_calls],
            "metrics": self._metrics.items,
        }
        self._metrics.items = []
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
