"""Model-free stand-ins for the speech adapters: unit tests and the model-free integration tests use these.

They honour the same contracts as the real adapters (base.py), including the one-stream-per-model rule, so a test
that passes with them exercises the orchestrator's own logic. Nothing here imports mlx.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator

import numpy as np

from .base import StreamingTranscriber, Synthesizer, Transcriber


class FakeTranscriber(Transcriber):
    """Returns scripted transcripts, one per utterance (`script`), else `text`; silence returns ""."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.text = str(settings.get("text", ""))
        self.script: deque[str] = deque(settings.get("script") or [])
        self.calls: list[float] = []          # seconds of audio per call

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(audio.size / self.sample_rate)
        if not audio.size or float(np.abs(audio).max()) < 0.01:
            return ""
        return self.script.popleft() if self.script else self.text


class _FakeSession:
    def __init__(self, text: str, words_per_s: float, rate: int, min_heard_s: float = 0.0):
        self._words = text.split()
        self._min_heard_s = min_heard_s
        self._rate = rate
        self._per_word = 1.0 / words_per_s
        self._heard_s = 0.0
        self.loud_s = 0.0               # loud samples fed, counted one by one (heard_s counts whole loud chunks)
        self._emitted = 0
        self._closed = False
        self._done = False
        self._lock = threading.Lock()
        self._quiet_s = 0.0             # quiet audio fed since the last loud sample
        self.quiet_tail_at_close_s: float | None = None

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, samples: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("streaming input is closed")
            loud = samples.size and float(np.abs(samples).max()) >= 0.01
            self.loud_s += float(np.count_nonzero(np.abs(samples) >= 0.01)) / self._rate
            if loud:
                self._heard_s += samples.size / self._rate
                self._quiet_s = 0.0
            else:
                self._quiet_s += samples.size / self._rate

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.quiet_tail_at_close_s = self._quiet_s

    def step(self) -> list[str]:
        with self._lock:
            target = len(self._words) if self._closed else min(len(self._words), int(self._heard_s / self._per_word))
            if self._heard_s == 0 or self.loud_s < self._min_heard_s:
                target = 0
            out = [(" " if self._emitted + i else "") + w for i, w in enumerate(self._words[self._emitted:target])]
            self._emitted = max(self._emitted, target)
            if self._closed:
                self._done = True
            return out

    def cancel(self) -> None:
        with self._lock:
            self._closed = self._done = True


class FakeStreamingTranscriber(StreamingTranscriber):
    """A live session that reveals scripted words as loud audio arrives (words_per_s), the rest on close."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.text = str(settings.get("text", ""))
        self.script: deque[str] = deque(settings.get("script") or [])
        self.words_per_s = float(settings.get("words_per_s", 3.0))
        # a session that heard less loud audio than this returns no words, as Nemotron does for a short word alone
        self.min_heard_s = float(settings.get("min_heard_s", 0.0))
        self.sessions = 0
        self.opened: list[_FakeSession] = []

    def open(self) -> _FakeSession:
        self.sessions += 1
        text = self.script.popleft() if self.script else self.text
        self.last_session = _FakeSession(text, self.words_per_s, self.sample_rate, min_heard_s=self.min_heard_s)
        self.opened.append(self.last_session)
        return self.last_session


class FakeSynthesizer(Synthesizer):
    """Leading silence then a quiet tone, `seconds_per_char` long per character, in 80 ms chunks.

    Records every text it was asked to speak and whether each stream ran to the end; refuses a second concurrent
    stream on the same object (the Qwen3-TTS rule the service must respect)."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.sample_rate = int(settings.get("sample_rate", 24000))
        self.seconds_per_char = float(settings.get("seconds_per_char", 0.06))
        self.lead_s = float(settings.get("lead_s", 0.1))
        self.rtf = float(settings.get("rtf", 0.0))      # > 0: take this fraction of each chunk's duration to make it
        self.spoken: list[str] = []
        self.finished: list[bool] = []
        self.closed_early = 0
        self._active = False

    def stream(self, text: str) -> Iterator[np.ndarray]:
        if self._active:
            raise RuntimeError("FakeSynthesizer: a second stream started while one is active")
        self._active = True
        self.spoken.append(text)
        self.finished.append(False)
        idx = len(self.finished) - 1
        n = int(0.08 * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        total = max(0.3, len(text) * self.seconds_per_char)
        chunks = int(np.ceil(total / 0.08))
        lead = int(round(self.lead_s / 0.08))
        try:
            for i in range(chunks):
                if self.rtf:
                    time.sleep(0.08 * self.rtf)
                if i < lead:
                    yield np.zeros(n, np.float32)
                else:
                    yield (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            self.finished[idx] = True
        except GeneratorExit:
            self.closed_early += 1
            raise
        finally:
            self._active = False
