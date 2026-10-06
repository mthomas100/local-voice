"""Nemotron 3.5 ASR streaming 0.6B 8-bit through mlx-audio 0.5.7's live session (the default ears).

Measured here (09b §3), feeding 320 ms chunks in real time: partials arrive in bursts each time a 1.12 s encoder
chunk completes, and the final transcript 53-54 ms after the audio ends; 0.74 GiB of MLX memory. The session's
`feed`/`close` are thread-safe; `step` and `cancel` must run on one consumer thread, here the MLX worker (09c §5).
`step` returns bare text deltas with no final marker; completion is `done`.
"""
from __future__ import annotations

import time

import numpy as np

from .base import StreamingTranscriber


class _Session:
    """Wraps mlx-audio's NemotronStreamingSession with this project's step budget."""

    def __init__(self, inner, max_decode_tokens: int):
        self._inner = inner
        self._budget = max_decode_tokens

    @property
    def done(self) -> bool:
        return bool(self._inner.done)

    def feed(self, samples: np.ndarray) -> None:
        self._inner.feed(np.asarray(samples, dtype=np.float32).reshape(-1))

    def close(self) -> None:
        self._inner.close()

    def step(self) -> list[str]:
        return list(self._inner.step(max_decode_tokens=self._budget))

    def cancel(self) -> None:
        self._inner.cancel()


class NemotronStreamingEngine(StreamingTranscriber):
    sample_rate = 16000

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.model = None
        self.max_decode_tokens = int(settings.get("max_decode_tokens", 8))

    def load(self) -> None:
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model

        t0 = time.monotonic()
        self.model = load_model(self.settings["model"])
        mx.eval(self.model.parameters())
        rate = int(self.model.preprocessor_config.sample_rate)
        if rate != self.sample_rate:
            raise RuntimeError(f"{self.settings['model']} wants {rate} Hz; the pipeline feeds {self.sample_rate} Hz")
        self.load_s = round(time.monotonic() - t0, 3)

    def open(self) -> _Session:
        # Construction may do MLX work (the session protocol says so), hence the MLX thread.
        return _Session(self.model.create_streaming_session(), self.max_decode_tokens)

    def warm(self) -> None:
        # The first encoder chunks include kernel compilation (230-326 ms per step in phase 2, then 31-56 ms).
        t = np.arange(int(2.4 * self.sample_rate)) / self.sample_rate
        s = self.open()
        s.feed((0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
        s.close()
        for _ in range(10_000):
            if s.done:
                break
            s.step()
        import mlx.core as mx

        mx.clear_cache()
