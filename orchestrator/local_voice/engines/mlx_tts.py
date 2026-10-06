"""Any other mlx-audio TTS model that streams with `stream=True, streaming_interval=...` (Pocket TTS, Marvis).

Measured here (09b §2) at interval 0.32: Pocket TTS 18 ms to the first chunk (leading silence 0.04-0.26 s), Marvis
117 ms (no leading silence). Settings: model, voice (null = the model's default), streaming_interval, and a
`generate` table of extra keyword arguments; arguments a model's generate() does not take are dropped.
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from .base import Synthesizer, accepted_kwargs, to_float32


class MLXStreamingTTSEngine(Synthesizer):
    def __init__(self, settings: dict):
        super().__init__(settings)
        self.model = None

    def load(self) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        t0 = time.monotonic()
        self.model = load_model(self.settings["model"])
        mx.eval(self.model.parameters())
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self.load_s = round(time.monotonic() - t0, 3)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        import mlx.core as mx

        s = self.settings
        kw = dict(s.get("generate") or {})
        kw.update(stream=True, streaming_interval=float(s.get("streaming_interval", 0.32)))
        if s.get("voice"):
            kw["voice"] = s["voice"]
        gen = self.model.generate(text=text, **accepted_kwargs(self.model.generate, kw))
        finished = False
        try:
            for r in gen:
                yield to_float32(r.audio)
            finished = True
        finally:
            if not finished:
                gen.close()
            mx.clear_cache()

    def warm(self) -> None:
        for _ in self.stream("Hello there, this is a warm-up sentence."):
            pass
