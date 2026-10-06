"""Parakeet TDT 0.6B v3 through mlx-audio 0.5.7, one call per utterance (the alternative ears, kind: segmented).

Measured here (09b §3): 23 ms for a 2.48 s clip, load about 1 s, 2.4 GiB of MLX memory. mlx-audio's Parakeet has no
live session, so it runs once per VAD-delimited utterance. It takes an `mx.array` that is already mono float at
16 kHz and never resamples an array; a NumPy array fails at its `astype(mx.bfloat16)` (09c §4).
"""
from __future__ import annotations

import time

import numpy as np

from .base import Transcriber


class ParakeetEngine(Transcriber):
    sample_rate = 16000

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.model = None

    def load(self) -> None:
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model

        t0 = time.monotonic()
        self.model = load_model(self.settings["model"])
        mx.eval(self.model.parameters())
        rate = int(getattr(getattr(self.model, "preprocessor_config", None), "sample_rate", 16000))
        if rate != self.sample_rate:
            raise RuntimeError(f"{self.settings['model']} wants {rate} Hz; the pipeline feeds {self.sample_rate} Hz")
        self.load_s = round(time.monotonic() - t0, 3)

    def warm(self) -> None:
        # Kernel compilation lands on the first call: 1.4 s cold vs 0.05 s warm (pi dictate, 2026-09-13).
        t = np.arange(self.sample_rate) / self.sample_rate
        self.transcribe((0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        import mlx.core as mx

        if not audio.size:
            return ""
        result = self.model.generate(mx.array(np.ascontiguousarray(audio, dtype=np.float32)))
        text = getattr(result, "text", result if isinstance(result, str) else "")
        mx.clear_cache()   # this process shares the GPU with the LLM rig; hand cached buffers back
        return " ".join(str(text or "").split())
