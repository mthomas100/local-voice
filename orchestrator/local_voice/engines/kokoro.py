"""Kokoro 82M through mlx-audio 0.5.7 (the fallback voice: the latency floor, no cloning, no instruct).

Measured here (09b §2, §5): 74-79 ms for a whole sentence, RTF 0.02, 0.3 GiB; its first call in a process costs
0.8-4.3 s (kernels and G2P), so it is warmed at start. Kokoro has no streaming inside a segment, so a long sentence
is cut at clause punctuation (`clause_chars`) and each piece is one call, which brings the first audio forward.

Kokoro's G2P is misaki (the `kokoro` extra in pyproject.toml). misaki's EspeakFallback aborts the process on this
Mac (espeakng-loader's data path is baked in at build time), so load() makes its constructor raise and KokoroPipeline
takes its own except branch: out-of-dictionary words are skipped rather than spoken (measure/bench/setup.sh, 09c §6).
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator

import numpy as np

from .base import Synthesizer, to_float32

CLAUSE = re.compile(r"(?<=[,;:—])\s+")


def _disable_espeak_fallback() -> None:
    try:
        import misaki.espeak as esp
    except Exception:  # noqa: BLE001 - misaki missing is reported by load_model itself
        return

    class _Disabled:
        def __init__(self, *a, **k):
            raise RuntimeError("EspeakFallback disabled (espeak-ng data path crash on this Mac)")

    esp.EspeakFallback = _Disabled


def clause_pieces(text: str, max_chars: int) -> list[str]:
    """Cut text at clause punctuation into pieces of at most about max_chars (never mid-clause)."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    out, cur = [], ""
    for part in CLAUSE.split(text):
        if cur and len(cur) + 1 + len(part) > max_chars:
            out.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    if cur:
        out.append(cur)
    return out


class KokoroEngine(Synthesizer):
    sample_rate = 24000

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.model = None
        self.clause_chars = int(settings.get("clause_chars", 90))

    def load(self) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        _disable_espeak_fallback()
        t0 = time.monotonic()
        self.model = load_model(self.settings["model"])
        mx.eval(self.model.parameters())
        self.sample_rate = int(getattr(getattr(self.model, "config", None), "sample_rate", 24000))
        self.load_s = round(time.monotonic() - t0, 3)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        import mlx.core as mx

        s = self.settings
        try:
            for piece in clause_pieces(text, self.clause_chars):
                # split_pattern that never matches: one segment per call, we already cut the text.
                for r in self.model.generate(piece, voice=s.get("voice") or "af_heart", speed=float(s.get("speed", 1.0)),
                                             lang_code=s.get("lang_code") or "a", split_pattern=r"\n\n\n+"):
                    yield to_float32(r.audio)
        finally:
            mx.clear_cache()

    def warm(self) -> None:
        for _ in self.stream("Hello there, this is a warm-up sentence."):
            pass
