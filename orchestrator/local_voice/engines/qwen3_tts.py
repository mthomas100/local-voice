"""Qwen3-TTS CustomVoice through mlx-audio 0.5.7, streamed (the default voice: 1.7B, speaker Ryan).

Measured here (../measure/): first chunk after 103 ms at streaming_interval 0.32, 0.08 s of leading silence, RTF 0.26-0.28,
4.3 GiB active and 4.75 GiB peak while streaming. What the adapter does about the footguns (measure/09c §1-2):

- `streaming_interval` is passed (default 2.0 costs 0.3-0.4 s of first audio); 0.32 s = 4 codec frames per chunk.
- The model's own sampling is passed explicitly, with repetition_penalty >= 1.05: at 1.0 the 0.6B ran to the token
  cap and emitted a minute of silence.
- `max_tokens` is per call, in 12.5 Hz codec frames: frames_per_char x characters + frames_min. The service sends one
  sentence per call (CustomVoice ignores split_pattern), so a runaway costs one sentence.
- Barge-in: the stream is a plain generator with no cancel flag. Closing ours closes mlx-audio's at its paused
  yield, then resets the decoder's streaming state and clears MLX's cache; the library does both only after a
  normal finish. All streams of one model share the decoder state: one stream per model at a time (the service
  holds a lock).
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from .base import Synthesizer, to_float32

WARM_TEXT = "Hello there, this is a warm-up sentence."


class Qwen3TTSEngine(Synthesizer):
    sample_rate = 24000

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.model = None
        s = self.settings
        if float(s.get("repetition_penalty", 1.05)) < 1.05:
            raise ValueError("qwen3 repetition_penalty must be >= 1.05 (1.0 runs away)")
        self.frames_per_char = float(s.get("frames_per_char", 1.6))
        self.frames_min = int(s.get("frames_min", 40))
        self.last: dict = {}

    def load(self) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        t0 = time.monotonic()
        self.model = load_model(self.settings["model"])
        mx.eval(self.model.parameters())
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        voice = self.settings.get("voice")
        speakers = getattr(self.model, "supported_speakers", None) or []
        if speakers and voice and voice.lower() not in [str(x).lower() for x in speakers]:
            raise ValueError(f"voice {voice!r} is not a speaker of {self.settings['model']}: {', '.join(map(str, speakers))}")
        self.load_s = round(time.monotonic() - t0, 3)

    def max_tokens_for(self, text: str) -> int:
        return int(len(text) * self.frames_per_char) + self.frames_min

    def _kwargs(self, text: str) -> dict:
        s = self.settings
        kw = dict(voice=s.get("voice"), lang_code=s.get("lang_code") or "auto",
                  temperature=float(s.get("temperature", 0.9)), top_k=int(s.get("top_k", 50)),
                  top_p=float(s.get("top_p", 1.0)), repetition_penalty=float(s.get("repetition_penalty", 1.05)),
                  max_tokens=self.max_tokens_for(text), stream=True,
                  streaming_interval=float(s.get("streaming_interval", 0.32)))
        if s.get("instruct"):
            kw["instruct"] = str(s["instruct"])
        return kw

    def stream(self, text: str) -> Iterator[np.ndarray]:
        import mlx.core as mx

        kw = self._kwargs(text)
        gen = self.model.generate(text=text, **kw)
        frames = 0
        finished = False
        try:
            for r in gen:
                frames += int(getattr(r, "token_count", 0) or 0)
                yield to_float32(r.audio)   # np.asarray realises the chunk here, on the MLX thread
            finished = True
        finally:
            self.last = {"text_chars": len(text), "frames": frames, "max_tokens": kw["max_tokens"],
                         "hit_cap": frames >= kw["max_tokens"], "finished": finished}
            if not finished:
                gen.close()
                decoder = getattr(getattr(self.model, "speech_tokenizer", None), "decoder", None)
                if decoder is not None and hasattr(decoder, "reset_streaming_state"):
                    decoder.reset_streaming_state()
                mx.clear_cache()

    def warm(self) -> None:
        for _ in self.stream(WARM_TEXT):
            pass

    def describe(self) -> dict:
        d = super().describe()
        d.update(voice=self.settings.get("voice"), streaming_interval=self.settings.get("streaming_interval"))
        return d
