"""Speech-like test voices for the recording tests (recorder.py, tools/session_report.py; 2026-10-05), model-free.

FakeSynthesizer says every sentence as the same 220 Hz tone restarted every 80 ms chunk: periodic, so a reply's echo
correlates with the reply at every 80 ms of lag and one sentence cannot be told from another by its audio. These
voices keep its timing (leading silence, 80 ms chunks, seconds per character) but sound like a voice to a correlator:
band-limited noise under a syllable envelope, seeded by the text, so each sentence has its own audio and the same
sentence always the same. `voice` names a speaker: the same words in another voice are different audio (the
read-aloud control: a person reading the agent's line is not a replay of it)."""
from __future__ import annotations

import time
import zlib
from collections.abc import Iterator

import numpy as np

from local_voice.engines.base import Synthesizer


def speech_like(text: str, rate: int, *, seconds_per_char: float = 0.03, level: float = 0.1, voice: str = "agent",
                lead_s: float = 0.0) -> np.ndarray:
    """float32 audio `lead_s` of silence then about len(text) * seconds_per_char of speech-like noise at RMS `level`:
    white noise smoothed to about 4 kHz under a 4-6 Hz syllable envelope (a syllable gap every 0.2 s or so), the seed
    the crc32 of voice and text."""
    rng = np.random.default_rng(zlib.crc32(f"{voice}:{text}".encode()))
    n = max(int(0.3 * rate), int(len(text) * seconds_per_char * rate))
    x = rng.standard_normal(n + 4)
    x = np.convolve(x, np.ones(4) / 4, mode="valid")[:n]            # a gentle low-pass (first null at rate / 4)
    t = np.arange(n) / rate
    syl = rng.uniform(4.0, 6.0)
    env = np.clip(np.sin(2 * np.pi * syl * t + rng.uniform(0, np.pi)), 0, None) ** 0.5
    y = x * env
    y *= level / max(float(np.sqrt(np.mean(y * y))), 1e-9)
    return np.concatenate([np.zeros(int(round(lead_s * rate)), np.float32), y.astype(np.float32)])


class NoiseSynthesizer(Synthesizer):
    """FakeSynthesizer's contract and timing (80 ms chunks, `lead_s` of silence first, `rtf`, one stream at a time,
    `spoken`/`finished`), each sentence's sound from speech_like."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.sample_rate = int(settings.get("sample_rate", 24000))
        self.seconds_per_char = float(settings.get("seconds_per_char", 0.03))
        self.lead_s = float(settings.get("lead_s", 0.08))
        self.level = float(settings.get("level", 0.1))
        self.voice = str(settings.get("voice", "agent"))
        self.rtf = float(settings.get("rtf", 0.0))
        self.spoken: list[str] = []
        self.finished: list[bool] = []
        self._active = False

    def audio(self, text: str) -> np.ndarray:
        return speech_like(text, self.sample_rate, seconds_per_char=self.seconds_per_char, level=self.level,
                           voice=self.voice, lead_s=self.lead_s)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        if self._active:
            raise RuntimeError("NoiseSynthesizer: a second stream started while one is active")
        self._active = True
        self.spoken.append(text)
        self.finished.append(False)
        idx = len(self.finished) - 1
        a = self.audio(text)
        n = int(0.08 * self.sample_rate)
        try:
            for i in range(0, len(a), n):
                if self.rtf:
                    time.sleep(0.08 * self.rtf)
                yield a[i:i + n]
            self.finished[idx] = True
        finally:
            self._active = False
