"""The contracts every speech model adapter implements, and small helpers they share.

An adapter is one class named by config.yaml (`impl: module:Class`) and built with its settings dict. Every
method that touches a model runs on the MLX worker thread (mlx_worker.py), never on the event loop: `load`,
`warm`, `transcribe`, `open`/`step`, and creating and advancing a synthesis stream. Keep this module free of mlx
imports so the fakes and the unit tests never load MLX.
"""
from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np


class Engine:
    """Base for adapters: settings in, a model loaded on the MLX thread, a describe() for /v1/status."""

    sample_rate: int = 16000

    def __init__(self, settings: dict[str, Any]):
        self.settings = dict(settings)
        self.load_s: float | None = None

    def load(self) -> None:
        """Load weights. MLX thread."""

    def warm(self) -> None:
        """Run once on dummy input so the first real call does not pay kernel compilation. MLX thread."""

    def describe(self) -> dict[str, Any]:
        return {"impl": type(self).__name__, "model": self.settings.get("model"), "load_s": self.load_s}


class Transcriber(Engine):
    """Per-utterance STT (kind: segmented). `transcribe` gets float32 mono at `sample_rate` (16 kHz)."""

    def transcribe(self, audio: np.ndarray) -> str:
        raise NotImplementedError


class StreamSession(Protocol):
    """A live transcription session (mlx-audio's StreamingSession shape, 09c §5).

    feed() and close() may run on any thread; step() and cancel() only on the MLX thread. step() returns text
    deltas (append-only); `done` turns true once a closed session has decoded everything."""

    @property
    def done(self) -> bool: ...

    def feed(self, samples: np.ndarray) -> None: ...

    def close(self) -> None: ...

    def step(self) -> list[str]: ...

    def cancel(self) -> None: ...


class StreamingTranscriber(Engine):
    """Live STT (kind: streaming): open() a session per utterance on the MLX thread."""

    def open(self) -> StreamSession:
        raise NotImplementedError


class Synthesizer(Engine):
    """TTS. `stream(text)` returns a generator of float32 mono chunks at `sample_rate`.

    The generator is created and advanced on the MLX thread, one chunk per call. Closing it early (barge-in) must
    leave the model ready for the next utterance: the adapter's own `finally` releases what the model keeps (for
    Qwen3-TTS: close mlx-audio's generator, reset the decoder's streaming state, clear MLX's cache; 09c §2).
    One stream per model object at a time: the service holds a lock across a whole utterance."""

    sample_rate: int = 24000

    def stream(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError


def accepted_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The subset of kwargs that fn's signature takes (all of them if it has **kwargs)."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def to_float32(audio: Any) -> np.ndarray:
    """An mx.array or array-like to a flat float32 NumPy array (realised on the calling thread)."""
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
