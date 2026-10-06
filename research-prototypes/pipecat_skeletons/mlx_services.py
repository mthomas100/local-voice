"""Custom STT and TTS services for mlx-audio, shaped by Pipecat 1.12.0's base classes.

Construct-only skeletons: the engines are protocols, and the fakes at the bottom let everything construct and
import without loading a model. The real engines (Parakeet per utterance, Qwen3-TTS streaming, Kokoro) belong
to the orchestrator and follow research note 09c for the mlx-audio 0.5.7 calls.

Facts this file relies on (pipecat-ai 1.12.0, file:line in the installed package):
- SegmentedSTTService buffers audio between VADUserStartedSpeakingFrame and VADUserStoppedSpeakingFrame,
  pads trailing_silence_secs (default 0.5) of silence, and queues the segment to ONE background task that
  calls run_stt (stt_service.py:827-854, 932-968). run_stt therefore never blocks frame flow, but a blocking
  call inside it would still block the event loop: offload it.
- wants_wav_segments=False hands run_stt raw PCM16 instead of a WAV container (stt_service.py:895-904, 949-954).
- Every TranscriptionFrame from a segmented service is marked finalized=True (stt_service.py:906-918), which
  lets TurnAnalyzerUserTurnStopStrategy end the turn as soon as Smart Turn says "complete" and the transcript
  is in, instead of waiting out ttfs_p99_latency (turn_analyzer_user_turn_stop_strategy.py:277-284, 328-364).
- Pass ttfs_p99_latency explicitly; unset it defaults to 1.0 s with a warning (stt_service.py:560-576,
  stt_latency.py:38). It is the fallback wait when no finalized transcript arrives.
- run_tts is iterated inside the TTS service's frame-processing task (tts_service.py:1430,
  1460-1487), so an InterruptionFrame cancels it between yields (frame_processor.py:1129-1156). The executor
  call already running finishes anyway; the stop flag below ends the engine's own loop at the next chunk.
- push_start_frame=True makes the base create the audio context, start TTFB and emit TTSStartedFrame
  (tts_service.py:1387-1393); the base stops TTFB and measures TTFA (time to first AUDIBLE sample, with the
  leading silence split out) on the first audio frame (tts_service.py:1847-1854, metrics.py:41-61).
- A synchronous run_tts (frames yielded from inside it) gets its TTSStoppedFrame appended when the turn's
  LLMFullResponseEndFrame arrives (tts_service.py:828-841), not after the 3 s context idle timeout.
- Three audio contexts in a row with no audio mark the service unusable
  (max_consecutive_zero_audio_contexts=3, tts_service.py:172-174, 230-238). Never return zero audio for
  non-empty text; trim silence, do not drop everything.
- pipecat.audio.utils.detect_speech_onset (audio/utils.py:327-367) is the same RMS onset detector the TTFA
  metric uses; it trims the leading silence that Qwen3-TTS 0.6B adds (~0.42 s, measured in measure/09-phase1-report.md).
"""

from __future__ import annotations

import threading
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Protocol

import numpy as np
from loguru import logger

from pipecat.audio.utils import detect_speech_onset
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601

from mlx_thread import run_mlx, submit_mlx

# --------------------------------------------------------------------------------------------------- STT


class TranscribeEngine(Protocol):
    """A blocking, MLX-backed transcriber. Called only on the MLX thread."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe float32 mono audio in [-1, 1] and return the text."""
        ...


class MLXSegmentedSTTService(SegmentedSTTService):
    """Per-utterance STT (Parakeet TDT 0.6B v3 in the plan) with all MLX work on the MLX thread."""

    def __init__(
        self,
        *,
        engine: TranscribeEngine,
        ttfs_p99_latency: float = 0.35,
        trailing_silence_secs: float = 0.3,
        **kwargs,
    ):
        """Initialize.

        Args:
            engine: The transcriber (Parakeet via mlx-audio in the plan; a fake in tests).
            ttfs_p99_latency: Fallback wait after VAD stop when no finalized transcript arrives. Segmented
                transcripts are always finalized, so this only matters if transcription fails; 0.35 s is a
                placeholder until the benchmark measures Parakeet's p99 on this Mac.
            trailing_silence_secs: Silence appended to each segment (Pipecat's default is 0.5 s). It costs
                compute, not wall-clock waiting; tune it once the benchmark shows clipped last words or not.
            **kwargs: Passed to SegmentedSTTService (sample_rate, settings, ...).
        """
        kwargs.setdefault("settings", STTSettings(model=None))
        super().__init__(
            ttfs_p99_latency=ttfs_p99_latency,
            trailing_silence_secs=trailing_silence_secs,
            **kwargs,
        )
        self._engine = engine

    @property
    def wants_wav_segments(self) -> bool:
        """Receive raw PCM16, not a WAV container (no temp file, no header parsing)."""
        return False

    def can_generate_metrics(self) -> bool:
        """Report processing time and the finalized-transcript TTFB."""
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Transcribe one VAD segment (raw PCM16 at self.sample_rate) on the MLX thread."""
        await self.start_processing_metrics()
        try:
            pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            text = (await run_mlx(self._engine.transcribe, pcm, self.sample_rate)).strip()
        except Exception as e:  # a failed segment must not take the session down
            await self.stop_processing_metrics()
            yield ErrorFrame(error=f"{self}: transcription failed: {e}")
            return
        await self.stop_processing_metrics()
        if text:
            # finalized=True is set for us by SegmentedSTTService.push_frame (stt_service.py:906-918).
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601())


# --------------------------------------------------------------------------------------------------- TTS


class SpeechEngine(Protocol):
    """A streaming, MLX-backed synthesizer. Every method is called only on the MLX thread."""

    sample_rate: int

    def stream(self, text: str, stop: threading.Event) -> Iterator[np.ndarray]:
        """Return a generator of float32 mono chunks in [-1, 1].

        The generator is created and advanced on the MLX thread, one chunk per call. It should check `stop`
        between chunks and return early when it is set (barge-in). For Qwen3-TTS this wraps mlx-audio's
        generate(..., stream=True, streaming_interval=0.32) with the model's own sampling and
        repetition_penalty >= 1.05 (it over-generates without it); for Kokoro, one sentence per call.
        """
        ...


def float_to_pcm16(chunk: np.ndarray) -> bytes:
    """Clamp float32 audio to [-1, 1] and convert to little-endian PCM16 bytes."""
    return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class MLXStreamingTTSService(TTSService):
    """Streaming TTS over an MLX engine, one chunk per MLX-thread call, leading silence trimmed."""

    def __init__(
        self,
        *,
        engine: SpeechEngine,
        trim_leading_silence: bool = True,
        keep_before_onset_ms: float = 30.0,
        max_lead_scan_secs: float = 0.8,
        **kwargs,
    ):
        """Initialize.

        Args:
            engine: The synthesizer (Qwen3-TTS 1.7B CustomVoice in the plan; Kokoro as fallback).
            trim_leading_silence: Drop the silence before the first audible sample of each utterance.
            keep_before_onset_ms: Audio kept before the detected onset so plosives are not clipped.
            max_lead_scan_secs: Stop looking for an onset after this much audio and play what there is.
            **kwargs: Passed to TTSService. Useful ones: sample_rate (the pipeline's output rate, 24000),
                skip_aggregator_types=["code"], text_filters=[MarkdownTextFilter()].
        """
        kwargs.setdefault("settings", TTSSettings(model=None, voice=None, language=None))
        super().__init__(push_start_frame=True, push_stop_frames=True, **kwargs)
        self._engine = engine
        self._trim = trim_leading_silence
        self._keep_ms = keep_before_onset_ms
        self._max_scan = max_lead_scan_secs

    def can_generate_metrics(self) -> bool:
        """Report TTFB and TTFA (the base measures both, tts_service.py:1847-1854)."""
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Synthesize `text`, yielding audio as the engine produces it."""
        stop = threading.Event()
        engine_rate = self._engine.sample_rate

        def make_stream() -> Iterator[np.ndarray]:
            return self._engine.stream(text, stop)

        def next_chunk(gen: Iterator[np.ndarray]) -> np.ndarray | None:
            try:
                return next(gen)
            except StopIteration:
                return None

        async def pcm_chunks() -> AsyncIterator[bytes]:
            gen = await run_mlx(make_stream)
            lead = bytearray()
            scanning = self._trim
            try:
                while True:
                    chunk = await run_mlx(next_chunk, gen)
                    if chunk is None:
                        break
                    pcm = float_to_pcm16(chunk)
                    if not scanning:
                        yield pcm
                        continue
                    lead.extend(pcm)
                    onset = detect_speech_onset(bytes(lead), engine_rate)
                    if onset is None and len(lead) < self._max_scan * engine_rate * 2:
                        continue  # keep scanning; nothing audible yet
                    scanning = False
                    start = 0 if onset is None else max(0, onset - int(self._keep_ms / 1000 * engine_rate))
                    yield bytes(lead[start * 2 :])
                    lead.clear()
                if lead:  # an utterance shorter than the scan window, or all near-silent: play it as is
                    yield bytes(lead)
            finally:
                # Runs on normal end, on error, and when an interruption cancels this task. Do not await
                # here: the interruption waits for this task to finish cancelling (frame_processor.py:
                # 1256-1260 -> base_object.py:154-165, a 1 s wait_for), so anything slow delays the
                # InterruptionFrame reaching the output transport. Stop the engine loop, then close the
                # generator on the MLX thread after whatever chunk is running there.
                stop.set()
                submit_mlx(getattr(gen, "close", lambda: None))

        try:
            async for frame in self._stream_audio_frames_from_iterator(
                pcm_chunks(), in_sample_rate=engine_rate, context_id=context_id
            ):
                yield frame
        except Exception as e:
            logger.error(f"{self}: synthesis failed: {e}")
            yield ErrorFrame(error=f"{self}: synthesis failed: {e}")


# --------------------------------------------------------------------------------------- fakes (no model)


class FakeTranscribeEngine:
    """Returns a fixed transcript; for construct checks and unit tests."""

    def __init__(self, text: str = "what does my knowledge base say about the hold gate"):
        """Initialize with the transcript to return."""
        self.text = text

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return the fixed transcript."""
        return self.text if audio.size else ""


class FakeSpeechEngine:
    """Emits 0.3 s of leading silence, then a quiet tone, in 80 ms chunks; no model involved."""

    def __init__(self, sample_rate: int = 24000, seconds: float = 1.0):
        """Initialize with an output rate and an utterance length."""
        self.sample_rate = sample_rate
        self.seconds = seconds

    def stream(self, text: str, stop: threading.Event) -> Iterator[np.ndarray]:
        """Yield silence then a 220 Hz tone, checking `stop` between chunks."""
        n = int(0.08 * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        for i in range(int(self.seconds / 0.08)):
            if stop.is_set():
                return
            yield np.zeros(n, np.float32) if i < 4 else (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
