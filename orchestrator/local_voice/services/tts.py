"""A Pipecat TTS service over the MLX synthesizers: one chunk per MLX-thread call, leading silence trimmed.

Pipecat 1.12.0 facts used here (note 04c §6, tts_service.py):
- run_tts is iterated inside the service's process task, and `_push_tts_frames` awaits the whole generator before the
  next sentence is taken, so within one pipeline sentence N+1 starts after sentence N's generator ends. Across
  pipelines (a phone and a browser at once) a lock per engine keeps one stream per model object (09c §2).
- An InterruptionFrame cancels the process task between yields. The cleanup below never awaits: the interruption
  waits for this task to finish cancelling (at most 1 s), so it only submits closing the generator to the MLX thread,
  where it runs right after the chunk in progress (09c §2: "at most one chunk late").
- Contexts that produce no audio count toward writing the service off after three in a row; a hold legitimately
  silences contexts, so that check is off here (max_consecutive_zero_audio_contexts=0) and the hold coordinator
  speaks the busy notice instead.

Leading silence: only the first sentence of a reply is trimmed, because that is where it adds to the wait;
later sentences keep theirs as the pause between sentences. The onset detector is Pipecat's own (the one its TTFA
metric uses).

Sentence grouping (tts.group, off by default; the rule is speech_text.group_sentences'): a sentence of fewer than
`group_min_words` words is held and said in one generation with the sentence after it. An early live test
(2026-10-05) opened with "Hey!" generated alone, where Qwen3-TTS is known to go off, and every
generation restarts the voice's pitch and pace. Sentences arrive one at a time, so what is held cannot wait for long:
it goes alone at the end of the response (LLMFullResponseEndFrame, EndFrame), before anything else in the stream (a
TTSSpeakFrame such as an acknowledgement, fenced code, a protocol message), or `group_hold_s` after it came with nothing
behind it (the model's words before a tool call: the tool may run for seconds). An interruption drops it, unsaid,
like the rest of the cut reply. Only sentence aggregations are joined, never code (skipped, so it would vanish).
Unlike group_sentences, a short LAST sentence is said alone ("...saved in your notes." then "Done."): joining it to
the sentence before would hold every sentence back for the next. tools/tts_quality_bench.py measures the setting.

The echo guard (echo_guard.py) learns every sentence said here: run_tts records it once the voice has accepted it.

The session recorder (recorder.py, config.yaml record:) gets one "tts" event per generation: its text, when it was asked
for and when its first audio came, its length, the seams between the engine's chunks (tools/tts_bench_scorers.py
compares the spectral flux there with mid-chunk points), how it ended, and a fingerprint, the generation's own first
0.1 s from 10 ms before its first audible sample (-40 dBFS) as PCM16: the bytes the output transport sends are these
bytes (Pipecat resamples nothing when the rates match, base_output.py 1.12), so tools/session_report.py finds each
generation in playback.wav exactly and cuts its clip where it starts.
"""
from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from pipecat.audio.utils import detect_speech_onset
from pipecat.frames.frames import (AggregatedTextFrame, ControlFrame, ErrorFrame, Frame, InterruptionFrame, SystemFrame,
                                   TTSTextFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import AggregationType
from pipecat.utils.text.base_text_filter import BaseTextFilter

from ..engines.base import Synthesizer, float_to_pcm16
from ..mlx_worker import GpuHeldError, MLXWorker
from ..speech_text import speakable, word_count


class SpeakableTextFilter(BaseTextFilter):
    """Every sentence as it should be spoken (speech_text.speakable: no Markdown). Pipecat applies text filters to what
    it synthesizes only: the captions (`reply_text`) and the turn log keep the model's text."""

    async def filter(self, text: str) -> str:
        return speakable(text)


def engine_lock(engine: Synthesizer) -> asyncio.Lock:
    """One stream per model object, process-wide (every pipeline shares the loaded engines). Kept on the engine
    itself, so a lock never outlives its model or crosses event loops."""
    lock = getattr(engine, "_stream_lock", None)
    if lock is None:
        lock = engine._stream_lock = asyncio.Lock()
    return lock


def _next(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


FP_SAMPLES = 2400            # a generation's fingerprint: 0.1 s at 24 kHz ...
FP_LEAD = 240                # ... from 10 ms before its first audible sample ...
FP_ONSET = 328               # ... at -40 dBFS (0.01 of full scale, the echo mixer's and client.speech_end_s's line)
FP_SCAN = 3 * 24000          # looked for in the first 3 s at most


@dataclass
class _Take:
    """One generation as the session recorder sees it (module docstring)."""
    text: str
    context_id: str
    first_of_reply: bool
    t_start: float = field(default_factory=time.monotonic)
    t_first: float | None = None
    samples: int = 0                     # at the service's rate, as yielded
    engine_samples: int = 0              # at the engine's rate
    seams: list[int] = field(default_factory=list)
    head: bytearray = field(default_factory=bytearray)

    def engine_chunk(self, nbytes: int) -> None:
        if self.engine_samples:
            self.seams.append(self.engine_samples)
        self.engine_samples += nbytes // 2

    def audio(self, pcm: bytes) -> None:
        if self.t_first is None and pcm:
            self.t_first = time.monotonic()
        if len(self.head) < 2 * FP_SCAN:
            self.head += pcm[: 2 * FP_SCAN - len(self.head)]
        self.samples += len(pcm) // 2

    def fields(self, status: str, rate: int, engine_rate: int) -> dict:
        a = np.frombuffer(bytes(self.head[: len(self.head) // 2 * 2]), dtype="<i2")
        loud = np.flatnonzero(np.abs(a.astype(np.int32)) >= FP_ONSET)
        at = max(0, int(loud[0]) - FP_LEAD) if loud.size else 0
        k = rate / engine_rate if engine_rate else 1.0
        now = time.monotonic()
        return {"text": self.text, "context_id": self.context_id, "first_of_reply": self.first_of_reply,
                "status": status, "rate": rate, "samples": self.samples,
                "first_audio_s": None if self.t_first is None else round(self.t_first - self.t_start, 4),
                "dur_s": round(now - self.t_start, 4), "seams": [int(round(s * k)) for s in self.seams],
                "fp_at": at, "fp": base64.b64encode(a[at:at + FP_SAMPLES].tobytes()).decode("ascii")}


@dataclass
class _SayHeld(ControlFrame):
    """Queued to the service itself `group_hold_s` after a short sentence was held: said alone if still held then.
    Through the input queue, so it keeps its place among the frames that came meanwhile."""
    n: int = 0


class MLXTTSService(TTSService):
    def __init__(self, *, engine: Synthesizer, worker: MLXWorker, trim_leading_silence: bool = True,
                 keep_before_onset_ms: float = 30.0, max_lead_scan_secs: float = 0.8,
                 on_held: Callable[[str], Awaitable[None]] | None = None, group_min_words: int = 0,
                 group_hold_s: float = 0.5, **kwargs):
        kwargs.setdefault("settings", TTSSettings(model=engine.settings.get("model"), voice=engine.settings.get("voice"),
                                                  language=None))
        kwargs.setdefault("max_consecutive_zero_audio_contexts", 0)
        kwargs.setdefault("text_filters", [SpeakableTextFilter()])
        super().__init__(push_start_frame=True, push_stop_frames=True, **kwargs)
        self._engine = engine
        self._worker = worker
        self._trim = trim_leading_silence
        self._keep_ms = keep_before_onset_ms
        self._max_scan = max_lead_scan_secs
        self._on_held = on_held
        self._audio_contexts_seen: set[str] = set()
        self.last_first_chunk_ms: float | None = None   # run_tts start to the first audible chunk (for /v1/status)
        self.held = False
        self.echo_guard = None                          # echo_guard.EchoGuard, set by the pipeline when echo.guard is on
        self.recorder = None                            # recorder.SessionRecorder, when recording (record.enabled)
        self._takes = 0
        self._group_min = group_min_words
        self._group_hold_s = group_hold_s
        self._short: AggregatedTextFrame | None = None  # the short sentence waiting for the next one
        self._short_n = 0                               # bumped whenever it changes, so a stale _SayHeld does nothing
        self._short_timer: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        return True

    # ------------------------------------------------------------------------------------- sentence grouping

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if self._group_min <= 0 or direction != FrameDirection.DOWNSTREAM:
            await super().process_frame(frame, direction)
            return
        if isinstance(frame, _SayHeld):
            if frame.n == self._short_n and self._short is not None:
                await self._say_short(f"nothing came within {self._group_hold_s} s")
            return
        if isinstance(frame, InterruptionFrame):
            self._drop_short()
        elif isinstance(frame, SystemFrame):
            pass                         # out of band (audio, metrics, urgent messages): no place in the order
        elif self._groupable(frame):
            if self._short is not None:
                frame = _joined(self._short, frame)
                self._forget_short()
            if word_count(frame.text) < self._group_min:
                self._hold_short(frame)
                return
        elif self._short is not None:
            await self._say_short(f"before {type(frame).__name__}")
        await super().process_frame(frame, direction)

    @staticmethod
    def _groupable(frame: Frame) -> bool:
        return (isinstance(frame, AggregatedTextFrame) and not isinstance(frame, TTSTextFrame)
                and frame.aggregated_by == AggregationType.SENTENCE and not frame.skip_tts)

    def _hold_short(self, frame: AggregatedTextFrame) -> None:
        self._forget_short()
        self._short = frame
        n = self._short_n

        async def later():
            await asyncio.sleep(self._group_hold_s)
            if self._short_n == n:
                await self.queue_frame(_SayHeld(n=n))
        self._short_timer = self.create_task(later(), name="tts_group_hold")

    async def _say_short(self, why: str) -> None:
        frame = self._short
        self._forget_short()
        logger.debug(f"{self}: saying {frame.text.strip()!r} alone ({why})")
        await super().process_frame(frame, FrameDirection.DOWNSTREAM)

    def _drop_short(self) -> None:
        if self._short is not None:
            logger.debug(f"{self}: interrupted: {self._short.text.strip()!r} is not said")
        self._forget_short()

    def _forget_short(self) -> None:
        self._short = None
        self._short_n += 1
        t, self._short_timer = self._short_timer, None
        if t is not None and t is not asyncio.current_task() and not t.done():
            t.cancel()

    # ---------------------------------------------------------------------------------------------- speech

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        first_of_reply = context_id not in self._audio_contexts_seen
        self._audio_contexts_seen.add(context_id)
        if len(self._audio_contexts_seen) > 256:
            self._audio_contexts_seen = {context_id}
        trim = self._trim and first_of_reply
        rate = self._engine.sample_rate
        t0 = time.monotonic()
        lock = engine_lock(self._engine)
        take = _Take(text, context_id, first_of_reply) if self.recorder is not None else None
        status = "cut"                     # until the generation runs to its end (an interruption closes it early)

        async def pcm() -> AsyncIterator[bytes]:
            async with lock:
                gen = None
                try:
                    gen = await self._worker.run(self._engine.stream, text)
                    if self.echo_guard is not None:
                        self.echo_guard.said(text)   # what is spoken, as spoken; a held GPU raised before this line
                    lead = bytearray()
                    scanning = trim
                    first = True
                    while True:
                        chunk = await self._worker.run(_next, gen)
                        if chunk is None:
                            break
                        data = float_to_pcm16(np.asarray(chunk, dtype=np.float32))
                        if scanning:
                            lead.extend(data)
                            onset = detect_speech_onset(bytes(lead), rate)
                            if onset is None and len(lead) < self._max_scan * rate * 2:
                                continue
                            scanning = False
                            start = 0 if onset is None else max(0, onset - int(self._keep_ms / 1000 * rate))
                            data, lead = bytes(lead[start * 2:]), bytearray()
                        if first and data:
                            first = False
                            self.last_first_chunk_ms = (time.monotonic() - t0) * 1000
                        if take is not None:
                            take.engine_chunk(len(data))
                        yield data
                    if lead:   # shorter than the scan window, or near-silent throughout: play it as it is
                        if take is not None:
                            take.engine_chunk(len(lead))
                        yield bytes(lead)
                finally:
                    if gen is not None:
                        # never awaited (see the module docstring); runs on the MLX thread after the chunk in progress
                        self._worker.submit(getattr(gen, "close", lambda: None))

        try:
            async for frame in self._stream_audio_frames_from_iterator(pcm(), in_sample_rate=rate, context_id=context_id):
                if take is not None:
                    take.audio(frame.audio)
                yield frame
            self.held = False
            status = "done"
        except GpuHeldError as e:
            status = "held"
            self.held = True
            logger.info(f"{self}: not speaking, {e}")
            if self._on_held:
                await self._on_held(str(e))
        except Exception as e:  # noqa: BLE001
            status = "failed"
            logger.exception(f"{self}: synthesis failed")
            yield ErrorFrame(error=f"{self}: synthesis failed: {e}")
        finally:
            if take is not None:
                self._record_take(take, status, rate)

    def _record_take(self, take: _Take, status: str, engine_rate: int) -> None:
        """The generation's "tts" event (module docstring); never raises into the speech path."""
        rec = self.recorder
        try:
            self._takes += 1
            rec.event("tts", at=take.t_start, gen=self._takes, **take.fields(status, self.sample_rate, engine_rate))
        except Exception as e:  # noqa: BLE001
            logger.error(f"{self}: recording the generation failed: {e}")


def _joined(a: AggregatedTextFrame, b: AggregatedTextFrame) -> AggregatedTextFrame:
    """Two sentence aggregations as one: one generation, one caption, one entry in the assistant's context."""
    out = AggregatedTextFrame(text=f"{a.text.strip()} {b.text.strip()}", aggregated_by=b.aggregated_by,
                              raw_text=f"{(a.raw_text or a.text).strip()} {(b.raw_text or b.text).strip()}")
    out.append_to_context = b.append_to_context
    out.skip_tts = b.skip_tts
    return out
