"""Pipecat STT services over the MLX speech adapters: a live one (Nemotron) and a per-utterance one (Parakeet).

Both run every model call on the MLX worker (mlx_worker.py) and keep the user's audio when the hold gate refuses
GPU work: the utterance goes to the session's hold coordinator, which transcribes it once the gate opens (PROTOCOL.md
"Held GPU": keep the transcript if it can, run the turn when the hold ends).

Pipecat 1.12.0 facts used here (note 04c §6): VAD frames reach the STT twice, downstream from the transport for
push-to-talk and upstream from the user aggregator for open mic, so both directions are handled; a TranscriptionFrame
marked finalized lets the Smart Turn stop strategy end the turn without waiting out ttfs_p99_latency.

The echo guard's verdict is asked before a transcript is pushed, interim or final (`transcript_gate`, the echo filter's
gate, set by pipeline.build_session when echo.guard is on): Pipecat's RTVI observer captions the browser page from a
transcript's first push, which is this service's, so echo the filter after it dropped still reached the page as the
person's words (2026-10-05). Dropped echo now never leaves the service; what passes is marked judged (echo_guard.JUDGED)
and the filter lets it by.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.utils.time import time_now_iso8601

from ..echo_guard import JUDGED
from ..engines.base import StreamingTranscriber, Transcriber, pcm16_to_float
from ..mlx_worker import GpuHeldError, MLXWorker
from . import store_settings

HeldAudio = Callable[[bytes, str], Awaitable[None]]   # (pcm16 at 16 kHz, why)
TranscriptGate = Callable[[str, bool], Awaitable[str | None]]   # (text, final) -> the text to push, or None (dropped)


class _EchoGated:
    """The echo guard's question before a push (module docstring), shared by both services."""

    transcript_gate: TranscriptGate | None = None

    async def _judge(self, text: str, final: bool) -> str | None:
        gate = self.transcript_gate
        return text if gate is None else await gate(text, final)

    def _judged(self, frame: Frame) -> Frame:
        if self.transcript_gate is not None:
            frame.metadata[JUDGED] = True
        return frame


# Silence fed to a streaming session just before close(): the right context its last chunk needs. A live segment ends
# 0.2 s after the speech (the VAD's stop_secs), and with that little audio after it Nemotron dropped the last word
# depending on where the chunks fell ("Which animal says moo?" -> "Which animal says" in the 2026-10-05 e2e runs; the
# same recorded audio decoded with 0.1 s more silence gave "moo", and "yellow" got its full stop back). Zeros cost no
# waiting: they are decoded in milliseconds. The segmented path pads the same way (trailing_silence_s).
TAIL_PAD_S = 0.3


async def transcribe_buffer(worker: MLXWorker, engine, pcm: bytes, *, guarded: bool = True,
                            tail_pad_s: float = TAIL_PAD_S, rate: int = 16000) -> str:
    """Transcribe a whole utterance with either kind of engine (used when a held turn resumes)."""
    audio = pcm16_to_float(pcm)
    if isinstance(engine, StreamingTranscriber):
        def run() -> str:
            s = engine.open()
            s.feed(audio)
            if tail_pad_s > 0:
                s.feed(np.zeros(int(tail_pad_s * rate), dtype=np.float32))
            s.close()
            parts: list[str] = []
            for _ in range(100_000):
                if s.done:
                    break
                parts += s.step()
            return "".join(parts)
        text = await worker.run(run, guarded=guarded)
    else:
        text = await worker.run(engine.transcribe, audio, guarded=guarded)
    return " ".join(text.split())


class SegmentedMLXSTTService(_EchoGated, SegmentedSTTService):
    """One transcription per VAD segment (Parakeet), on the MLX thread."""

    def __init__(self, *, engine: Transcriber, worker: MLXWorker, on_held_audio: HeldAudio | None = None,
                 trailing_silence_secs: float = 0.3, ttfs_p99_latency: float = 0.25, **kwargs):
        kwargs.setdefault("settings", store_settings(STTSettings(model=engine.settings.get("model"))))
        super().__init__(trailing_silence_secs=trailing_silence_secs, ttfs_p99_latency=ttfs_p99_latency, **kwargs)
        self._engine = engine
        self._worker = worker
        self._on_held_audio = on_held_audio
        self.last_stt_ms: float | None = None

    @property
    def wants_wav_segments(self) -> bool:
        return False   # raw PCM16 in, no WAV header to strip

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        await self.start_processing_metrics()
        t0 = time.monotonic()
        try:
            text = await self._worker.run(self._engine.transcribe, pcm16_to_float(audio))
        except GpuHeldError as e:
            await self.stop_processing_metrics()
            if self._on_held_audio:
                await self._on_held_audio(audio, str(e))
            return
        except Exception as e:  # noqa: BLE001 - a failed segment must not end the session
            await self.stop_processing_metrics()
            yield ErrorFrame(error=f"{self}: transcription failed: {e}")
            return
        self.last_stt_ms = (time.monotonic() - t0) * 1000
        await self.stop_processing_metrics()
        text = " ".join(text.split())
        heard = await self._judge(text, True) if text else None    # None: no words, or echo the guard dropped
        if heard:   # one transcript per VAD segment, final by construction: the turn may end on it at once
            yield self._judged(TranscriptionFrame(heard, self._user_id, time_now_iso8601(), finalized=True))


class StreamingMLXSTTService(_EchoGated, STTService):
    """A live session per VAD segment (Nemotron 3.5 streaming): partials while speaking, the final on VAD stop.

    The session opens on VADUserStartedSpeakingFrame with the last `preroll_s` of audio (the VAD fires after its
    start_secs of speech), takes every frame until VADUserStoppedSpeakingFrame, then closes; a stepper task runs
    `step()` on the MLX thread whenever audio or the close arrives and pushes InterimTranscriptionFrame for captions
    and one finalized TranscriptionFrame when the session is done (53-54 ms after the audio ends, measured).
    """

    def __init__(self, *, engine: StreamingTranscriber, worker: MLXWorker, on_held_audio: HeldAudio | None = None,
                 preroll_s: float = 0.5, tail_pad_s: float = TAIL_PAD_S, ttfs_p99_latency: float = 0.25,
                 carry_s: float = 0.0, **kwargs):
        """carry_s: a segment that gave no words is decoded again with the next one when that starts within this
        many seconds (0 = off). Nemotron returns nothing for a short word on its own: "Wait," said before a pause
        was lost from "Wait, does it work in winter?" (2026-10-05); decoded together with what follows, it has the
        context it lacked. Nothing was said for the empty segment, so nothing is said twice."""
        kwargs.setdefault("settings", store_settings(STTSettings(model=engine.settings.get("model"))))
        super().__init__(ttfs_p99_latency=ttfs_p99_latency, **kwargs)
        self._engine = engine
        self._worker = worker
        self._on_held_audio = on_held_audio
        self._preroll_s = preroll_s
        self._tail_pad_s = tail_pad_s
        self._carry_s = carry_s
        self._preroll = bytearray()
        self._carry: bytearray | None = None      # an empty segment's audio and everything since
        self._carry_since = 0.0
        self._utt: _Utterance | None = None
        self.last_stt_ms: float | None = None
        self.carried = 0                           # segments decoded again with the next
        self.keep_turn_audio = False               # the tone hook wants the turn's speech (tone_hook.py)
        self._turn_audio = bytearray()

    def take_turn_audio(self) -> bytes:
        """The speech of the segments transcribed since the last call (the tone hook's input), then forgotten."""
        out, self._turn_audio = bytes(self._turn_audio), bytearray()
        return out

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Unused: audio is handled in process_audio_frame."""
        yield None

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection):
        if self._muted or not frame.audio:
            return
        u = self._utt
        if u is not None and not u.closed:
            u.add(frame.audio)
        else:
            self._preroll += frame.audio
            keep = int(self._preroll_s * self.sample_rate) * 2
            if len(self._preroll) > keep:
                del self._preroll[: len(self._preroll) - keep]
            if self._carry is not None:
                if time.monotonic() - self._carry_since > self._carry_s:
                    self._carry = None     # nothing followed soon enough: it was not the start of a sentence
                else:
                    self._carry += frame.audio

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._begin()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._end()

    async def _begin(self):
        if self._utt is not None and not self._utt.closed:
            return   # both copies of a broadcast VAD frame arrive; the first one opened it
        if self._carry is not None:
            pre, self._carry = bytes(self._carry), None
            self.carried += 1
        else:
            pre = bytes(self._preroll)
        self._preroll.clear()
        u = _Utterance(self)
        self._utt = u
        u.add(pre)
        u.task = self.create_task(u.run(), name="stt_utterance")

    async def _end(self):
        u = self._utt
        if u is None or u.closed:
            return
        u.close()

    async def stop(self, frame: EndFrame):
        await self._drop()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._drop()
        await super().cancel(frame)

    async def _drop(self):
        u, self._utt = self._utt, None
        if u is not None:
            await u.discard()

    async def _push_final(self, text: str, started: float):
        self.last_stt_ms = (time.monotonic() - started) * 1000
        # finalized: the turn can end now (stt_service.py push_frame stops the STT TTFB metric on it)
        await self.push_frame(self._judged(TranscriptionFrame(text, self._user_id, time_now_iso8601(), finalized=True)))


class _Utterance:
    """One VAD segment's live session. Audio kept whole, so a hold mid-utterance can still be transcribed later."""

    def __init__(self, svc: StreamingMLXSTTService):
        self.svc = svc
        self.pcm = bytearray()
        self.session = None
        self.closed = False
        self.held: str | None = None
        self.text = ""
        self.closed_at = 0.0
        self.session_closed = False   # session.close() has been called (not merely asked for by the VAD)
        self.wake = asyncio.Event()
        self.task: asyncio.Task | None = None
        self._fed = 0

    def add(self, pcm: bytes) -> None:
        if pcm:
            self.pcm += pcm
            self.wake.set()

    def close(self) -> None:
        self.closed = True
        self.closed_at = time.monotonic()
        self.wake.set()

    async def discard(self):
        if self.task and not self.task.done():
            await self.svc.cancel_task(self.task)
        if self.session is not None:
            self.svc._worker.submit(self.session.cancel)
            self.session = None

    async def run(self):
        svc = self.svc
        try:
            try:
                self.session = await svc._worker.run(svc._engine.open)
            except GpuHeldError as e:
                self.held = str(e)
            while self.held is None:
                await self.wake.wait()
                self.wake.clear()
                # feed what arrived (thread-safe in the session), then step while there is work
                new = bytes(self.pcm[self._fed:])
                self._fed = len(self.pcm)
                if new:
                    self.session.feed(pcm16_to_float(new))
                if self.closed and not self.session_closed:
                    if svc._tail_pad_s > 0:   # right context for the last word (TAIL_PAD_S)
                        self.session.feed(np.zeros(int(svc._tail_pad_s * svc.sample_rate), dtype=np.float32))
                    self.session.close()
                    self.session_closed = True
                try:
                    progressed = await self._drain()
                except GpuHeldError as e:
                    self.held = str(e)
                    break
                if self.session.done:
                    break
                if progressed and not self.closed:
                    heard = await svc._judge(self.text, False)
                    if heard:
                        await svc.push_frame(svc._judged(InterimTranscriptionFrame(heard, svc._user_id,
                                                                                   time_now_iso8601())))
            if self.held is not None:
                if self.session is not None:
                    svc._worker.submit(self.session.cancel)
                    self.session = None
                await self._wait_closed()
                if svc._on_held_audio:
                    await svc._on_held_audio(bytes(self.pcm), self.held)
                return
            text = " ".join(self.text.split())
            heard = await svc._judge(text, True) if text else None    # None: no words, or echo the guard dropped
            if svc.keep_turn_audio and (heard or not text):            # the tone hook hears the person, not echo
                svc._turn_audio += self.pcm
                del svc._turn_audio[: max(0, len(svc._turn_audio) - 60 * 16000 * 2)]   # at most a minute
            if heard:
                await svc._push_final(heard, self.closed_at)
            elif text:
                await svc.stop_ttfb_metrics()     # what the final would have stopped (stt_service.py push_frame)
            elif svc._carry_s > 0 and svc._utt is self:
                svc._carry, svc._carry_since = bytearray(self.pcm), time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - a failed utterance must not end the session
            logger.exception(f"{svc}: live transcription failed")
            await svc.push_error(f"{svc}: live transcription failed: {e}", exception=e)

    async def _wait_closed(self):
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()

    async def _drain(self) -> bool:
        """step() until the session has nothing left to do for the audio fed so far; True if text arrived.

        Keyed on whether session.close() was called, not on the VAD's stop: a stop that lands while this loop runs must
        not keep it stepping a session that was never closed (and so never finishes). That cost 2.4 s before the final
        transcript of a short first segment in the 2026-10-05 e2e run, enough for the turn to end without it."""
        svc = self.svc
        progressed = False
        for _ in range(10_000):
            deltas = await svc._worker.run(self.session.step)
            if deltas:
                self.text += "".join(deltas)
                progressed = True
            if self.session.done:
                break
            if not deltas and not _pending(self.session) and not self.session_closed:
                break
            if self.closed and not self.session_closed:
                break   # the VAD stopped meanwhile: go back, feed the rest and close the session, then drain
        return progressed


def _pending(session) -> bool:
    """Whether a live session still has decode work for the audio already fed.

    mlx-audio 0.5.7's NemotronStreamingSession has no public "pending" query: step() returns [] both when it waits
    for a whole 1.12 s encoder chunk and when it only spent its token budget on blanks. The private queues answer it
    (session.py:36-49, 80-106); a session type without them (the fake) is treated as having nothing pending."""
    inner = getattr(session, "_inner", session)
    encoded = getattr(inner, "_encoded", None)
    if encoded:
        return True
    queued = getattr(inner, "_queued", 0)
    enc = getattr(inner, "_encoder", None)
    if enc is not None and queued:
        hop = inner.model.preprocessor_config.hop_length
        return queued >= enc.chunk_mel * hop
    return False
