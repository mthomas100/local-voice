"""Per-connection behaviour around the GPU hold: the pre-rendered notice, the held turn, and its resumption.

The Mac's rule (2026-10-04): while the hold gate is draining or held nothing may load a model, run
speech inference or call the LLM. So during a hold the session answers with a notice rendered once at startup
(played as raw PCM, no model involved), keeps what the person said (their audio when the STT was refused, their
words when the agent was), and runs the turn when /hold/wait-open returns (PROTOCOL.md "Held GPU"). A reply already
being spoken when a hold begins is cut off with an interruption, so the agent is told what was heard, and resumed the
same way.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from loguru import logger

from pipecat.frames.frames import (Frame, InterruptionFrame, LLMMessagesAppendFrame, TTSAudioRawFrame,
                                   TTSStartedFrame, TTSStoppedFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .hold import HoldMonitor, HoldState
from .mlx_worker import GpuHeldError
from .turnlog import now_local

RESUME_AFTER_CUTOFF = ("(The Mac's GPU was taken for a render while you were answering, so your reply stopped. "
                       "Answer the last request again, briefly.)")


@dataclass
class Notice:
    """A pre-rendered spoken notice: PCM16 mono."""
    text: str
    pcm: bytes
    rate: int

    @property
    def seconds(self) -> float:
        return len(self.pcm) / 2 / self.rate


class NoticePlayer(FrameProcessor):
    """Sits right after the TTS and plays pre-rendered PCM as if the TTS had produced it (no model call)."""

    CHUNK_S = 0.04

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.played: list[str] = []
        self._after_interruption: Notice | None = None

    def play_after_interruption(self, notice: Notice) -> None:
        """Play `notice` right behind the next InterruptionFrame. Frames pushed before an interruption has passed are
        dropped by it downstream, so a notice that follows a cut-off must wait for the cut-off to go by."""
        self._after_interruption = notice

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, InterruptionFrame) and direction == FrameDirection.DOWNSTREAM and self._after_interruption:
            notice, self._after_interruption = self._after_interruption, None
            await self.play(notice)

    async def play(self, notice: Notice) -> None:
        ctx = f"notice-{uuid.uuid4().hex[:8]}"
        self.played.append(notice.text)
        await self.push_frame(TTSStartedFrame(context_id=ctx, append_to_context=False))
        step = int(self.CHUNK_S * notice.rate) * 2
        for i in range(0, len(notice.pcm), step):
            await self.push_frame(TTSAudioRawFrame(notice.pcm[i:i + step], notice.rate, 1, context_id=ctx))
        await self.push_frame(TTSStoppedFrame(context_id=ctx))


class HoldCoordinator:
    """One per connection. Everything the pipeline could not do because of a hold comes here."""

    def __init__(self, *, hold: HoldMonitor, notice: Notice | None, player: NoticePlayer,
                 send: Callable[[dict], Awaitable[None]],
                 transcribe: Callable[[bytes], Awaitable[str]],
                 queue_frame: Callable[[Frame], Awaitable[None]],
                 interrupt: Callable[[], Awaitable[None]] | None = None):
        self.hold = hold
        self.notice = notice
        self.player = player
        self.send = send
        self.transcribe = transcribe
        self.queue_frame = queue_frame
        self.interrupt = interrupt
        self.held_audio: list[tuple[bytes, str]] = []     # (pcm16 at 16 kHz, when the speech started, ISO 8601)
        self.held_text: list[tuple[str, dict]] = []        # (words, how and when they came in)
        self.cut_off = False
        self.notified = False
        self.resumes = 0
        self._hold_sent: tuple[str, str] | None = None
        self._resume_task: asyncio.Task | None = None
        self._unsubscribe = hold.subscribe(self._on_phase)

    def close(self) -> None:
        self._unsubscribe()
        if self._resume_task and not self._resume_task.done():
            self._resume_task.cancel()

    @property
    def pending(self) -> bool:
        return bool(self.held_audio or self.held_text or self.cut_off)

    async def _send_hold(self, phase: str, why: str) -> None:
        if self._hold_sent != (phase, why):
            self._hold_sent = (phase, why)
            await self.send({"t": "hold", "phase": phase, "why": why})

    async def _on_phase(self, old: HoldState, new: HoldState) -> None:
        await self._send_hold(new.phase if new.phase != "absent" else "open", new.holder or new.why)
        if new.allows_gpu:
            self.notified = False

    async def _notify(self, why: str, *, play: bool = True) -> None:
        if self.notified:
            return
        self.notified = True
        st = self.hold.state
        await self._send_hold(st.phase if not st.allows_gpu else "held", st.holder or st.why or why)
        await self.send({"t": "state", "v": "held"})
        if play and self.notice is not None:
            await self.player.play(self.notice)

    def _ensure_resume(self) -> None:
        if self._resume_task is None or self._resume_task.done():
            self._resume_task = asyncio.get_running_loop().create_task(self._resume(), name="hold-resume")

    # -- what the pipeline reports

    async def held_audio_in(self, pcm: bytes, why: str) -> None:
        logger.info(f"hold: kept {len(pcm) / 32000:.1f} s of speech for later ({why})")
        started = now_local() - timedelta(seconds=len(pcm) / 32000)
        self.held_audio.append((pcm, started.isoformat()))
        await self._notify(why)
        self._ensure_resume()

    async def held_text_in(self, text: str, why: str, meta: dict | None = None) -> None:
        logger.info(f"hold: kept the request for later ({why})")
        self.held_text.append((text, dict(meta or {})))
        await self._notify(why)
        self._ensure_resume()

    async def speech_refused(self, why: str) -> None:
        """The TTS was refused mid-reply: cut the reply off (the agent learns what was heard) and resume later.

        Called from inside the TTS's frame-processing task, which the interruption below cancels, so the work runs in
        a task of its own and this returns at once."""
        if self.cut_off:
            return
        self.cut_off = True
        asyncio.get_running_loop().create_task(self._cut_off(why), name="hold-cut-off")

    async def _cut_off(self, why: str) -> None:
        try:
            if self.interrupt is not None:
                if self.notice is not None and not self.notified:
                    self.player.play_after_interruption(self.notice)
                await self._notify(why, play=self.interrupt is None)
                await self.interrupt()
            else:
                await self._notify(why)
            self._ensure_resume()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"hold: cutting the reply off failed: {e}")

    async def pi_waiting(self, message: str) -> None:
        """hold.ts made the Pi run wait inside the turn; it goes on by itself when the hold ends."""
        await self._notify(message)

    # -- the way back

    async def _resume(self) -> None:
        try:
            while True:
                st = await self.hold.wait_open()
                if st.allows_gpu:
                    break
            self.notified = False
            await self.send({"t": "state", "v": "thinking"})
            texts: list[str] = []
            meta: dict = {}
            while self.held_audio:
                pcm, started = self.held_audio[0]
                try:
                    text = await self.transcribe(pcm)
                except GpuHeldError as e:   # held again before we got to it: wait for the next opening
                    await self._notify(str(e))
                    self._ensure_resume_later()
                    return
                self.held_audio.pop(0)
                if text:
                    texts.append(text)
                    meta = meta or {"input": "voice", "t_start": started}
                    await self.send({"t": "transcript", "final": True, "text": text})
            for text, m in self.held_text:
                texts.append(text)
                meta = meta or m
            self.held_text = []
            if self.cut_off:
                self.cut_off = False
                if not texts:
                    texts.append(RESUME_AFTER_CUTOFF)
                    meta = {"system": True}     # the orchestrator's words, not the person's: not in the turn log
            text = " ".join(t for t in texts if t).strip()
            if text:
                self.resumes += 1
                logger.info(f"hold: the gate is open again; running the held turn ({len(text)} chars)")
                msg = {"role": "user", "content": text, "lv": meta}
                await self.queue_frame(LLMMessagesAppendFrame(messages=[msg], run_llm=True))
            else:
                await self.send({"t": "state", "v": "listening"})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception(f"hold: resuming failed: {e}")

    def _ensure_resume_later(self) -> None:
        self._resume_task = asyncio.get_running_loop().create_task(self._resume(), name="hold-resume")
