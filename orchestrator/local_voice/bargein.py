"""Barge-in that stops the reply at the first sign of speech, without interrupting on every cough.

Pipecat interrupts when the VAD reaches SPEAKING: Silero needs `start_secs` (5 frames of 32 ms at 0.15) of frames it
scores as speech. Fricatives score low, so "Stop" was interrupted 336-346 ms after it began in the e2e runs: Silero
gives the "st" 0.2-0.3 and the vowel 0.84 at 208 ms (2026-10-05, tools/bargein_bench.py). A shorter start_secs is
faster but interrupts on more coughs and sneezes (bench: 24 of 260 ESC-50 controls at 0.15, 37 at 0.1, 43 at 0.064).

So the reply's audio stops being sent at the first frame scored as speech (`cue_confidence`, `cue_frames`) while a
reply is playing; the client's playout buffer is only the transport's 40 ms pacing, so the person hears it stop at
once. The interruption itself still waits for the VAD to confirm (start_secs, as before): then Pipecat broadcasts
it, the client gets `interrupt`, and the held audio is dropped. If the VAD has not confirmed within `resume_after_s`
(and nothing has sounded like speech for a moment), the reply carries on from where it stopped. A cough costs a short
pause, not the reply. Bench, 110 interruptions in 5 voices: the pause comes at a median of 72 ms (max 240) after
speech starts, against 275 ms (max 860) for the interrupt; on 260 ESC-50 controls it pauses on 92 (coughs, sneezes,
laughs) and interrupts on the same 24 as before.

Protocol v1 only: the WebSocket output paces audio in real time, so holding it back stops the client. Push-to-talk
has no VAD and is unaffected. The browser path (SmallWebRTC) keeps Pipecat's behaviour.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, InterruptionFrame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.websocket.fastapi import FastAPIWebsocketOutputTransport, FastAPIWebsocketTransport

# How long after a frame scored as speech a pause may still be held: Silero scores the gaps between a fricative's
# noise and the vowel below the cue, and a paused reply must not resume in the middle of a word the VAD is about to
# confirm.
HANGOVER_S = 0.1
# Reply audio is "playing" this long after the last chunk was sent (the transport sends 40 ms chunks in real time).
AUDIO_ACTIVE_S = 0.12
# Unconfirmed pauses allowed in a window before pausing stops for the rest of it: speech-like sound that never becomes
# speech the VAD confirms (a television, a distant voice scoring between cue_confidence and the VAD's confidence)
# would otherwise stutter the reply over and over. Past the budget, the VAD's interruption is all there is, as before.
UNCONFIRMED_BUDGET = 2
UNCONFIRMED_WINDOW_S = 10.0


@dataclass
class PauseSettings:
    cue_confidence: float = 0.5
    cue_frames: int = 1
    min_volume: float = 0.5          # the VAD's own loudness floor (turn.vad.min_volume)
    resume_after_s: float = 0.4
    max_pause_s: float = 1.5         # never hold a reply longer than this without a confirmed interruption


@dataclass
class PauseEvent:
    kind: str                        # paused | resumed | confirmed
    at: float                        # time.monotonic()
    held_ms: float | None = None


@dataclass
class ReplyPause:
    """The pause gate between the VAD (which reports every frame it scores) and the output transport (which waits at
    the gate before sending each audio chunk). Everything runs on the event loop except frame_threadsafe()."""
    settings: PauseSettings = field(default_factory=PauseSettings)
    loop: asyncio.AbstractEventLoop | None = None
    on_event: Callable[[PauseEvent], None] | None = None

    def __post_init__(self):
        self.loop = self.loop or asyncio.get_running_loop()
        self._open = asyncio.Event()
        self._open.set()
        self.generation = 0          # bumped by every interruption; audio held across one belongs to a cut reply
        self.paused_at: float | None = None
        self.events: list[PauseEvent] = []
        self._run = 0
        self._last_speechy = 0.0
        self._audio_until = 0.0
        self._timer: asyncio.TimerHandle | None = None
        self._unconfirmed: list[float] = []   # when recent pauses ended without a confirmed interruption
        self._holding = False      # the turn-start strategy waits for the words (turn_start.py): no timed resume
        self._echo = False         # the speech still heard is the agent's own voice (turn_start.py): never pause on it

    # -- from the VAD (its executor thread)

    def frame_threadsafe(self, confidence: float, volume: float) -> None:
        self.loop.call_soon_threadsafe(self.frame, confidence, volume, time.monotonic())

    def frame(self, confidence: float, volume: float, at: float | None = None) -> None:
        at = time.monotonic() if at is None else at
        s = self.settings
        speechy = confidence >= s.cue_confidence and volume >= s.min_volume
        if speechy:
            self._last_speechy = at
        if self.paused_at is not None or self._echo:
            return
        self._run = self._run + 1 if speechy else 0
        # "playing" is judged now, on the loop: a frame scored just before an interruption can be handled just after
        # it (seen in the e2e of 2026-10-05 13:31, which paused again 37 ms after the confirmed interruption)
        if self._run >= s.cue_frames and self.playing() and self._budget_left():
            self._pause(at)

    def _budget_left(self) -> bool:
        now = time.monotonic()
        self._unconfirmed = [t for t in self._unconfirmed if now - t < UNCONFIRMED_WINDOW_S]
        return len(self._unconfirmed) < UNCONFIRMED_BUDGET

    # -- from the output transport

    def playing(self, at: float | None = None) -> bool:
        return (time.monotonic() if at is None else at) < self._audio_until

    def audio_sent(self) -> None:
        self._audio_until = time.monotonic() + AUDIO_ACTIVE_S

    async def wait_open(self) -> None:
        await self._open.wait()

    def interrupting(self) -> None:
        """An InterruptionFrame reached the output: the VAD confirmed (or the turn was cut off another way). Nothing
        is playing after it. The gate stays shut until interrupted(): opened now, the held writer would drop its own
        chunk and then send the next one still queued before Pipecat empties the queue (e2e 2026-10-05 13:35: stale
        audio went out 17 ms after the interruption, and the cue paused on it)."""
        self.generation += 1
        self._audio_until = 0.0
        self._echo = False

    def interrupted(self) -> None:
        """The output has dropped the cut reply's audio; the gate opens for the next reply."""
        if self.paused_at is not None:
            self._release("confirmed")

    # -- from the turn-start strategy (turn_start.py)

    def hold_for_words(self) -> None:
        """The VAD heard speech while the agent was busy, and the words decide: echo of the agent's own voice and a
        backchannel ("mm-hm") resume the reply, anything else interrupts it. A pause in progress no longer resumes on
        its timer, only at `max_pause_s` (the strategy decides before that). No pause in progress: nothing to hold, the
        reply plays on while the words come (the browser path, or a spent pause budget)."""
        self._holding = self.paused_at is not None

    def resume_now(self, *, echo: bool = False) -> None:
        """The words were echo or a backchannel: the reply carries on now. echo: the speech the VAD still hears is the
        agent's own voice coming back, so it does not pause the reply again until speech_over(). Otherwise every echo
        word after the resume would pause it anew (a replayed reply runs for seconds), each pause held up to
        max_pause_s, until two had spent the false-alarm budget: a stuttering reply. The person's own words after the
        echo are the strategy's to catch (they start the turn, which interrupts)."""
        self._holding = False
        self._echo = self._echo or echo
        if self.paused_at is not None:
            self._unconfirmed.append(time.monotonic())
            self._release("resumed")

    def speech_over(self) -> None:
        """The speech the strategy was deciding on is over (its hold ended): the next sound may pause the reply."""
        self._echo = False

    def stop_holding(self) -> None:
        """The decision went to an interruption (or nothing is held any more): the interruption releases the pause."""
        self._holding = False
        self._echo = False

    # -- inside

    def _emit(self, ev: PauseEvent) -> None:
        self.events.append(ev)
        if self.on_event:
            self.on_event(ev)

    def _pause(self, at: float) -> None:
        self._open.clear()
        self.paused_at = at
        self._emit(PauseEvent("paused", at))
        self._timer = self.loop.call_later(self.settings.resume_after_s, self._check)

    def _check(self) -> None:
        if self.paused_at is None:
            return
        now = time.monotonic()
        held = now - self.paused_at < self.settings.max_pause_s
        if held and (self._holding or now - self._last_speechy < HANGOVER_S):
            self._timer = self.loop.call_later(HANGOVER_S, self._check)
            return
        self._holding = False
        self._unconfirmed.append(now)
        self._release("resumed")

    def _release(self, kind: str) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        now = time.monotonic()
        held = None if self.paused_at is None else (now - self.paused_at) * 1000
        self.paused_at = None
        self._holding = False
        self._run = 0
        self._open.set()
        self._emit(PauseEvent(kind, now, held))


class SpeechCueMixin:
    """For a Pipecat VADAnalyzer: report each analysed frame's confidence and smoothed volume to `cue_sink` as soon
    as it is scored (in the analyzer's executor thread). VADAnalyzer._run_analyzer (1.12.0) calls voice_confidence and
    then _get_smoothed_volume once per frame, in that order."""
    cue_sink: Callable[[float, float], None] | None = None
    _cue_confidence: float = 0.0

    def voice_confidence(self, buffer: bytes) -> float:
        c = super().voice_confidence(buffer)  # type: ignore[misc]
        self._cue_confidence = float(np.asarray(c, dtype=np.float32).ravel()[0]) if np.size(c) else 0.0
        return c

    def _get_smoothed_volume(self, audio: bytes) -> float:
        v = super()._get_smoothed_volume(audio)  # type: ignore[misc]
        sink = self.cue_sink
        if sink is not None:
            sink(self._cue_confidence, float(v))
        return v


def cue_silero(**kwargs):
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    class CueSileroVADAnalyzer(SpeechCueMixin, SileroVADAnalyzer):
        pass

    return CueSileroVADAnalyzer(**kwargs)


class PausableWebsocketOutput(FastAPIWebsocketOutputTransport):
    """Pipecat's WebSocket output, with the pause gate in front of every audio chunk it sends.

    on_audio_sent(pcm, rate, at): told of every chunk that went out, `at` the moment it was sent (after the gate; the
    session recorder's playback track, recorder.py). It must not raise or block; a failure in it is logged, never
    passed to the transport."""

    reply_pause: ReplyPause | None = None
    on_audio_sent: Callable[[bytes, int, float], None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        p = self.reply_pause
        if isinstance(frame, InterruptionFrame) and p is not None:
            p.interrupting()
            try:
                await super().process_frame(frame, direction)   # Pipecat drops the queued audio, restarts the writer
            finally:
                p.interrupted()
            return
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        p = self.reply_pause
        if p is not None:
            gen = p.generation
            await p.wait_open()
            if p.generation != gen:
                return False             # held across an interruption: it belongs to the reply that was cut off
            p.audio_sent()
        at = time.monotonic()            # Pipecat's write sends at once, then sleeps out the chunk (fastapi.py 1.12)
        ok = await super().write_audio_frame(frame)
        sent = self.on_audio_sent
        if ok and sent is not None:
            try:
                sent(frame.audio, frame.sample_rate, at)
            except Exception as e:  # noqa: BLE001 - a listener never breaks the audio path
                logger.error(f"{self}: on_audio_sent failed: {e}")
        return ok


class PausableWebsocketTransport(FastAPIWebsocketTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the same construction as FastAPIWebsocketTransport.__init__ (1.12.0), with the gated output in its place
        self._output = PausableWebsocketOutput(self, self._client, self._params, name=self._output_name)


def log_events(device: str) -> Callable[[PauseEvent], None]:
    def log(ev: PauseEvent) -> None:
        if ev.kind == "paused":
            logger.info(f"barge-in {device}: speech cue, reply paused")
        else:
            logger.info(f"barge-in {device}: {ev.kind} after {ev.held_ms:.0f} ms")
    return log
