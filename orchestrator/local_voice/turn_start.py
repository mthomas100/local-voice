"""When the person starts speaking while the agent is busy, the words decide whether the turn starts, not the VAD.

What an early live test showed (2026-10-05, the browser page on the Mac's speakers; tools/echo_turns.py over
its turn log): 5 of 19 voice turns were the agent's own words, the last ~10 s of one reply heard again 6-59 s after
the replies ended. Two of the five began while the agent was busy (0.4 s into a reply; during generation) and cut it:
Pipecat starts the turn, and interrupts, at the VAD's start, before anything is known about the words. Silero started
no turn during any of the eleven replies (Chrome's echo canceller held the live echo); replayed audio is beyond any
canceller's reach, so the check is on the words (echo_guard.py).

BusyHoldStartStrategy takes the place of Pipecat's default start strategies (VAD, then transcription) where
echo.hold_for_words applies (hold_applies):

- agent idle: a VAD start starts the turn at once, and a transcript with no turn starts one (the VAD missed it), as
  the defaults do.
- a spoken yes/no pending (a permission question asked aloud): never held, the turn starts at the VAD start. "Yes" is
  a backchannel word, and an answer must never be swallowed as one.
- agent busy (the bot speaking, followed from the bot frames; or its run in progress, AgentActivity): a VAD start
  holds. The reply pause (bargein.py), when one is in progress, waits for the decision (hold_for_words). The words
  decide:
  - echo the filter reports (echo_guard.echo_filter's on_echo), or a fragment (EchoGuard.fragment_is_echo): the
    reply resumes. On an echo interim the hold goes on (the person's own words may follow); the echo final ends it.
  - a final of backchannel words only ("Mm-hmm.", "Yeah."): the reply resumes and the words leave the aggregation
    (trigger_reset_aggregation), so they never open the next turn's text.
  - other words, interim or final: the turn starts, which interrupts.
  - no words: after max_hold_s with no words and no echo while the VAD still hears speech, the turn starts; once the
    VAD has stopped, final_wait_s with nothing applies no_words (resume or interrupt).
  - the agent no longer busy: any words start the turn.
  A fragment or a backchannel on an interim decides nothing yet ("Yeah, but what about...", "What" of "What time is
  it?"): the final, or more words, does.
- the tail (echo.tail_s, AgentActivity): for that long after the bot stops speaking a VAD start still holds and the
  filter still drops echo, since live echo lags the speaker. Nothing plays then, so there is nothing to resume: a
  final the filter lets through starts the turn, a backchannel or a fragment included (an interim still waits, so a
  longer echo's first words cannot start a turn before the filter has heard enough to call it echo). The spoken cap's
  offer ("Want me to go on?") is answered within a second by "Yes." or "Go on.", which the busy rules would swallow as
  a backchannel and a fragment; an echo of one or two words just after a reply goes through as a turn instead, as echo
  does while idle (only the client's canceller missing it lets live echo reach the recogniser at all: Chrome's held all
  of it in an early live test).

Why the interims decide: a turn started on a final after the VAD stop is ended by TurnAnalyzerUserTurnStopStrategy
0.3 s later (its fallback: ttfs_p99 0.5 minus stop_secs 0.2) without asking Smart Turn, while a turn started on an
interim, with the VAD still hearing speech, leaves Smart Turn the end. Nemotron's interims come about every 1.12 s of
audio, the first soon after the VAD start (the STT's 1 s preroll), its final 0.23-0.25 s after the VAD stop: real
words over a reply mostly interrupt at the first interim. That is still later than the VAD start, and the protocol
v1 `interrupt` would go past M1 DoD 4's 300 ms: hence `browser`, the default, which leaves the apps alone (they run
Apple's voice processing, and their reply pause already stops the audio ~72 ms into speech). Not measured yet on
real models: the planted-echo bench gives the numbers once the GPU is free (2026-10-05: model-free tests only).

Pipecat 1.12 facts this rests on (.venv/.../pipecat/turns/): UserTurnController.process_frame runs every start
strategy, then every stop strategy, on each frame, turn or no turn (STOP ends that loop only); stop strategies'
triggers are ignored while no turn is on. LLMUserAggregator appends every final's text to its aggregation before the
controller sees the frame, and on_reset_aggregation clears it. A turn start calls handle_user_turn_started on every
strategy, then the aggregator broadcasts the interruption. A strategy's own timer task may trigger, as Pipecat's
stop strategies do.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InterimTranscriptionFrame,
                                   TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

from .echo_guard import EchoGuard, words
from .speech_text import is_backchannel

HOLD_MODES = ("off", "browser", "all")


def hold_applies(mode: str, *, protocol_v1: bool, mic: str) -> bool:
    """echo.hold_for_words for one connection: `all` every open-mic entry, `browser` the SmallWebRTC page only, `off`
    none. Never push-to-talk: there the client's own start is the person taking the turn. Under `all` a protocol v1
    barge-in's `interrupt` waits for the first words, past milestone M1's target ("`interrupt` within 300 ms of
    speech start"); the apps do not need it (Apple's voice processing, and the reply pause stops the audio ~72 ms in)."""
    if mic != "vad":
        return False
    return mode == "all" or (mode == "browser" and not protocol_v1)


class AgentActivity:
    """Whether the agent is busy: one answer for the echo filter and the turn-start strategy, so the filter never
    drops words the strategy would treat as the person's, or the reverse.

    run_state() says what the agent's run is doing: "confirm" (a spoken yes/no waits for its answer), "running" or
    "idle" (pipeline.build_session builds it from the agent). The bot's speech is followed here, from the
    BotStarted/StoppedSpeakingFrame the output transport sends upstream past both of them (saw()): a reply's run
    usually ends 0.2-0.84 s after its audio starts (tests/e2e/test_reply_timing.py, 2026-10-05), so most of a reply
    plays with no run in progress.

    The tail (echo.tail_s): the agent counts as busy for `tail_s` after the bot stops speaking. Pipecat says "Bot
    stopped speaking" when the server's audio has gone out (base_output.py 1.12: the TTSStoppedFrame after the last
    chunk written), but live echo lags it by the whole echo path (the client's playout, the room, the microphone, the
    uplink): in the harness the echo of short replies started 137 ms after it (the planted-echo bench,
    2026-10-05), when the agent was idle, so the guard let it through as a turn; and an echo begun during a reply has its
    final transcript 0.25-0.45 s after the echo ends (the VAD's 0.2 s stop, then Nemotron's 23-73 ms), after the reply
    has ended. Live echo cannot outlast the reply by more than its path's delay, so 1.0 s covers delays up to about
    0.5 s. While only the tail makes it busy nothing is playing: in_tail() lets the strategy tell that apart."""

    def __init__(self, run_state: Callable[[], str] = lambda: "idle", *, tail_s: float = 0.0,
                 clock: Callable[[], float] = time.monotonic):
        self.run_state = run_state
        self.tail_s = tail_s
        self.bot_speaking = False
        self.stopped_at: float | None = None      # the last "Bot stopped speaking" (clock seconds)
        self._clock = clock

    def saw(self, frame: Frame) -> None:
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self.bot_speaking:             # the strategy and the filter both see it: the first one sets the time
                self.stopped_at = self._clock()
            self.bot_speaking = False

    def confirm_pending(self) -> bool:
        return self.run_state() == "confirm"

    def in_tail(self) -> bool:
        """The bot stopped speaking less than tail_s ago (whatever the run does)."""
        return (not self.bot_speaking and self.stopped_at is not None and self.tail_s > 0
                and self._clock() - self.stopped_at < self.tail_s)

    def playing(self) -> bool:
        """Busy with something a held reply can go back to: the bot speaking or the run in progress, not the tail."""
        return self.bot_speaking or self.run_state() != "idle"

    def busy(self) -> bool:
        return self.playing() or self.in_tail()

    def why(self) -> str:
        if self.bot_speaking:
            return "speaking"
        state = self.run_state()
        if state == "idle" and self.in_tail():
            return f"just done speaking (its echo may still come back for {self.tail_s:g} s)"
        return {"running": "thinking or running tools", "confirm": "waiting for a yes or no"}.get(state, "idle")


@dataclass
class Decision:
    """One decision of the strategy, kept for the tests and logged at INFO."""
    action: str            # start | hold | resume | drop
    reason: str            # REASONS' keys
    text: str = ""         # the transcript that decided, if any
    final: bool | None = None
    held_s: float = 0.0    # how long the hold had lasted
    at: float = field(default_factory=time.monotonic)


REASONS = {
    "idle": "the agent is idle",
    "confirm": "a spoken yes/no is pending",
    "busy": "speech while the agent is {why}",
    "words": "the person's own words",
    "echo": "the agent's own speech heard back",
    "fragment": "a fragment of the agent's last sentences",
    "backchannel": "a backchannel",
    "no_words": "no words came",
    "max_hold": "speech with no words for {max_hold_s} s",
    "no_longer_busy": "the agent is no longer busy",
    "tail": "words just after the agent stopped speaking, with nothing playing to resume",
}


@dataclass
class _Hold:
    since: float
    vad_speaking: bool
    text: str = ""                    # the last transcript, undecided (a fragment or a backchannel interim)
    echo: bool = False                # the filter reported echo during this hold
    echo_resumed: bool = False        # ... and the reply was resumed for it
    max_timer: asyncio.Task | None = None
    final_timer: asyncio.Task | None = None


class BusyHoldStartStrategy(BaseUserTurnStartStrategy):
    """The module docstring's rules. `pause` (bargein.ReplyPause) is set by the pipeline once the output exists, and
    only on protocol v1 with the pause on; `guard` (for fragments) when echo.guard is on. `decisions` keeps them all."""

    def __init__(self, *, activity: AgentActivity, guard: EchoGuard | None = None, max_hold_s: float = 1.5,
                 final_wait_s: float = 0.6, no_words: str = "resume", **kwargs):
        super().__init__(**kwargs)
        if no_words not in ("resume", "interrupt"):
            raise ValueError(f"no_words must be resume or interrupt, not {no_words!r}")
        self.activity = activity
        self.guard = guard
        self.pause = None
        self.max_hold_s = max_hold_s
        self.final_wait_s = final_wait_s
        self.no_words = no_words
        self.decisions: list[Decision] = []
        self.on_decision: Callable[[Decision], None] | None = None   # the session recorder's hook (recorder.py)
        self._hold: _Hold | None = None
        self._in_turn = False
        self._vad_speaking = False

    # -- Pipecat's hooks

    async def handle_user_turn_started(self):
        self._in_turn = True
        self._end_hold()
        if self.pause is not None:
            self.pause.stop_holding()

    async def handle_user_turn_stopped(self):
        self._in_turn = False

    async def cleanup(self):
        self._end_hold()
        await super().cleanup()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        self.activity.saw(frame)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            return await self._vad_started()
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_stopped()
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            return await self._transcript(frame.text, isinstance(frame, TranscriptionFrame))
        return ProcessFrameResult.CONTINUE

    # -- from the echo filter (its own task: nothing here awaits the aggregator)

    async def on_echo(self, final: bool, rest: str) -> None:
        """echo_guard.echo_filter dropped a transcript as the agent's own speech while it was busy (rest "") or cut it
        to the person's own words around the echo (rest: that frame goes on and decides when it gets here)."""
        h = self._hold
        if h is None or self._in_turn:
            return
        h.echo = True
        if rest:
            return
        if final:
            self._resume("echo", final=True, echo=True)
        elif not h.echo_resumed:
            h.echo_resumed = True
            self._resume("echo", final=False, echo=True, end=False)

    # -- inside

    async def _vad_started(self) -> ProcessFrameResult:
        self._vad_speaking = True
        if self._in_turn:
            return ProcessFrameResult.CONTINUE        # a pause inside the person's own turn
        h = self._hold
        if h is not None:                             # speech again within the hold: the words still decide
            h.vad_speaking = True
            _cancel(h.final_timer)
            h.final_timer = None
            _cancel(h.max_timer)
            h.max_timer = self._arm(self._max_hold(h))
            return ProcessFrameResult.CONTINUE
        if self.activity.confirm_pending():
            return await self._start("confirm")
        if not self.activity.busy():
            return await self._start("idle")
        self._begin_hold(vad_speaking=True)
        return ProcessFrameResult.CONTINUE

    def _vad_stopped(self) -> None:
        self._vad_speaking = False
        h = self._hold
        if h is None or self._in_turn:
            return
        h.vad_speaking = False
        _cancel(h.max_timer)
        h.max_timer = None
        _cancel(h.final_timer)
        h.final_timer = self._arm(self._no_final(h))

    async def _transcript(self, text: str, final: bool) -> ProcessFrameResult:
        if not words(text) or self._in_turn:
            return ProcessFrameResult.CONTINUE
        h = self._hold
        if self.activity.confirm_pending():
            return await self._start("confirm", text, final)
        if not self.activity.busy():
            return await self._start("idle" if h is None else "no_longer_busy", text, final)
        kind = self._kind(text)
        if kind == "words":
            return await self._start("words", text, final)
        if final and not self.activity.playing():
            # the tail: the filter dropped what was echo before it got here, so this is the person's (an interim still
            # waits: the first words of a longer echo come as a fragment before the filter can call them echo)
            return await self._start("tail", text, final)
        if final:
            # resumed; out of the aggregation, which took the text before the controller showed it to us
            self._resume(kind, text, final=True)
            await self.trigger_reset_aggregation()
            return ProcessFrameResult.CONTINUE
        if h is None:     # an interim with no hold: the VAD missed this speech's start
            h = self._begin_hold(vad_speaking=self._vad_speaking)
        h.text = text
        self._record("hold", kind, text, final=False)
        return ProcessFrameResult.CONTINUE

    def _kind(self, text: str) -> str:
        if is_backchannel(text):
            return "backchannel"
        if self.guard is not None and self.guard.fragment_is_echo(text):
            return "fragment"
        return "words"

    def _begin_hold(self, *, vad_speaking: bool) -> _Hold:
        h = self._hold = _Hold(since=time.monotonic(), vad_speaking=vad_speaking)
        if self.pause is not None:
            self.pause.hold_for_words()
        if vad_speaking:
            h.max_timer = self._arm(self._max_hold(h))
        else:
            h.final_timer = self._arm(self._no_final(h))
        self._record("hold", "busy")
        return h

    async def _max_hold(self, h: _Hold) -> None:
        await asyncio.sleep(self.max_hold_s)
        if self._hold is not h or self._in_turn:
            return
        h.max_timer = None
        if h.vad_speaking and not h.text and not h.echo:
            await self._start("max_hold")

    async def _no_final(self, h: _Hold) -> None:
        await asyncio.sleep(self.final_wait_s)
        if self._hold is not h or self._in_turn or h.vad_speaking:
            return
        h.final_timer = None
        if not self.activity.busy():
            self._record("drop", "no_longer_busy", h.text)    # nothing to resume, no words to start a turn with
            self._end_hold()
        elif h.echo:
            self._resume("echo")
        elif h.text:                       # its final never came: the last interim decides
            kind = self._kind(h.text)      # (what was said since may have changed a fragment)
            if kind == "words":
                await self._start(kind, h.text, final=False)
            elif not self.activity.playing():
                await self._start("tail", h.text, final=False)
            else:
                self._resume(kind, h.text, final=False)
        elif self.no_words == "interrupt":
            await self._start("no_words")
        else:
            self._resume("no_words")

    async def _start(self, reason: str, text: str = "", final: bool | None = None) -> ProcessFrameResult:
        self._record("start", reason, text, final)
        self._end_hold()
        if self.pause is not None:
            self.pause.stop_holding()
        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP

    def _resume(self, reason: str, text: str = "", *, final: bool | None = None, echo: bool = False,
                end: bool = True) -> None:
        self._record("resume", reason, text, final)
        if self.pause is not None:
            self.pause.resume_now(echo=echo)
        if end:
            self._end_hold()

    def _end_hold(self) -> None:
        h, self._hold = self._hold, None
        if h is not None:
            _cancel(h.max_timer)
            _cancel(h.final_timer)
            if self.pause is not None:
                self.pause.speech_over()

    def _arm(self, coro) -> asyncio.Task:
        return self.task_manager.create_task(coro, f"{self}::hold_timer")

    def _record(self, action: str, reason: str, text: str = "", final: bool | None = None) -> None:
        h = self._hold
        held = 0.0 if h is None else time.monotonic() - h.since
        self.decisions.append(Decision(action, reason, text, final, round(held, 3)))
        del self.decisions[:-200]
        if self.on_decision is not None:
            try:
                self.on_decision(self.decisions[-1])
            except Exception as e:  # noqa: BLE001 - a listener never breaks the turn taking
                logger.error(f"{self}: on_decision failed: {e}")
        why = REASONS[reason].format(why=self.activity.why(), max_hold_s=self.max_hold_s)
        said = "" if not text else f" {'final' if final else 'interim'} {text!r}"
        logger.info(f"{self}: {action} ({why}) after holding {held:.2f} s{said}")


def _cancel(task: asyncio.Task | None) -> None:
    """Cancel a timer, never the one running now (a decision taken from inside it ends its own hold)."""
    if task is not None and task is not asyncio.current_task() and not task.done():
        task.cancel()
