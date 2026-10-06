"""The final transcript as a second opinion on Smart Turn, in both directions (2026-10-05).

Smart Turn v3.2 hears only the audio. tools/smart_turn_bench.py found it calling 24 of 100 finished `say` requests
INCOMPLETE, often with a probability near zero ("What does a heat pump do?", Karen: 0.018), and every such turn waits
for the analyzer's silence fallback (turn.smart_turn.stop_secs, 3 s): in the e2e run of 14:07 that answer started
4.3 s after the person stopped. A lower probability bar cannot fix that. And at a pause spliced into the middle of a
sentence ("What's the difference between ... a heat pump and a furnace?") it said COMPLETE in 32 of 80 cases
(tools/turn_end_bench.py), which cuts the person off. Nemotron punctuates, so the words can say both things:

- a final transcript that ends a sentence (?, . or !) and not on a word no sentence ends on ("the", "of", "and",
  "um"): `ends_turn`; with Smart Turn unsure, the turn ends `punctuation_stop_secs` after the VAD's stop;
- one that ends on a comma, a dash, an ellipsis or such a word: `goes_on`; with Smart Turn sure, the turn still waits
  `veto_wait_secs` for the rest of the sentence.

Anything else leaves Smart Turn's verdict alone. config.yaml turn.smart_turn.words sets both; tools/turn_end_bench.py
measures them on `say` speech, with the real recogniser's punctuation. Both stay off (the recogniser punctuates
fragments too); a third rule is on:

- a final transcript the orchestrator answers itself, complete as said (a space or mode switch alone, "note this"
  alone, a bare yes or no to a permission question; `is_command`, the agent's): with Smart Turn unsure, the turn ends
  `command_stop_secs` after the VAD's stop. Smart Turn called "Back home.", "Just talk." and "Yes." INCOMPLETE in the
  M3 e2e of 2026-10-05 16:22, and each waited for the 2 s fallback.

And a fourth, on by default since it can only keep a turn open, never end one early:

- a final transcript whose last word is a filler or a joining word ("um", "uh", "like", "and", "so", "but", "the":
  `HOLD_WORDS`), whatever punctuation the recogniser put after it: with Smart Turn sure, the turn still waits for the
  silence fallback (`hold_secs`, turn.smart_turn.stop_secs). Rambling requests are full
  of them: "Could you go to my journal. And uh. Just put a Um. So. I think like a short note that just. ..."
  (tools/turn_end_bench.py measures it on spliced `say` speech with fillers before pauses).
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable

from loguru import logger

from pipecat.frames.frames import (Frame, TranscriptionFrame, VADUserStartedSpeakingFrame,
                                   VADUserStoppedSpeakingFrame)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.types import ProcessFrameResult

SENTENCE_END = re.compile(r"[?.!]['\"”’)]*\s*$")
# Words a sentence does not end on: articles, possessives, prepositions that take an object, conjunctions, fillers.
# A transcript ending on one of these is a sentence still going on, whatever punctuation the recogniser put after it.
# Words that can end a sentence ("in", "on", "that", "her", "it", "you") are left out on purpose.
DANGLING = frozenset("""a an the my your our their his its and or but nor so because if whether than of to for with
about from into onto upon via like between among through toward towards across against during without within behind
beyond versus including regarding um uh er erm hmm mm""".split())
GOES_ON = re.compile(r"(,|;|:|-|–|—|\.\.\.|…)['\"”’)]*\s*$")
# Words a person trails off on before going on: the dangling words, and "just" ("a short note that just. Call the
# plumber ...").
HOLD_WORDS = DANGLING | {"just"}
# A question may end on one of these ("What does it look like?", "Where did that come from?"): such a question is done.
PREPOSITIONS = frozenset("""of to for with about from into onto upon via like between among through toward towards
across against during without within behind beyond versus including regarding""".split())


def holds(text: str, *, except_questions: bool = False) -> bool:
    """A final transcript that stops on a filler or a joining word: the person is not done ("And um.", "Like.").
    except_questions: not when it is a question ending on a preposition ("What does a heat pump look like?")."""
    if last_word(text) not in HOLD_WORDS:
        return False
    return not (except_questions and text.rstrip().endswith("?") and last_word(text) in PREPOSITIONS)


def last_word(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.lower())
    return words[-1] if words else ""


def ends_turn(text: str) -> bool:
    """A final transcript that reads as a finished sentence."""
    t = text.strip()
    return bool(t) and bool(SENTENCE_END.search(t)) and last_word(t) not in DANGLING


def goes_on(text: str, *, unpunctuated: bool = False) -> bool:
    """A final transcript that reads as a sentence still going on: it ends on a comma, a dash or an ellipsis, or on a
    dangling word; with `unpunctuated`, also when it has no punctuation at its end at all."""
    t = text.strip()
    if not t:
        return False
    if GOES_ON.search(t) or last_word(t) in DANGLING:
        return True
    return unpunctuated and not SENTENCE_END.search(t)


class WordsTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Pipecat's Smart Turn stop strategy, with the final transcript as a second opinion in both directions:

    - early end: Smart Turn said INCOMPLETE, and the final transcript `ends_turn`: the turn ends `punctuation_stop_secs`
      after the VAD stop instead of after the analyzer's silence fallback (3 s);
    - veto: Smart Turn said COMPLETE, but the final transcript `goes_on` (a comma, a dangling word; with
      `veto_unpunctuated`, no punctuation at all): the turn waits `veto_wait_secs` more for the rest of the sentence.

    Either is off when its setting is None. Pinned to pipecat-ai 1.12.0's TurnAnalyzerUserTurnStopStrategy
    (turns/user_stop/turn_analyzer_user_turn_stop_strategy.py): `_text` holds the last final transcript,
    `_turn_complete` the analyzer's verdict for the latest VAD stop, `_transcript_finalized` whether a final came
    after it, `_vad_user_speaking` the VAD; `_maybe_trigger_user_turn_stopped()` ends the turn when the verdict is
    COMPLETE and a final is in, through `trigger_user_turn_stopped()`. The analyzer's own silence fallback also ends
    the turn through it; that path is never vetoed (its COMPLETE does not come from the model at a VAD stop)."""

    def __init__(self, *, punctuation_stop_secs: float | None = None, veto_wait_secs: float | None = None,
                 veto_unpunctuated: bool = False, command_stop_secs: float | None = None,
                 hold_secs: float | None = None, hold_except_questions: bool = True,
                 is_command: Callable[[str], bool] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._command_secs = command_stop_secs
        self.is_command = is_command               # set by the pipeline once the agent exists
        self._punct_secs = punctuation_stop_secs
        self._veto_secs = veto_wait_secs
        self._veto_unpunct = veto_unpunctuated
        self._hold_secs = hold_secs                 # a COMPLETE on a filler or joining word waits this long (the fallback)
        self._hold_q = hold_except_questions
        self.holds = 0
        self._stopped_at: float | None = None     # monotonic time of the latest VAD stop
        self._final_after_stop = False            # a final transcript arrived after it
        self._model_complete = False              # the analyzer said COMPLETE at that stop
        self._timer: asyncio.Task | None = None   # the early end, or the end after a veto
        self.early_ends = 0
        self.command_ends = 0
        self.vetoes = 0

    # The base class ends the turn from inside its own handlers, so what the veto needs is recorded before they run.
    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._stopped_at = time.monotonic()
        self._final_after_stop = False
        self._model_complete = False
        await super()._handle_vad_user_stopped_speaking(frame)
        await self._maybe_end_early()

    async def _handle_prediction_result(self, result):
        # The model's verdict at a VAD stop carries its metrics; the analyzer's silence fallback comes with none.
        self._model_complete = bool(result is not None and getattr(result, "is_complete", False))
        await super()._handle_prediction_result(result)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if frame.finalized and self._stopped_at is not None:
            self._final_after_stop = True
        await super()._handle_transcription(frame)
        await self._maybe_end_early()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._forget()
        return await super().process_frame(frame)

    async def handle_user_turn_started(self):
        await super().handle_user_turn_started()
        await self._forget()

    async def handle_user_turn_stopped(self):
        await super().handle_user_turn_stopped()
        await self._forget()

    async def cleanup(self):
        await self._cancel_timer()
        await super().cleanup()

    async def trigger_user_turn_stopped(self):
        if self._model_complete and self._final_after_stop and self._stopped_at is not None:
            wait = None
            if self._veto_secs is not None and goes_on(self._text, unpunctuated=self._veto_unpunct):
                wait = self._veto_secs
                self.vetoes += 1
            if self._hold_secs is not None and holds(self._text, except_questions=self._hold_q):
                wait = max(wait or 0.0, self._hold_secs)
                self.holds += 1
            if wait is not None:
                self._model_complete = False
                self._turn_complete = False
                logger.debug(f"{self}: Smart Turn said complete, but {self._text!r} goes on: waiting {wait} s")
                await self._arm(self._stopped_at + wait)
                return
        await self._cancel_timer()
        await super().trigger_user_turn_stopped()

    async def _forget(self):
        self._stopped_at, self._final_after_stop, self._model_complete = None, False, False
        await self._cancel_timer()

    async def _cancel_timer(self):
        if self._timer is not None and self._timer is not asyncio.current_task():
            await self.task_manager.cancel_task(self._timer)
        self._timer = None

    async def _maybe_end_early(self):
        """After the analyzer said INCOMPLETE for this stop and a final transcript that is a command, or (with the
        punctuation rule on) ends a sentence, is in."""
        if (self._timer is not None or self._stopped_at is None or not self._final_after_stop or self._turn_complete
                or self._model_complete or self._vad_user_speaking):
            return
        if self._command_secs is not None and self.is_command is not None and self.is_command(self._text):
            secs, why = self._command_secs, "is a command the orchestrator answers"
            self.command_ends += 1
        elif self._punct_secs is not None and ends_turn(self._text):
            secs, why = self._punct_secs, "ends a sentence"
            self.early_ends += 1
        else:
            return
        logger.debug(f"{self}: {self._text!r} {why} while Smart Turn was unsure: ending the turn in {secs} s")
        await self._arm(self._stopped_at + secs)

    async def _arm(self, at: float):
        await self._cancel_timer()
        self._timer = self.task_manager.create_task(self._end_at(at), f"{self}::end_at")

    async def _end_at(self, at: float):
        try:
            await asyncio.sleep(max(0.0, at - time.monotonic()))
        except asyncio.CancelledError:
            return
        self._timer = None
        if self._vad_user_speaking:
            return
        self._model_complete = False       # a decision made here is final: never vetoed again
        self._turn_complete = True
        self._transcript_finalized = True
        await self._maybe_trigger_user_turn_stopped()
