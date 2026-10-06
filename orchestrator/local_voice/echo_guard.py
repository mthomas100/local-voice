"""The echo guard: the agent's own words heard back through the microphone are not the person's words.

What an early live test showed (2026-10-05, the browser page in Chrome on the Mac's speakers; the orchestrator
log and the turn log): five of twelve user turns were fragments of the agent's own sentences, and two of them cut off
a reply. None of them came while a reply was playing: Silero started no turn during any of the eleven
replies (about 150 s of speech), so Chrome's echo canceller held the live echo. They were the last ten seconds of one
reply (sent once) coming back into the microphone three times, 6 s, 59 s and 158 s
after the server stopped sending it, in its own order and pacing: audio played again by something outside the server,
which no echo canceller can remove (its reference is long gone). So the check is on the words: a transcript that is
mostly a stretch of what the agent said in the last few minutes is dropped before it becomes a turn.

Matching is a local alignment of words (Smith-Waterman: equal words +2, a different word or a gap -1), so the
recogniser's slips ("want a full breakdown" for "want the full breakdown?") and its own sentence breaks ("each. One.")
still match, while a few common words scattered over a long reply do not. A turn counts as echo when at least
`min_words` of its words align and they are at least `min_share` of its words: the person quoting the agent inside
their own question ("what do you mean by a compost heap layer?") keeps most of their words and is kept.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")
MATCH, MISMATCH, GAP = 2, -1, -1


def words(text: str | None) -> list[str]:
    """Lower-case words without punctuation; "~/.pi/agent" is "pi", "agent"; "don't" stays one word."""
    return _WORD.findall((text or "").lower().replace("’", "'"))


def align(heard: list[str], said: list[str]) -> tuple[int, int, int, int, int]:
    """The best local alignment of `heard` inside `said`: (equal words aligned, said start, said end, heard start,
    heard end), the spans as slice bounds.

    O(len(heard) x len(said)) in plain Python: about 15k cells for a 50-word transcript against a 300-word window."""
    best, best_cell = 0, (0, 0, 0, 0, 0)
    n = len(said)
    prev_s = [0] * (n + 1)
    prev_m = [0] * (n + 1)      # equal words on the best path ending at the cell
    prev_b = [0] * (n + 1)      # where in `said` that path starts
    prev_h = [0] * (n + 1)      # where in `heard` it starts
    for i in range(1, len(heard) + 1):
        cur_s = [0] * (n + 1)
        cur_m = [0] * (n + 1)
        cur_b = [0] * (n + 1)
        cur_h = [0] * (n + 1)
        w = heard[i - 1]
        for j in range(1, n + 1):
            eq = w == said[j - 1]
            diag = prev_s[j - 1] + (MATCH if eq else MISMATCH)
            up = prev_s[j] + GAP          # a heard word with no counterpart (a recogniser insertion)
            left = cur_s[j - 1] + GAP     # a said word not heard
            s = max(0, diag, up, left)
            if s == 0:
                continue
            if s == diag:
                fresh = prev_s[j - 1] == 0
                m = prev_m[j - 1] + (1 if eq else 0)
                b, h = (j - 1, i - 1) if fresh else (prev_b[j - 1], prev_h[j - 1])
            elif s == up:
                m, b, h = prev_m[j], prev_b[j], prev_h[j]
            else:
                m, b, h = cur_m[j - 1], cur_b[j - 1], cur_h[j - 1]
            cur_s[j], cur_m[j], cur_b[j], cur_h[j] = s, m, b, h
            if s > best:
                best, best_cell = s, (m, b, j, h, i)
        prev_s, prev_m, prev_b, prev_h = cur_s, cur_m, cur_b, cur_h
    return best_cell


@dataclass
class EchoMatch:
    heard_words: int
    matched: int                 # heard words aligned to equal words of the agent's speech
    said: str                    # the agent's sentence (or reply) it matched best
    said_at: float | None = None # when that was said (seconds, the guard's clock), if known; EchoGuard.match gives the
                                 # time of the sentence the aligned words start in, not of the reply's first sentence
    span: str = ""               # the agent's words the transcript aligned to
    heard_span: tuple[int, int] = (0, 0)   # the transcript's words (slice bounds) that aligned
    said_span: tuple[int, int] = (0, 0)    # the words of `said` (slice bounds) they aligned to
    heard_at: float | None = None          # when the filter let it through (the guard's clock): the idle case only

    @property
    def share(self) -> float:
        return self.matched / self.heard_words if self.heard_words else 0.0

    def is_echo(self, min_words: int = 3, min_share: float = 0.6) -> bool:
        return self.matched >= min_words and self.share >= min_share


def best_match(heard: str, candidates: list[tuple[str, float | None]]) -> EchoMatch | None:
    """The candidate (text, when) whose words `heard` aligns to best; None when `heard` has no words."""
    hw = words(heard)
    if not hw:
        return None
    best: EchoMatch | None = None
    for text, at in candidates:
        sw = words(text)
        if not sw:
            continue
        m, b, e, hb, he = align(hw, sw)
        if best is None or m > best.matched:
            best = EchoMatch(heard_words=len(hw), matched=m, said=text, said_at=at, span=" ".join(sw[b:e]),
                             heard_span=(hb, he), said_span=(b, e))
    return best or EchoMatch(heard_words=len(hw), matched=0, said="")


class EchoGuard:
    """What the agent said lately, and whether a transcript is that speech heard back. One per connection: the TTS
    service records every sentence it speaks (`said`), the transcript filter asks (`judge`).

    Sentences are kept for `window_s` (an early live test had echoes up to 158 s after the speech) and joined in
    order before matching, so an echo that runs across a sentence break still aligns as one stretch.

    `fragment_sentences`: how many of the last sentences said a fragment (fragment_is_echo) is checked against."""

    def __init__(self, *, window_s: float = 300.0, min_words: int = 3, min_share: float = 0.6,
                 fragment_sentences: int = 3, max_sentences: int = 200, clock=time.monotonic):
        self.window_s = window_s
        self.min_words = min_words
        self.min_share = min_share
        self.fragment_sentences = fragment_sentences
        self._clock = clock
        self._said: deque[tuple[float, str]] = deque(maxlen=max_sentences)
        # bounded: a connection can stay open all day
        self.dropped: deque[tuple[str, EchoMatch]] = deque(maxlen=100)   # echo dropped while the agent was busy
        self.passed: deque[tuple[str, EchoMatch]] = deque(maxlen=100)    # echo heard while it was idle, let through
        self._noted: set[int] = set()       # id() of the passed matches a turn_note already reported

    def said(self, text: str, at: float | None = None) -> None:
        if text and text.strip():
            self._said.append((self._clock() if at is None else at, text.strip()))

    def recent(self, now: float | None = None) -> list[tuple[float, str]]:
        now = self._clock() if now is None else now
        return [(t, s) for t, s in self._said if now - t <= self.window_s]

    def match(self, heard: str, now: float | None = None) -> EchoMatch | None:
        """The best alignment of `heard` against the speech of the window, read as one text per reply-sized run:
        consecutive sentences said less than 30 s apart are joined (a reply), so an echo across sentences aligns.
        said_at is when the sentence the aligned words start in was said (the echoes in an early live test were the last ten
        seconds of a thirty-second reply: the reply's start would put them 20 s too early)."""
        runs: list[list[tuple[float, str]]] = []
        for t, s in self.recent(now):
            if runs and t - runs[-1][-1][0] <= 30.0:
                runs[-1].append((t, s))
            else:
                runs.append([(t, s)])
        m = best_match(heard, [(" ".join(s for _, s in run), run[0][0]) for run in runs])
        if m is not None and m.matched:
            run = next((r for r in runs if r[0][0] == m.said_at), None)   # runs start more than 30 s apart
            first = m.said_span[0]
            for t, s in run or []:
                n = len(words(s))
                if first < n:
                    m.said_at = t
                    break
                first -= n
        return m

    def judge(self, heard: str, now: float | None = None) -> EchoMatch | None:
        """The match when `heard` is echo, else None."""
        m = self.match(heard, now)
        if m is not None and m.is_echo(self.min_words, self.min_share):
            self.dropped.append((heard, m))
            return m
        return None

    def fragment_is_echo(self, heard: str, now: float | None = None) -> bool:
        """A transcript too short for `judge` (1 to min_words - 1 words) whose every word is among the words of the
        last `fragment_sentences` sentences said within the window. Only the turn-start strategy asks, and only while
        the agent is busy: the reply pause stops the reply's audio at the first speech frame (bargein.py, a median
        72 ms in), so live echo that gets past the client's canceller is cut short and reaches the recogniser as a
        word or two ("Paris." of "The capital is Paris."), under the 3 words echo needs. The cost: a one-word barge-in
        the agent itself said in those sentences ("no" after "No, I couldn't find it.") resumes the reply instead."""
        hw = words(heard)
        if not 0 < len(hw) < self.min_words:
            return False
        pool = {w for _, s in self.recent(now)[-self.fragment_sentences:] for w in words(s)}
        return all(w in pool for w in hw)

    def turn_note(self, user_text: str, now: float | None = None) -> tuple[str | None, dict | None]:
        """For the agent, once a turn's words are in: when some of them matched the agent's own recent speech while it
        was idle (`passed`: the filter let them through), a one-line note for the model and a record for the turn log;
        (None, None) otherwise. The note does not say which: the person may have read a line aloud to ask about it, or
        the microphone heard the speakers (5 of the 19 voice turns of an early live test were the agent's own words,
        6-59 s after the replies ended). Each passed transcript is noted
        once, in the turn whose words contain it."""
        now = self._clock() if now is None else now
        turn = f" {' '.join(words(user_text))} "
        hits: list[tuple[str, EchoMatch]] = []
        for heard, m in self.passed:
            h = " ".join(words(heard))
            if id(m) not in self._noted and h and f" {h} " in turn:
                hits.append((heard, m))
                self._noted.add(id(m))
        if not hits:
            return None, None
        best = max(hits, key=lambda hm: hm[1].matched)[1]
        ago = None if best.said_at is None else max(0.0, (best.heard_at or now) - best.said_at)
        when = "earlier" if ago is None else f"{ago:.0f} s ago"
        heard_words = sum(m.heard_words for _, m in hits)
        which = "These words" if heard_words >= len(words(user_text)) else \
            "Some of these words (" + " ... ".join(f'"{h.strip()}"' for h, _ in hits) + ")"
        note = (f"({which} match what you said {when}; the person may be quoting you, or the microphone heard your own "
                f"voice.)")
        record = {"matched_words": sum(m.matched for _, m in hits), "heard_words": heard_words,
                  "said_ago_s": None if ago is None else round(ago, 1), "said": best.span}
        return note, record


def without_span(text: str, span: tuple[int, int]) -> str:
    """`text` with the words span[0]:span[1] (as numbered by `words`) cut out, punctuation around the rest kept."""
    spans = [m.span() for m in _WORD.finditer(text.lower().replace("’", "'"))]
    a, b = span
    if not spans or a >= b:
        return text.strip()
    cut_from = spans[a][0] if a < len(spans) else len(text)
    cut_to = spans[b][0] if b < len(spans) else len(text)
    return " ".join(f"{text[:cut_from]} {text[cut_to:]}".split()).strip(" ,;:-—–")


class EchoFilter:
    """Built by echo_filter(): a FrameProcessor between the STT and everything that reads the person's words."""


# Frame.metadata key the STT services set on a transcript the echo gate has already judged (services/stt.py): the
# filter passes such a frame on as it is, so no transcript is judged (and logged, and reported to the strategy) twice.
JUDGED = "lv_echo_judged"


def echo_filter(guard: EchoGuard, *, busy=lambda: True, on_echo=None, keep_words: int = 3, watch=None):
    """A Pipecat processor between the STT and everything that reads the person's words.

    While the agent is busy (`busy()`: speaking, or thinking and running tools), a transcript (final or interim) that
    is the agent's own recent speech is dropped before the captions, the turn strategies and the context see it, and
    `on_echo(final, rest)` tells the turn-start strategy (turn_start.py) that what it waits on was echo, so replayed
    audio never cuts a reply. Echo followed or preceded by the person's own words ("...layer on top. What does that
    mean?") keeps those words when there are at least `keep_words` of them.

    While the agent is idle nothing is dropped: the person may have read the agent's lines aloud to ask about them
    (nothing in an early live test showed what produced the echo turns, and the replies to them were sensible
    follow-ups). The words go on as the person's; the match is kept in `guard.passed` for the agent's note
    to the model and the turn log (EchoGuard.turn_note). An idle interim is not even matched: only finals are kept.

    The verdict is gate(text, final). The STT services ask it before they push a transcript (services/stt.py's
    transcript_gate, wired by pipeline.build_session), because whatever an STT pushes is already seen: Pipecat's RTVI
    observer captions the browser page from a transcript's first push, the STT's (rtvi/observer.py 1.12), so echo this
    processor dropped still showed on the page as the person's words. A transcript that reaches this processor unjudged
    (any other source) is judged here, as before.

    `watch(frame)` sees every frame that passes, both ways: turn_start.AgentActivity follows the bot's speech from the
    BotStarted/StoppedSpeakingFrame the output transport sends upstream through here (pipeline.build_session)."""
    from loguru import logger
    from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class _EchoFilter(FrameProcessor, EchoFilter):
        def __init__(self):
            super().__init__(name="EchoFilter")
            self.guard = guard
            self.dropped_finals = 0
            self.record = None       # the session recorder's verdict hook, when recording (recorder.py)

        async def gate(self, text: str, final: bool) -> str | None:
            """What of a transcript goes on: all of it, the person's own words around echo, or None (echo heard while
            the agent is busy, dropped). Logs, tells the strategy (on_echo) and keeps the match, as the filter did."""
            is_busy = busy()
            m = guard.match(text) if (final or is_busy) else None
            if m is None or not m.is_echo(guard.min_words, guard.min_share):
                self._note(text, final, is_busy, "kept", text, m)
                return text
            if not is_busy:
                if final:
                    m.heard_at = guard._clock()
                    guard.passed.append((text, m))
                    logger.info(f"{self}: words matching the agent's own speech while it was idle "
                                f"({m.matched}/{m.heard_words} of {m.said[:60]!r}): kept {text!r}")
                self._note(text, final, is_busy, "idle_echo", text, m)
                return text
            if final:
                guard.dropped.append((text, m))
            rest = without_span(text, m.heard_span)
            rest = rest if len(words(rest)) >= keep_words else ""
            if final:
                self.dropped_finals += 1
                ago = "" if m.said_at is None else f" said {guard._clock() - m.said_at:.0f} s ago"
                logger.info(f"{self}: echo of the agent's own speech ({m.matched}/{m.heard_words} words of "
                            f"{m.said[:60]!r}{ago}): dropped {text!r}" + (f", kept {rest!r}" if rest else ""))
            self._note(text, final, is_busy, "cut" if rest else "dropped", rest or None, m)
            if on_echo is not None:
                await on_echo(final, rest)
            return rest or None

        def _note(self, text: str, final: bool, is_busy: bool, verdict: str, after: str | None,
                  m: EchoMatch | None) -> None:
            """The verdict, before and after, for the session recorder (never raises into the speech path)."""
            if self.record is None:
                return
            try:
                rec = {"final": final, "text": text, "after": after, "verdict": verdict, "busy": is_busy}
                if m is not None and m.matched:
                    rec.update(matched=m.matched, heard_words=m.heard_words, said=m.span,
                               said_ago_s=None if m.said_at is None else round(guard._clock() - m.said_at, 2))
                self.record(rec)
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self}: recording the verdict failed: {e}")

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if watch is not None:
                watch(frame)
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (TranscriptionFrame,
                                                                              InterimTranscriptionFrame)) \
                    and not frame.metadata.get(JUDGED):
                text = await self.gate(frame.text, isinstance(frame, TranscriptionFrame))
                if text is None:
                    return
                frame.text = text
            await self.push_frame(frame, direction)

    return _EchoFilter()
