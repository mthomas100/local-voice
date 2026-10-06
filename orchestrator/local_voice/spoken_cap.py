"""The spoken cap: a reply is said up to the last sentence that ends within about `words` words, and the person is then
asked whether to go on (2026-10-05). The persona asked for one or two short sentences and qwen38 still wrote 49-133
words, 16-58 s of speech, for "What does a heat pump do?" and "How much does one cost?" (tool expedition
test runs, 2026-10-05). The model's whole text stays in Pi's context; the next prompt says where the speech
stopped (pi_rpc.cut_note), so "yes" goes on from there.

Every sentence is held until it ends, then said if the reply stays within the cap; otherwise the reply is cut before
it. Holding costs no time: Pipecat's TTS synthesises whole sentences only, so it waits for the end of each sentence
anyway. A sentence that would go over the cap while nothing has been said in its block of text (the reply's first, or
the first after a tool call) ends at a clause break within the cap instead, with a full stop: qwen38's 16.4 s
heat-pump answer was one sentence of 49 words.
"""
from __future__ import annotations

import re

from .speech_text import speakable

# A sentence ends at . ! ? or … (and any closing quotes or brackets) followed by white space or the end of the text so
# far. A full stop at the end of the text so far is not trusted: the next text or the end of the block decides it
# (SpokenCap.boundary). "It costs 3." may go on "5 dollars", and a path or a file name streams in pieces: in an early
# live test a delta ended at "I run from ~/." and the voice said that alone, then the rest of the path. Waiting costs one delta (tens of ms); ! ? and … are trusted at once.
_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
# a full stop after these is not a sentence end ("Mt. Elbrus", "Dr. Smith"); nor after a word with a dot inside it
# ("e.g.", "U.S.") or after a number that starts a line ("1. Eggs")
_ABBREV = frozenset("mr mrs ms dr st mt vs jr sr prof approx fig".split())
_LAST_WORD = re.compile(r"\S*$")
# a clause break inside a sentence: a comma, semicolon or colon, or a dash between spaces; the cut goes before it
_CLAUSE = re.compile(r"[,;:](?=\s)|\s+[—–-]+(?=\s)")
MIN_PART = 8    # a sentence cut at a clause break keeps at least this many words ("Well." alone is no answer)

# Asked for something long: the cap is `long_words` instead ("tell me a story", "go on", "in detail"). Not "the
# whole": the recogniser heard "the hold gate" as "the whole gate" (e2e 2026-10-05 17:16).
_LONG_ASK = re.compile(r"\b(?:stor(?:y|ies)|poem|in (?:more )?detail|detailed|tell me (?:more|everything)|more about|"
                       r"go on|keep going|carry on|continue|read (?:me|it|out|that|them|the whole)|"
                       r"all of it|step by step|walk me through|a long|longer)\b", re.I)


def wants_more(text: str) -> bool:
    """The person asked for a long answer, or for more of the last one."""
    return bool(_LONG_ASK.search(text or ""))


def _spoken(token: str) -> int:
    """About how many words a token takes to say: a number by its digits ("5,642" is "five thousand six hundred
    forty-two"; qwen38's Elbrus answer, 57 written words with three numbers, took 31.4 s to say, 2026-10-05)."""
    if not any(c.isalnum() for c in token):
        return 0
    digits = sum(1 for c in token if c in "123456789")
    if not digits:
        return 1
    return max(1, round(1.5 * digits)) + (1 if any(c in token for c in "%$€£") else 0)


def words(text: str) -> int:
    """Spoken words: the tokens with a letter or digit in them once the Markdown the voice skips is gone, a number
    counted by its digits."""
    return sum(_spoken(t) for t in speakable(text).split())


def sentence_end(text: str) -> int | None:
    """The index just past the first sentence end in `text`, or None."""
    for m in _END.finditer(text):
        if m.group().startswith("."):
            w = _LAST_WORD.search(text, 0, m.start())
            word, head = w.group(), text[:w.start()].rstrip(" \t")
            if word and (word.lower() in _ABBREV or "." in word or (word.isdigit() and (not head or head[-1] == "\n"))):
                continue
            if m.end() == len(text):
                continue
        return m.end()
    return None


def clause_cut(sentence: str, budget: int) -> str | None:
    """The sentence ended at its last clause break within `budget` words (else its first after it), with a full stop;
    None when it has no clause break that leaves MIN_PART words."""
    best = None
    for m in _CLAUSE.finditer(sentence):
        n = words(sentence[:m.start()])
        if n < MIN_PART:
            continue
        if n > budget:
            best = best if best is not None else m.start()
            break
        best = m.start()
    if best is None:
        return None
    return sentence[:best].rstrip() + "."


class SpokenCap:
    """One reply's model text in, the text to speak out. `words` 0 says everything."""

    def __init__(self, words: int):
        self.limit = int(words)
        self.said = 0          # words released so far
        self.spoken = ""       # everything released so far
        self.cut = False       # the reply was cut: nothing more of it is said
        self._buf = ""         # the sentence in progress
        self._block = 0        # words released in this block of text (since the last tool call)

    def feed(self, text: str) -> str:
        if self.limit <= 0:
            self.spoken += text
            return text
        if self.cut or not text:
            return ""
        self._buf += text
        out = ""
        while not self.cut:
            end = sentence_end(self._buf)
            if end is None:
                break
            out += self._take(end)
        self.spoken += out
        return out

    def boundary(self) -> str:
        """The model's block of text is over (a tool call starts, its message ended, or the run is over): what is
        held is a whole sentence."""
        if self.limit <= 0 or self.cut:
            return ""
        out = self._take(len(self._buf)) if self._buf.strip() else ""
        self._buf, self._block = "", 0
        self.spoken += out
        return out

    def _take(self, end: int) -> str:
        """A whole sentence (_buf[:end]) has come: release it, or cut the reply there."""
        sentence, self._buf = self._buf[:end], self._buf[end:]
        n = words(sentence)
        if n == 0 or self.said + n <= self.limit:
            self.said += n
            self._block += n
            return sentence
        out = ""
        if self._block == 0:
            # nothing said yet in this block: say the sentence, ended at a clause break within the cap if it has one
            part = clause_cut(sentence, max(self.limit - self.said, MIN_PART))
            out = sentence if part is None else part
            self.said += words(out)
            if part is None:
                self._block += n
                return out
        self.cut, self._buf = True, ""
        return out
