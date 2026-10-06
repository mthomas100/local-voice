"""What the transcript says about delivery: words, syllables, fillers, repetitions, cut-offs.

Counts are only as verbatim as the speech recogniser: one that drops "um" (many are trained to) shows fewer fillers
than were said. The baseline is the person's own under the same recogniser, so the comparison holds while the
recogniser stays the same; change it and the filler baseline should be reset (README).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FILLERS = frozenset({"um", "umm", "uhm", "uh", "uhh", "er", "erm", "err", "ah", "ahh", "hmm", "hm", "mm", "mmm",
                     "eh"})
# Logged, never used in a hint: each is also an ordinary word ("I like it", "a kind of tea").
DISCOURSE = ("you know", "i mean", "sort of", "kind of", "like", "basically", "literally")

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*-?")


@dataclass
class TextCounts:
    words: int
    syllables: int
    fillers: int
    repetitions: int
    cutoffs: int
    discourse: int


def tokens(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in _TOKEN.findall(text or "")]


def syllables(word: str) -> int:
    """A vowel-group estimate (about nine words in ten exact for English); consistent, which is what a comparison
    against the same person's own baseline needs. Digits count one each."""
    w = word.lower().rstrip("-")
    if w.isdigit():
        return len(w)
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def count(text: str) -> TextCounts:
    toks = tokens(text)
    fill = sum(1 for t in toks if t in FILLERS)
    rep = sum(1 for a, b in zip(toks, toks[1:]) if a == b and a not in FILLERS and not a.endswith("-"))
    cut = sum(1 for t in toks if t.endswith("-"))
    low = " " + " ".join(toks) + " "
    disc = sum(low.count(f" {d} ") for d in DISCOURSE)
    return TextCounts(words=len(toks), syllables=sum(syllables(t) for t in toks), fillers=fill, repetitions=rep,
                      cutoffs=cut, discourse=disc)
