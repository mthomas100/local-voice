"""Turn the model's JSON into facts and reflections the writers may use, or drop each item with a reason.

The model proposes; this module decides. Every citation must name a turn the pass showed. Every quote must be found
in the cited turn's words, and what gets written is the exact span of the source (so a curly quote or a re-cased
word in the model's copy can never change the person's words). Life reflections that state a verdict, a diagnosis
or advice in the model's own words are dropped (Atlas AGENTS.md: observations with evidence, never verdicts).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .turnlog import Turn

KINDS = ("decision", "preference", "finding", "todo", "question")

_FOLD = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "“": '"', "”": '"',
         "„": '"', "″": '"', "–": "-", "—": "-", "−": "-", " ": " ", "…": "..."}


def _normalise(s: str) -> tuple[str, list[int]]:
    """Fold quotes, dashes, ellipses, whitespace and case; keep a map from each output char to its source index."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(s):
        rep = _FOLD.get(ch, ch)
        if rep.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
            continue
        prev_space = False
        for c in rep.casefold():
            out.append(c)
            idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def find_span(source: str, quote: str) -> str | None:
    """The exact text of `source` that matches `quote` up to quotes, dashes, spacing and case; None if absent."""
    q, _ = _normalise(quote.strip().strip("\"'“”‘’").strip())
    if len(q) < 3:
        return None
    s, idx = _normalise(source)
    at = s.find(q)
    if at < 0:
        return None
    start, end = idx[at], idx[at + len(q) - 1] + 1
    return source[start:end].strip()


def parse_json_reply(text: str) -> dict[str, Any]:
    """The first-to-last-brace object of the reply, tolerating code fences and chatter around it."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(t[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("the reply's JSON is not an object")
    return obj


def one_line(s: Any, limit: int) -> str:
    s = " ".join(str(s or "").split()).strip().lstrip("-*• ").strip()
    if len(s) > limit:
        s = s[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return s


_MARKUP = re.compile(r"[`*_#<>|\[\]{}]")


def clean_spoken(s: Any, max_words: int) -> str | None:
    """Plain words for TTS, at most `max_words`, cut at a sentence end when possible. None means NO_REPLY."""
    raw = str(s or "")
    if re.sub(r"[\s_.!*`\"']", "", raw).upper() in ("", "NOREPLY"):
        return None
    text = " ".join(_MARKUP.sub(" ", raw.replace("NO_REPLY", "")).split())
    if not text:
        return None
    words = text.split()
    if len(words) > max_words:
        cut = " ".join(words[:max_words])
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: end + 1] if end > len(cut) // 3 else cut.rstrip(",;:") + "."
    return text or None


# --- the technical pass -------------------------------------------------------------------------------------------

@dataclass
class Fact:
    text: str
    kind: str
    cites: list[str]
    quote: str | None = None


@dataclass
class TechProposal:
    facts: list[Fact] = field(default_factory=list)
    spoken: str | None = None
    dropped: list[str] = field(default_factory=list)


def _cites(item: dict[str, Any], handles: dict[str, Turn], key: str = "cites") -> tuple[list[str], list[str]]:
    raw = item.get(key)
    if isinstance(raw, str):
        raw = [raw]
    good, bad = [], []
    for c in raw or []:
        h = str(c).strip().strip("[]").upper().replace(" ", "")
        (good if h in handles else bad).append(h)
    return list(dict.fromkeys(good)), bad


def validate_tech(obj: dict[str, Any], handles: dict[str, Turn], *, max_facts: int, max_words: int) -> TechProposal:
    out = TechProposal(spoken=clean_spoken(obj.get("spoken"), max_words))
    items = obj.get("facts") if isinstance(obj.get("facts"), list) else []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            out.dropped.append(f"fact {i + 1}: not an object")
            continue
        text = one_line(it.get("text"), 240)
        if len(text) < 8:
            out.dropped.append(f"fact {i + 1}: no text")
            continue
        cites, bad = _cites(it, handles)
        if not cites:
            out.dropped.append(f"fact {i + 1} ({text[:60]}): cites no turn it was shown {bad or ''}".rstrip())
            continue
        quote = None
        if it.get("quote"):
            for h in cites:
                t = handles[h]
                quote = find_span(t.user_text, str(it["quote"])) or find_span(t.spoken_text, str(it["quote"]))
                if quote:
                    break
            if not quote:
                # A quote that is not there means the model claimed evidence it does not have: drop the fact.
                out.dropped.append(f"fact {i + 1} ({text[:60]}): quote not found in {', '.join(cites)}")
                continue
        kind = str(it.get("kind") or "").strip().lower()
        out.facts.append(Fact(text=text, kind=kind if kind in KINDS else "finding", cites=cites, quote=quote))
        if len(out.facts) >= max_facts:
            if i + 1 < len(items):
                out.dropped.append(f"{len(items) - i - 1} more facts over the limit of {max_facts}")
            break
    return out


# --- the life pass ------------------------------------------------------------------------------------------------

CLINICAL = re.compile(
    r"\b(depress\w*|anxiety|anxious|disorder\w*|diagnos\w*|symptom\w*|trauma\w*|therap\w*|mental (?:health|illness)|"
    r"burn-?out|burned out|burnt out|ptsd|adhd|ocd|bipolar|panic attacks?|cognitive decline|addict\w*|medicat\w*|"
    r"illness)\b", re.I)
VERDICT = re.compile(
    r"\byou(?:'re|’re| are| seem| sound| feel| must be| look| were)\s+(?:\w+\s+){0,2}?"
    r"(sad|angry|anxious|stressed|depressed|upset|happy|excited|tired|exhausted|frustrated|overwhelmed|lonely|"
    r"afraid|scared|worried|nervous|unhappy|hopeless|lost|stuck|struggling|insecure|avoiding)\b", re.I)
ADVICE = re.compile(r"\byou (?:should|need to|must|ought to|have to)\b|\btry (?:to|and)\b|\bmake sure\b", re.I)
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*(?:/[a-z0-9][a-z0-9-]*)?$")


@dataclass
class Quote:
    handle: str
    text: str            # exact source span


@dataclass
class Reflection:
    title: str
    quotes: list[Quote]
    notice: str
    questions: list[str]
    tags: list[str]


@dataclass
class LifeProposal:
    reflections: list[Reflection] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def judgement_problem(own_text: str, their_words: str, *, advice: bool = True) -> str | None:
    """Why the model's own sentence is a verdict, a diagnosis or advice. A feeling word the person used themselves
    may be named back to them ("you said 'anxious' twice"); one they did not use may not."""
    theirs = their_words.casefold()
    for rx, what in ((CLINICAL, "a clinical word"), (VERDICT, "a verdict about how they feel")):
        for m in rx.finditer(own_text):
            if m.group(1).casefold() not in theirs:
                return f"{what} ({m.group(0)!r})"
    m = ADVICE.search(own_text) if advice else None
    if m:
        return f"advice ({m.group(0)!r})"
    return None


def validate_life(obj: dict[str, Any], handles: dict[str, Turn], words: dict[str, str], *, known_tags: set[str],
                  max_reflections: int) -> LifeProposal:
    """`words[handle]` is the exact text saved in Atlas for that turn, already checked to be in the note."""
    out = LifeProposal()
    items = obj.get("reflections") if isinstance(obj.get("reflections"), list) else []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            out.dropped.append(f"reflection {i + 1}: not an object")
            continue
        title = one_line(it.get("title"), 60).replace("/", "-")
        quotes: list[Quote] = []
        for q in it.get("quotes") or []:
            if not isinstance(q, dict):
                continue
            h = str(q.get("turn") or q.get("handle") or "").strip().strip("[]").upper()
            if h not in handles or h not in words:
                out.dropped.append(f"reflection {i + 1}: a quote cites {h or 'no turn'}, which it was not shown")
                continue
            span = find_span(words[h], str(q.get("text") or ""))
            if not span:
                out.dropped.append(f"reflection {i + 1}: a quote is not in {h}'s words")
                continue
            if all(span != x.text for x in quotes):
                quotes.append(Quote(handle=h, text=span))
        if not quotes:
            out.dropped.append(f"reflection {i + 1} ({title}): no quote survived the check")
            continue
        theirs = " ".join(q.text for q in quotes)
        notice = one_line(it.get("notice"), 300)
        questions = [one_line(x, 200) for x in (it.get("questions") or [])[:2] if one_line(x, 200)]
        problem = None
        for own, is_question in [(title, False), (notice, False), *((q, True) for q in questions)]:
            # Questions may say "try"; advice is looked for in the title and the notice only.
            problem = judgement_problem(own, theirs, advice=not is_question)
            if problem:
                break
        if problem:
            out.dropped.append(f"reflection {i + 1} ({title}): {problem}")
            continue
        if not notice and not questions:
            out.dropped.append(f"reflection {i + 1} ({title}): nothing noticed and nothing asked")
            continue
        tags = []
        for t in it.get("tags") or []:
            t = str(t).strip().lstrip("#").lower().replace(" ", "-")
            if TAG_RE.match(t) and (t in known_tags or t.startswith("area/")) and t not in tags:
                tags.append(t)
        out.reflections.append(Reflection(title=title or f"Reflection {i + 1}", quotes=quotes, notice=notice,
                                          questions=questions, tags=tags[:3]))
        if len(out.reflections) >= max_reflections:
            break
    return out
