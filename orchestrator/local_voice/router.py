"""The router (SPACES.md "Switching" and "Modes"): what the person's words ask of the orchestrator itself, decided on
the final transcript before any model call.

- A space switch: a space's trigger phrase alone ("my journal"), or after a switching verb ("go to", "switch to",
  "open", "take me to", "back to", "head over to", "jump into", "bring up" ...), itself after any request phrase or
  filler people put first ("I'd like you to", "can you", "let's", "okay, um,"). The words after the phrase are the new
  space's first prompt. A phrase that names two spaces is asked about, never guessed. A trigger followed by a noun that
  makes it part of a longer name ("my atlas folder", "the journal entry", "my notes app") is not a switch: those words
  go to the model, which has the space_switch tool (voice_mode.ts) for a switch said less directly ("go where my journal
  is"). A rambling request after a request phrase ("I want you to go to my journal. And uh. ...") used
  not to switch, and the home agent browsed the journal folder instead (2026-10-05); tools/router_bench.py measures the phrasings in
  tests/fixtures/router_phrasings.yaml, planted negatives included.
- A mode switch: "act mode" (tools that change things, each still gated), "just talk" / "conversation mode".
- Journal capture in the Atlas space: "note this: ..." or "journal this ..." saves the words that follow verbatim
  (atlas.py capture --via voice) before the reply; "just take notes" saves every utterance until "stop taking notes".

Everything here is pure text handling; agent.py acts on the result. Matching is on the lowercased words with
punctuation stripped at the edges, so the recogniser's commas and full stops do not get in the way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .spaces import Spaces

# What people say before the verb: fillers, and request phrases (Nemotron writes a spoken "I'd like you to" as "I like
# you to"). Only a verb or a trigger alone makes a switch; these never do on their own.
_FILLER = r"(?:um|uh|er|erm|hmm|well|so|okay|ok|alright|right|now|and|hey|please|just|actually)"
_REQUEST = (r"(?:can you|could you|would you|will you|can we|could we|let's|let us|i'd like you to|i would like you to|"
            r"i like you to|i want you to|i need you to|i'd like to|i would like to|i like to|i want to|i wanna|i need to|"
            r"we need to|we should|time to|go ahead and)")
_PRE = rf"(?:(?:{_FILLER}|{_REQUEST})\s+)*"
_VERB = (r"(?:go(?:\s+back)?\s+(?:to|into)|go\s+back|go|switch(?:\s+(?:back|over))?\s+(?:to|into)|switch|"
         r"move(?:\s+back)?\s+(?:to|into)|head(?:\s+(?:back|over))?\s+(?:to|into)|jump(?:\s+back)?\s+(?:to|into)|"
         r"take\s+me(?:\s+back)?(?:\s+(?:to|into))?|bring\s+me(?:\s+back)?\s+(?:to|into)|back\s+to|over\s+to|"
         r"open(?:\s+up)?|bring\s+up|pull\s+up|enter)")
# A word that, right after a trigger, makes the trigger part of a longer name: "my atlas folder", "the journal entry",
# "my notes app", "the kb page" are things in a space (or elsewhere), not the space.
_NAME_GOES_ON = frozenset("""folder folders file files entry entries page pages app apps application directory dir
repo repository document documents doc docs item items list template templates tab window link links settings
club note notes""".split())
_SEP = r"[\s,.;:!?-]*"
_ACT = re.compile(r"^(?:(?:ok(?:ay)?|so|now|please)\s+)*(?:switch\s+to\s+|go\s+to\s+|use\s+)?act(?:ion)?\s+mode$")
_TALK = re.compile(r"^(?:(?:ok(?:ay)?|so|now|please|let's)\s+)*(?:(?:switch\s+to\s+|go\s+to\s+|back\s+to\s+)?"
                   r"(?:conversation|talk|chat)\s+mode|just\s+(?:talk|chat)|let's\s+just\s+(?:talk|chat))$")
_COURTESY = {"", "please", "now", "then", "thanks", "thank you", "for me", "please thanks"}
_NOTE = re.compile(r"^\s*(?:(?:ok(?:ay)?|so|please)[\s,]+)*(?:note|journal|jot\s+down|write\s+down)\s+this\b[\s,.;:!?-]*",
                   re.IGNORECASE)
_NOTES_ON = re.compile(r"^(?:(?:ok(?:ay)?|so|now|please)\s+)*(?:just\s+)?(?:take|taking)\s+notes$")
_NOTES_OFF = re.compile(r"^(?:(?:ok(?:ay)?|so|now|please)\s+)*(?:stop|done|finish(?:ed)?)\s+(?:taking\s+)?notes$")


def norm(text: str) -> str:
    """Lowercase words with the recogniser's punctuation between and around them reduced to single spaces."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^\w'\s-]", " ", t)
    return " ".join(t.split())


@dataclass
class Route:
    kind: str                         # none | switch | ask | mode | notes_on | notes_off
    space: str | None = None          # switch: where to
    rest: str = ""                    # switch: the words after the phrase, the new space's first prompt
    candidates: list[str] = field(default_factory=list)   # ask: the spaces it could mean
    mode: str | None = None           # mode: conversation | act


def _trigger_index(spaces: Spaces) -> list[tuple[str, str]]:
    """(normalised trigger, space), longest first so "back home" wins over "home"."""
    out = [(norm(t), name) for name, sp in spaces.spaces.items() for t in sp.triggers]
    out += [(norm(name), name) for name in spaces.spaces]      # a space's own name is a trigger too
    seen, uniq = set(), []
    for t, n in sorted(out, key=lambda x: -len(x[0])):
        if t and (t, n) not in seen:
            seen.add((t, n))
            uniq.append((t, n))
    return uniq


def route(text: str, spaces: Spaces) -> Route:
    """What these words ask of the orchestrator (module docstring). Words that ask nothing of it: Route("none")."""
    n = norm(text)
    if not n:
        return Route("none")
    if _ACT.match(n):
        return Route("mode", mode="act")
    if _TALK.match(n):
        return Route("mode", mode="conversation")
    if _NOTES_ON.match(n):
        return Route("notes_on")
    if _NOTES_OFF.match(n):
        return Route("notes_off")
    index = _trigger_index(spaces)
    alts = "|".join(re.escape(t) for t, _ in index)
    by_trigger = {t: name for t, name in index}
    # a phrase alone, or after a switching verb, at the start of the utterance (after any filler or request phrase)
    m = re.match(rf"^{_PRE}(?:(?P<verb>{_VERB})\s+)?(?:the\s+|my\s+)?(?P<t>{alts})(?=$|\s)(?P<rest>.*)$", n)
    if not m:
        return Route("none")
    after = m.group("rest").split()
    if after and after[0] in _NAME_GOES_ON and not _punctuated_after(text, m.group("t")):
        return Route("none")            # "open my atlas folder": a thing in or near the space, not the space
    alone = m.group("rest").strip() in _COURTESY
    if not (alone or m.group("verb")):
        return Route("none")            # "home is where..." or "the kb says..." is a sentence, not a switch
    first = by_trigger[m.group("t")]
    # "go to my journal or the kb": two spaces named in the phrase itself is a question back
    tail = m.group("rest")
    other = re.match(rf"^\s*(?:or|and|,)\s+(?:to\s+)?(?:the\s+|my\s+)?(?P<t>{alts})(?=$|\s)", tail)
    if other and by_trigger[other.group("t")] != first:
        return Route("ask", candidates=[first, by_trigger[other.group("t")]])
    return Route("switch", space=first, rest="" if alone else _rest_of(text, m.group("t")))


def _punctuated_after(text: str, trigger: str) -> bool:
    """The recogniser put punctuation right after the trigger ("Go to my journal, note this: ..."): the name ends there."""
    pat = r"\W+".join(re.escape(w) for w in trigger.split())
    m = re.search(rf"\b{pat}\b\s*[,.;:!?—–-]", text, re.IGNORECASE)
    return m is not None


def _rest_of(text: str, trigger: str) -> str:
    """The original words after the trigger phrase (punctuation and a joining "and"/"then" dropped), as said."""
    words = trigger.split()
    pat = r"\W+".join(re.escape(w) for w in words)
    m = re.search(rf"\b{pat}\b", text, re.IGNORECASE)
    if not m:
        return ""
    rest = text[m.end():]
    # the joining words and fillers between the switch and the request ("..., and um, just make ...")
    rest = re.sub(r"^(?:[\s,.;:!?-]+|(?:and|then|so|also|um|uh|er|erm)\b)*", "", rest, flags=re.IGNORECASE)
    return rest.strip()


def answer_to_ask(text: str, candidates: list[str], spaces: Spaces) -> str | None:
    """The person's answer to "Do you mean A or B?": the one candidate their words name, or None."""
    n = f" {norm(text)} "
    hits = []
    for name in candidates:
        sp = spaces[name]
        names = [name, *sp.triggers, sp.description] + sp.description.split()[-1:]
        if any(f" {norm(x)} " in n for x in names if norm(x)):
            hits.append(name)
    return hits[0] if len(hits) == 1 else None


def musing(text: str) -> str | None:
    """"note this: <words>": the words to save verbatim (as said, after the cue), or "" when the cue came alone.
    None when the utterance is not a note."""
    m = _NOTE.match(text)
    if not m:
        return None
    return text[m.end():].strip()
