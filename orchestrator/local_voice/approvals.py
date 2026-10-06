"""Approvals (PROTOCOL.md "Approvals", 2026-10-05): what the orchestrator does with a tool call voice_gate.ts asks about.

An early version asked aloud "May I change your knowledge base? It starts with kb new. Yes or no?", heard
nothing for 20 s, and gave up: it said too little about what would happen. What coding agents give is better: exactly
what will happen, and a choice the person can see and pick. So voice_gate.ts sends each question as one select dialog whose
title is the approval (summary, action with the exact command or path and a preview, session scope) and whose options
are the choices; this module turns it into the protocol's `confirm_request` (a card on a client that can show one), the
spoken question, and back into a choice from a button or from the person's words:

- "yes" (alone, or with words that only agree: "yes, go ahead and create it") is allow_once; "yes, for this session"
  (or "don't ask again") is allow_session when the gate offered it; "no" (alone, or "no thanks") is deny;
- anything else is deny_said: not approved, and the words go to the model as the person's next message (a coding
  agent's "no, and tell it what to do instead"), so "yes, but call it X" never approves the call as it was;
- an echo of the question, or "wait" / "hold on" alone, answers nothing yet.

Session grants live here, per voice session and space, and cover exactly the scope the gate named (the same tool and
command prefix or folder); the gate offers none for a compound command, a delete or the network. Deterministic text
rules on purpose: an LLM in the final control path of a permission is what the Home Assistant consensus warns against.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import Config, ConfigError, _get
from .pi_rpc import ConfirmRequest, SelectRequest, UIRequest

ALLOW = ("allow_once", "allow_session")
# kb verbs that only read (voice_gate.ts KB_READ_VERBS): such a kb call is a lookup, acknowledged aloud; any other verb
# changes the kb and is asked about, so no "Let me look that up." goes before the question
KB_READ_VERBS = frozenset({"search", "trace", "stale"})


@dataclass
class ApprovalSettings:
    """config.yaml agent.approvals."""
    card_clients: tuple[str, ...] = ("iphone", "mac", "browser")
    card_wait_s: float = 120.0          # PROTOCOL.md: at least 120 s while a client that shows the card is connected
    reask_after_s: float = 60.0         # ... and the question is said once more before giving up
    voice_wait_s: float = 20.0          # no card: a question nobody can see waits this long
    ask: str = "I'd like to {summary}. Shall I?"
    ask_again: str = "Shall I {short}? Yes or no."
    denied: str = "Okay, I didn't {short}."
    timed_out: str = "I didn't hear an answer, so I didn't {short}."
    session: str = "Okay. For the rest of this session I won't ask again about {label}."
    labels: dict[str, str] = field(default_factory=lambda: {
        "allow_once": "Do it", "allow_session": "Allow {label} for the rest of this session", "deny": "Don't"})

    @property
    def longest_wait_s(self) -> float:
        return max(self.card_wait_s, self.voice_wait_s)


def settings(cfg: Config) -> ApprovalSettings:
    raw = cfg.raw
    d = ApprovalSettings()
    if _get(raw, "agent.approvals", dict, required=False) is None:
        return d
    out = ApprovalSettings(
        card_clients=tuple(str(x) for x in _get(raw, "agent.approvals.card_clients", list, required=False,
                                                 default=list(d.card_clients))),
        card_wait_s=_get(raw, "agent.approvals.card_wait_s", float, required=False, default=d.card_wait_s),
        reask_after_s=_get(raw, "agent.approvals.reask_after_s", float, required=False, default=d.reask_after_s),
        voice_wait_s=_get(raw, "agent.approvals.voice_wait_s", float, required=False, default=d.voice_wait_s),
        labels={**d.labels, **{str(k): str(v) for k, v in (_get(raw, "agent.approvals.labels", dict, required=False,
                                                                 default={}) or {}).items()}},
        **{k: _get(raw, f"agent.approvals.{k}", str, required=False, default=getattr(d, k))
           for k in ("ask", "ask_again", "denied", "timed_out", "session")})
    if out.card_wait_s <= 0 or out.voice_wait_s <= 0 or out.reask_after_s < 0:
        raise ConfigError("agent.approvals: card_wait_s and voice_wait_s must be > 0, reask_after_s >= 0")
    return out


@dataclass
class Approval:
    """One question, from voice_gate.ts's select (kind "select") or any other extension's plain confirm."""
    id: str
    kind: str                       # select | confirm
    title: str
    message: str
    summary: str
    short: str
    action: dict[str, Any]
    scope: dict[str, str] | None    # {key, label}: what allow_session would cover
    options: list[str]              # the choice ids the gate offered, in order

    @property
    def tool(self) -> str:
        return str(self.action.get("tool") or "")

    def choices(self, s: ApprovalSettings) -> list[dict[str, str]]:
        label = (self.scope or {}).get("label", "this")
        return [{"id": o, "label": s.labels.get(o, o).format(label=label)} for o in self.options]

    def request_message(self, s: ApprovalSettings, timeout_ms: int) -> dict[str, Any]:
        """The protocol's confirm_request (v1 fields, then the Approvals fields)."""
        return {"t": "confirm_request", "id": self.id, "title": self.title, "message": self.message,
                "timeout_ms": timeout_ms, "summary": self.summary, "action": self.action,
                "choices": self.choices(s)}


def from_request(ev: UIRequest) -> Approval | None:
    """voice_gate.ts's select (its title is the approval as JSON), or a plain confirm from any other extension. None
    for a select that is not an approval (it has no spoken form)."""
    if isinstance(ev, SelectRequest):
        try:
            d = json.loads(ev.title)
        except ValueError:
            return None
        if not isinstance(d, dict) or d.get("lv") != "approval":
            return None
        action = d.get("action") if isinstance(d.get("action"), dict) else {}
        scope = d.get("scope") if isinstance(d.get("scope"), dict) and d["scope"].get("key") else None
        options = [str(o) for o in ev.options] or ["allow_once", "deny"]
        if scope is None:
            options = [o for o in options if o != "allow_session"]
        return Approval(id=ev.id, kind="select", title=str(d.get("title") or ""), message=str(d.get("message") or ""),
                        summary=str(d.get("summary") or d.get("title") or "go ahead"),
                        short=str(d.get("short") or "do that"), action=action, scope=scope, options=options)
    if isinstance(ev, ConfirmRequest):
        title = (ev.title or "May I go ahead?").strip()
        detail = (ev.message or "").strip()
        summary = re.sub(r"^May I\s+", "", title).rstrip("?").strip() or "go ahead"
        return Approval(id=ev.id, kind="confirm", title=title, message=detail,
                        summary=summary[:1].upper() + summary[1:] + (f": {detail.splitlines()[0]}" if detail else "") + ".",
                        short="do that", action={"tool": "", "effect": "run", "command": detail or None, "path": None,
                                                 "cwd": "", "space": "", "mode": "", "preview": None},
                        scope=None, options=["allow_once", "deny"])
    return None


# ------------------------------------------------------------------------------------------------ spoken lines

_EXT_SPELL = re.compile(r"^[b-df-hj-np-tv-z]{1,3}$")    # txt, md, py, js, csv: said letter by letter


def speakable_names(text: str) -> str:
    """File names said as words: "Hello.txt" is "Hello dot T X T", so a dictated name can be checked by ear (an early live test
    heard "Plantxt" and "Hello TXT"). A full stop at the end of a sentence is left alone."""
    def one(m: re.Match) -> str:
        ext = m.group(2)
        said = " ".join(ext.upper()) if _EXT_SPELL.match(ext.lower()) else ext
        return f"{m.group(1)} dot {said}"
    return re.sub(r"\b([\w-]+)\.([A-Za-z][A-Za-z0-9]{0,4})\b(?!\w|\.\w)", one, text)


def _phrase(summary: str) -> str:
    s = summary.strip().rstrip(".").strip()
    return s[:1].lower() + s[1:] if s[:2] != s[:2].upper() else s   # "Create a…" -> "create a…"; "KB…" stays


def _fill(template: str, a: Approval) -> str:
    return speakable_names(template.format(summary=_phrase(a.summary), short=a.short,
                                           label=(a.scope or {}).get("label", "this")))


def spoken_question(a: Approval, s: ApprovalSettings) -> str:
    return _fill(s.ask, a)


def spoken_again(a: Approval, s: ApprovalSettings) -> str:
    return _fill(s.ask_again, a)


def spoken_outcome(a: Approval, s: ApprovalSettings, choice: str) -> str | None:
    """What the voice says once the question is answered: nothing for a plain yes (the tool runs and the model says how
    it went), what the session grant covers, or plainly what was not done."""
    if choice == "allow_session" and a.scope:
        return _fill(s.session, a)
    if choice in ("deny", "deny_said"):
        return _fill(s.denied, a)
    if choice == "timeout":
        return _fill(s.timed_out, a)
    return None


# ------------------------------------------------------------------------------------------ the spoken answer

_WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "all right", "fine", "please", "please do",
       "do it", "go ahead", "go for it", "absolutely", "of course", "affirmative", "correct", "yes please",
       "sure thing", "why not", "sounds good", "that's fine", "allow it", "allow", "approve", "approved", "confirm",
       "confirmed", "proceed", "make it so"}
NO = {"no", "nope", "nah", "never", "don't", "do not", "dont", "stop", "cancel", "never mind", "nevermind", "no thanks",
      "no thank you", "negative", "not now", "skip it", "leave it", "don't do it", "do not do it", "deny", "denied",
      "please don't", "please do not", "please stop", "please cancel", "please no"}
HOLD = {"wait", "hold on", "hang on", "one sec", "one second", "just a second", "give me a second", "let me think",
        "um", "uh", "hmm", "er", "erm", "mm"}
DONT_ASK = ("don't ask me again", "don't ask again", "do not ask me again", "do not ask again")
SESSION = ("for the rest of this session", "for the rest of the session", "for this session", "for the session",
           "this session", *DONT_ASK, "from now on", "every time", "always")
# Words that only agree with, or point at, the thing asked about, after a yes ("yes, go ahead and add it"); any other
# word makes the reply one of its own (deny_said), so "yes, but call it X" or "yes, in my inbox" never approves the
# call as it stands. Kept short on purpose: a missing word costs one more question, an extra one could approve a change.
AGREE = set("""please thanks thank you go ahead do it that that's this these those them one fine sure okay ok yes yeah
yep yup sounds good great perfect right correct absolutely of course and now then just all alright it's is so for
create write run make add save change changes edit file page note entry issue command proceed allow approve confirm
i you can""".split())
# words that only refuse, after a no ("no, don't do it")
REFUSE = set("""thanks thank you no not now don't do it please stop cancel that this never mind nope leave skip for
the moment yet i dont""".split())


def words(text: str) -> list[str]:
    return _WORDS.findall(text.lower().replace("’", "'"))


def _lead(w: list[str], phrases: set[str]) -> int:
    """How many leading words form one of these phrases (the longest of up to 4 words), or 0."""
    for n in (4, 3, 2, 1):
        if len(w) >= n and " ".join(w[:n]) in phrases:
            return n
    return 0


def _strip_session(w: list[str]) -> tuple[list[str], bool]:
    text = f" {' '.join(w)} "
    for p in SESSION:
        if f" {p} " in text:
            return text.replace(f" {p} ", " ", 1).split(), True
    return w, False


def is_echo(w: list[str], question: str) -> bool:
    """Words that are mostly the question just asked: its echo, not the person (a speaker heard by the microphone)."""
    q = set(words(question))
    return len(w) >= 2 and sum(x in q for x in w) / len(w) >= 0.8


def heard_answer(text: str, offered: list[str], asked: str) -> str | None:
    """The choice these words make: allow_once, allow_session, deny or deny_said; None when they answer nothing yet
    (an echo of `asked`, the question as said aloud, or "wait" alone)."""
    w = words(text)
    if not w:
        return None
    joined = " ".join(w)
    if joined in HOLD:
        return None
    choice = _choice(["yes", *w] if joined.startswith(DONT_ASK) else w, offered)   # "don't ask again" agrees
    # a reply that is not a plain answer but only repeats the question is its echo (the speaker, heard by the
    # microphone): it answers nothing, so the question stays open
    return None if choice == "deny_said" and is_echo(w, asked) else choice


def _choice(w: list[str], offered: list[str]) -> str:
    ny, nn = _lead(w, YES), _lead(w, NO)
    if nn and nn >= ny:                 # the longer phrase wins: "please don't" is a no, "please" alone a yes
        return "deny" if all(x in REFUSE for x in w[nn:]) else "deny_said"
    rest, session = _strip_session(w[ny:])
    while ny and (n := _lead(rest, YES)):   # "sure, why not": yes after yes
        rest = rest[n:]
    if (ny or session) and all(x in AGREE for x in rest):   # a yes, or "for this session" / "always" said alone
        return "allow_session" if session and "allow_session" in offered else "allow_once"
    return "deny_said"


def is_clean_answer(text: str, offered: list[str]) -> bool:
    """A yes, a session yes or a no with nothing else in it: complete as said, so the turn need not wait for Smart
    Turn's silence fallback (turn_end.py). A reply that goes on ("yes, but...") is left to Smart Turn: a permission
    must never be decided on half a sentence."""
    return heard_answer(text, offered, "") in ("allow_once", "allow_session", "deny")


# ------------------------------------------------------------------------------------------- session grants

@dataclass
class Grant:
    session: str
    space: str
    key: str
    label: str
    since: str
    uses: int = 0

    def as_json(self) -> dict[str, Any]:
        return {"session": self.session, "space": self.space, "scope": self.key, "label": self.label,
                "since": self.since, "uses": self.uses}


class Grants:
    """allow_session answers, per voice session and space, each covering exactly one scope the gate named."""

    def __init__(self) -> None:
        self._by: dict[tuple[str, str, str], Grant] = {}

    def add(self, session: str, space: str, scope: dict[str, str]) -> Grant:
        key = (session, space, scope["key"])
        if key not in self._by:
            self._by[key] = Grant(session=session, space=space, key=scope["key"], label=scope.get("label", ""),
                                  since=datetime.now().astimezone().isoformat(timespec="seconds"))
        return self._by[key]

    def match(self, session: str, space: str, scope: dict[str, str] | None) -> Grant | None:
        return None if not scope else self._by.get((session, space, scope.get("key", "")))

    def drop_session(self, session: str) -> None:
        for k in [k for k in self._by if k[0] == session]:
            del self._by[k]

    def as_json(self) -> list[dict[str, Any]]:
        return [g.as_json() for g in self._by.values()]
