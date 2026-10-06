"""What the model is asked in each pass, and how a day's turns are shown to it.

Two passes on purpose: the technical pass never sees a turn of the Atlas space, and the life pass sees only the
person's own words that are already saved in Atlas. Turns are shown with short handles (S1T3) so the model can cite
them without copying long ids; the validator maps handles back and refuses citations of turns it was not shown.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .turnlog import Turn

TECH_SYSTEM = """\
You are the background reflection step of the owner's own voice agent, running on their Mac while nobody is
using it. You read one day of the owner's voice conversations and pick out the few technical facts worth keeping:
decisions they made, preferences about how their tools or the voice agent should behave, findings about their Mac,
models, code or projects, and things they said they still need to do.

Rules:
- Technical only. Never record anything about their personal life, health, feelings, family, friends or
  relationships; skip such turns entirely.
- What the owner said or clearly agreed to counts. What the assistant said counts only when the owner accepted it
  or acted on it.
- One fact per item: one plain sentence in your own words, under 200 characters.
- "cites" lists the handles (like "S1T3") of the turns the fact comes from. Cite only turns you used.
- "quote" is optional. If you give one, copy a few words exactly, character for character, from a cited turn.
- Most days have nothing worth keeping. An empty list is a good answer.
- "spoken" is what the agent will say aloud at the start of the owner's next conversation: at most two short
  sentences about the most useful facts, in plain spoken words, with no lists, symbols, file paths or code. If
  nothing is worth saying, "spoken" is exactly "NO_REPLY".

Reply with one JSON object and nothing else, in this shape:
{"facts": [{"text": "...", "kind": "decision", "cites": ["S1T3"], "quote": "..."}], "spoken": "NO_REPLY"}
"kind" is one of: decision, preference, finding, todo, question.
"""

LIFE_SYSTEM = """\
You help the owner see their own words. Below is what they said aloud to their voice agent on one day and chose
to keep in their journal. You may draft short reflections for their journal's inbox: their words, what you notice,
and a question worth sitting with. They will accept or reject each one.

Rules:
- Their words are never rewritten. Every quote is copied exactly, character for character, from one turn, and
  cites that turn's handle (like "S2T1").
- Observations with evidence, never verdicts: "you mentioned work twice today, once about ... and once about ...",
  never "you are stressed". Do not name a feeling they did not name themselves. No diagnosing and no advice.
- One entry is not a pattern. Do not invent connections. If nothing stands out, return no reflections.
- Questions open things up; they do not lead.
- At most {max_reflections} reflections. Each has a short title, one to four quotes, one "notice" sentence and at
  most two questions.
- Tags: reuse these when one fits: {tags}. Life areas look like area/work or area/health. At most three.

Reply with one JSON object and nothing else, in this shape:
{{"reflections": [{{"title": "...", "quotes": [{{"turn": "S2T1", "text": "exact words"}}], "notice": "...",
  "questions": ["..."], "tags": ["area/work"]}}]}}
"""

TECH_TASK = "Reply with the JSON object your instructions describe, for the conversations above."
LIFE_TASK = "Reply with the JSON object your instructions describe, for the words above."


@dataclass
class Shown:
    """The turns a pass showed the model, by handle, and the text it was given."""
    handles: dict[str, Turn]
    text: str
    omitted: int = 0


def handles_for(turns: list[Turn]) -> dict[str, Turn]:
    """S<n>T<m>: sessions numbered in order of first appearance, turns by their own number."""
    order: dict[str, int] = {}
    out: dict[str, Turn] = {}
    for t in turns:
        n = order.setdefault(t.session, len(order) + 1)
        out[f"S{n}T{t.turn}"] = t
    return out


def weekday(day: str) -> str:
    return date.fromisoformat(day).strftime("%A")


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def render_tech(day: str, turns: list[Turn], max_chars: int) -> Shown:
    """The technical pass's input. Over budget, the oldest replies shrink first, then the oldest turns go."""
    handles = handles_for(turns)
    reply_cap = 4000

    def build(items: list[tuple[str, Turn]], cap: int) -> str:
        head = (f"Voice conversations on {weekday(day)} {day} (technical pass). Each turn starts with its handle, "
                f"the time, the space and the mode.\n")
        blocks = []
        for h, t in items:
            tools = ", ".join(f"{n} {'ok' if ok else 'failed'}" for n, ok in t.tools)
            reply = t.spoken_text
            cut = " (interrupted; this is what the owner heard)" if t.interrupted and t.heard_text is not None else ""
            lines = [f"[{h}] {t.t_start.strftime('%H:%M')} · space {t.space} · {t.mode}",
                     f"OWNER: {t.user_text}", f"ASSISTANT{cut}: {_clip(reply, cap) if reply else '(no reply)'}"]
            if tools:
                lines.append(f"TOOLS: {tools}")
            blocks.append("\n".join(lines))
        return head + "\n" + "\n\n".join(blocks) + "\n"

    items = list(handles.items())
    text = build(items, reply_cap)
    while len(text) > max_chars and reply_cap > 200:
        reply_cap //= 2
        text = build(items, reply_cap)
    omitted = 0
    while len(text) > max_chars and len(items) > 1:
        items.pop(0)
        omitted += 1
        text = build(items, reply_cap)
    if omitted:
        text = f"({omitted} earlier turns omitted to fit.)\n" + text
    return Shown(handles=dict(items), text=text, omitted=omitted)


def render_life(day: str, quotable: list[tuple[Turn, str]], max_chars: int) -> Shown:
    """The life pass's input: only the person's saved words, never the assistant's replies."""
    handles = handles_for([t for t, _ in quotable])
    by_turn = {(t.session, t.turn): words for t, words in quotable}
    items = list(handles.items())

    def build(items: list[tuple[str, Turn]]) -> str:
        head = (f"What the owner said aloud on {weekday(day)} {day} and kept in their journal. Each entry starts "
                f"with its handle and the time.\n")
        blocks = [f"[{h}] {t.t_start.strftime('%H:%M')}\n{by_turn[(t.session, t.turn)]}" for h, t in items]
        return head + "\n" + "\n\n".join(blocks) + "\n"

    text = build(items)
    omitted = 0
    while len(text) > max_chars and len(items) > 1:
        items.pop(0)
        omitted += 1
        text = build(items)
    return Shown(handles=dict(items), text=text, omitted=omitted)
