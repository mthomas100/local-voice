"""One whole conversation with real models, the way a person would use it (judge the whole
conversation, not the checkboxes): small talk, a tool lookup, an interruption mid-reply, a slow knowledge-base
question acknowledged first, a switch to the journal (a clone), a musing captured verbatim, back home, act mode
with a spoken yes and a silence for two writes, back to talk, and goodbye. One connection, one continuous Pi
conversation, `say` speech at real-time pace and room silence between turns, every root on a clone
(e2e_fixtures.orch_spaces).

Every beat is recorded with its timings (end of speech to first audio and to the end of the turn, the server's
latency stages, load) whatever happens; a beat that goes wrong is noted and the conversation carries on, and the test
fails at the end listing every problem. Machine ground truth only: the facts planted in the scratch home, the files
the permission was for, the journal's bytes, the follow-up's known answer.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from e2e_fixtures import RUN_DIR, kb_clone, orch_spaces, run_dir  # noqa: F401 - fixtures
from local_voice.router import musing
from test_spaces import confirm_turn, named
from test_turn import Turn, ask, check_gpu, connect, record, said

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio(loop_scope="module")]

DEVICE = "e2e-conversation"
SHOPPING = "eggs\noat milk\ncoffee beans\n"
STORY = "Tell me a long story about a lighthouse keeper and his cat with lots of detail."
FOLLOW = "Stop. What is the capital of Italy?"
KB_Q = "What does my knowledge base say about the hold gate?"
MUSE = "Note this, I want to spend more time outside this autumn, maybe a walk before breakfast."


class Beats:
    def __init__(self, o):
        self.o = o
        self.rows: list[dict] = []
        self.problems: list[str] = []

    def server_line(self) -> dict | None:
        cl = self.o.clients.get(DEVICE)
        return cl.session.latency.lines[-1] if cl and cl.session.latency.lines else None

    def last_reply(self) -> dict | None:
        """The last run: words written by the model and said (the spoken cap's view), and how it was acknowledged."""
        cl = self.o.clients.get(DEVICE)
        return cl.session.agent.last_reply if cl else None

    def add(self, beat: str, data: dict, *checks: tuple[bool, str]) -> None:
        bad = [why for ok, why in checks if not ok]
        self.problems += [f"{beat}: {why}" for why in bad]
        row = {"beat": beat, "ok": not bad, "problems": bad, "space": self.o.hub.active, "mode": self.o.hub.mode,
               "reply": self.last_reply(), **data}
        self.rows.append(row)
        record(f"conversation:{beat}", **row)

    def turn(self, beat: str, t: Turn, *checks: tuple[bool, str], **extra) -> Turn:
        self.add(beat, {**asdict(t), "server": self.server_line(), **extra}, *checks)
        return t


async def test_a_whole_conversation(orch_spaces):
    check_gpu()
    o = orch_spaces
    home = o.hub.spaces["home"].root
    atlas = o.hub.spaces["atlas"].root
    assert home.is_relative_to(o.e2e_scratch) and atlas.is_relative_to(o.e2e_scratch)
    (home / "shopping-list.md").write_text(SHOPPING)
    lines = ["Hi there, how is your day going?", "What's on my shopping list?", STORY, FOLLOW, KB_Q,
             "Go to my journal.", MUSE, "Back home.", "Act mode.",
             "Please create a file called plan.txt that says walk every day.", "Yes.",
             "Please create a file called later.txt that says maybe later.", "Just talk.",
             "Thanks, that's all for now."]
    for s in lines:
        said(s)
    b = Beats(o)
    started = time.strftime("%H:%M:%S")
    c = await connect(o, DEVICE)
    try:
        t = await ask(c, lines[0])
        b.turn("small talk", t, (not t.tools, f"tools {t.tools}"), (len(t.reply_text.split()) <= 60,
               f"{len(t.reply_text.split())} words"))
        check_gpu()

        t = await ask(c, lines[1], wait_s=120)
        b.turn("tool lookup", t, (bool(t.tools), "no tool"),
               (any(w in t.reply_text.lower() for w in ("eggs", "oat milk", "coffee")), "the list's items not said"))
        check_gpu()

        # an interruption mid-reply: the story is cut 2 s into its audio by the follow-up question
        start = c.now()
        await c.stream(said(STORY))
        quiet = asyncio.create_task(c.silence(90))
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=90, since=start)
        while True:
            r = c.current
            if r is not None and r.first_audio_at is not None and c.now() - r.first_audio_at > 2.0:
                break
            await asyncio.sleep(0.02)
        quiet.cancel()
        story, speech_at = r, c.now()
        follow = asyncio.create_task(ask(c, FOLLOW))
        intr = await c.wait_for(lambda m: m.get("t") == "interrupt", timeout=15, since=speech_at)
        intr_ms = round(1000 * (next(at for at, m in c.messages if m is intr) - speech_at))
        silent = story.silent_from(speech_at)
        t = await follow
        agent = o.clients[DEVICE].session.agent
        b.turn("interruption", t, (intr_ms <= 600, f"interrupt {intr_ms} ms after speech"),
               ("rome" in t.reply_text.lower(), "no Rome in the answer"),
               story_text="".join(story.text)[:300], story_played_ms=story.played_ms,
               audio_stop_ms=None if silent is None else round(1000 * (silent - speech_at)), interrupt_ms=intr_ms,
               silent_runs=dict(agent.silent_runs))
        check_gpu()

        t = await ask(c, KB_Q, wait_s=150)
        # the canned acknowledgement, or the model's own words said before its tool call (then no canned one follows)
        how = (b.last_reply() or {}).get("ack")
        ack = o.cfg.acks.get(t.tools[0], o.cfg.acks["default"]) if t.tools and how == "canned" else ""
        b.turn("slow kb question", t, (bool(t.tools), "no tool"),
               (how in ("canned", "model") and t.reply_text.startswith(ack),
                f"not acknowledged first ({how}): {t.reply_text[:60]!r}"),
               (len(t.reply_text[len(ack):].split()) >= 4, "no answer after the acknowledgement"))
        check_gpu()

        before = {p.name: p.read_bytes() for p in (atlas / "journal").glob("*.md")}
        t = await ask(c, "Go to my journal.")
        b.turn("to the journal", t, (o.hub.active == "atlas", f"space {o.hub.active}"),
               (t.reply_text.startswith("Switching to"), "no spoken switch"))
        t = await ask(c, MUSE, wait_s=120)
        words = musing(t.transcript) or ""
        today = f"{time.strftime('%Y-%m-%d')}.md"
        after = {p.name: p.read_bytes() for p in (atlas / "journal").glob("*.md")}
        changed = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
        old, new = before.get(today, b""), after.get(today, b"")
        appended = new[len(old):].decode() if new.startswith(old) else None
        # a pause inside the musing can split it into two utterances (16:57), each saved as its own block: every
        # part must be there word for word, in order, and nothing else
        parts = [x for x in re.split(r"(?<=[.!?])\s+", words) if x]
        at = [appended.find(x) if appended else -1 for x in parts]
        b.turn("musing captured", t, (bool(words), "no musing in the transcript"), (changed == [today], f"changed {changed}"),
               (appended is not None and -1 not in at and at == sorted(at) and appended.endswith(f"{parts[-1]}\n"),
                "not appended word for word"), words=words, appended=appended)
        t = await ask(c, "Back home.")
        b.turn("back home", t, (o.hub.active == "home", f"space {o.hub.active}"),
               (t.reply_text.startswith("Back on"), "no spoken switch"))
        check_gpu()

        t = await ask(c, "Act mode.")
        b.turn("act mode", t, (o.hub.mode == "act", f"mode {o.hub.mode}"))
        yes = await confirm_turn(c, lines[9], "Yes.")
        plan = named(home, "plan")
        b.add("write, spoken yes", {**yes, "file": plan}, (bool(yes.get("question")), "no permission question"),
              (plan is not None and "walk" in plan.lower(), "plan.txt not written"))
        check_gpu()
        no = await confirm_turn(c, lines[11], None)
        later = named(home, "later")
        b.add("write, silence", {**no, "file": later}, (bool(no.get("question")), "no permission question"),
              (later is None, "later.txt was written"))
        t = await ask(c, "Just talk.")
        b.turn("just talk", t, (o.hub.mode == "conversation", f"mode {o.hub.mode}"))
        t = await ask(c, lines[13])
        b.turn("goodbye", t, (bool(t.reply_text.strip()), "no reply"))
    finally:
        await c.close()
    voice = [r for r in b.rows if r.get("eos_to_first_audio_ms") is not None]
    summary = {"window": f"{started}-{time.strftime('%H:%M:%S')}", "beats": len(b.rows),
               "problems": b.problems,
               "first_audio_ms": {r["beat"]: r["eos_to_first_audio_ms"] for r in voice},
               "llm_ttft_ms": {r["beat"]: (r.get("server") or {}).get("stages", {}).get("llm_ttft_ms") for r in voice},
               "reply_audio_s": {r["beat"]: r.get("reply_audio_s") for r in voice},
               "reply": {r["beat"]: r.get("reply") for r in b.rows},
               "user_turn_ms": {r["beat"]: (r.get("server") or {}).get("user_turn_ms") for r in voice},
               "load1": [r.get("load1") for r in voice]}
    record("conversation-summary", **summary)
    (RUN_DIR / "conversation.json").write_text(json.dumps({"summary": summary, "beats": b.rows}, indent=1, default=str))
    assert not b.problems, "\n".join(b.problems)
