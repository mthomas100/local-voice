"""A synthetic day of voice sessions in the turn-log format of brain/TURN_LOG.md, written the way the orchestrator
will write it, plus the Atlas captures it would have made (through the clone's own `atlas.py capture --via voice`).

Sunday 2026-10-04:
  s-20261004-0914-a1  home   09:14-09:20  four technical turns (a kb question, a voice decision, a preference, a todo)
  s-20261004-1330-b2  home   13:30-13:31  an interrupted reply, then "thanks"
  s-20261004-2102-c3  atlas  21:02-21:07  two musings captured into Atlas (one with "note this:" left off), a question
plus one line that is not JSON, one record of another type, and one duplicate record.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DAY = "2026-10-04"
TZ = "-07:00"

MUSING_1 = ("been thinking about how stretched thin I feel with work lately, I keep saying yes to things before "
            "I've even thought about them")
MUSING_2_SAID = ("note this: walked by the river after dinner and it was the first time all week my head felt "
                 "quiet")
MUSING_2_SAVED = "walked by the river after dinner and it was the first time all week my head felt quiet"


def _t(hm: str, s: int = 0) -> str:
    return f"{DAY}T{hm}:{s:02d}.000{TZ}"


def turn(session, n, start, end, space, user, reply, *, client="iphone", tools=(), heard=None, atlas=None,
         mode="conversation"):
    return {"v": 1, "type": "turn", "session": session, "turn": n, "t_start": start, "t_end": end, "client": client,
            "space": space, "mode": mode, "input": "voice", "user_text": user, "reply_text": reply,
            "heard_text": heard, "interrupted": heard is not None,
            "tools": [{"name": t, "ok": True} for t in tools], "atlas": atlas, "tone": None}


def day_records(atlas_root: Path | None) -> list[dict]:
    a, b, c = "s-20261004-0914-a1", "s-20261004-1330-b2", "s-20261004-2102-c3"
    root = str(atlas_root) if atlas_root else "/nonexistent/atlas"
    note = f"journal/{DAY}.md"
    return [
        turn(a, 1, _t("09:14", 3), _t("09:14", 19), "home", "what does my knowledge base say about the hold gate",
             "It says the hold gate sits in front of llama-swap on port 8090 and lets one job hold the GPU at a time.",
             tools=("kb",)),
        turn(a, 2, _t("09:16", 2), _t("09:16", 9), "home",
             "okay let's keep kokoro as the fallback voice and ryan as the main one",
             "Got it: Ryan stays the main voice and Kokoro is the fallback."),
        turn(a, 3, _t("09:18"), _t("09:18", 4), "home", "and keep your answers to two sentences when we're just talking",
             "Will do."),
        turn(a, 4, _t("09:20"), _t("09:20", 6), "home",
             "remind me tomorrow to check why smart turn called my question incomplete",
             "I can't set reminders yet, but I've noted it."),
        turn(b, 1, _t("13:30"), _t("13:30", 12), "home", "how long did the voicechat benchmark take",
             "The VoiceChat 11B benchmark ran for sixty-four seconds at about sixty milliseconds per frame, which is "
             "comfortably real time on this Mac.", client="mac",
             heard="The VoiceChat 11B benchmark ran for sixty-four seconds"),
        turn(b, 2, _t("13:31"), _t("13:31", 3), "home", "thanks that's all", "Anytime.", client="mac"),
        turn(c, 1, _t("21:02"), _t("21:02", 20), "atlas", MUSING_1, "Saved. That sounds like a lot to carry.",
             atlas={"root": root, "path": note, "by": "orchestrator", "text": None}),
        turn(c, 2, _t("21:05"), _t("21:05", 15), "atlas", MUSING_2_SAID, "Saved to today's note.",
             atlas={"root": root, "path": note, "by": "orchestrator", "text": MUSING_2_SAVED}),
        turn(c, 3, _t("21:07"), _t("21:07", 30), "atlas", "what did I write last week about sleep",
             "Last week you wrote twice about sleep, both times about waking early."),
    ]


def write_day(turns_dir: Path, atlas_root: Path | None, *, extra_lines: bool = True) -> Path:
    turns_dir.mkdir(parents=True, exist_ok=True)
    recs = day_records(atlas_root)
    lines = [json.dumps(r) for r in recs]
    if extra_lines:
        lines.insert(2, '{"v": 1, "type": "turn", "session": "broken')                 # cut off mid-write
        lines.insert(0, json.dumps({"v": 1, "type": "session_start", "session": recs[0]["session"]}))
        lines.append(json.dumps(recs[1]))                                                 # a duplicate
    p = turns_dir / f"{DAY}.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def capture_into_atlas(atlas_root: Path) -> None:
    """What the orchestrator does at conversation time: verbatim capture through the repo's atlas.py, then commit
    (the person's words are committed in the clone so the brain's later commit can be checked to touch nothing else)."""
    for words in (MUSING_1, MUSING_2_SAVED):
        subprocess.run([sys.executable, "atlas.py", "capture", "--via", "voice", "--to", f"journal/{DAY}.md"],
                       input=words, text=True, cwd=atlas_root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=atlas_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "test: voice captures"], cwd=atlas_root, check=True)


def write_next_day_turn(turns_dir: Path, hm: str = "08:00") -> Path:
    rec = turn("s-20261005-0800-d4", 1, f"2026-10-05T{hm}:00.000{TZ}", f"2026-10-05T{hm}:09.000{TZ}", "home",
               "good morning", "Morning.")
    p = turns_dir / "2026-10-05.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    return p


# What the stub should answer, by pass. Handles: technical pass S1 = a1 (T1-T4), S2 = b2 (T1-T2); life pass S1 = c3.
TECH_REPLY = {
    "facts": [
        {"text": "Kokoro is the fallback voice and Ryan stays the main voice.", "kind": "decision",
         "cites": ["S1T2"], "quote": "keep kokoro as the fallback voice"},
        {"text": "Spoken replies in conversation mode should be at most two sentences.", "kind": "preference",
         "cites": ["S1T3"], "quote": "keep your answers to two sentences"},
        {"text": "Check why Smart Turn judged a finished question incomplete.", "kind": "todo", "cites": ["S1T4"]},
        {"text": "The hold gate lets two jobs share the GPU.", "kind": "finding", "cites": ["S1T1"],
         "quote": "lets two jobs share the GPU"},                                      # not said: dropped
        {"text": "VoiceChat runs at twice real time.", "kind": "finding", "cites": ["S9T9"]},  # not shown: dropped
    ],
    "spoken": "Kokoro stays the fallback voice, and you asked for replies of two sentences at most.",
}
LIFE_REPLY = {
    "reflections": [
        {"title": "Saying yes, and a quiet head",
         "quotes": [{"turn": "S1T1", "text": "I keep saying yes to things before I’ve even thought about them"},
                    {"turn": "S1T2", "text": "the first time all week my head felt quiet"}],
         "notice": "You talked about work once and about the river walk once; the walk is the only time this week "
                   "you described your head as quiet.",
         "questions": ["What was different about the walk?"], "tags": ["area/work", "Not A Tag!"]},
        {"title": "Burnout", "quotes": [{"turn": "S1T1", "text": "stretched thin"}],
         "notice": "You seem stressed and close to burnout.", "questions": []},              # a verdict: dropped
        {"title": "Made up", "quotes": [{"turn": "S1T1", "text": "I hate my job"}],          # not their words: dropped
         "notice": "x", "questions": ["y?"]},
    ],
}
