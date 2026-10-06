"""M3 definitions of done 3 and 4 with real models, on the real spaces.yaml with every root on a
clone (e2e_fixtures.orch_spaces):

3. voice_gate.ts: a call that needs permission is spoken as a question and runs only after a spoken yes; silence
   cancels (a scripted yes and a scripted silence, both `say` speech or its absence);
4. Atlas on a clone: a musing is captured verbatim by `atlas.py capture --via voice` before the reply, and `atlas.py
   finish` passes its words-untouched check; the turn log's `atlas` field says where the words went.

Plus DoD 2 and 5 as they look on real models: the router's switches spoken, and /v1/status during the run.
Machine ground truth only: the files the permission was for exist or not; the journal's bytes before and after.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from e2e_fixtures import RUN_DIR, gpu_still_ours, kb_clone, orch_spaces, run_dir  # noqa: F401 - fixtures
from local_voice.client import V1Client, speech_end_s
from local_voice.router import musing
from test_turn import ask, check_gpu, connect, record, said

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio(loop_scope="module")]


async def played_out(c: V1Client, since: float, quiet_s: float = 0.7, timeout: float = 30.0) -> None:
    """Until the reply audio received since `since` has played out at the client and nothing new came for quiet_s."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ends = [seg[1] for r in c.replies.values() if r.started_at is not None and r.started_at >= since - 60
                for seg in r.segments if seg[1] >= since]
        if ends and max(ends) + quiet_s < c.now():
            return
        await asyncio.sleep(0.05)


def named(folder: Path, word: str) -> str | None:
    """The text of the file in `folder` whose name contains `word` (any case), or None."""
    hits = [p for p in folder.iterdir() if p.is_file() and word in p.name.lower()]
    return hits[0].read_text() if hits else None


async def confirm_turn(c: V1Client, request: str, answer: str | None, *, wait_s: float = 120.0) -> dict:
    """Speak a request that needs permission; once the confirm_request has come and its spoken question has played,
    speak `answer`, or stay silent (None) until the gate's timeout. Returns what happened, timed from the request's
    end of speech."""
    pcm = said(request)
    if answer is not None:
        said(answer)
    start = c.now()
    await c.stream(pcm)
    eos = start + speech_end_s(pcm)
    tail = asyncio.create_task(c.silence(wait_s + 30))
    out: dict = {"said": request, "answer": answer}
    try:
        q = await c.wait_for(lambda m: m.get("t") == "confirm_request", timeout=wait_s, since=eos)
        q_at = next(at for at, m in c.messages if m is q)
        out.update(question=q, question_after_ms=round(1000 * (q_at - eos)))
        await played_out(c, q_at)
        if answer is not None:
            tail.cancel()
            a_pcm = said(answer)
            a_start = c.now()
            await c.stream(a_pcm)
            out["answered_after_question_ms"] = round(1000 * (a_start - q_at))
            tail = asyncio.create_task(c.silence(wait_s + 30))
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn" and bool(c.replies.get(m.get("reply_id"))),
                               timeout=wait_s, since=q_at)
        end_at = next(at for at, m in c.messages if m is end)
        out["total_ms"] = round(1000 * (end_at - eos))
        replies = sorted((r for r in c.replies.values() if r.started_at is not None and r.started_at >= start),
                         key=lambda r: r.started_at)
        out["reply_text"] = " | ".join("".join(r.text) for r in replies)
        out["tools"] = [(m.get("phase"), m.get("name"), m.get("ok")) for at, m in c.messages
                        if at >= eos and m.get("t") == "tool"]
        out["transcripts"] = [m["text"] for at, m in c.messages if at >= start and m.get("t") == "transcript"
                              and m.get("final")]
    finally:
        tail.cancel()
    return out


async def test_a_permission_question_runs_on_a_spoken_yes_and_silence_refuses(orch_spaces):
    """DoD 3. Act mode in the home space (tier ask: every write needs a spoken yes), on an empty scratch home."""
    check_gpu()
    o = orch_spaces
    home = o.hub.spaces["home"].root
    assert home.is_relative_to(o.e2e_scratch), home          # never the real home folder
    said("Act mode.")
    said("Just talk.")
    c = await connect(o, "e2e-m3-confirm")
    try:
        t = await ask(c, "Act mode.")
        record("m3-act-mode", **asdict(t), mode=o.hub.mode)
        assert o.hub.mode == "act", t
        check_gpu()
        yes = await confirm_turn(c, "Please create a file called hello.txt that says hello.", "Yes.")
        # the file's name is what the recogniser made of "hello.txt" ("Hello TXT" in the 16:17 run)
        yes["file"] = named(home, "hello")
        record("m3-confirm-yes", **yes)
        check_gpu()
        no = await confirm_turn(c, "Please create a file called goodbye.txt that says goodbye.", None)
        no["file"] = named(home, "goodbye")
        log = (o.state_dir / "pi-logs" / "home.log").read_text()
        no["refused_in_tool_result"] = "did not approve" in log
        record("m3-confirm-silence", **no)
        t = await ask(c, "Just talk.")
        record("m3-just-talk", **asdict(t), mode=o.hub.mode)
    finally:
        await c.close()
    assert yes.get("question") and "hello" in json.dumps(yes["question"]).lower(), yes
    assert yes["file"] is not None and "hello" in yes["file"].lower(), yes
    assert no.get("question"), no
    assert no["file"] is None, no
    assert no["refused_in_tool_result"], "the silence was not reported to the model as a refusal"
    assert o.hub.mode == "conversation"


async def test_a_musing_is_captured_verbatim_into_atlas_on_a_clone(orch_spaces):
    """DoD 4. "Go to my journal", then a musing: the orchestrator saves the words after the cue, exactly as they were
    recognised, with atlas.py capture --via voice before the model is prompted; nothing else in the journal changes;
    atlas.py start then finish --via voice pass; the turn log's atlas field points at the note."""
    check_gpu()
    o = orch_spaces
    clone = o.hub.spaces["atlas"].root
    assert clone.is_relative_to(o.e2e_scratch) and (clone / ".git").exists(), clone
    journal = clone / "journal"
    before = {p.name: p.read_bytes() for p in journal.glob("*.md")}
    today = f"{time.strftime('%Y-%m-%d')}.md"
    musing_text = "Note this, the garden was quiet this morning and I kept thinking about the old oak tree."
    for s in ("Go to my journal.", musing_text, "Back home."):
        said(s)
    c = await connect(o, "e2e-m3-atlas")
    try:
        go = await ask(c, "Go to my journal.")
        space_after_go = o.hub.active
        check_gpu()
        note = await ask(c, musing_text, wait_s=120)
        async with httpx.AsyncClient() as h:     # the server runs on this event loop: never a blocking call
            status = (await h.get(o.e2e_status, timeout=5)).json()
        back = await ask(c, "Back home.")
    finally:
        await c.close()
    words = musing(note.transcript)
    after = {p.name: p.read_bytes() for p in journal.glob("*.md")}
    changed = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
    old, new = before.get(today, b""), after.get(today, b"")
    appended = new[len(old):].decode() if new.startswith(old) else None
    start = subprocess.run(["python3", "atlas.py", "start"], cwd=clone, capture_output=True, text=True)
    fin = subprocess.run(["python3", "atlas.py", "finish", "--via", "voice"], cwd=clone, capture_output=True, text=True)
    rows = [json.loads(x) for p in sorted((RUN_DIR / "turns").glob("*.jsonl")) for x in p.read_text().splitlines()]
    row = next((x for x in reversed(rows) if x.get("atlas")), None)
    record("m3-atlas", go=asdict(go), space_after_go=space_after_go, note=asdict(note), back=asdict(back),
           words=words, journal_changed=changed, appended=appended, start_rc=start.returncode,
           finish_rc=fin.returncode, finish_out=(fin.stdout + fin.stderr)[-400:], turn_log_atlas=row and row["atlas"],
           status={k: status.get(k) for k in ("space", "mode", "tier", "model", "tool", "hold", "last_turn")})
    assert space_after_go == "atlas" and go.reply_text.startswith("Switching to"), go
    assert words, f"no musing in the transcript: {note.transcript!r}"
    assert changed == [today], changed                                # nothing but today's note changed
    assert appended is not None and appended.endswith(f"{words}\n"), appended    # appended, byte for byte
    assert "said to voice" in appended
    assert fin.returncode == 0, fin.stdout + fin.stderr
    # atlas.text is null when the whole utterance was the musing (the cue said alone, then the words: TURN_LOG.md)
    assert row is not None and row["atlas"]["by"] == "orchestrator", row
    assert (row["atlas"]["text"] if row["atlas"]["text"] is not None else row["user_text"]) == words, row
    assert row["atlas"]["path"] == f"journal/{today}" and row["space"] == "atlas"
    assert status["space"] == "atlas" and status["tier"] == "trusted" and status["last_turn"]["eos_to_first_audio_ms"] > 0
    assert o.hub.active == "home" and back.reply_text.startswith("Back on"), back
