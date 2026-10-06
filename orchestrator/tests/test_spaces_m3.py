"""M3, model-free: whole turns over protocol v1 against the stub LLM, with several spaces. The
router's switches and its question back, the `space` and `mode` messages and their errors, act mode by voice (and its
tools, and no lookup budget there), a permission question answered by a spoken yes and by silence, and Atlas capture
on a git clone of an Atlas journal (ATLAS_REPO) with atlas.py finish's untouched check."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from harness import rig
from local_voice.client import tone_pcm
from local_voice.router import musing, route

ATLAS = Path(os.environ.get("ATLAS_REPO", "~/atlas")).expanduser()   # an Atlas-style journal repo, cloned, never written
DEFAULTS = {"model": "local/qwen38", "thinking": "off", "mode": "conversation", "tier": "ask", "voice": "ryan"}


def spaces_file(tmp_path: Path, journal_root: Path | None = None) -> dict:
    work = tmp_path / "work"
    notes = tmp_path / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "todo.md").write_text("buy bread\n")
    spec = {"home": {"root": str(work), "description": "the scratch folder", "triggers": ["home", "back home"],
                     "skills": [], "tools": ["read", "ls"], "act_tools": ["bash"], "tier": "ask"},
            "notes": {"root": str(notes), "description": "the notes folder", "triggers": ["notes folder", "my notes"],
                      "skills": [], "tools": ["read", "ls"], "tier": "readonly"}}
    if journal_root is not None:
        spec["journal"] = {"root": str(journal_root), "description": "your journal", "triggers": ["my journal"],
                           "skills": "auto", "tools": ["read", "ls", "bash"], "tier": "trusted",
                           "bash_allow": [["python3", "atlas.py", "capture"], ["python3", "atlas.py", "start"],
                                          ["python3", "atlas.py", "finish"]]}
    path = tmp_path / "spaces-m3.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "defaults": DEFAULTS, "spaces": spec}))
    return {"agent": {"spaces_file": str(path)}}


def test_the_router_reads_switches_modes_and_notes():
    from local_voice.config import load_config
    from local_voice.spaces import load_spaces
    sp = load_spaces(load_config())
    assert route("Go to my journal.", sp).space == "atlas"
    r = route("Switch to the knowledge base and search for the hold gate.", sp)
    assert (r.kind, r.space, r.rest) == ("switch", "kb", "search for the hold gate.")
    assert route("Back home, please.", sp).space == "home"
    assert route("The kb says what about heat pumps?", sp).kind == "none"      # a sentence, not a switch
    assert route("Go to my journal or the kb.", sp).candidates == ["atlas", "kb"]
    assert route("Act mode.", sp).mode == "act" and route("Just talk.", sp).mode == "conversation"
    assert route("go to sleep", sp).kind == "none"
    assert musing("Note this: the garden was quiet… like $5") == "the garden was quiet… like $5"
    assert musing("What did I note?") is None


async def _text_turn(c, text: str, since: float | None = None) -> dict:
    t0 = c.now() if since is None else since
    await c.send({"t": "text", "text": text})
    return await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=20, since=t0)


async def test_a_trigger_phrase_switches_space_and_the_rest_is_its_first_prompt(tmp_path):
    async with rig(tmp_path, overrides=spaces_file(tmp_path)) as r:
        r.stub.script([{"text": "There is a todo file.", "delay_ms": 5}])
        c = await r.client()
        t0 = c.now()
        await _text_turn(c, "Go to my notes and list the files.")
        sp = await c.wait_for(lambda m: m.get("t") == "space" and m.get("name") == "notes", timeout=5, since=t0)
        assert sp["tier"] == "readonly" and sp["description"] == "the notes folder"
        assert r.user_texts() == ["list the files."]
        assert r.tts.spoken[1:] == ["Switching to the notes folder.", "There is a todo file."]
        assert r.orch.hub.active == "notes" and set(r.orch.hub.pool.children) == {"notes"}   # home never started
        await _text_turn(c, "Back home.")
        assert r.orch.hub.active == "home" and r.tts.spoken[-1] == "Back on the scratch folder."
        assert len(r.stub.requests()) == 1, "a bare switch makes no model call"
        await c.close()


async def test_a_phrase_naming_two_spaces_is_asked_about(tmp_path):
    async with rig(tmp_path, overrides=spaces_file(tmp_path, journal_root=tmp_path / "work")) as r:
        c = await r.client()
        await _text_turn(c, "Switch to my notes or my journal.")
        assert r.tts.spoken[-1] == "Do you mean the notes folder or your journal?"
        assert r.orch.hub.active == "home"
        await _text_turn(c, "The notes folder.")
        assert r.orch.hub.active == "notes" and r.tts.spoken[-1] == "Switching to the notes folder."
        assert r.stub.requests() == []
        await c.close()


async def test_space_and_mode_messages_are_answered_or_refused(tmp_path):
    async with rig(tmp_path, overrides=spaces_file(tmp_path)) as r:
        c = await r.client()
        t0 = c.now()
        await c.send({"t": "space", "name": "notes"})
        sp = await c.wait_for(lambda m: m.get("t") == "space" and m.get("name") == "notes", timeout=5, since=t0)
        assert sp["mode"] == "conversation"
        await c.send({"t": "space", "name": "nowhere"})
        err = await c.wait_for(lambda m: m.get("t") == "error", timeout=5, since=t0)
        assert err["code"] == "space_unknown"
        await c.send({"t": "mode", "name": "party"})
        err = await c.wait_for(lambda m: m.get("t") == "error" and m.get("code") == "mode_unknown", timeout=5, since=t0)
        t1 = c.now()
        await c.send({"t": "mode", "name": "act"})
        sp = await c.wait_for(lambda m: m.get("t") == "space" and m.get("mode") == "act", timeout=5, since=t1)
        assert sp["name"] == "notes"
        shutil.rmtree(tmp_path / "work")
        await c.send({"t": "space", "name": "home"})
        err = await c.wait_for(lambda m: m.get("t") == "error" and m.get("code") == "space_unavailable", timeout=5, since=t1)
        assert r.orch.hub.active == "notes"
        await c.close()


async def test_act_mode_by_voice_widens_the_tools_and_lifts_the_lookup_budget(tmp_path):
    ov = spaces_file(tmp_path)
    ov["agent"]["tool_budget"] = {"calls": 1, "seconds": 0}
    ls = {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}], "delay_ms": 5}
    async with rig(tmp_path, overrides=ov) as r:
        r.stub.script([{"text": "Hi.", "delay_ms": 5}, ls, ls, ls, {"text": "Three looks.", "delay_ms": 5},
                       {"text": "Hi again.", "delay_ms": 5}])
        c = await r.client()
        await _text_turn(c, "Hello.")
        tools = lambda i: [t["function"]["name"] for t in r.stub.requests()[i]["body"]["tools"]]
        assert tools(0) == ["read", "ls", "space_switch"]       # space_switch: the model's switch, in both modes
        await _text_turn(c, "Act mode.")
        assert r.orch.hub.mode == "act" and r.tts.spoken[-1].startswith("Act mode.")
        await _text_turn(c, "Look three times.")
        reqs = r.stub.requests()
        assert tools(1) == ["read", "ls", "space_switch", "bash"]
        results = [m.get("content") for m in reqs[4]["body"]["messages"] if m.get("role") == "tool"]
        assert len(results) == 3 and not any("Stop looking" in json.dumps(x) for x in results)
        await _text_turn(c, "Just talk.")
        await _text_turn(c, "Hello again.")
        assert tools(-1) == ["read", "ls", "space_switch"]
        await c.close()


@pytest.mark.parametrize("answer", ["yes", "silence"])
async def test_a_permission_question_runs_on_a_spoken_yes_and_silence_cancels(tmp_path, answer):
    """M3 DoD 3, model-free: voice_gate.ts asks, the orchestrator speaks the question and hears the answer."""
    ov = spaces_file(tmp_path)
    ov["agent"]["approvals"] = {"voice_wait_s": 2.5}                     # the test client shows no card
    ov["agent"]["progress"] = {"after_s": 0.5, "text": "Still looking."}   # due while the question waits
    cmd = "echo hello > made.txt"
    async with rig(tmp_path, stt_script=["make a file", "yes"], overrides=ov) as r:
        r.stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": cmd}}], "delay_ms": 5},
                       {"text": "Done.", "delay_ms": 5}])
        c = await r.client()
        await _text_turn(c, "Act mode.")
        t0 = c.now()
        await c.speak(tone_pcm(0.8))
        tail = asyncio.create_task(c.silence(1.0))
        q = await c.wait_for(lambda m: m.get("t") == "confirm_request", timeout=10, since=t0)
        assert q["title"] == "May I run a command?" and cmd in q["message"]
        await tail
        if answer == "yes":
            await c.speak(tone_pcm(0.5))
        tail = asyncio.create_task(c.silence(8.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=c.now())
        tail.cancel()
        await c.close()
    made = (tmp_path / "work" / "made.txt")
    if answer == "yes":
        assert made.read_text() == "hello\n"
    else:   # silence: refused aloud by the orchestrator, and the run ended without another model call
        assert not made.exists() and len(r.stub.requests()) == 1
        assert r.tts.spoken[-1] == "I didn't hear an answer, so I didn't run that command."
    question = 'I\'d like to run a shell command in the scratch folder, starting with "echo hello". Shall I?'
    assert question in r.tts.spoken, "the question was spoken, saying what would run and where"
    # the agent is waiting for an answer, not looking: "Still looking." talked over the person's yes (e2e 16:17)
    assert "Still looking." not in r.tts.spoken, r.tts.spoken


@pytest.mark.skipif(not (ATLAS / "atlas.py").exists(), reason="set ATLAS_REPO to an Atlas journal repo to clone")
async def test_atlas_musing_is_captured_verbatim_before_the_reply_on_a_clone(tmp_path):
    """M3 DoD 4: on a git clone of the real Atlas (never the real one), the orchestrator saves the words itself with
    atlas.py capture --via voice before the model replies, atlas.py finish passes its words-untouched check, and the
    turn log's atlas field says where they went."""
    clone = tmp_path / "atlas"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ATLAS), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "remote", "set-url", "--push", "origin", "DISABLED"], check=True)
    assert clone.resolve() != ATLAS.resolve()
    words = "the garden was quiet… like $5 & `odd` — 'really' quiet"
    async with rig(tmp_path, overrides=spaces_file(tmp_path, journal_root=clone)) as r:
        r.stub.script([{"text": "A quiet garden.", "delay_ms": 5}])
        c = await r.client()
        await _text_turn(c, "Go to my journal.")
        await _text_turn(c, f"Note this: {words}")
        await asyncio.sleep(1.0)
        await c.close()
        prompt = r.user_texts()[-1]
        spoken = list(r.tts.spoken)
    assert prompt.startswith(f"Note this: {words}") and "already saved word for word" in prompt
    # the orchestrator says so at once, before the model's reply (whose first request may be a cold prefill)
    assert spoken[-2:] == ["Saved.", "A quiet garden."] and 'they were told "Saved."' in prompt
    note = (clone / "journal" / f"{time.strftime('%Y-%m-%d')}.md").read_text()
    assert note.endswith(f"{words}\n")
    assert subprocess.run(["python3", "atlas.py", "start"], cwd=clone, capture_output=True).returncode == 0
    fin = subprocess.run(["python3", "atlas.py", "finish", "--via", "voice"], cwd=clone, capture_output=True, text=True)
    assert fin.returncode == 0, fin.stdout + fin.stderr
    rows = [json.loads(x) for p in (tmp_path / "turns").glob("*.jsonl") for x in p.read_text().splitlines()]
    muse = next(x for x in rows if x["user_text"].startswith("Note this"))
    assert muse["space"] == "journal" and muse["atlas"]["by"] == "orchestrator" and muse["atlas"]["text"] == words
    assert muse["atlas"]["path"] == f"journal/{time.strftime('%Y-%m-%d')}.md" and muse["atlas"]["root"] == str(clone)


@pytest.mark.skipif(not (ATLAS / "atlas.py").exists(), reason="set ATLAS_REPO to an Atlas journal repo to clone")
async def test_a_cue_said_alone_makes_the_next_utterance_the_musing(tmp_path):
    """Smart Turn ends the turn at the pause after "Note this," (e2e 2026-10-05 16:18): the orchestrator says "Go
    ahead." without a model call and saves the next utterance word for word; a capture the model runs itself is in
    the turn log too, as `by: model` (TURN_LOG.md, the fallback path)."""
    clone = tmp_path / "atlas"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ATLAS), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "remote", "set-url", "--push", "origin", "DISABLED"], check=True)
    words = "the garden was quiet this morning"
    own = "a second thought, saved by the model"
    cmd = f"cd {clone} && python3 atlas.py capture --via voice <<'ATLAS_END'\n{own}\nATLAS_END"
    async with rig(tmp_path, overrides=spaces_file(tmp_path, journal_root=clone)) as r:
        r.stub.script([{"text": "Saved. A quiet garden.", "delay_ms": 5},
                       {"tool_calls": [{"name": "bash", "arguments": {"command": cmd}}], "delay_ms": 5},
                       {"text": "Saved that too.", "delay_ms": 5}])
        c = await r.client()
        await _text_turn(c, "Go to my journal.")
        await _text_turn(c, "Note this.")
        assert r.tts.spoken[-1] == "Go ahead." and r.stub.requests() == []
        await _text_turn(c, words)
        assert "already saved word for word" in r.user_texts()[-1]
        await _text_turn(c, own)
        await asyncio.sleep(1.0)
        await c.close()
    note = (clone / "journal" / f"{time.strftime('%Y-%m-%d')}.md").read_text()
    assert note.count(words) == 1 and note.count(own) == 1 and note.endswith(f"{own}\n")
    rows = [json.loads(x) for p in (tmp_path / "turns").glob("*.jsonl") for x in p.read_text().splitlines()]
    saved = {x["user_text"]: x["atlas"] for x in rows if x.get("atlas")}
    assert saved[words]["by"] == "orchestrator" and saved[words]["text"] is None, saved
    assert saved[own]["by"] == "model" and saved[own]["text"] is None, saved
    assert saved[own]["path"] == f"journal/{time.strftime('%Y-%m-%d')}.md"


@pytest.mark.skipif(not (ATLAS / "atlas.py").exists(), reason="set ATLAS_REPO to an Atlas journal repo to clone")
async def test_a_musing_split_by_the_turn_end_is_saved_whole(tmp_path):
    """The turn end took a pause inside a musing for its end ("...this autumn." / "Maybe a walk before breakfast.",
    e2e 2026-10-05 16:57: the second part went to the model, which asked to run its own capture and was refused at
    the permission timeout). Speech that starts within journal_join_s of a saved musing's turn end is saved with it;
    after a longer pause it is a new turn for the model."""
    clone = tmp_path / "atlas"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ATLAS), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "remote", "set-url", "--push", "origin", "DISABLED"], check=True)
    first, second, later = "Note this, I want more time outside.", "Maybe a walk before breakfast.", "What time is it?"
    async with rig(tmp_path, stt_script=["Go to my journal.", first, second, later],
                   overrides=spaces_file(tmp_path, journal_root=clone)) as r:
        # the first reply waits 6 s on the model, a stand-in for the journal child's cold prefill (7.7 s at 17:17)
        r.stub.script([{"text": "A good plan.", "delay_ms": 5, "pre_delay_ms": 6000},
                       {"text": "A walk, then.", "delay_ms": 5}, {"text": "It is noon.", "delay_ms": 5}])
        c = await r.client()

        async def say(pause_s: float) -> float:
            await c.silence(pause_s)
            at = c.now()
            await c.speak(tone_pcm(0.8))
            await c.wait_for(lambda m: m.get("t") == "transcript" and m.get("final"), timeout=10, since=at)
            return at

        tail = asyncio.create_task(c.silence(30.0))
        await say(0.2)
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        tail.cancel()
        await say(0.5)                                  # the musing
        went_on = await say(1.0)                        # goes on after the turn ended (the harness: 0.4 s after the VAD stop), while the first run waits on the model
        at = c.now()
        tail = asyncio.create_task(c.silence(30.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id"), timeout=15, since=at)
        tail.cancel()
        # the second part's "Saved." (the first part's may be cut off by the second part's speech)
        saved_after = max(t for t, m in c.messages if t >= went_on and m.get("t") == "reply_text"
                          and m.get("delta", "").strip() == "Saved.") - at
        await say(3.0)                                  # a new turn
        at = c.now()
        tail = asyncio.create_task(c.silence(30.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=at)
        tail.cancel()
        await c.close()
        prompts = r.user_texts()
    note = (clone / "journal" / f"{time.strftime('%Y-%m-%d')}.md").read_text()
    assert note.count("I want more time outside.") == 1 and note.endswith(f"{second}\n"), note[-300:]
    assert later not in note
    assert prompts[-1] == later and all("already saved" in p for p in prompts[-3:-1]), prompts
    assert saved_after < 1.5, f"Saved. came {saved_after:.2f} s after the words, behind the first run"
