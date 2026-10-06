"""The router and the model's space_switch (SPACES.md "Switching"), model-free.

A rambling request after a request phrase ("I want you to go to my journal. And uh. ...") used not to
switch (2026-10-05), because the router took a switching verb only at the very start of the words; the home agent browsed the
journal folder and said writing needs act mode. The router now takes request phrases and fillers before the verb and
more verbs, and a trigger followed by a noun ("my atlas folder") is not a switch; tests/fixtures/router_phrasings.yaml
holds the phrasings and the planted negatives (tools/router_bench.py prints the same numbers). What it does not take, the model can still switch
with its space_switch tool (voice_mode.ts): the orchestrator then switches and sends the person's own words on."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from conftest import EXT
from harness import rig
from local_voice.config import load_config
from local_voice.router import route
from local_voice.spaces import load_spaces

PHRASINGS = yaml.safe_load((Path(__file__).parent / "fixtures/router_phrasings.yaml").read_text())
SPACES = load_spaces(load_config(), check_paths=False)


@pytest.mark.parametrize("case", PHRASINGS["switches"], ids=lambda c: c["say"][:40])
def test_a_switch_said_naturally_is_taken(case):
    r = route(case["say"], SPACES)
    if case["want"] == "ask":
        assert r.kind == "ask", r
    else:
        assert (r.kind, r.space, r.rest) == ("switch", case["want"], case.get("rest", "")), r


@pytest.mark.parametrize("case", PHRASINGS["negatives"], ids=lambda c: c["say"][:40])
def test_words_that_are_not_a_switch_are_left_to_the_model(case):
    assert route(case["say"], SPACES).kind == "none"


async def test_the_space_switch_tool_reports_and_ends_the_run(make_child, stub):
    spaces = [{"name": "notes", "description": "the notes folder"}, {"name": "home", "description": "your Mac"}]
    child = await make_child(tools=["read", "space_switch"], extensions=[EXT / "voice_mode.ts"],
                             env={"VOICE_SPACES": json.dumps(spaces), "VOICE_SPACE": "home", "VOICE_TOOLS": "read,space_switch"})
    tools = None
    stub.script([{"tool_calls": [{"name": "space_switch", "arguments": {"space": "notes", "request": "list them"}}]},
                 {"text": "UNEXPECTED"}])
    n0 = len(stub.requests())
    r = await child.run_turn("can we do this where my notes live")
    tools = [t["function"]["name"] for t in stub.requests()[-1]["body"]["tools"]]
    assert tools == ["read", "space_switch"]                  # only the other spaces' tool, active in conversation mode
    desc = [t for t in stub.requests()[-1]["body"]["tools"] if t["function"]["name"] == "space_switch"][0]["function"]["description"]
    assert "notes (the notes folder)" in desc and "home (your Mac)" not in desc
    assert r.tool_results[0][:2] == ("space_switch", True) and "Switching to the notes folder" in r.tool_results[0][2]
    assert len(stub.requests()) - n0 == 1                     # the run ended with the tool: no words after it
    stub.script([{"tool_calls": [{"name": "space_switch", "arguments": {"space": "garage"}}]}, {"text": "There is none."}])
    r = await child.run_turn("go to the garage")
    assert r.tool_results[0][1] is False and 'no space "garage"' in r.tool_results[0][2]


def _two_spaces(tmp_path) -> dict:
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    (notes / "todo.md").write_text("buy bread\n")
    spec = {"home": {"root": str(tmp_path / "work"), "description": "the scratch folder", "triggers": ["home"],
                     "skills": [], "tools": ["read", "ls"], "tier": "ask"},
            "notes": {"root": str(notes), "description": "the notes folder", "triggers": ["my notes"],
                      "skills": [], "tools": ["read", "ls"], "tier": "readonly"}}
    path = tmp_path / "spaces-two.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "defaults": {"model": "local/qwen38", "thinking": "off",
                                                                "mode": "conversation", "tier": "ask", "voice": "ryan"},
                                    "spaces": spec}))
    return {"agent": {"spaces_file": str(path)}}


async def _turn(c, text: str) -> dict:
    t0 = c.now()
    await c.send({"t": "text", "text": text})
    return await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=20, since=t0)


@pytest.mark.parametrize("request_there,runs_there", [("list the files", True), ("", False)])
async def test_the_models_switch_moves_the_conversation_and_the_persons_words_go_on(tmp_path, request_there, runs_there):
    words = "Can we do this where my notes live instead, and list what's there?"
    async with rig(tmp_path, overrides=_two_spaces(tmp_path)) as r:
        r.stub.script([{"tool_calls": [{"name": "space_switch", "arguments": {"space": "notes", "request": request_there}}],
                        "delay_ms": 5},
                       {"text": "There is a todo file.", "delay_ms": 5}])
        c = await r.client()
        t0 = c.now()
        await _turn(c, words)
        sp = await c.wait_for(lambda m: m.get("t") == "space" and m.get("name") == "notes", timeout=10, since=t0)
        assert sp["description"] == "the notes folder" and r.orch.hub.active == "notes"
        if runs_there:
            await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=20, since=c.now())
        await c.close()
        texts = r.user_texts()
    assert r.tts.spoken[-2 if runs_there else -1:] == (["Switching to the notes folder.", "There is a todo file."]
                                                        if runs_there else ["Switching to the notes folder."])
    assert "One moment." not in r.tts.spoken                     # no acknowledgement for the switch itself
    # the person's own words, not the model's "list the files", are the notes space's first prompt
    assert texts == ([words, words] if runs_there else [words])
