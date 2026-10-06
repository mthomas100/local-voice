"""Conversation mode's budget for looking things up (voice_gate.ts, config agent.tool_budget), the progress notice and
the no-answer line (agent.py): whole spoken turns against the stub LLM. Act mode's exemption is tested with the mode
switch (test_spaces_m3.py)."""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from harness import rig
from local_voice.client import tone_pcm

LS = {"tool_calls": [{"name": "ls", "arguments": {"path": "."}}], "delay_ms": 300}


async def _one_turn(r) -> None:
    c = await r.client()
    await c.speak(tone_pcm(0.8))
    tail = asyncio.create_task(c.silence(10.0))
    await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=25)
    tail.cancel()
    await c.close()


def _tool_results(req: dict) -> list[str]:
    out = []
    for m in req["body"]["messages"]:
        if m.get("role") == "tool":
            c = m.get("content")
            out.append(c if isinstance(c, str) else json.dumps(c))
    return out


async def test_over_budget_the_model_is_told_to_answer_with_what_it_has(tmp_path):
    ov = {"agent": {"tool_budget": {"calls": 2, "seconds": 0}, "progress": {"after_s": 0.5, "text": "Still looking."}}}
    async with rig(tmp_path, stt_text="how much does one cost", overrides=ov) as r:
        r.stub.script([LS, LS, LS, {"text": "I don't have a price to hand; want me to look further?", "delay_ms": 5}])
        await _one_turn(r)
        reqs = r.stub.requests()
    assert len(reqs) == 4
    results = _tool_results(reqs[3])
    assert len(results) == 3
    assert "Stop looking" not in results[0] and "Stop looking" not in results[1]
    assert "Stop looking: you have spent 2 tool calls" in results[2]
    spoken = r.tts.spoken[1:]
    assert spoken[0] == "Let me look." and "Still looking." in spoken
    assert spoken[-1] == "I don't have a price to hand; want me to look further?"


async def test_a_model_that_keeps_looking_is_stopped_and_the_person_hears_so(tmp_path):
    ov = {"agent": {"tool_budget": {"calls": 1, "seconds": 0}, "progress": {"after_s": 0}}}
    async with rig(tmp_path, stt_text="how much does one cost", overrides=ov) as r:
        r.stub.script([LS] * 8, default=LS)
        await _one_turn(r)
        reqs = r.stub.requests()
    # one call allowed, then three refusals; the third ends the run, so no fifth request is made
    assert len(reqs) == 4
    assert r.tts.spoken[1:] == ["Let me look.", r.orch.cfg.no_answer_text]


def test_the_budget_rule():
    script = """
import { overBudget } from "./pi/voice_gate.ts";
const out = [overBudget(1, 1, 2, 20), overBudget(2, 1, 2, 20), overBudget(0, 25, 2, 20), overBudget(9, 99, 0, 0)];
console.log(JSON.stringify(out));
"""
    r = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                       capture_output=True, text=True, cwd=str(Path(__file__).parents[1]))
    assert r.returncode == 0, r.stderr
    a, b, c, d = json.loads(r.stdout)
    assert a is None and d is None
    assert b.startswith("Stop looking: you have spent 2 tool calls") and c.startswith("Stop looking: you have spent 25 seconds")
