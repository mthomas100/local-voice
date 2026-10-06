"""voice_gate.ts and the kb tool's verbs: read-only verbs run, write verbs ask, a verb kb does not have is refused at
once with a pointer (qwen38 called `kb read <page>` in the 2026-10-05 e2e run and waited 20 s on a spoken question), and
so is a `kb new` kb would refuse (an early live test was asked about `kb new Issue ...`; Issue is not a type)."""
from __future__ import annotations

import pytest

from conftest import EXT, HERE
from local_voice.pi_rpc import SelectRequest

pytestmark = pytest.mark.needs_pi
FAKE_KB = HERE / "tests/fixtures/fake_kb_tool.ts"


def answer(value, log):
    async def on_ui(child, ev):
        log.append(ev)
        return value
    return on_ui


@pytest.mark.parametrize("tier,args,ran,asked,says", [
    ("ask", ["search", "hold gate"], True, False, "ran kb search hold gate"),
    ("ask", ["trace", "/wiki/x.md"], True, False, "ran kb trace"),
    ("ask", ["read", "/wiki/x.md"], False, False, 'kb has no "read" verb'),
    ("ask", ["new", "Concept", "x"], False, True, "did not approve"),
    ("ask", ["new", "Issue", "x"], False, False, "not \"Issue\""),
    ("readonly", ["new", "Concept", "x"], False, False, "this space is read-only"),
    ("readonly", ["search", "x"], True, False, "ran kb search x"),
])
async def test_kb_verbs(make_child, stub, tmp_path, tier, args, ran, asked, says):
    seen = []
    child = await make_child(tools=["read", "kb"], extensions=[FAKE_KB, EXT / "voice_gate.ts"], on_ui=answer("deny", seen),
                             env={"VOICE_TIER": tier, "VOICE_CONFIRM_TIMEOUT_MS": "3000", "KB_HOME": str(tmp_path / "kb")})
    stub.script([{"tool_calls": [{"name": "kb", "arguments": {"args": args}}]}, {"text": "ok"}])
    r = await child.run_turn("kb please")
    assert r.tool_results[0][1] is ran
    assert any(isinstance(e, SelectRequest) for e in seen) is asked
    assert says in r.tool_results[0][2]
