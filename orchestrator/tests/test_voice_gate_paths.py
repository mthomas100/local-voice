"""voice_gate.ts speed rules (M1 e2e, 2026-10-05: one tool turn took 2 min 10 s). A path as kb prints it
("/wiki/decisions/x.md") reads the page instead of failing with ENOENT; find or grep over the whole home folder or
the disk is refused with a pointer (one such find took 105 s)."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import EXT

pytestmark = pytest.mark.needs_pi


def tool_text(stub) -> str:
    return str([m for m in stub.requests()[-1]["body"]["messages"] if m["role"] == "tool"][-1]["content"])


@pytest.fixture
def kb(tmp_path) -> Path:
    page = tmp_path / "kb/wiki/decisions/gate.md"
    page.parent.mkdir(parents=True)
    page.write_text("The hold gate pauses model calls while a render runs.\n")
    return tmp_path / "kb"


async def gated(make_child, kb, tools):
    return await make_child(tools=tools, extensions=[EXT / "voice_gate.ts"], env={"KB_HOME": str(kb)})


async def test_a_path_as_kb_prints_it_reads_the_page(make_child, stub, kb):
    child = await gated(make_child, kb, ["read"])
    stub.script([{"tool_calls": [{"name": "read", "arguments": {"path": "/wiki/decisions/gate.md"}}]}, {"text": "ok"}])
    r = await child.run_turn("what does the kb say about the gate")
    assert r.tool_results[0][1] is True
    assert "pauses model calls while a render runs" in tool_text(stub)


async def test_real_and_missing_paths_are_left_alone(make_child, stub, kb, tmp_path):
    real = tmp_path / "notes.txt"
    real.write_text("a real file\n")
    child = await gated(make_child, kb, ["read"])
    stub.script([{"tool_calls": [{"name": "read", "arguments": {"path": str(real)}}]}, {"text": "ok"},
                 {"tool_calls": [{"name": "read", "arguments": {"path": "/wiki/decisions/missing.md"}}]}, {"text": "ok"}])
    await child.run_turn("read my notes")
    assert "a real file" in tool_text(stub)
    r = await child.run_turn("read a page that is not there")
    assert r.tool_results[0][1] is False and "ENOENT" in tool_text(stub)


@pytest.mark.parametrize("tool,where,says", [
    ("find", "~", "whole home folder"),
    ("find", str(Path.home()), "whole home folder"),
    ("grep", "/", "whole disk"),
])
async def test_find_or_grep_over_home_or_the_disk_is_refused(make_child, stub, kb, tool, where, says):
    child = await gated(make_child, kb, ["find", "grep"])
    stub.script([{"tool_calls": [{"name": tool, "arguments": {"pattern": "gate.md", "path": where}}]}, {"text": "ok"}])
    r = await child.run_turn("where is it")
    assert r.tool_results[0][1] is False
    assert says in tool_text(stub) and "takes minutes" in tool_text(stub)


async def test_find_in_a_folder_runs(make_child, stub, kb):
    child = await gated(make_child, kb, ["find"])
    stub.script([{"tool_calls": [{"name": "find", "arguments": {"pattern": "*.md", "path": str(kb)}}]}, {"text": "ok"}])
    r = await child.run_turn("find the pages")
    assert r.tool_results[0][1] is True and "gate.md" in tool_text(stub)
