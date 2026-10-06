"""The voice gate, mode switching, sessions, the space pool, the hold gate and the Atlas flow."""
import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from conftest import EXT, free_port, write_agent_dir
from local_voice.pi_rpc import (Notify, SelectRequest, PiChild, Settled, SpaceConfig, SpacePool, Status, TextDelta, ToolEnd,
                           atlas_capture_command, heredoc_command)

pytestmark = pytest.mark.needs_pi
GATE = EXT / "voice_gate.ts"


def answer(value, delay=0.05, log=None):
    """An on_ui handler; True / False answer the gate's approval (a select) as allow_once / deny."""
    async def on_ui(child, ev):
        if log is not None:
            log.append(ev)
        await asyncio.sleep(delay)
        if isinstance(ev, SelectRequest) and isinstance(value, bool):
            return "allow_once" if value else "deny"
        return value
    return on_ui


@pytest.mark.parametrize("value,delay,ran", [(True, 0.05, True), (False, 0.05, False), (None, 0.05, False), (True, 2.5, False)],
                         ids=["approve", "deny", "cancel", "answer-after-timeout"])
async def test_ask_tier_confirms_by_voice(make_child, stub, value, delay, ran):
    seen = []
    child = await make_child(tools=["read", "bash"], extensions=[GATE], on_ui=answer(value, delay, seen),
                             env={"VOICE_TIER": "ask", "VOICE_CONFIRM_TIMEOUT_MS": "1500"})
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "echo gated"}}]}, {"text": "ok"}])
    r = await child.run_turn("run echo")
    assert len(seen) == 1 and isinstance(seen[0], SelectRequest) and seen[0].timeout_ms == 1500
    assert r.tool_results[0][1] is ran
    if not ran:
        assert "did not approve" in r.tool_results[0][2]


async def test_readonly_tier_refuses_without_asking(make_child, stub):
    seen = []
    child = await make_child(tools=["read", "bash"], extensions=[GATE], on_ui=answer(True, 0, seen), env={"VOICE_TIER": "readonly"})
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "echo nope"}}]}, {"text": "ok"}])
    r = await child.run_turn("run")
    assert not seen and r.tool_results == [("bash", False, "this space is read-only: no shell commands")]


async def test_trusted_tier_allowlist_cannot_be_stretched(make_child, stub, workdir):
    seen = []
    (workdir / "keep.txt").write_text("keep")
    child = await make_child(tools=["read", "bash"], extensions=[GATE], on_ui=answer(False, 0, seen),
                             env={"VOICE_TIER": "trusted", "VOICE_BASH_ALLOW": json.dumps(["echo *", "tee out.txt"])})
    cases = [("echo allowed", True, False),
             ("echo allowed; rm keep.txt", False, True),
             ("echo $(rm keep.txt)", False, True),
             (heredoc_command("tee out.txt", "words with $HOME and `ticks`"), True, False),
             ("tee out.txt <<'ATLAS_END'\nx\nATLAS_END\nrm keep.txt\nATLAS_END", False, True),
             ("tee out.txt <<ATLAS_END\nx $(rm keep.txt)\nATLAS_END", False, True)]
    for command, ran, asked in cases:
        n = len(seen)
        stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]}, {"text": "ok"}])
        r = await child.run_turn("run")
        assert (r.tool_results[0][1], len(seen) > n) == (ran, asked), command
    assert (workdir / "keep.txt").exists()
    assert (workdir / "out.txt").read_text() == "words with $HOME and `ticks`\n"


async def test_trusted_tier_accepts_a_cd_into_the_root_and_nothing_wider(make_child, stub, workdir):
    """qwen38 wrapped every Atlas capture in `cd <root> && python3 atlas.py capture ...` (05d); that must run
    without asking, and no other use of cd or chaining may ride along."""
    seen = []
    (workdir / "keep.txt").write_text("keep")
    child = await make_child(tools=["read", "bash"], extensions=[GATE], on_ui=answer(False, 0, seen),
                             env={"VOICE_TIER": "trusted", "VOICE_BASH_ALLOW": json.dumps(["echo *", "tee out.txt"])})
    root = str(workdir)
    cases = [(f"cd {root} && echo allowed", True, False),
             (f'cd "{root}" && echo allowed', True, False),
             (f"cd '{root}' && echo allowed", True, False),
             ("cd sub && echo allowed", True, False),
             ("cd . && " + heredoc_command("tee out.txt", "words with $HOME"), True, False),
             ('cd "$(pwd)" && echo here', True, False),
             ("cd $PWD && echo here", True, False),
             ('cd "$(pwd)/.." && echo up', False, True),
             ('cd "$(rm keep.txt)" && echo sneaky', False, True),
             ("cd /tmp && echo outside", False, True),
             ("cd .. && echo outside", False, True),
             ('cd "$HOME" && echo expanded', False, True),
             (f"cd {root} && echo x; rm keep.txt", False, True),
             (f"cd {root} && cd /tmp && echo twice", False, True),
             (f"cd {root} || echo either", False, True)]
    for command, ran, asked in cases:
        n = len(seen)
        stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": command}}]}, {"text": "ok"}])
        r = await child.run_turn("run")
        assert (r.tool_results[0][1], len(seen) > n) == (ran, asked), command
    assert (workdir / "keep.txt").exists()
    assert (workdir / "out.txt").read_text() == "words with $HOME\n"


async def test_ui_handler_failure_cancels_the_dialog(make_child, stub):
    async def broken(child, ev):
        raise RuntimeError("speech engine crashed")
    child = await make_child(tools=["bash"], extensions=[GATE], on_ui=broken, env={"VOICE_TIER": "ask", "VOICE_CONFIRM_TIMEOUT_MS": "20000"})
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "echo x"}}]}, {"text": "ok"}])
    t = time.monotonic()
    r = await child.run_turn("run")
    assert r.tool_results[0][1] is False and time.monotonic() - t < 5   # cancelled at once, not after 20 s


async def test_voice_mode_switch_changes_the_tool_set(make_child, stub):
    child = await make_child(tools=["read", "ls", "bash"], extensions=[EXT / "voice_mode.ts"], env={"VOICE_TOOLS": "read,ls"})
    stub.script([{"text": "a"}, {"text": "b"}, {"text": "c"}])
    await child.run_turn("one")
    tools = lambda: [t["function"]["name"] for t in stub.requests()[-1]["body"]["tools"]]
    assert tools() == ["read", "ls"]
    assert await child.prompt("/voice-mode act") == "handled"
    await asyncio.sleep(0.2)
    assert any(isinstance(e, Status) and e.key == "voice-mode" and e.text.startswith("act") for e in child.drain())
    await child.run_turn("two")
    assert tools() == ["read", "ls", "bash"]
    await child.prompt("/voice-mode conversation")
    await child.run_turn("three")
    assert tools() == ["read", "ls"]


async def test_session_dir_resume_and_switch(tmp_path, agent_dir, workdir, stub):
    sdir = tmp_path / "sessions"
    cfg = lambda **kw: SpaceConfig(name="home", root=workdir, tools=["read"], session_dir=sdir, session_name="voice-home",
                                   env={"PI_CODING_AGENT_DIR": str(agent_dir)}, **kw)
    a = PiChild(cfg())
    await a.start()
    stub.script([{"text": "noted"}])
    await a.run_turn("remember ostrich")
    file_a = (await a.get_state())["sessionFile"]
    await a.close()
    b = PiChild(cfg())                       # --continue: same file, same context
    st = await b.start()
    assert st["sessionFile"] == file_a and st["sessionName"] == "voice-home" and st["messageCount"] == 3
    stub.script([{"text": "ostrich"}])
    await b.run_turn("which word?")
    assert any("remember ostrich" in json.dumps(m) for m in stub.requests()[-1]["body"]["messages"])
    await b.close()
    c = PiChild(cfg(resume=False))           # a new session, then switch into the old one
    st = await c.start()
    assert st["messageCount"] == 0
    assert await c.switch_session(file_a)
    assert (await c.get_state())["messageCount"] == 5
    await c.close()


async def test_pool_starts_lazily_and_switches(tmp_path, agent_dir, workdir, stub):
    other = tmp_path / "other"
    other.mkdir()
    env = {"PI_CODING_AGENT_DIR": str(agent_dir)}
    pool = SpacePool({"home": SpaceConfig(name="home", root=workdir, tools=["read"], env=env),
                      "other": SpaceConfig(name="other", root=other, tools=["ls"], env=env)}, transcript_dir=tmp_path)
    try:
        assert pool.children == {}
        home = await pool.switch("home")
        assert pool.active == "home" and list(pool.children) == ["home"]
        o = await pool.get("other")
        assert o is not home and (await o.get_state())["model"]["id"] == "qwen38"
        stub.script([{"text": "from other"}])
        r = await o.run_turn("hi")
        assert r.text == "from other"
        assert "<cwd>\n" + str(other) in stub.requests()[-1]["body"]["messages"][0]["content"]
    finally:
        await pool.close()


HOLD_BIN = Path.home() / "repos/local-rig/hold/hold"
HOLD_TS = Path.home() / "repos/local-rig/extensions/hold.ts"


@pytest.mark.skipif(not (HOLD_BIN.exists() and HOLD_TS.exists()), reason="needs ~/repos/local-rig hold binary and hold.ts")
async def test_hold_gate_makes_the_turn_wait(tmp_path, stub, workdir):
    port = free_port()
    gate_url = f"http://127.0.0.1:{port}"
    assert port not in (8090, 8091)
    gate = subprocess.Popen([str(HOLD_BIN), "gate", "--listen", f"127.0.0.1:{port}", "--upstream", stub.base_url[:-3],
                             "--state", str(tmp_path / "state.json"), "--no-implicit", "--drain-timeout", "30s"],
                            stdout=open(tmp_path / "gate.log", "w"), stderr=subprocess.STDOUT)
    hold = lambda *a: subprocess.run([str(HOLD_BIN), *a], env={**os.environ, "HOLD_GATE": gate_url}, capture_output=True, text=True, timeout=30)
    child = None
    try:
        await asyncio.sleep(0.8)
        ag = write_agent_dir(tmp_path / "pi-agent-gated", f"{gate_url}/v1")
        child = PiChild(SpaceConfig(name="gated", root=workdir, tools=["read"], extensions=[HOLD_TS], session_name="voice-test",
                                    env={"PI_CODING_AGENT_DIR": str(ag), "HOLD_GATE": gate_url}))
        await child.start()
        assert hold("on", "bridge test").returncode == 0
        stub.script([{"text": "after the hold"}])
        n0 = len(stub.requests())
        await child.prompt("are you there")
        evs = []
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            try:
                evs.append(await asyncio.wait_for(child.events.get(), t_end - time.monotonic()))
            except asyncio.TimeoutError:
                break
        assert any(isinstance(e, Notify) and "held" in e.message for e in evs)
        assert any(isinstance(e, Status) and e.key == "hold" and e.text for e in evs)
        assert len(stub.requests()) == n0
        assert hold("off").returncode == 0
        while not any(isinstance(e, Settled) for e in evs):
            evs.append(await asyncio.wait_for(child.events.get(), 30))
        assert "".join(e.text for e in evs if isinstance(e, TextDelta)) == "after the hold"
    finally:
        if child:
            await child.close()
        gate.terminate()
        gate.wait(5)


ATLAS_REPO = os.environ.get("ATLAS_REPO")


@pytest.mark.skipif(not ATLAS_REPO, reason="set ATLAS_REPO to an Atlas git repo (it is cloned; nothing is written to it)")
async def test_atlas_muse_is_captured_verbatim_before_the_reply(tmp_path, agent_dir, stub):
    clone = tmp_path / "atlas"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", ATLAS_REPO, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "remote", "set-url", "--push", "origin", "DISABLED"], check=True)
    assert clone.resolve() != Path(ATLAS_REPO).resolve() and (clone / "atlas.py").exists()
    transcript = "um so the the garden was quiet… like $5 & `odd` — 'really' quiet"
    child = PiChild(SpaceConfig(name="atlas", root=clone, tools=["read", "grep", "find", "ls", "bash"], extensions=[GATE],
                                env={"PI_CODING_AGENT_DIR": str(agent_dir), "VOICE_TIER": "trusted",
                                     "VOICE_BASH_ALLOW": json.dumps(["python3 atlas.py *"])}))
    try:
        await child.start()
        skills = {c["name"] for c in await child.get_commands() if c["source"] == "skill"}
        assert {"skill:atlas-journal", "skill:atlas-process", "skill:atlas-review", "skill:atlas-file"} <= skills
        stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": atlas_capture_command(transcript)}}]},
                     {"text": "Saved."}])
        r = await child.run_turn(transcript)
        assert "AGENTS.md" in stub.requests()[0]["body"]["messages"][0]["content"]
        end = next(e.at for e in r.events if isinstance(e, ToolEnd))
        first = next(e.at for e in r.events if isinstance(e, TextDelta))
        assert r.tool_results[0][1] and end < first
        note = next((clone / "journal").glob(f"{time.strftime('%Y-%m-%d')}.md")).read_text()
        assert note.endswith(f" · said to voice\n\n{transcript}\n")
        assert subprocess.run(["python3", "atlas.py", "start"], cwd=clone, capture_output=True).returncode == 0
        fin = subprocess.run(["python3", "atlas.py", "finish", "--via", "voice"], cwd=clone, capture_output=True, text=True)
        assert fin.returncode == 0, fin.stdout + fin.stderr
    finally:
        await child.close()
