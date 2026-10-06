"""M1 DoD 5 against the real `hold` binary run as a test gate on a scratch port (~/repos/local-rig/hold): while it is
held the session plays the busy notice and makes no model call, of any kind; after `hold off` the kept turn runs.
The gate fronts the stub LLM, so Pi's hold.ts waits on it too. Nothing touches the live gate on :8090."""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from harness import free_port, rig
from local_voice.client import tone_pcm
from stub_llm import StubLLM

pytestmark = pytest.mark.needs_pi
HOLD_BIN = Path.home() / "repos/local-rig/hold/hold"


@pytest.mark.skipif(not HOLD_BIN.exists(), reason="needs ~/repos/local-rig/hold/hold")
async def test_a_held_test_gate_gets_the_notice_and_no_model_call(tmp_path):
    stub = StubLLM(0, tmp_path / "stub").start()
    port = free_port()
    assert port not in (8090, 8091)
    gate_url = f"http://127.0.0.1:{port}"
    gate = subprocess.Popen([str(HOLD_BIN), "gate", "--listen", f"127.0.0.1:{port}", "--upstream", stub.base_url[:-3],
                             "--state", str(tmp_path / "hold-state.json"), "--no-implicit", "--drain-timeout", "30s"],
                            stdout=open(tmp_path / "gate.log", "w"), stderr=subprocess.STDOUT)

    def hold(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(HOLD_BIN), *args], env={**os.environ, "HOLD_GATE": gate_url}, capture_output=True,
                              text=True, timeout=30)

    try:
        await asyncio.sleep(0.8)
        async with rig(tmp_path, stt_text="are you there", gate_url=gate_url, llm_url=f"{gate_url}/v1", stub=stub) as r:
            r.stub.script([{"text": "Back with you now."}])
            assert hold("on", "voice M1 DoD 5 test").returncode == 0
            await asyncio.sleep(0.4)
            assert r.orch.hold.phase == "held"
            c = await r.client()
            assert c.welcome["state"] == "held"
            calls = r.orch.runtime.worker.calls
            await c.speak(tone_pcm(0.8))
            quiet = asyncio.create_task(c.silence(12.0))
            await c.wait_for(lambda m: m.get("t") == "state" and m.get("v") == "held", timeout=10)
            notice = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
            assert c.replies[notice["reply_id"]].bytes > 0
            await asyncio.sleep(1.0)
            assert r.stub.requests() == [] and r.orch.runtime.worker.calls == calls and r.stt.calls == []
            opened = c.now()
            assert hold("off").returncode == 0
            await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=20, since=opened)
            quiet.cancel()
            assert r.user_texts() == ["are you there"] and r.tts.spoken[-1] == "Back with you now."
            await c.close()
    finally:
        gate.terminate()
        gate.wait(5)
        stub.stop()
