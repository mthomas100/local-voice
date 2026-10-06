"""A whole orchestrator for the model-free integration tests: real Pipecat pipelines, a real Pi child per space and the
real protocol v1 server on a scratch port, with fake speech adapters (no MLX), an energy VAD and a timer for the turn
(no ONNX), the stub LLM instead of the rig, and a fake hold gate. Nothing reaches :8090, ~/.pi/agent (only its
models.json is read, to derive the throwaway agent dir), ~/kb or Atlas."""
from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fake_gate import FakeGate
from local_voice.client import V1Client
from local_voice.config import load_config
from local_voice.server import Orchestrator
from local_voice.testing import model_free_turn
from stub_llm import StubLLM


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Rig:
    orch: Orchestrator
    stub: StubLLM
    gate: FakeGate
    url: str
    status_url: str
    workdir: Path
    serve_task: asyncio.Task

    @property
    def tts(self):
        return self.orch.runtime.tts_engine

    @property
    def stt(self):
        return self.orch.runtime.stt_engine

    def user_texts(self) -> list[str]:
        """The user message content of every request the stub LLM received, in order."""
        out = []
        for r in self.stub.requests():
            msgs = r["body"]["messages"]
            last = msgs[-1]
            c = last.get("content")
            out.append(c if isinstance(c, str) else "".join(p.get("text", "") for p in c or []))
        return out

    async def client(self, **kw: Any) -> V1Client:
        c = V1Client(self.url, **kw)
        await c.connect()
        return c


def write_spaces(path: Path, root: Path, **home: Any) -> Path:
    spec = {"version": 1,
            "defaults": {"model": "local/qwen38", "thinking": "off", "mode": "conversation", "tier": "ask", "voice": "ryan"},
            "spaces": {"home": {"root": str(root), "description": "the scratch folder", "triggers": ["home"],
                                "skills": [], "tools": ["read", "ls"], "act_tools": ["bash"], "tier": "ask", **home}}}
    path.write_text(yaml.safe_dump(spec))
    return path


@contextlib.asynccontextmanager
async def rig(tmp_path: Path, *, stt: str = "fake", stt_script: list[str] | None = None, stt_text: str = "hello there",
              overrides: dict | None = None, spaces: dict | None = None,
              speech_timeout: float = 0.4, tts_rtf: float = 0.0, gate_url: str | None = None,
              llm_url: str | None = None, stub: StubLLM | None = None, turn_factory=None) -> AsyncIterator[Rig]:
    """gate_url: a real test gate instead of the fake one; llm_url: where Pi's provider points (default the stub);
    turn_factory: mic -> TurnSetup, instead of the energy VAD and a speech timeout."""
    stub = stub or StubLLM(0, tmp_path / "stub").start()
    gate = FakeGate().start()
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    (work / "README.md").write_text("Scratch README.\nLine two.\n")
    spaces_file = write_spaces(tmp_path / "spaces.yaml", work, **(spaces or {}))
    port = free_port()
    stt_adapter = {"fake": {"impl": "local_voice.engines.fakes:FakeTranscriber", "kind": "segmented",
                            "text": stt_text, "script": stt_script or []},
                   "fake_stream": {"impl": "local_voice.engines.fakes:FakeStreamingTranscriber", "kind": "streaming",
                                   "text": stt_text, "script": stt_script or [], "words_per_s": 4}}
    ov = {"state_dir": str(tmp_path / "state"),
          "server": {"port": port, "hosts": ["127.0.0.1"], "allowed_logins": [], "browser": {"enabled": False}},
          "hold": {"gate": gate_url or gate.url, "poll_s": 0.1},
          "stt": {"adapter": stt, "adapters": stt_adapter},
          "tts": {"adapter": "fake", "adapters": {"fake": {"impl": "local_voice.engines.fakes:FakeSynthesizer",
                                                           "seconds_per_char": 0.03, "lead_s": 0.08,
                                                           "rtf": tts_rtf}}},
          "agent": {"spaces_file": str(spaces_file)},
          # the brain's interface under the test's own dir: never the real turn log or digests
          "brain": {"turn_log_dir": str(tmp_path / "turns"), "state_dir": str(tmp_path / "brain"),
                    "heard_wait_s": 0.8, "digest_idle_s": 0.5}}
    cfg = load_config(overrides=_merge(ov, overrides or {}))
    for p in (cfg.turn_log_dir, cfg.brain_state_dir, cfg.state_dir):
        assert p is None or tmp_path in p.parents, f"a test would write outside its tmp dir: {p}"
    orch = Orchestrator(cfg, use_mlx=False,
                        turn_factory=turn_factory or (lambda mic: model_free_turn(mic, speech_timeout=speech_timeout)),
                        base_url=llm_url or stub.base_url, allowed_logins=set())
    await orch.start()
    task = asyncio.create_task(orch.serve())
    for _ in range(100):
        await asyncio.sleep(0.05)
        try:
            _r, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            break
        except OSError:
            continue
    r = Rig(orch=orch, stub=stub, gate=gate, url=f"ws://127.0.0.1:{port}/v1/voice",
            status_url=f"http://127.0.0.1:{port}/v1/status", workdir=work, serve_task=task)
    try:
        yield r
    finally:
        await orch.stop()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, 5)
        stub.stop()
        gate.stop()


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
