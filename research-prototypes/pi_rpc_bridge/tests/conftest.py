"""Fixtures: a stub LLM on a free port, a throwaway PI_CODING_AGENT_DIR pointed at it, a scratch working dir, and
a factory for Pi RPC children that are always closed. Nothing here touches 127.0.0.1:8090, ~/.pi/agent or ~/kb."""
from __future__ import annotations

import json
import shutil
import socket
import sys
from pathlib import Path

import pytest
import pytest_asyncio

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from pi_rpc_bridge import PiChild, SpaceConfig  # noqa: E402
from stub_llm import StubLLM  # noqa: E402

EXT = HERE / "extensions"
PI = shutil.which("pi")

# The rig's `local` provider as ~/.pi/agent/models.json declared it on 2026-10-05, pointed at the stub instead.
COMPAT = {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": True,
          "supportsUsageInStreaming": True, "maxTokensField": "max_tokens", "supportsStrictMode": False,
          "thinkingFormat": "deepseek", "requiresReasoningContentOnAssistantMessages": True}
MODELS = [
    {"id": "qwen38", "name": "Qwen3.8 Flash Next (stub)", "reasoning": True, "input": ["text", "image"],
     "contextWindow": 245760, "maxTokens": 16384, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
     "thinkingLevelMap": {"off": "none", "minimal": "low", "low": "low", "medium": "medium", "high": "high",
                          "xhigh": "xhigh", "max": "xhigh"}},
    {"id": "qwen27-262k", "name": "Qwen3.8-27B 262K (stub)", "reasoning": True, "input": ["text", "image"],
     "contextWindow": 245760, "maxTokens": 16384, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}},
]


def pytest_collection_modifyitems(config, items):
    if PI is None:
        skip = pytest.mark.skip(reason="pi is not on PATH")
        for item in items:
            if "needs_pi" in item.keywords:
                item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_pi: starts a real pi --mode rpc child")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_agent_dir(d: Path, base_url: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    assert ":8090" not in base_url and ":8091" not in base_url
    (d / "models.json").write_text(json.dumps({"providers": {"local": {
        "baseUrl": base_url, "api": "openai-completions", "apiKey": "local-rig", "compat": COMPAT, "models": MODELS}}}))
    (d / "settings.json").write_text(json.dumps({"defaultProvider": "local", "defaultModel": "qwen38", "packages": []}))
    return d


@pytest.fixture
def stub(tmp_path):
    s = StubLLM(0, tmp_path / "stub").start()
    yield s
    s.stop()


@pytest.fixture
def agent_dir(tmp_path, stub):
    return write_agent_dir(tmp_path / "pi-agent", stub.base_url)


@pytest.fixture
def workdir(tmp_path):
    w = tmp_path / "work"
    (w / "sub").mkdir(parents=True)
    (w / "README.md").write_text("Scratch README.\nLine two.\nLine three.\n")
    (w / "sub" / "a.txt").write_text("alpha\n")
    (w / "sub" / "b.txt").write_text("beta\n")
    return w


@pytest_asyncio.fixture
async def make_child(tmp_path, agent_dir, workdir):
    children: list[PiChild] = []

    async def _make(*, on_ui=None, env=None, name="t", root=None, **cfg):
        cfg.setdefault("tools", ["read", "ls"])
        config = SpaceConfig(name=name, root=root or workdir, env={"PI_CODING_AGENT_DIR": str(agent_dir), **(env or {})}, **cfg)
        child = PiChild(config, on_ui=on_ui, transcript=tmp_path / f"{name}-{len(children)}.log")
        children.append(child)
        await child.start()
        return child

    yield _make
    for c in children:
        await c.close()
