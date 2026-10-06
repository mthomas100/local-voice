"""Fixtures for the brain's tests: the stub LLM from the due-diligence prototypes, a throwaway Pi agent dir pointed at
it, git clones of ~/kb and of Atlas in temporary directories (no remote, so nothing can be pushed), a fake
gpu_clear.sh, and a Config whose every path is inside the test's tmp dir.

Nothing here reaches 127.0.0.1:8090, ~/.pi/agent, the real knowledge base or the real journal: the clones are read
from, never written back. The real kb and Atlas repos are only `git clone`d; set KB_HOME and ATLAS_REPO to point at
them (tests that need one skip without it).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BRAIN = Path(__file__).resolve().parent.parent
REPO = BRAIN.parent
STUB_DIR = REPO / "research-prototypes" / "pi_rpc_bridge"
sys.path.insert(0, str(BRAIN))
sys.path.insert(0, str(STUB_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_voice_brain import config as cfgmod  # noqa: E402
from stub_llm import StubLLM  # noqa: E402

REAL_KB = Path(os.environ.get("KB_HOME", Path.home() / "kb"))
REAL_ATLAS = Path(os.environ.get("ATLAS_REPO", Path.home() / "atlas"))
PI = shutil.which("pi")

# The rig's `local` provider as ~/.pi/agent/models.json declared it on 2026-10-05 (read, not copied: the compat
# block is restated here), pointed at the stub. The qwen27 row's own compat is what the fallback test varies.
PROVIDER_COMPAT = {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": True,
                   "supportsUsageInStreaming": True, "maxTokensField": "max_tokens", "supportsStrictMode": False,
                   "thinkingFormat": "deepseek", "requiresReasoningContentOnAssistantMessages": True}


def _row(mid: str, compat: dict | None = None) -> dict:
    row = {"id": mid, "name": f"{mid} (stub)", "reasoning": True, "input": ["text"], "contextWindow": 245760,
           "maxTokens": 16384, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}
    if compat:
        row["compat"] = compat
    return row


def write_agent_dir(d: Path, base_url: str, *, qwen27_compat: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    assert ":8090" not in base_url and ":8091" not in base_url
    models = [_row("qwen38"), _row("qwen27-262k", qwen27_compat)]
    (d / "models.json").write_text(json.dumps({"providers": {"local": {
        "baseUrl": base_url, "api": "openai-completions", "apiKey": "local-rig", "compat": PROVIDER_COMPAT,
        "models": models}}}))
    (d / "settings.json").write_text(json.dumps({"defaultProvider": "local", "defaultModel": "qwen38", "packages": []}))
    return d


def pytest_collection_modifyitems(config, items):
    reasons = {"needs_pi": (PI is None, "pi is not on PATH"),
               "needs_kb": (not (REAL_KB / "bin" / "kb").exists(), "no kb repo to clone (set KB_HOME)"),
               "needs_atlas": (not (REAL_ATLAS / "atlas.py").exists(), "no Atlas repo to clone (set ATLAS_REPO)")}
    for item in items:
        for mark, (skip, why) in reasons.items():
            if skip and mark in item.keywords:
                item.add_marker(pytest.mark.skip(reason=why))


@pytest.fixture
def stub(tmp_path):
    s = StubLLM(0, tmp_path / "stub").start()
    yield s
    s.stop()


@pytest.fixture
def agent_dir(tmp_path, stub):
    return write_agent_dir(tmp_path / "pi-agent", stub.base_url)


def git_clone(src: Path, dest: Path) -> Path:
    subprocess.run(["git", "clone", "-q", str(src), str(dest)], check=True)
    # No remote at all: a push from a test cannot reach the real repo.
    subprocess.run(["git", "-C", str(dest), "remote", "remove", "origin"], check=True)
    return dest


@pytest.fixture
def kb_clone(tmp_path):
    return git_clone(REAL_KB, tmp_path / "kb")


@pytest.fixture
def atlas_clone(tmp_path):
    return git_clone(REAL_ATLAS, tmp_path / "atlas")


def fake_gpu_clear(path: Path, *, ok: bool = True) -> Path:
    """A stand-in for measure/bench/gpu_clear.sh that records each call; flip it with set_gpu()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flag = path.with_suffix(".state")
    flag.write_text("CLEAR" if ok else "BUSY")
    path.write_text(f"""#!/bin/zsh
st=$(cat {flag})
echo call >> {path.with_suffix('.calls')}
echo "$(date +%H:%M:%S) gpu_jobs=0 hold=open llm_last_post_ago=999s → $st"
[ "$st" = CLEAR ]
""")
    path.chmod(0o755)
    return path


def set_gpu(path: Path, ok: bool) -> None:
    path.with_suffix(".state").write_text("CLEAR" if ok else "BUSY")


def gpu_calls(path: Path) -> int:
    p = path.with_suffix(".calls")
    return len(p.read_text().splitlines()) if p.exists() else 0


def make_config(tmp_path: Path, *, agent_dir: Path | None = None, kb: Path | None = None, atlas: Path | None = None,
                gpu_ok: bool = True, **over) -> cfgmod.Config:
    """The shipped config.toml with every path moved into tmp_path and the gates made deterministic."""
    import tomllib
    raw = tomllib.loads(cfgmod.DEFAULT_CONFIG.read_text())
    gpu = fake_gpu_clear(tmp_path / "bin" / "gpu_clear.sh", ok=gpu_ok)
    raw["paths"].update(turns_dir=str(tmp_path / "state" / "turns"), state_dir=str(tmp_path / "state" / "brain"))
    raw["gates"].update(gpu_clear=str(gpu), user_idle_minutes=0, orchestrator_status="http://127.0.0.1:9/v1/status",
                        gpu_wait_s=0, gpu_poll_s=0)
    raw["llm"].update(pi_bin=PI or "/nonexistent/pi", agent_dir=str(agent_dir or tmp_path / "no-agent"),
                      session_dir=str(tmp_path / "state" / "brain" / "pi-sessions"), timeout_s=60)
    raw["kb"].update(home=str(kb or tmp_path / "no-kb"), enabled=kb is not None)
    raw["atlas"].update(root=str(atlas or tmp_path / "no-atlas"), enabled=atlas is not None)
    for dotted, value in over.items():
        section, key = dotted.split("__", 1)
        raw[section][key] = value
    assert "/kb" not in raw["kb"]["home"] or str(tmp_path) in raw["kb"]["home"]
    return cfgmod.from_dict(raw, cfgmod.DEFAULT_CONFIG)
