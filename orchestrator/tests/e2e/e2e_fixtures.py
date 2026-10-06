"""Fixtures for the M1 end-to-end tests (imported by test_turn.py; not a conftest.py, whose module name would
shadow tests/conftest.py for the prototype suite's `from conftest import ...`): real models (Nemotron, Qwen3-TTS 1.7B, Silero, Smart Turn, qwen38 through the hold gate),
the real protocol v1 server, `say` speech streamed at real-time pace by the protocol v1 test client.

GPU rule: nothing here runs unless measure/bench/gpu_clear.sh exits 0 right before; between tests
its gpu_jobs and hold fields are re-checked (its LLM-idle field cannot be, since these tests are LLM traffic). The kb
is a fresh clone (kb.ts writes a session digest), searched with ripgrep so the shared qmd index is not updated.
Nothing writes to the knowledge base, ~/.pi/agent, the journal, llama-swap's config or launchd. Results and logs go to
<repo>/state/orchestrator-e2e/<stamp>/ (gitignored).

Run:  .venv/bin/python -m pytest -m e2e tests/e2e -s
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

# every model is cached beforehand; the e2e run must never download one
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import pytest_asyncio

ORCH = Path(__file__).resolve().parents[2]
REPO = ORCH.parent
GPU_CLEAR = REPO / "measure/bench/gpu_clear.sh"
STAMP = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = REPO / "state" / "orchestrator-e2e" / STAMP


def gpu_line() -> tuple[int, str]:
    r = subprocess.run([str(GPU_CLEAR)], capture_output=True, text=True, timeout=30)
    return r.returncode, (r.stdout or r.stderr).strip()


def gpu_still_ours() -> str | None:
    """None while no render or hold has appeared since the run started; otherwise why to stop."""
    _, line = gpu_line()
    jobs = re.search(r"gpu_jobs=(\d+)", line)
    hold = re.search(r"hold=(\w+)", line)
    if jobs and jobs.group(1) != "0":
        return f"a GPU job started: {line}"
    if hold and hold.group(1) not in ("open", "absent"):
        return f"the hold gate is {hold.group(1)}: {line}"
    return None


@pytest.fixture(scope="session")
def run_dir() -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR


@pytest.fixture(scope="session")
def kb_clone(run_dir) -> Path:
    """Imported into each test module, so pytest runs it once per module even at session scope: the first module's
    clone is reused (a second `git clone` into it failed, 2026-10-05 16:53)."""
    dst = run_dir / "kb"
    if not (dst / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(Path.home() / "kb"), str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "remote", "set-url", "--push", "origin", "DISABLED-voice-e2e-clone"], check=True)
    assert dst.resolve() != (Path.home() / "kb").resolve()
    return dst


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def orch(run_dir, kb_clone):
    async for o in _serve(run_dir, kb_clone):
        yield o


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def orch_spaces(request, run_dir, kb_clone):
    """M3: the real spaces.yaml with every root moved off real data the way `./run.sh --scratch` does it
    (scratch.scratch_spaces): Atlas and this repo cloned under <run dir>/m3/spaces, the kb space on the kb clone, the
    home folder an empty <run dir>/m3/home. The permission question times out after 10 s instead of 20. Each test
    module gets its own state dir, so its Pi children start a fresh conversation instead of resuming another's."""
    from local_voice.scratch import scratch_spaces

    spaces = scratch_spaces(ORCH / "spaces.yaml", run_dir / "m3", kb_clone)
    name = request.module.__name__.rsplit(".", 1)[-1]
    async for o in _serve(run_dir, kb_clone, {"state_dir": str(run_dir / f"state-{name}"),
                                              "agent": {"spaces_file": str(spaces)}, "pi": {"confirm_timeout_ms": 10000}}):
        o.e2e_scratch = run_dir / "m3"
        yield o


_SESSION = {"cleared": False}    # gpu_clear.sh passed at this session's first server start


async def _serve(run_dir, kb_clone, extra: dict | None = None):
    code, line = gpu_line()
    (run_dir / "gpu.log").open("a").write(f"{time.strftime('%H:%M:%S')} start: {line}\n")
    # gpu_clear.sh reads BUSY for 120 s after any LLM call, ours included: a later module of the same session only
    # needs no other GPU job and an open gate (its test_reply_timing was skipped at 16:58:42 for our own traffic)
    if code != 0 and not (_SESSION["cleared"] and gpu_still_ours() is None):
        pytest.skip(f"GPU not clear, nothing run: {line}")
    _SESSION["cleared"] = True
    from harness import free_port
    from local_voice.config import deep_merge, load_config
    from local_voice.server import Orchestrator, setup_logging

    port = free_port()
    cfg = load_config(overrides=deep_merge({
        "state_dir": str(run_dir / "state"),
        "server": {"port": port, "hosts": ["127.0.0.1"], "allowed_logins": [], "browser": {"enabled": False}},
        "pi": {"env": {"KB_HOME": str(kb_clone), "KB_SEARCH_BACKEND": "rg", "UV_OFFLINE": "1"}},
        # synthetic speech is not a real conversation: its turn log and digests stay in the run dir
        "brain": {"turn_log_dir": str(run_dir / "turns"), "state_dir": str(run_dir / "brain")},
    }, extra or {}))
    setup_logging(cfg.state_dir, "INFO")
    o = Orchestrator(cfg)
    t0 = time.monotonic()
    timings = await o.start()
    timings["start_total_s"] = round(time.monotonic() - t0, 3)
    (cfg.state_dir / "startup.json").write_text(json.dumps(timings, indent=1))
    import asyncio

    task = asyncio.create_task(o.serve())
    for _ in range(100):
        await asyncio.sleep(0.05)
        try:
            _r, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            break
        except OSError:
            continue
    o.e2e_url = f"ws://127.0.0.1:{port}/v1/voice"
    o.e2e_status = f"http://127.0.0.1:{port}/v1/status"
    o.e2e_timings = timings
    try:
        yield o
    finally:
        await o.stop()
        task.cancel()
