"""Fixtures for the protocol v1 end-to-end tests: a mock server per test, and the lvclient binary.

Every run keeps its evidence (both JSONL logs, the client's recorded output) under MockServer/runs/<stamp>/<test>/,
which git ignores.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

APPLE = pathlib.Path(__file__).resolve().parents[2]
KIT = APPLE / "LocalVoiceKit"
LVCLIENT = KIT / ".build" / "debug" / "lvclient"
MOCK = APPLE / "MockServer" / "mock_server.py"
STAMP = time.strftime("%Y%m%d-%H%M%S")
RUNS = APPLE / "MockServer" / "runs" / STAMP


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Mock:
    def __init__(self, run_dir: pathlib.Path, flags: list[str], name: str = "server"):
        self.log = run_dir / f"{name}.jsonl"
        self.proc = subprocess.Popen(
            [sys.executable, str(MOCK), "--port", "0", "--log", str(self.log), *flags],
            stdout=subprocess.PIPE, stderr=open(run_dir / f"{name}.err", "w"), text=True)
        line = self.proc.stdout.readline()
        self.port = json.loads(line)["listening"]
        self.url = f"ws://127.0.0.1:{self.port}/v1/voice"

    def records(self) -> list[dict]:
        return read_jsonl(self.log)

    def events(self, name: str) -> list[dict]:
        return [r for r in self.records() if r["event"] == name]

    def received(self, t: str | None = None) -> list[dict]:
        msgs = [r["msg"] for r in self.records() if r["event"] == "in"]
        return [m for m in msgs if t is None or m.get("t") == t]

    def sent(self, t: str | None = None) -> list[dict]:
        msgs = [r["msg"] for r in self.records() if r["event"] == "out"]
        return [m for m in msgs if t is None or m.get("t") == t]

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class ClientRun:
    def __init__(self, proc: subprocess.CompletedProcess | None, log: pathlib.Path, wav: pathlib.Path):
        self.proc, self.log, self.wav = proc, log, wav

    @property
    def records(self) -> list[dict]:
        return read_jsonl(self.log)

    @property
    def summary(self) -> dict:
        s = [r for r in self.records if r["event"] == "summary"]
        assert s, f"no summary in {self.log}"
        return s[-1]

    def recv(self, t: str | None = None) -> list[dict]:
        return [r["msg"] for r in self.records if r["event"] == "recv" and (t is None or r["msg"]["t"] == t)]

    def sent(self, t: str | None = None) -> list[dict]:
        return [r["msg"] for r in self.records if r["event"] == "sent" and (t is None or r["msg"]["t"] == t)]

    def events(self, name: str) -> list[dict]:
        return [r for r in self.records if r["event"] == name]


@pytest.fixture(scope="session")
def lvclient() -> pathlib.Path:
    jobs = ["-j", os.environ["LV_JOBS"]] if os.environ.get("LV_JOBS") else []  # ../build.sh's job cap
    build = subprocess.run(["swift", "build", *jobs, "--product", "lvclient"], cwd=KIT, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    return LVCLIENT


@pytest.fixture
def run_dir(request) -> pathlib.Path:
    d = RUNS / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def mock(run_dir):
    started: list[Mock] = []

    def start(*flags: str) -> Mock:
        m = Mock(run_dir, list(flags), name="server" if not started else f"server{len(started) + 1}")
        started.append(m)
        return m

    yield start
    for m in started:
        m.stop()


@pytest.fixture
def client(lvclient, run_dir):
    """Runs lvclient to completion (or starts it in the background with wait=False)."""

    def run(url: str, script: str, *, mic: str = "ptt", device: str = "e2e", name: str = "client",
            audio: str = "headless", extra: list[str] | None = None, timeout: float = 60, wait: bool = True):
        log, wav = run_dir / f"{name}.jsonl", run_dir / f"{name}.wav"
        argv = [str(lvclient), "--url", url, "--device", device, "--mic", mic, "--audio", audio,
                "--script", script, "--log", str(log), "--timeout", str(timeout), *(extra or [])]
        if audio == "headless":
            argv += ["--record-output", str(wav)]
        if not wait:
            return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), \
                ClientRun(None, log, wav)
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 30)
        (run_dir / f"{name}.out").write_text(proc.stdout + proc.stderr)
        return ClientRun(proc, log, wav)

    return run
