"""The idle gates the job must pass before it may think (M4 is idle-gated; the GPU rule: no model call while the GPU is held or busy).

The order is cheapest first. gpu_clear.sh is the law for every LLM call and cannot be bypassed: `--force` skips only
the quiet, user-idle and orchestrator checks.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from . import turnlog


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str

    def __str__(self) -> str:
        return f"{self.name}: {'ok' if self.ok else 'BLOCKED'} ({self.detail})"


def gpu_clear(script: Path, timeout_s: float = 30) -> Gate:
    """Run measure/bench/gpu_clear.sh; exit 0 is CLEAR. Missing or failing to run is BUSY, never a pass."""
    if not script.is_file():
        return Gate("gpu_clear", False, f"{script} is missing, so the GPU cannot be shown to be clear")
    try:
        r = subprocess.run([str(script)], capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        return Gate("gpu_clear", False, f"could not run {script.name}: {e}")
    line = (r.stdout.strip().splitlines() or [r.stderr.strip() or f"exit {r.returncode}"])[-1]
    return Gate("gpu_clear", r.returncode == 0, line)


def wait_gpu_clear(script: Path, wait_s: float, poll_s: float, *, sleep: Callable[[float], None] = time.sleep,
                   log: Callable[[str], None] = lambda s: None) -> Gate:
    """Poll gpu_clear until it passes or `wait_s` runs out. Used between our own passes only: right after our own
    request gpu_clear says BUSY for 120 s because it watches llama-swap's last POST, and it cannot tell whose."""
    deadline = time.monotonic() + max(0.0, wait_s)
    g = gpu_clear(script)
    while not g.ok and time.monotonic() < deadline:
        log(f"waiting for the GPU: {g.detail}")
        sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
        g = gpu_clear(script)
    return g


def quiet(turns_dir: Path, now: datetime, minutes: float) -> Gate:
    last = turnlog.last_activity(turns_dir)
    if last is None:
        return Gate("quiet", True, "no turns logged")
    ago = now - last
    ok = ago >= timedelta(minutes=minutes)
    return Gate("quiet", ok, f"last voice turn {int(ago.total_seconds() // 60)} min ago (need {minutes:g})")


_HID_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')


def hid_idle_seconds(ioreg_output: str | None = None) -> float | None:
    """Seconds since the last keyboard, mouse or trackpad event, from `ioreg -c IOHIDSystem` (nanoseconds)."""
    if ioreg_output is None:
        try:
            ioreg_output = subprocess.run(["/usr/sbin/ioreg", "-c", "IOHIDSystem", "-d", "4"], capture_output=True,
                                          text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
    m = _HID_RE.search(ioreg_output or "")
    return int(m.group(1)) / 1e9 if m else None


def user_idle(minutes: float, ioreg_output: str | None = None) -> Gate:
    if minutes <= 0:
        return Gate("user_idle", True, "check off")
    s = hid_idle_seconds(ioreg_output)
    if s is None:
        return Gate("user_idle", False, "HID idle time unreadable")
    return Gate("user_idle", s >= minutes * 60, f"no keyboard or mouse for {int(s // 60)} min (need {minutes:g})")


def orchestrator(url: str, busy_states: list[str], timeout_s: float = 2.0) -> Gate:
    """GET /v1/status. Not running is fine; a turn under way (thinking, speaking) is not."""
    if not url:
        return Gate("orchestrator", True, "check off")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:  # noqa: S310 (loopback URL from config)
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return Gate("orchestrator", True, "not running")
    except ValueError:
        return Gate("orchestrator", False, "status is not JSON")
    state = str(data.get("state", "")) if isinstance(data, dict) else ""
    if state in busy_states:
        return Gate("orchestrator", False, f"state {state}")
    return Gate("orchestrator", True, f"state {state or 'unknown'}")
