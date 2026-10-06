"""The orchestrator's view of the Mac's GPU lock (the hold gate of the companion local-rig repo, docs/hold.md).

The gate knows about model calls through :8090, not about this process's own MLX speech models ("alive is not
busy"), so the orchestrator checks it itself: a background poll keeps `phase` fresh for the MLX thread's
guard, `check()` asks the gate right before every turn, and `wait_open()` blocks on the gate's own long poll until
model calls may pass again.

Phases from `GET /hold/status`: `open` (model calls pass), `draining` (a hold is next; nothing new may start),
`held` (a hold is granted, or a GPU job runs outside any hold). `absent` means the gate did not answer: like
gpu_clear.sh, that counts as allowed (no gate installed), and it is reported on /v1/status.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
from loguru import logger

ALLOWED = ("open", "absent")


@dataclass
class HoldState:
    phase: str = "unknown"
    why: str = ""
    holder: str = ""              # kind and reason of the first hold, for the spoken and shown notice
    checked_at: float = field(default_factory=time.monotonic)

    @property
    def allows_gpu(self) -> bool:
        return self.phase in ALLOWED

    def as_json(self) -> dict:
        return {"phase": self.phase, "why": self.why or self.holder}


def parse_status(data: dict) -> HoldState:
    holds = data.get("holds") or []
    first = holds[0] if holds else {}
    holder = " ".join(x for x in (str(first.get("kind") or ""), str(first.get("reason") or "")) if x).strip()
    if not holder and data.get("detected"):
        d = data["detected"][0]
        holder = str(d.get("name") or d.get("cmd") or "a GPU job")
    return HoldState(phase=str(data.get("phase") or "unknown"), why=str(data.get("why") or ""), holder=holder)


Listener = Callable[[HoldState, HoldState], Awaitable[None] | None]


class HoldMonitor:
    def __init__(self, gate: str, *, poll_s: float = 1.0, timeout_s: float = 2.0, who: str = "local-voice"):
        self.gate = gate.rstrip("/")
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.who = who
        self.state = HoldState()
        self._listeners: list[Listener] = []
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    # -- the guard the MLX thread calls (any thread; reads one attribute)

    def gpu_refusal(self) -> str | None:
        st = self.state
        if st.allows_gpu:
            return None
        return f"hold gate is {st.phase}" + (f" ({st.holder or st.why})" if (st.holder or st.why) else "")

    @property
    def phase(self) -> str:
        return self.state.phase

    def subscribe(self, fn: Listener) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    # -- HTTP

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def check(self) -> HoldState:
        """Ask the gate now (before every turn). Updates `state` and tells listeners about a change."""
        try:
            r = await self._http().get(f"{self.gate}/hold/status")
            new = parse_status(r.json()) if r.status_code == 200 else HoldState(phase="absent", why=f"HTTP {r.status_code}")
        except (httpx.HTTPError, ValueError) as e:
            new = HoldState(phase="absent", why=f"gate unreachable: {type(e).__name__}")
        await self._set(new)
        return new

    async def wait_open(self, timeout_s: float | None = None) -> HoldState:
        """Block until model calls may pass (the gate's /hold/wait-open long poll, 60 s per request)."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            st = await self.check()
            if st.allows_gpu:
                return st
            left = 60.0 if deadline is None else min(60.0, deadline - time.monotonic())
            if left <= 0:
                return st
            try:
                await self._http().get(f"{self.gate}/hold/wait-open", params={"timeout": f"{left:.1f}", "who": self.who},
                                       timeout=left + 5)
            except httpx.HTTPError:
                await asyncio.sleep(min(self.poll_s, max(left, 0.05)))

    async def _set(self, new: HoldState) -> None:
        old, self.state = self.state, new
        if (old.phase, old.holder) != (new.phase, new.holder):
            if new.phase != "open":
                logger.info(f"hold gate: {old.phase} -> {new.phase} {new.holder or new.why}".rstrip())
            elif old.phase not in ("unknown",):
                logger.info(f"hold gate: {old.phase} -> open")
            for fn in list(self._listeners):
                try:
                    res = fn(old, new)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:  # noqa: BLE001 - one bad listener must not stop the monitor
                    logger.warning(f"hold listener failed: {e!r}")

    # -- background poll

    async def start(self) -> HoldState:
        st = await self.check()
        if self._task is None:
            self._task = asyncio.create_task(self._poll(), name="hold-poll")
        return st

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self.poll_s)
            await self.check()

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None
