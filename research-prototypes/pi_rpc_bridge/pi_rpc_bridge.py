#!/usr/bin/env python3
"""pi_rpc_bridge: drive Pi's JSONL RPC mode from asyncio, one Pi child per space (05c prototype, 2026-10-05).

What it does, all verified against Pi 0.99.1 and a stub LLM (research note 05c):
  * spawns `pi --mode rpc` with the space's cwd, model, thinking level, tools, skills, extensions and
    session directory (RPC has no cwd or env field, so a space is a process);
  * sends commands with ids and matches responses by id (Pi answers asynchronously, so order is not
    a guarantee);
  * turns stdout records into typed events: text and thinking deltas, tool calls as the model starts
    them, tool execution start/update/end, message and agent lifecycle, queue updates, extension UI
    dialogs and notices, extension errors, retries, process exit;
  * prompt / steer / follow_up / abort / clear_queue, plus interrupt() and the fire-and-forget
    request_interrupt(): clear the queue, abort, and tell the model on the next prompt what the user
    actually heard; every event carries the turn it was read in, and turn() quiesces first (waits for the
    aborted run to settle, drains its leftovers), so a stale agent_settled can never end a later turn;
  * answers extension confirm/select/input dialogs through an async callback (the voice layer speaks
    the question and listens), or leaves them to the consumer; a dialog with a timeout auto-resolves
    inside Pi, so a slow answer is harmless;
  * SpacePool keeps one child per space, starts it lazily, resumes its session with --continue.

Pitfalls handled here (see 05c): asyncio's StreamReader splits lines only on b"\\n" but caps a line at
64 KiB by default, while Pi's message_end, tool and get_messages records exceed that, so the limit is
raised; stdout and stderr are drained continuously because Pi blocks when a pipe fills; the child gets
its own process group so a hard kill also takes the bash tool's children.

Standard library only (Python 3.11+).
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

STREAM_LIMIT = 64 * 1024 * 1024
# Inherited variables that would make a child act as some other session: kb's session identity, Claude
# Code's env-file seam, herdr's pane binding. Pi sets its own PI_* variables for its bash tool.
SCRUB_ENV = ("KB_SESSION", "KB_ACTOR", "CLAUDE_ENV_FILE", "HERDR_ENV", "HERDR_SOCKET_PATH", "HERDR_PANE_ID",
             "PI_SESSION_ID", "PI_SESSION_FILE")
# Startup network activity off for every child: catalog refreshes and the pi.dev version check.
QUIET_ENV = {"PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1"}


# --------------------------------------------------------------------------------------------- config

@dataclass
class SpaceConfig:
    """How to start the Pi child for one space (the fields SPACES.md derives from spaces.yaml)."""
    name: str
    root: Path
    model: str = "local/qwen38"
    thinking: str | None = "off"
    tools: list[str] | None = None                 # --tools allowlist; None = Pi's default set
    skills: str | list[str | Path] = "auto"        # "auto" = discovery; a list = --no-skills + --skill each
    extensions: list[str | Path] = field(default_factory=list)   # explicit -e paths (load even with discovery off)
    discover_extensions: bool = False              # False = --no-extensions (only the explicit ones load)
    prompt_templates: bool = False                 # False = --no-prompt-templates
    context_files: bool = True                     # False = --no-context-files (AGENTS.md, CLAUDE.md)
    append_system_prompt: list[str] = field(default_factory=list)
    session_dir: Path | None = None                # None = --no-session
    session_name: str | None = None                # --name
    resume: bool = True                            # --continue when session_dir already holds a session
    approve: bool = True                           # --approve: trust the root's .pi/.agents files
    env: dict[str, str] = field(default_factory=dict)
    pi_bin: str = "pi"
    extra_args: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        a = [self.pi_bin, "--mode", "rpc", "--model", self.model]
        if self.thinking:
            a += ["--thinking", self.thinking]
        if self.tools is not None:
            a += ["--tools", ",".join(self.tools)] if self.tools else ["--no-tools"]
        if self.skills != "auto":
            a += ["--no-skills"]
            for s in self.skills:
                a += ["--skill", str(s)]
        if not self.discover_extensions:
            a += ["--no-extensions"]
        for e in self.extensions:
            a += ["--extension", str(e)]
        if not self.prompt_templates:
            a += ["--no-prompt-templates"]
        if not self.context_files:
            a += ["--no-context-files"]
        for text in self.append_system_prompt:
            a += ["--append-system-prompt", text]
        if self.session_dir is None:
            a += ["--no-session"]
        else:
            a += ["--session-dir", str(self.session_dir)]
            if self.resume and any(Path(self.session_dir).rglob("*.jsonl")):
                a += ["--continue"]
        if self.session_name:
            a += ["--name", self.session_name]
        a += ["--approve" if self.approve else "--no-approve"]
        return a + list(self.extra_args)

    def child_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
        env.update(QUIET_ENV)
        env.update(self.env)
        return env


# --------------------------------------------------------------------------------------------- events

@dataclass
class Event:
    raw: dict
    at: float = field(default_factory=time.monotonic, kw_only=True)
    seq: int = field(default=0, kw_only=True)    # order in which the bridge read it
    turn: int = field(default=0, kw_only=True)   # the PiChild turn number current when it was read


@dataclass
class TextDelta(Event):
    text: str
    index: int


@dataclass
class ThinkingDelta(Event):
    text: str


@dataclass
class ToolCallStarted(Event):
    """The model began emitting a tool call (message_update toolcall_start): the earliest moment to
    acknowledge, before its arguments have streamed (a capture's arguments are the whole transcript)."""
    call_id: str
    name: str


@dataclass
class ToolStart(Event):
    call_id: str
    name: str
    args: dict


@dataclass
class ToolUpdate(Event):
    call_id: str
    name: str
    text: str


@dataclass
class ToolEnd(Event):
    call_id: str
    name: str
    ok: bool
    text: str


@dataclass
class MessageStart(Event):
    role: str


@dataclass
class MessageEnd(Event):
    role: str
    text: str
    stop_reason: str | None
    usage: dict | None
    error: str | None


@dataclass
class TurnEnd(Event):
    pass


@dataclass
class AgentStart(Event):
    pass


@dataclass
class AgentEnd(Event):
    will_retry: bool


@dataclass
class Settled(Event):
    """agent_settled: Pi will not continue on its own. The end of a spoken turn."""


@dataclass
class QueueUpdate(Event):
    steering: list
    follow_up: list


@dataclass
class UIRequest(Event):
    id: str
    method: str


@dataclass
class ConfirmRequest(UIRequest):
    title: str
    message: str
    timeout_ms: int | None


@dataclass
class SelectRequest(UIRequest):
    title: str
    options: list
    timeout_ms: int | None


@dataclass
class InputRequest(UIRequest):
    title: str


@dataclass
class Notify(UIRequest):
    message: str
    level: str


@dataclass
class Status(UIRequest):
    key: str
    text: str | None


@dataclass
class ExtensionError(Event):
    path: str
    event: str
    error: str


@dataclass
class Retry(Event):
    kind: str


@dataclass
class Exited(Event):
    code: int | None


@dataclass
class Other(Event):
    type: str


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def to_event(rec: dict) -> Event:
    t = rec.get("type")
    if t == "message_update":
        e = rec.get("assistantMessageEvent") or {}
        et = e.get("type")
        if et == "text_delta":
            return TextDelta(rec, text=e.get("delta", ""), index=e.get("contentIndex", 0))
        if et == "thinking_delta":
            return ThinkingDelta(rec, text=e.get("delta", ""))
        if et == "toolcall_start":
            return ToolCallStarted(rec, call_id=e.get("id", ""), name=e.get("toolName", ""))
        return Other(rec, type=f"message_update.{et}")
    if t == "message_start":
        return MessageStart(rec, role=(rec.get("message") or {}).get("role", ""))
    if t == "message_end":
        m = rec.get("message") or {}
        return MessageEnd(rec, role=m.get("role", ""), text=_text_of(m.get("content")), stop_reason=m.get("stopReason"),
                          usage=m.get("usage"), error=m.get("errorMessage"))
    if t == "tool_execution_start":
        return ToolStart(rec, call_id=rec.get("toolCallId", ""), name=rec.get("toolName", ""), args=rec.get("args") or {})
    if t == "tool_execution_update":
        return ToolUpdate(rec, call_id=rec.get("toolCallId", ""), name=rec.get("toolName", ""),
                          text=_text_of((rec.get("partialResult") or {}).get("content")))
    if t == "tool_execution_end":
        return ToolEnd(rec, call_id=rec.get("toolCallId", ""), name=rec.get("toolName", ""), ok=not rec.get("isError"),
                       text=_text_of((rec.get("result") or {}).get("content")))
    if t == "turn_end":
        return TurnEnd(rec)
    if t == "agent_start":
        return AgentStart(rec)
    if t == "agent_end":
        return AgentEnd(rec, will_retry=bool(rec.get("willRetry")))
    if t == "agent_settled":
        return Settled(rec)
    if t == "queue_update":
        return QueueUpdate(rec, steering=rec.get("steering") or [], follow_up=rec.get("followUp") or [])
    if t == "extension_ui_request":
        m, rid = rec.get("method", ""), rec.get("id", "")
        if m == "confirm":
            return ConfirmRequest(rec, id=rid, method=m, title=rec.get("title", ""), message=rec.get("message", ""),
                                  timeout_ms=rec.get("timeout"))
        if m == "select":
            return SelectRequest(rec, id=rid, method=m, title=rec.get("title", ""), options=rec.get("options") or [],
                                 timeout_ms=rec.get("timeout"))
        if m in ("input", "editor"):
            return InputRequest(rec, id=rid, method=m, title=rec.get("title", ""))
        if m == "notify":
            return Notify(rec, id=rid, method=m, message=rec.get("message", ""), level=rec.get("notifyType", "info"))
        if m == "setStatus":
            return Status(rec, id=rid, method=m, key=rec.get("statusKey", ""), text=rec.get("statusText"))
        return UIRequest(rec, id=rid, method=m)
    if t == "extension_error":
        return ExtensionError(rec, path=rec.get("extensionPath", ""), event=rec.get("event", ""), error=rec.get("error", ""))
    if t in ("auto_retry_start", "auto_retry_end"):
        return Retry(rec, kind=t)
    return Other(rec, type=str(t))


# --------------------------------------------------------------------------------------------- child

class PiError(Exception):
    pass


class PiCommandError(PiError):
    def __init__(self, command: str, error: str | None):
        super().__init__(f"{command}: {error}")
        self.command, self.error = command, error


class PiExitedError(PiError):
    pass


class PiBusyError(PiError):
    """A run is still open (it did not settle within the settle timeout), so a new turn cannot start."""


UIHandler = Callable[["PiChild", UIRequest], Awaitable[Any]]


@dataclass
class TurnResult:
    """What one spoken turn produced, for tests and measurements."""
    disposition: str
    text: str = ""
    thinking: str = ""
    tools: list[tuple[str, dict]] = field(default_factory=list)
    tool_results: list[tuple[str, bool, str]] = field(default_factory=list)
    stop_reason: str | None = None
    error: str | None = None
    first_text_s: float | None = None     # prompt sent -> first text delta
    first_tool_s: float | None = None     # prompt sent -> first toolcall_start
    total_s: float | None = None          # prompt sent -> agent_settled
    events: list[Event] = field(default_factory=list)


class PiChild:
    """One `pi --mode rpc` process. Single consumer of its event stream.

    Turn hygiene (05c E10, after the 04c review): Pi writes an aborted run's tail (message_end "aborted",
    turn_end, agent_end, agent_settled) before abort's own response, so those records sit in the queue when an
    abort returns, and a fire-and-forget abort may still be in flight when the next turn starts. Therefore:
    every event is stamped with the turn it was read in; turn() first quiesces (waits for any pending
    interrupt, waits until no run is open, drains leftovers into `stray`), then reads only events stamped
    with its own turn number. An event from an earlier turn can never end a later one.
    """

    def __init__(self, cfg: SpaceConfig, *, on_ui: UIHandler | None = None, transcript: Path | None = None,
                 settle_timeout: float = 10.0):
        self.cfg = cfg
        self.on_ui = on_ui
        self.transcript = transcript
        self.settle_timeout = settle_timeout
        self.proc: asyncio.subprocess.Process | None = None
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.stray: deque[Event] = deque(maxlen=500)   # events no turn claimed (drained leftovers, stale tags)
        self.stderr_tail: list[str] = []
        self.startup_s: float | None = None
        self._ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future] = {}
        self._tasks: list[asyncio.Task] = []
        self._t0 = 0.0
        self._wlock = asyncio.Lock()
        self._busy = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._turn = 0                      # the turn number events are stamped with
        self._seq = 0
        self._open_dialogs: set[str] = set()
        self._interrupting: asyncio.Task | None = None
        self._heard: str | None = None   # set by interrupt(): what the user heard before barging in
        self.last_used = time.monotonic()

    # -- process

    async def start(self, timeout: float = 30.0) -> dict:
        root = Path(self.cfg.root).expanduser()
        if not root.is_dir():
            raise PiError(f"space {self.cfg.name}: root {root} is not a directory")
        if self.cfg.session_dir is not None:
            Path(self.cfg.session_dir).mkdir(parents=True, exist_ok=True)
        argv = self.cfg.argv()
        self._t0 = time.monotonic()
        self._record("##", json.dumps({"argv": argv, "cwd": str(root)}))
        self.proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(root), env=self.cfg.child_env(), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=STREAM_LIMIT,
            start_new_session=True)
        self._tasks = [asyncio.create_task(self._read_stdout()), asyncio.create_task(self._read_stderr())]
        state = await self.command("get_state", timeout=timeout)
        self.startup_s = time.monotonic() - self._t0
        return state

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def busy(self) -> bool:
        """True from the moment a turn is sent (or Pi starts a run on its own) until its agent_settled."""
        return self._busy

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        if value:
            self._idle.clear()
        else:
            self._idle.set()

    async def close(self, timeout: float = 10.0) -> int | None:
        """Orderly shutdown: close stdin (Pi disposes its runtime), then escalate."""
        if not self.proc:
            return None
        if self._interrupting and not self._interrupting.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._interrupting), 3)
        if self.proc.stdin and not self.proc.stdin.is_closing():
            self.proc.stdin.close()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout)
        except asyncio.TimeoutError:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, sig)
                try:
                    await asyncio.wait_for(self.proc.wait(), 3)
                    break
                except asyncio.TimeoutError:
                    continue
        for t in self._tasks:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(t, 2)
        return self.proc.returncode

    # -- I/O

    def _record(self, direction: str, line: str) -> None:
        if not self.transcript:
            return
        ms = (time.monotonic() - self._t0) * 1000 if self._t0 else 0.0
        wall = time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(f"{wall} +{ms:8.1f}ms {direction} {line.rstrip()}\n")

    async def _send(self, obj: dict) -> None:
        if not self.alive or not self.proc or not self.proc.stdin:
            raise PiExitedError(f"space {self.cfg.name}: pi is not running")
        line = json.dumps(obj, ensure_ascii=False)
        async with self._wlock:
            self._record(">>", line)
            self.proc.stdin.write(line.encode("utf-8") + b"\n")
            await self.proc.stdin.drain()

    async def _put(self, ev: Event) -> None:
        self._seq += 1
        ev.seq, ev.turn = self._seq, self._turn
        await self.events.put(ev)

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            try:
                line = await self.proc.stdout.readline()
            except ValueError as e:  # a record over STREAM_LIMIT
                await self._put(Other({"type": "bridge_error", "error": str(e)}, type="bridge_error"))
                continue
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
            if not text.strip():
                continue
            self._record("<<", text)
            try:
                rec = json.loads(text)
            except ValueError:
                await self._put(Other({"type": "non_json", "line": text}, type="non_json"))
                continue
            if rec.get("type") == "response":
                fut = self._pending.pop(rec.get("id"), None) if rec.get("id") else None
                if fut and not fut.done():
                    fut.set_result(rec)
                else:  # a parse error has no id
                    await self._put(Other(rec, type="response_unmatched"))
                continue
            ev = to_event(rec)
            if isinstance(ev, AgentStart):
                self._set_busy(True)
            elif isinstance(ev, Settled):
                self._open_dialogs.clear()   # every dialog of a run is resolved once it settles
                self._set_busy(False)
            if isinstance(ev, UIRequest) and ev.method in ("confirm", "select", "input", "editor"):
                self._open_dialogs.add(ev.id)
                if self.on_ui:
                    asyncio.create_task(self._answer_ui(ev))
            await self._put(ev)
        code = await self.proc.wait()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(PiExitedError(f"space {self.cfg.name}: pi exited ({code})"))
        self._pending.clear()
        self._set_busy(False)
        await self._put(Exited({"type": "exited", "code": code}, code=code))

    async def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            self._record("!!", text)
            self.stderr_tail = (self.stderr_tail + [text])[-200:]

    async def _answer_ui(self, ev: UIRequest) -> None:
        try:
            answer = await self.on_ui(self, ev)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001 - a failing handler must not leave Pi waiting
            self._record("##", f"ui handler failed: {e!r}")
            answer = None
        with contextlib.suppress(PiError):
            if answer is None:
                await self.cancel_ui(ev.id)
            elif isinstance(ev, ConfirmRequest):
                await self.answer_confirm(ev.id, bool(answer))
            else:
                self._open_dialogs.discard(ev.id)
                await self._send({"type": "extension_ui_response", "id": ev.id, "value": answer})

    # -- commands

    async def _begin(self, type_: str, **fields: Any) -> tuple[str, asyncio.Future]:
        """Write one command and return (id, future of its response) without waiting for the response."""
        cid = f"b{next(self._ids)}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[cid] = fut
        try:
            await self._send({"id": cid, "type": type_, **fields})
        except BaseException:
            self._pending.pop(cid, None)
            raise
        return cid, fut

    async def command(self, type_: str, timeout: float | None = 30.0, **fields: Any) -> Any:
        """Send one command and wait for its response (timeout None = as long as it takes)."""
        cid, fut = await self._begin(type_, **fields)
        try:
            resp = await (asyncio.wait_for(fut, timeout) if timeout else fut)
        finally:
            self._pending.pop(cid, None)
        self.last_used = time.monotonic()
        if not resp.get("success"):
            raise PiCommandError(type_, resp.get("error"))
        return resp.get("data")

    async def prompt(self, text: str, streaming_behavior: str | None = None, images: list | None = None,
                     timeout: float | None = None) -> str:
        """The raw prompt command. Use turn() for a user turn: it quiesces first and filters stale events.

        Pi answers `prompt` only after every before_agent_start handler has returned (05c): film-rig.ts holds
        there for a whole render, with no response and no agent_start meanwhile. Hence no timeout by default;
        watch the event stream (a Notify says why) instead."""
        if self._heard is not None and streaming_behavior is None:
            text = interrupted_note(self._heard) + text
            self._heard = None
        fields: dict[str, Any] = {"message": text}
        if streaming_behavior:
            fields["streamingBehavior"] = streaming_behavior
        if images:
            fields["images"] = images
        data = await self.command("prompt", timeout=timeout, **fields)
        return (data or {}).get("disposition", "")

    async def steer(self, text: str) -> str:
        return ((await self.command("steer", message=text)) or {}).get("disposition", "")

    async def follow_up(self, text: str) -> str:
        return ((await self.command("follow_up", message=text)) or {}).get("disposition", "")

    async def abort(self, timeout: float | None = 30.0) -> None:
        await self.command("abort", timeout=timeout)

    async def clear_queue(self) -> tuple[list, list]:
        d = await self.command("clear_queue") or {}
        return d.get("steering", []), d.get("followUp", [])

    # -- interruption

    def request_interrupt(self, heard: str | None = None) -> asyncio.Task:
        """Barge-in without waiting: returns at once with the background task doing it. Safe on a
        cancellation path (Pipecat cancels the turn task on an interruption and waits at most 1 s for it).
        The next turn() waits for this task and for the aborted run to settle before it prompts. `heard`
        (what the user actually heard) prefixes that next prompt. Repeated calls share one task."""
        if heard is not None:
            self._heard = heard
        if self._interrupting is None or self._interrupting.done():
            self._interrupting = asyncio.get_running_loop().create_task(self._interrupt_now(), name=f"interrupt-{self.cfg.name}")
        return self._interrupting

    async def _interrupt_now(self) -> tuple[list, list]:
        queued: tuple[list, list] = ([], [])
        if not self.alive:
            return queued
        with contextlib.suppress(PiError):
            queued = await self.clear_queue()   # first, or the abort would let queued messages run (Pi rpc docs)
        if not self._busy:
            return queued
        cid = None
        try:
            cid, fut = await self._begin("abort")
            # The abort is on its way first, so a dialog cancelled now cannot let the run reach another model
            # call. Extensions that pass ctx.signal to ctx.ui.confirm are dismissed by the abort itself; this
            # covers those that don't, which would otherwise hold the abort until their dialog times out.
            for ui_id in list(self._open_dialogs):
                with contextlib.suppress(PiError):
                    await self.cancel_ui(ui_id)
            await asyncio.wait_for(fut, self.settle_timeout)
        except (PiError, asyncio.TimeoutError) as e:
            self._record("##", f"interrupt: {e!r}")
        finally:
            if cid:
                self._pending.pop(cid, None)
        return queued

    async def interrupt(self, heard: str | None = None, timeout: float | None = None) -> tuple[list, list]:
        """Barge-in, awaited: clear queued steer/follow-up text (returned), abort the run, wait until it
        settles, and drain what it left behind. Afterwards `busy` is False and the queue is empty."""
        queued = await asyncio.shield(self.request_interrupt(heard))
        await self.quiesce(timeout)
        return queued

    async def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until no run is open (agent_settled seen, or the child exited). False on timeout."""
        if not self.alive:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), self.settle_timeout if timeout is None else timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def quiesce(self, timeout: float | None = None) -> list[Event]:
        """Make the stream clean for a new turn: wait for a pending interrupt, wait until idle, then drain
        every event left over (they are also kept in `stray`). Raises PiBusyError if a run will not settle."""
        t = self.settle_timeout if timeout is None else timeout
        task = self._interrupting
        if task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(task), t)
            if task.done():
                self._interrupting = None
        if not await self.wait_idle(t):
            raise PiBusyError(f"space {self.cfg.name}: the previous run did not settle within {t:.0f} s")
        return self.drain()

    # -- dialogs and the rest

    async def answer_confirm(self, ui_id: str, confirmed: bool) -> None:
        self._open_dialogs.discard(ui_id)
        await self._send({"type": "extension_ui_response", "id": ui_id, "confirmed": confirmed})

    async def cancel_ui(self, ui_id: str) -> None:
        self._open_dialogs.discard(ui_id)
        await self._send({"type": "extension_ui_response", "id": ui_id, "cancelled": True})

    async def get_state(self) -> dict:
        return await self.command("get_state")

    async def set_model(self, model: str) -> dict:
        provider, _, model_id = model.partition("/")
        return await self.command("set_model", provider=provider, modelId=model_id)

    async def set_thinking(self, level: str) -> None:
        await self.command("set_thinking_level", level=level)

    async def switch_session(self, path: str | Path) -> bool:
        d = await self.command("switch_session", sessionPath=str(path)) or {}
        return not d.get("cancelled", False)

    async def new_session(self) -> bool:
        d = await self.command("new_session") or {}
        return not d.get("cancelled", False)

    async def get_messages(self) -> list:
        return (await self.command("get_messages") or {}).get("messages", [])

    async def get_commands(self) -> list:
        return (await self.command("get_commands") or {}).get("commands", [])

    async def bash(self, command: str, exclude_from_context: bool = False, timeout: float = 120.0) -> dict:
        """Pi's RPC bash: runs in the space's cwd now, output joins the model context on the next prompt."""
        return await self.command("bash", timeout=timeout, command=command, excludeFromContext=exclude_from_context)

    # -- turns

    def drain(self) -> list[Event]:
        """Take whatever events are waiting (leftovers of an aborted run, notices between turns). They are
        also kept in `stray` for the dashboard."""
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        self.stray.extend(out)
        return out

    async def next_event(self, timeout: float | None = None) -> Event:
        return await asyncio.wait_for(self.events.get(), timeout)

    async def turn(self, text: str, timeout: float = 600.0, settle_timeout: float | None = None) -> AsyncIterator[Event]:
        """Send one user turn and yield its events through agent_settled.

        Quiesces first (see the class docstring), so it is safe right after request_interrupt() or
        interrupt(). Only events read during this turn are yielded. If an extension command or input handler
        consumed the prompt (disposition "handled"), nothing runs and nothing is yielded. If the consumer
        stops early (cancellation, break, aclose) before agent_settled, the run is interrupted in the
        background, so it never keeps going unobserved."""
        await self.quiesce(settle_timeout)
        self._turn += 1
        mine = self._turn
        self._set_busy(True)
        done = False
        try:
            disposition = await self.prompt(text)
            if disposition == "handled":
                done = True
                self._set_busy(False)
                return
            deadline = time.monotonic() + timeout
            while True:
                ev = await asyncio.wait_for(self.events.get(), max(0.01, deadline - time.monotonic()))
                if ev.turn != mine:
                    self.stray.append(ev)
                    continue
                yield ev
                if isinstance(ev, (Settled, Exited)):
                    done = True
                    return
        finally:
            if not done and self.alive:
                self.request_interrupt()

    async def run_turn(self, text: str, timeout: float = 600.0) -> TurnResult:
        t0 = time.monotonic()
        r = TurnResult(disposition="started")
        async for ev in self.turn(text, timeout):
            r.events.append(ev)
            if isinstance(ev, TextDelta):
                if r.first_text_s is None:
                    r.first_text_s = ev.at - t0
                r.text += ev.text
            elif isinstance(ev, ThinkingDelta):
                r.thinking += ev.text
            elif isinstance(ev, ToolCallStarted) and r.first_tool_s is None:
                r.first_tool_s = ev.at - t0
            elif isinstance(ev, ToolStart):
                r.tools.append((ev.name, ev.args))
            elif isinstance(ev, ToolEnd):
                r.tool_results.append((ev.name, ev.ok, ev.text))
            elif isinstance(ev, MessageEnd) and ev.role == "assistant":
                r.stop_reason, r.error = ev.stop_reason, ev.error
            elif isinstance(ev, Settled):
                r.total_s = ev.at - t0
        if not r.events:
            r.disposition = "handled"
        return r


def heredoc_command(head: str, text: str, marker: str = "ATLAS_END") -> str:
    """`head <<'MARK'` + text + `MARK`, with a quoted marker that does not occur as a line of the text, so the
    shell passes the text through untouched (Atlas: "Quote the heredoc marker so the shell can't change
    anything"). The voice gate accepts exactly this shape: the marker line once, at the end."""
    import secrets
    lines = set(text.split("\n"))
    while marker in lines:
        marker = f"ATLAS_END_{secrets.token_hex(3)}"
    return f"{head} <<'{marker}'\n{text}\n{marker}"


def atlas_capture_command(transcript: str, via: str = "voice", to: str | None = None) -> str:
    """The Atlas verbatim-capture command for one utterance (run from the Atlas root). atlas.py strips only
    trailing newlines from stdin and verifies the bytes it appended."""
    import shlex
    head = f"python3 atlas.py capture --via {shlex.quote(via)}" + (f" --to {shlex.quote(to)}" if to else "")
    return heredoc_command(head, transcript)


def interrupted_note(heard: str) -> str:
    heard = heard.strip()
    if not heard:
        return "(You were interrupted before the user heard any of your last reply.)\n\n"
    return f"(You were interrupted. The user heard only this much of your last reply: \"{heard}\")\n\n"


# --------------------------------------------------------------------------------------------- pool

class SpacePool:
    """One lazily started Pi child per space; the active one receives the user's turns."""

    def __init__(self, spaces: dict[str, SpaceConfig], *, on_ui: UIHandler | None = None,
                 transcript_dir: Path | None = None, idle_exit_s: float = 1800.0):
        self.spaces = spaces
        self.on_ui = on_ui
        self.transcript_dir = transcript_dir
        self.idle_exit_s = idle_exit_s
        self.children: dict[str, PiChild] = {}
        self.active: str | None = None
        self._locks: dict[str, asyncio.Lock] = {name: asyncio.Lock() for name in spaces}

    async def get(self, name: str) -> PiChild:
        if name not in self.spaces:
            raise KeyError(f"no space {name!r}; known: {', '.join(self.spaces)}")
        async with self._locks[name]:
            child = self.children.get(name)
            if child is None or not child.alive:
                tr = self.transcript_dir / f"{name}.log" if self.transcript_dir else None
                child = PiChild(self.spaces[name], on_ui=self.on_ui, transcript=tr)
                await child.start()
                self.children[name] = child
            return child

    async def switch(self, name: str) -> PiChild:
        child = await self.get(name)
        self.active = name
        return child

    async def reap_idle(self) -> list[str]:
        """Close children idle for idle_exit_s; their sessions resume with --continue next time."""
        now, closed = time.monotonic(), []
        for name, child in list(self.children.items()):
            if name != self.active and not child.busy and now - child.last_used > self.idle_exit_s:
                await child.close()
                del self.children[name]
                closed.append(name)
        return closed

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in self.children.values()), return_exceptions=True)
        self.children.clear()
