"""The turn log the background brain reads (brain/TURN_LOG.md §1, interface v1, written by the orchestrator).

One JSON line per user turn, appended when the turn is over (the agent settled, or the reply was interrupted or
cancelled), to <dir>/<local day the turn started>.jsonl: open, append one line, close; never rewritten. The person's
words are verbatim: `user_text` is exactly what the recogniser produced or what was typed, never a note the
orchestrator adds to Pi's prompt (the heard-text note, the "already saved" note, a resume instruction).

An interrupted turn waits a moment for the client's `played_ms`, which says what was actually heard; the record is
written when it arrives, or after `heard_wait_s`.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def now_local() -> datetime:
    """Local time with its UTC offset (TURN_LOG.md: an offset-less time is refused)."""
    return datetime.now().astimezone()


def iso(t: datetime) -> str:
    return t.isoformat(timespec="milliseconds")


@dataclass
class TurnEntry:
    session: str
    turn: int
    t_start: datetime
    client: str
    space: str
    mode: str
    input: str                                   # voice | text
    user_text: str
    reply_text: str = ""
    t_end: datetime | None = None
    heard_text: str | None = None
    interrupted: bool = False
    tools: list[dict[str, Any]] = field(default_factory=list)
    atlas: dict[str, Any] | None = None
    tone: dict[str, Any] | None = None
    reply_id: str | None = None                  # the protocol v1 reply this turn's speech went out in
    settled: bool = False                        # the agent's run is over; the reply may still be playing
    written: bool = False
    fallback: Any = None                         # the task that writes the record if no reply ever closes

    def tool_started(self, name: str) -> None:
        self.tools.append({"name": name, "ok": False})

    def tool_ended(self, name: str, ok: bool) -> None:
        for t in self.tools:
            if t["name"] == name and not t.get("_done"):
                t["ok"], t["_done"] = bool(ok), True
                return
        self.tools.append({"name": name, "ok": bool(ok), "_done": True})

    def record(self) -> dict[str, Any]:
        return {"v": 1, "type": "turn", "session": self.session, "turn": self.turn,
                "t_start": iso(self.t_start), "t_end": iso(self.t_end) if self.t_end else None,
                "client": self.client, "space": self.space, "mode": self.mode, "input": self.input,
                "user_text": self.user_text, "reply_text": self.reply_text,
                "heard_text": self.heard_text if self.interrupted else None, "interrupted": self.interrupted,
                "tools": [{"name": t["name"], "ok": t["ok"]} for t in self.tools],
                "atlas": self.atlas, "tone": self.tone}


class TurnLog:
    """Process-wide writer; numbers the turns of each session (a resumed session keeps counting)."""

    def __init__(self, directory: Path | None, *, heard_wait_s: float = 1.5, settle_fallback_s: float = 120.0):
        self.dir = directory
        self.heard_wait_s = heard_wait_s
        self.settle_fallback_s = settle_fallback_s
        self._counters: dict[str, int] = {}
        self.written: list[dict[str, Any]] = []      # this process's records, for tests and the status page

    def begin(self, *, session: str, t_start: datetime, client: str, space: str, mode: str, input: str,
              user_text: str) -> TurnEntry:
        n = self._counters.get(session, 0) + 1
        self._counters[session] = n
        return TurnEntry(session=session, turn=n, t_start=t_start, client=client, space=space, mode=mode,
                         input=input, user_text=user_text)

    def write(self, entry: TurnEntry) -> None:
        if entry.written:
            return
        entry.written = True
        if entry.fallback is not None and not entry.fallback.done() and entry.fallback is not asyncio.current_task():
            entry.fallback.cancel()
        if entry.t_end is None:
            entry.t_end = now_local()
        rec = entry.record()
        self.written.append(rec)
        if self.dir is None:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / f"{entry.t_start.strftime('%Y-%m-%d')}.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:   # a full disk must not end the conversation
            logger.error(f"turn log: could not append to {self.dir}: {e}")

    def write_after_heard(self, entry: TurnEntry) -> asyncio.Task:
        """An interrupted turn: write once `played_ms` refined heard_text, or after heard_wait_s."""
        async def later():
            await asyncio.sleep(self.heard_wait_s)
            self.write(entry)
        return asyncio.get_running_loop().create_task(later(), name=f"turnlog-{entry.session}-{entry.turn}")
