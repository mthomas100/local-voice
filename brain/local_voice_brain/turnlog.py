"""Read the orchestrator's turn log (brain/TURN_LOG.md §1): one JSONL file per local day, one record per user turn.

Malformed lines and records missing a required field are skipped and reported, never fatal: one bad line must not
block a day's digest.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REQUIRED = ("session", "turn", "t_start", "space", "user_text", "reply_text")


@dataclass(frozen=True)
class AtlasCapture:
    root: str
    path: str
    by: str
    text: str | None          # the captured bytes when they differ from user_text


@dataclass(frozen=True)
class Turn:
    session: str
    turn: int
    t_start: datetime
    t_end: datetime | None
    space: str
    mode: str
    client: str
    input: str
    user_text: str
    reply_text: str
    heard_text: str | None
    interrupted: bool
    tools: tuple[tuple[str, bool], ...]
    atlas: AtlasCapture | None
    raw: dict[str, Any] = field(compare=False, repr=False)

    @property
    def spoken_text(self) -> str:
        """What the person actually heard of the reply."""
        if self.interrupted and self.heard_text is not None:
            return self.heard_text
        return self.reply_text

    @property
    def captured_text(self) -> str | None:
        """The words saved into Atlas, if any."""
        if not self.atlas:
            return None
        return self.atlas.text if self.atlas.text is not None else self.user_text

    @property
    def last_time(self) -> datetime:
        return self.t_end or self.t_start


def _ts(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} is not a string")
    t = datetime.fromisoformat(value)
    if t.tzinfo is None:
        # An offset-less time means a different instant on every machine; refuse rather than guess (kb's rule too).
        raise ValueError(f"{name} has no UTC offset")
    return t


def parse_record(obj: Any) -> Turn:
    """One JSON object to a Turn; ValueError names what is wrong."""
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    if obj.get("type") != "turn":
        raise ValueError(f"type {obj.get('type')!r}")
    if obj.get("v") != 1:
        raise ValueError(f"unknown version {obj.get('v')!r}")
    for k in REQUIRED:
        if k not in obj or obj[k] is None:
            raise ValueError(f"missing {k}")
    if not isinstance(obj["turn"], int) or isinstance(obj["turn"], bool):
        raise ValueError("turn is not an integer")
    for k in ("session", "space", "user_text", "reply_text"):
        if not isinstance(obj[k], str):
            raise ValueError(f"{k} is not a string")
    tools = []
    for t in obj.get("tools") or []:
        if isinstance(t, dict) and isinstance(t.get("name"), str):
            tools.append((t["name"], bool(t.get("ok", True))))
    atlas = None
    a = obj.get("atlas")
    if isinstance(a, dict) and isinstance(a.get("root"), str) and isinstance(a.get("path"), str):
        text = a.get("text")
        atlas = AtlasCapture(root=a["root"], path=a["path"], by=str(a.get("by") or ""),
                             text=text if isinstance(text, str) else None)
    heard = obj.get("heard_text")
    return Turn(
        session=obj["session"], turn=obj["turn"], t_start=_ts(obj["t_start"], "t_start"),
        t_end=_ts(obj["t_end"], "t_end") if obj.get("t_end") else None,
        space=obj["space"], mode=str(obj.get("mode") or "conversation"), client=str(obj.get("client") or ""),
        input=str(obj.get("input") or "voice"), user_text=obj["user_text"], reply_text=obj["reply_text"],
        heard_text=heard if isinstance(heard, str) else None, interrupted=bool(obj.get("interrupted", False)),
        tools=tuple(tools), atlas=atlas, raw=obj,
    )


def day_file(turns_dir: Path, day: str) -> Path:
    return turns_dir / f"{day}.jsonl"


def days_available(turns_dir: Path) -> list[str]:
    if not turns_dir.is_dir():
        return []
    return sorted(p.stem for p in turns_dir.glob("*.jsonl") if DAY_RE.match(p.stem))


def read_day(turns_dir: Path, day: str) -> tuple[list[Turn], list[str]]:
    """The day's turns in time order (duplicates of one session+turn keep the last), and the problems found."""
    path = day_file(turns_dir, day)
    problems: list[str] = []
    turns: dict[tuple[str, int], Turn] = {}
    if not path.exists():
        return [], problems
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{path.name}:{n}: not JSON ({e.msg})")
                continue
            if isinstance(obj, dict) and obj.get("type") not in (None, "turn"):
                continue  # other record types are allowed and ignored
            try:
                t = parse_record(obj)
            except ValueError as e:
                problems.append(f"{path.name}:{n}: {e}")
                continue
            turns[(t.session, t.turn)] = t
    return sorted(turns.values(), key=lambda t: (t.t_start, t.session, t.turn)), problems


def last_activity(turns_dir: Path) -> datetime | None:
    """The newest t_end/t_start in the two newest day files (a session can cross midnight)."""
    newest: datetime | None = None
    for day in days_available(turns_dir)[-2:]:
        turns, _ = read_day(turns_dir, day)
        for t in turns:
            if newest is None or t.last_time > newest:
                newest = t.last_time
    return newest


def file_digest(turns_dir: Path, day: str) -> str:
    """A content hash of the day's file, so a rerun can tell whether the input changed."""
    import hashlib
    p = day_file(turns_dir, day)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else ""
