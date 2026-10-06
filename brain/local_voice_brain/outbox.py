"""The spoken-digest outbox (brain/TURN_LOG.md §2): the brain queues, the orchestrator speaks and marks delivered.

Standard library only and free of imports from this package, so the orchestrator can load this one file by path
(importlib) or copy it. Files: `<state_dir>/outbox/<day>.json`; delivered ones move to `outbox/delivered/`.

No `from __future__ import annotations` here, on purpose: with postponed (string) annotations, `dataclass` resolves
them through sys.modules, so loading the file by path without registering it there first failed with AttributeError
(found by the orchestrator agent, 2026-10-05). Python 3.12 evaluates these annotations natively.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

VERSION = 1


@dataclass
class Digest:
    """One queued spoken digest."""
    path: Path
    id: str
    day: str
    text: str
    created: datetime
    expires: datetime
    data: dict[str, Any] = field(default_factory=dict)

    def expired(self, now: datetime) -> bool:
        return now >= self.expires


def outbox_dir(state_dir: str | os.PathLike) -> Path:
    return Path(state_dir) / "outbox"


def _write_atomic(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def queue(state_dir: str | os.PathLike, *, day: str, text: str, now: datetime, expiry_days: float,
          cites: list[dict[str, Any]] | None = None, **extra: Any) -> Path:
    """Queue (or replace) the digest for `day`. `now` must carry a timezone."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    d = outbox_dir(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    obj = {"v": VERSION, "id": f"digest-{day}", "day": day, "created": now.isoformat(timespec="seconds"),
           "expires": (now + timedelta(days=expiry_days)).isoformat(timespec="seconds"), "text": text,
           "cites": cites or [], **extra}
    path = d / f"{day}.json"
    _write_atomic(path, obj)
    return path


def _load(path: Path) -> Digest | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return Digest(path=path, id=str(obj["id"]), day=str(obj["day"]), text=str(obj["text"]),
                      created=datetime.fromisoformat(obj["created"]), expires=datetime.fromisoformat(obj["expires"]),
                      data=obj)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pending(state_dir: str | os.PathLike, now: datetime) -> list[Digest]:
    """Digests not yet delivered and not expired, oldest day first. Unreadable files are skipped."""
    d = outbox_dir(state_dir)
    if not d.is_dir():
        return []
    out = [g for g in (_load(p) for p in sorted(d.glob("*.json"))) if g and not g.expired(now)]
    return sorted(out, key=lambda g: g.day)


def mark_delivered(path: str | os.PathLike, now: datetime) -> Path:
    """Move a digest to outbox/delivered/ with a `delivered` stamp. Call once it was played (or interrupted)."""
    src = Path(path)
    obj = json.loads(src.read_text(encoding="utf-8"))
    obj["delivered"] = now.isoformat(timespec="seconds")
    dest_dir = src.parent / "delivered"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    _write_atomic(dest, obj)
    src.unlink()
    return dest
