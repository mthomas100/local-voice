"""The job's own state under state/brain: the ledger of days, each day's saved proposal, the heartbeat log, the lock.

A day's proposal (what the model proposed, after validation) is saved before anything is written, so a run that
fails half way retries only the writes that failed, without asking the model again.
"""
from __future__ import annotations

import fcntl
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .turnlog import Turn
from .validate import Fact, LifeProposal, Quote, Reflection, TechProposal

DONE, RETRY, FAILED, SKIPPED = "done", "retry", "failed", "skipped"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Ledger:
    """{"days": {"YYYY-MM-DD": {"status", "attempts", "sinks", "last_error", "updated"}}}"""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def load(cls, state_dir: Path) -> "Ledger":
        p = state_dir / "ledger.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("days", {})
        return cls(p, data)

    def day(self, day: str) -> dict[str, Any]:
        return self.data["days"].setdefault(day, {"status": RETRY, "attempts": 0, "sinks": {}})

    def status(self, day: str) -> str | None:
        d = self.data["days"].get(day)
        return d.get("status") if d else None

    def save(self) -> None:
        _write_json(self.path, self.data)


def day_dir(state_dir: Path, day: str) -> Path:
    return state_dir / "days" / day


# --- proposals ----------------------------------------------------------------------------------------------------

def handle_map(handles: dict[str, Turn]) -> dict[str, list[Any]]:
    return {h: [t.session, t.turn] for h, t in handles.items()}


def resolve_handles(saved: dict[str, list[Any]], turns: list[Turn]) -> dict[str, Turn]:
    by_key = {(t.session, t.turn): t for t in turns}
    out = {}
    for h, (session, turn) in saved.items():
        if (session, turn) not in by_key:
            raise ValueError(f"saved handle {h} names a turn no longer in the log")
        out[h] = by_key[(session, turn)]
    return out


def save_proposal(state_dir: Path, day: str, obj: dict[str, Any]) -> None:
    _write_json(day_dir(state_dir, day) / "proposal.json", obj)


def load_proposal(state_dir: Path, day: str) -> dict[str, Any] | None:
    try:
        return json.loads((day_dir(state_dir, day) / "proposal.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tech_to_json(p: TechProposal) -> dict[str, Any]:
    return {"facts": [asdict(f) for f in p.facts], "spoken": p.spoken, "dropped": p.dropped}


def tech_from_json(d: dict[str, Any]) -> TechProposal:
    return TechProposal(facts=[Fact(**f) for f in d.get("facts", [])], spoken=d.get("spoken"),
                        dropped=list(d.get("dropped", [])))


def life_to_json(p: LifeProposal) -> dict[str, Any]:
    return {"reflections": [asdict(r) for r in p.reflections], "dropped": p.dropped}


def life_from_json(d: dict[str, Any]) -> LifeProposal:
    refl = []
    for r in d.get("reflections", []):
        refl.append(Reflection(title=r["title"], quotes=[Quote(**q) for q in r["quotes"]], notice=r["notice"],
                               questions=list(r["questions"]), tags=list(r["tags"])))
    return LifeProposal(reflections=refl, dropped=list(d.get("dropped", [])))


# --- heartbeat log and lock -------------------------------------------------------------------------------------

def log_heartbeat(state_dir: Path, now: datetime, outcome: str, detail: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "heartbeat.log", "a", encoding="utf-8") as fh:
        fh.write(f"{now.isoformat(timespec='seconds')} {outcome} {' '.join(detail.split())}\n")


class Lock:
    """A non-blocking flock: a second heartbeat while one runs simply exits."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def __exit__(self, *exc) -> None:
        if self.fh:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
