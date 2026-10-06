"""Technical facts into the kb, through kb verbs only (kb AGENTS.md: every bookkeeping step is a `kb` verb).

Where they go, and why: the kb's operational lane, as `kb session note` lines on the digest of the technical pass's
own Pi session (kb.ts registered and closed that session during the run). Not wiki pages: kb AGENTS.md's two-lane
rule says session material is operational memory and becomes knowledge only when the human promotes a distilled
note into raw/notes/, and "do not let raw transcripts become topic evidence". A note line carries the fact, its kind
and the voice turns it came from, so promoting one is a copy.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .turnlog import Turn
from .validate import Fact

_DROP_PREFIXES = ("KB_", "PI_", "CLAUDE", "HERDR")


class KbError(RuntimeError):
    def __init__(self, msg: str, written: int = 0):
        super().__init__(msg)
        self.written = written      # notes already appended before the failure: a retry skips them


@dataclass
class KbResult:
    digest: str          # bundle-relative path kb printed, e.g. /.sessions/digests/2026/10/pi-<id>.md
    lines: list[str]


def kb_env(kb_home: Path) -> dict[str, str]:
    # The session and actor are passed as flags, so nothing inherited may override them (kb resolution step 1).
    env = {k: v for k, v in os.environ.items() if not k.startswith(_DROP_PREFIXES)}
    env["KB_HOME"] = str(kb_home)
    return env


def cite(t: Turn) -> str:
    return f"voice session {t.session} turn {t.turn}, {t.t_start.strftime('%H:%M')}, space {t.space}"


def note_line(day: str, f: Fact, handles: dict[str, Turn]) -> str:
    where = "; ".join(cite(handles[h]) for h in f.cites)
    line = f"voice {day} · {f.kind}: {f.text} ({where})"
    if f.quote:
        line += f' — "{" ".join(f.quote.split())}"'
    return line


def note_facts(kb_bin: Path, kb_home: Path, *, session_id: str, actor: str, day: str, facts: list[Fact],
               handles: dict[str, Turn], skip: int = 0) -> KbResult:
    """Append one `kb session note` per fact; `skip` resumes after a partial earlier attempt."""
    if not kb_bin.is_file():
        raise KbError(f"kb binary missing: {kb_bin}")
    if not session_id.startswith("pi:"):
        raise KbError(f"not a pi session id: {session_id}")
    lines, digest = [], ""
    for f in facts[skip:]:
        line = note_line(day, f, handles)
        r = subprocess.run([str(kb_bin), "session", "note", "--session", session_id, "--actor", actor, line],
                           capture_output=True, text=True, cwd=str(kb_home), env=kb_env(kb_home), timeout=60)
        if r.returncode != 0:
            raise KbError(f"kb session note failed ({r.returncode}): {r.stderr.strip()[-300:]}", written=len(lines))
        digest = r.stdout.strip() or digest
        lines.append(line)
    return KbResult(digest=digest, lines=lines)


def digest_path(kb_home: Path, session_id: str) -> Path | None:
    """Where kb keeps the session's digest (kb finds it by name the same way)."""
    harness, _, native = session_id.partition(":")
    hits = sorted((kb_home / ".sessions" / "digests").glob(f"*/*/{harness}-{native}.md"))
    return hits[0] if hits else None
