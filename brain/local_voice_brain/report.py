"""The day's digest as a page for the owner: which sessions, what was kept where, what was said, what was dropped.

Written to state/brain/days/<day>/report.md (gitignored, beside the turn log it cites). It is the "morning digest
that cites yesterday's sessions" of milestone M4 in written form; the spoken form is the outbox entry.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .prompts import weekday

if TYPE_CHECKING:
    from .config import Config
    from .job import DayRun


def write_report(cfg: "Config", run: "DayRun", *, now: datetime, path: Path) -> Path:
    by_session: dict[str, list] = defaultdict(list)
    for t in run.turns:
        by_session[t.session].append(t)
    tech_ids = {t.session: h.split("T")[0] for h, t in run.tech_handles.items()}
    L = [f"# Voice digest, {weekday(run.day)} {run.day}", "",
         f"Written {now.isoformat(timespec='seconds')} by the voice brain · model `{run.model or 'none asked'}`"
         + (f" ({run.model_note})" if run.model_note else ""), ""]
    L += ["## Sessions", ""]
    for sid, ts in by_session.items():
        spaces = ", ".join(dict.fromkeys(t.space for t in ts))
        saved = sum(1 for t in ts if t.atlas)
        L.append(f"- `{sid}` {ts[0].t_start.strftime('%H:%M')}–{ts[-1].last_time.strftime('%H:%M')} · {spaces} · "
                 f"{len(ts)} turns" + (f", {saved} saved to Atlas" if saved else "")
                 + (f" (handle {tech_ids[sid]})" if sid in tech_ids else ""))
    if run.problems:
        L += ["", f"Skipped {len(run.problems)} unreadable turn-log lines: " + "; ".join(run.problems[:5])]
    L += ["", "## Technical facts", ""]
    if run.tech is None:
        L.append("No technical pass (no turns outside the Atlas space).")
    elif not run.tech.facts:
        L.append("None worth keeping.")
    else:
        where = f"kb digest `{run.kb_digest}`" if run.kb_digest else "not written to the kb"
        L.append(f"Kept as `kb session note` lines on kb session `{run.tech_session}` ({where}):")
        L.append("")
        for f in run.tech.facts:
            cites = "; ".join(f"`{run.tech_handles[h].session}` turn {run.tech_handles[h].turn} "
                              f"{run.tech_handles[h].t_start.strftime('%H:%M')}" for h in f.cites)
            L.append(f"- {f.kind}: {f.text} [{cites}]" + (f' — "{f.quote}"' if f.quote else ""))
    L += ["", "## Reflections", ""]
    if run.life is None:
        L.append("No life pass (nothing said in the Atlas space was saved there)." if not run.life_skipped
                 else "No life pass.")
    elif not run.life.reflections:
        L.append("None drafted.")
    else:
        page = run.atlas.page if run.atlas else "(not written)"
        commit = f", commit `{run.atlas.commit[:12]}`" if run.atlas and run.atlas.commit else ""
        L.append(f"In Atlas `{page}`{commit}:")
        L.append("")
        for r in run.life.reflections:
            L.append(f"- {r.title}: {len(r.quotes)} quotes from "
                     + ", ".join(sorted({f'{run.life_handles[q.handle].t_start.strftime("%H:%M")}' for q in r.quotes})))
    if run.life_skipped:
        L += ["", "Not quotable: " + "; ".join(run.life_skipped)]
    L += ["", "## Spoken digest", ""]
    L.append(f"> {run.spoken}" if run.spoken else "NO_REPLY: nothing queued.")
    if run.outbox_path:
        L.append(f"\nQueued at `{run.outbox_path}` for the next conversation.")
    dropped = (run.tech.dropped if run.tech else []) + (run.life.dropped if run.life else [])
    if dropped or run.warnings:
        L += ["", "## Dropped and warnings", ""]
        L += [f"- dropped: {d}" for d in dropped] + [f"- warning: {w}" for w in run.warnings]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
