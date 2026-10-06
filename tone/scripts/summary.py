"""Summarise the hint log, and with the orchestrator's turn log, compare the two A/B arms (the design: measure for a
week, then A/B with the hint stripped).

    uv run --project tone python tone/scripts/summary.py [--state ~/repos/local-voice/state] [--days 7]

Machine signals only, never a rating by a listener: how often a hint was due and for which features; and
per arm, how often the reply to that turn was interrupted, how long it was, and whether it asked a question (the
instruction tells the model to ask rather than assert). The verdict stays a judgement; this makes it an informed one.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


def jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except ValueError:
            continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(Path(__file__).resolve().parent.parent.parent / "state"))
    ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args()
    state = Path(a.state).expanduser()
    first = (date.today() - timedelta(days=a.days - 1)).isoformat()
    hints = [r for p in sorted((state / "tone").glob("hints-*.jsonl")) if p.stem[6:] >= first for r in jsonl(p)]
    if not hints:
        print(f"no hint log under {state / 'tone'} since {first}")
        return 0
    arms = Counter(r["arm"] for r in hints)
    feats = Counter(d["name"] for r in hints if r.get("due") for d in r["deviations"])
    ms = [r["ms"] for r in hints]
    print(f"{len(hints)} turns analysed since {first}; hint due on {sum(1 for r in hints if r.get('due'))}")
    print(f"arms: {dict(arms)}; features named when due: {dict(feats)}")
    print(f"analyze ms: median {statistics.median(ms):.1f}, max {max(ms):.1f}")
    turns = {}
    for p in sorted((state / "turns").glob("*.jsonl")):
        if p.stem >= first:
            for t in jsonl(p):
                if t.get("type") == "turn":
                    turns[(t["session"], t["turn"])] = t
    by_arm = defaultdict(list)
    for r in hints:
        t = turns.get((r["session"], r["turn"]))
        if t and r["arm"] in ("show", "strip"):
            by_arm[r["arm"]].append(t)
    for arm, ts in sorted(by_arm.items()):
        words = [len((t.get("reply_text") or "").split()) for t in ts]
        print(f"{arm:5s}: {len(ts)} turns; reply interrupted {sum(bool(t.get('interrupted')) for t in ts) / len(ts):.0%}; "
              f"reply words median {statistics.median(words)}; reply asks a question "
              f"{sum('?' in (t.get('reply_text') or '') for t in ts) / len(ts):.0%}")
    if not by_arm:
        print("no hinted turns found in the turn log yet (the orchestrator logs state/turns; brain/TURN_LOG.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
