"""Count echo turns in a turn log: user turns whose words are mostly what the agent said shortly before.

    .venv/bin/python tools/echo_turns.py ../state/turns/<day>.jsonl \
        --log ../state/orchestrator/logs/orchestrator-<day>.log [--window-s 300] [--session ID] [--json]

The turn log alone gives each earlier turn's `reply_text` (what was said) and its `t_end`. With the orchestrator's
log (`--log`), every sentence the TTS generated is a candidate with its own time (acks, offers and notices too), and
each echo turn is reported with whether a reply was playing when it started (Pipecat's "Bot started/stopped
speaking") and how long after the matched sentence it came. Model-free: words only (local_voice/echo_guard.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_voice.echo_guard import best_match  # noqa: E402

_LOG_T = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
_TTS = re.compile(r"Generating TTS \[(.*)\]\s*$")


def _log_time(line: str) -> float | None:
    m = _LOG_T.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()


def read_log(path: Path, t0: float, t1: float) -> tuple[list[tuple[float, str]], list[tuple[float, float]]]:
    """(sentences generated, (start, stop) intervals the bot was speaking) between t0 and t1 (local epoch seconds;
    the log's times are local and offset-less, like datetime.timestamp() of a naive time)."""
    said: list[tuple[float, str]] = []
    speaking: list[tuple[float, float]] = []
    started = None
    for line in path.read_text(errors="replace").splitlines():
        t = _log_time(line)
        if t is None or not (t0 <= t <= t1):
            continue
        m = _TTS.search(line)
        if m:
            said.append((t, m.group(1)))
        elif "Bot started speaking" in line:
            started = t if started is None else started
        elif "Bot stopped speaking" in line and started is not None:
            speaking.append((started, t))
            started = None
    if started is not None:
        speaking.append((started, t1))
    return said, speaking


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def detect(turns: list[dict], *, window_s: float, min_words: int, min_share: float,
           said: list[tuple[float, str]] | None = None, speaking: list[tuple[float, float]] | None = None) -> list[dict]:
    rows = []
    for k, turn in enumerate(turns):
        start = _epoch(turn["t_start"])
        if said is not None:
            cands = [(s, t) for t, s in said if 0 <= start - t <= window_s]
        else:
            cands = [(p.get("reply_text") or "", _epoch(p["t_end"])) for p in turns[:k]
                     if 0 <= start - _epoch(p["t_end"]) <= window_s]
        m = best_match(turn.get("user_text") or "", cands)
        row = {"turn": turn.get("turn"), "t_start": turn["t_start"], "user_text": turn.get("user_text"),
               "echo": bool(m and m.is_echo(min_words, min_share)), "matched": m.matched if m else 0,
               "heard_words": m.heard_words if m else 0, "share": round(m.share, 2) if m else 0.0,
               "said": m.said if m else "", "lag_s": round(start - m.said_at, 1) if m and m.said_at else None,
               "interrupted": turn.get("interrupted")}
        if speaking is not None:
            row["bot_speaking_at_start"] = any(a <= start <= b for a, b in speaking)
            ended = [b for a, b in speaking if b <= start]
            row["since_bot_stopped_s"] = round(start - max(ended), 1) if ended and not row["bot_speaking_at_start"] else None
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("turn_log", type=Path, nargs="+")
    ap.add_argument("--log", type=Path, help="the orchestrator log of the same day (TTS sentences, bot speaking)")
    ap.add_argument("--session", help="only this session id")
    ap.add_argument("--window-s", type=float, default=300.0)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--min-share", type=float, default=0.6)
    ap.add_argument("--json", action="store_true", help="one JSON line per turn")
    a = ap.parse_args(argv)

    turns = [json.loads(line) for p in a.turn_log for line in p.read_text().splitlines() if line.strip()]
    turns = [t for t in turns if t.get("type") == "turn" and t.get("input", "voice") == "voice"]
    sessions = sorted({t["session"] for t in turns}) if not a.session else [a.session]
    total = echoes = 0
    for sid in sessions:
        st = sorted((t for t in turns if t["session"] == sid), key=lambda t: t["t_start"])
        if not st:
            continue
        said = speaking = None
        if a.log:
            said, speaking = read_log(a.log, _epoch(st[0]["t_start"]) - a.window_s, _epoch(st[-1]["t_end"]) + 60)
        rows = detect(st, window_s=a.window_s, min_words=a.min_words, min_share=a.min_share, said=said,
                      speaking=speaking)
        n = sum(r["echo"] for r in rows)
        total, echoes = total + len(rows), echoes + n
        if a.json:
            for r in rows:
                print(json.dumps({"session": sid, **r}))
            continue
        print(f"{sid}: {n} echo turns of {len(rows)}")
        for r in rows:
            if r["echo"]:
                where = ""
                if "bot_speaking_at_start" in r:
                    where = ("during a reply" if r["bot_speaking_at_start"]
                             else f"{r['since_bot_stopped_s']} s after a reply ended")
                print(f"  turn {r['turn']} {r['t_start'][11:19]} ({where}; {r['lag_s']} s after the words were "
                      f"generated; {r['matched']}/{r['heard_words']} words): {r['user_text']!r}  <-  {r['said']!r}")
    if not a.json:
        print(f"all: {echoes} echo turns of {total} voice turns (window {a.window_s:.0f} s, "
              f">= {a.min_words} words and >= {a.min_share:.0%} of the turn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
