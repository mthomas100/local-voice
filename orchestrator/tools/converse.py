"""A scripted conversation against a running orchestrator (protocol v1), one line of timings per turn (2026-10-05).

Each argument is one turn, spoken with `say` (rendered before anything is timed) and streamed at real-time pace with
room silence around it, or typed when it starts with "type:". The client keeps its microphone open the whole time, as
the apps do, and waits for the turn's end_of_turn before the next one. Recorded per turn, from the client's side: the
transcript, end of speech to the final transcript and to the reply's first audio, the reply's text and audio length,
every `tool` message with its time after the end of speech, any `confirm_request`, and the turn's total time. Needs a
server (`./run.sh --scratch DIR --port 8771 --browser-port 7861`; the models and the LLM run there, so the server's
own gpu_clear.sh rule applies to starting it).

Usage:  .venv/bin/python tools/converse.py --url ws://127.0.0.1:8771/v1/voice --out DIR "What does a heat pump do?" ...
Writes DIR/converse.jsonl (one JSON line per turn) and prints a table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH))

from local_voice.client import V1Client, say_pcm, speech_end_s  # noqa: E402


async def turn(c: V1Client, beat: str, pcm: bytes | None, wait_s: float) -> dict:
    """One turn: speak (or type) it, then wait for the end_of_turn of a reply that started after it."""
    typed = pcm is None
    start = c.now()
    if typed:
        await c.send({"t": "text", "text": beat})
        eos = c.now()
    else:
        await c.stream(pcm)
        eos = start + speech_end_s(pcm)
    tail = asyncio.create_task(c.silence(wait_s + 5))
    row: dict = {"at": time.strftime("%H:%M:%S"), "input": "text" if typed else "voice", "said": beat}
    try:
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn" and (r := c.replies.get(m.get("reply_id")))
                               is not None and r.started_at is not None and r.started_at >= start,
                               timeout=wait_s, since=eos)
        end_at = next(at for at, m in c.messages if m is end)
    except TimeoutError:
        end_at = None
    finally:
        tail.cancel()
    msgs = [(at, m) for at, m in c.messages if at >= start]
    finals = [(at, m["text"]) for at, m in msgs if m.get("t") == "transcript" and m.get("final")]
    replies = sorted((r for r in c.replies.values() if r.started_at is not None and r.started_at >= start),
                     key=lambda r: r.started_at)
    first = next((r for r in replies if r.first_audio_at is not None), None)
    ms = lambda at: None if at is None else round(1000 * (at - eos))  # noqa: E731
    row.update(
        transcript=" ".join(t for _, t in finals) if not typed else None,
        eos_to_transcript_ms=ms(finals[-1][0]) if finals else None,
        eos_to_first_audio_ms=ms(first.first_audio_at) if first else None,
        total_ms=ms(end_at),
        timed_out=end_at is None,
        reply_text="".join("".join(r.text) for r in replies),
        reply_audio_s=round(sum(r.audio_s for r in replies), 2),
        tools=[{"phase": m["phase"], "name": m.get("name"), "ok": m.get("ok"), "after_ms": ms(at)}
               for at, m in msgs if m.get("t") == "tool"],
        confirm=[m.get("message") for _, m in msgs if m.get("t") == "confirm_request"],
        load1=round(os.getloadavg()[0], 2))
    return row


async def run(args) -> list[dict]:
    pcms = {b: (None if b.startswith("type:") else say_pcm(b, voice=args.voice)) for b in args.turns}
    c = V1Client(args.url, device=args.device)
    await c.connect()
    rows = []
    try:
        await c.silence(1.0)
        for b in args.turns:
            row = await turn(c, b.removeprefix("type:"), pcms[b], args.wait)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            await c.silence(args.gap)
    finally:
        await c.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("turns", nargs="+", help='each a turn: spoken with say, or typed when it starts with "type:"')
    ap.add_argument("--url", default="ws://127.0.0.1:8771/v1/voice")
    ap.add_argument("--device", default="converse")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--wait", type=float, default=150.0, help="seconds to wait for a turn's end_of_turn")
    ap.add_argument("--gap", type=float, default=1.5, help="seconds of room silence between turns")
    ap.add_argument("--out", type=Path, help="folder for converse.jsonl")
    args = ap.parse_args()
    rows = asyncio.run(run(args))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "converse.jsonl").open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        tools = ",".join(t["name"] for t in r["tools"] if t["phase"] == "start") or "-"
        print(f"{r['at']} {r['said'][:40]:40} first audio {r['eos_to_first_audio_ms']} ms, total {r['total_ms']} ms, "
              f"tools {tools}: {r['reply_text'][:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
