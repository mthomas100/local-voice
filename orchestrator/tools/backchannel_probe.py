"""Backchannel probe (Nemotron on the MLX thread: needs gpu_clear.sh first): what does the live recogniser return for
the first VAD segment of a person speaking over a reply? It decides whether a barge-in can be confirmed by its words
(the design: resume the reply on "mm-hmm", interrupt on anything else). Machine ground truth only: the barge-in bench's
`say` cases (tools/bargein_bench.py), 22 interruption phrases and 10 backchannels in 5 voices.

Each case's first segment is decoded as services/stt.py does it (config.yaml's preroll before the VAD start, the
segment, the tail pad). Reported per kind: how many come back empty, how many read as backchannels
(speech_text.is_backchannel), and how long after speech onset the segment's final would arrive at the latest
(its VAD stop; the decode itself is about 50 ms more).

Usage:  .venv/bin/python tools/backchannel_probe.py ../state/backchannel/<stamp>
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH))
sys.path.insert(0, str(ORCH / "tools"))

import bargein_bench as bb  # noqa: E402
from turn_end_bench import Recogniser, segments  # noqa: E402

from local_voice.speech_text import is_backchannel  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from local_voice.config import load_config
    cfg = load_config()
    vad = cfg.vad
    st = cfg.raw["stt"]["adapters"]["nemotron"]
    rec = Recogniser("nemotron", float(st.get("preroll_s", 1.0)), float(st.get("tail_pad_s", 0.3)))
    cases = []
    for v in bb.VOICES:
        for t in bb.PHRASES:
            cases.append(("interruption", v, t, bb.say_clip(t, v)))
        for t in bb.BACKCHANNELS:
            cases.append(("backchannel", v, t, bb.say_clip(t, v)))

    async def traces():
        rng = np.random.default_rng(0)
        return [await bb.trace(bb.Case(f"{k}:{v}:{t}", k, a), "silence", rng) for k, v, t, a in cases]

    rows = []
    for (kind, voice, text, audio), tr in zip(cases, asyncio.run(traces())):
        lead = np.zeros(int(bb.LEAD_S * bb.RATE), np.float32)
        full = np.concatenate([lead, audio, np.zeros(int(bb.TAIL_S * bb.RATE), np.float32)])
        segs = segments(tr, vad["confidence"], vad["start_secs"], vad["min_volume"], vad["stop_secs"])
        if not segs:
            rows.append({"kind": kind, "voice": voice, "text": text, "heard": None})
            continue
        a, b = segs[0]
        heard = rec.text(full, a + bb.LEAD_S, b + bb.LEAD_S, text)
        onset = bb.onset_s(audio)
        rows.append({"kind": kind, "voice": voice, "text": text, "heard": heard, "segments": len(segs),
                     "vad_start_s": round(a - onset, 3), "vad_stop_s": round(b - onset, 3),
                     "backchannel": is_backchannel(heard)})
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    lines = ["# Backchannel probe (Nemotron, first VAD segment)", "",
             "| kind | cases | VAD missed | empty | read as backchannel | final due (VAD stop after onset): median / max s |",
             "|---|---|---|---|---|---|"]
    for kind in ("interruption", "backchannel"):
        rs = [r for r in rows if r["kind"] == kind]
        got = [r for r in rs if r["heard"] is not None]
        stops = [r["vad_stop_s"] for r in got]
        lines.append(f"| {kind} | {len(rs)} | {len(rs) - len(got)} | {sum(1 for r in got if not r['heard'])} | "
                     f"{sum(1 for r in got if r['backchannel'])} | {statistics.median(stops):.2f} / {max(stops):.2f} |")
    lines += ["", "## What was heard", ""]
    for r in rows:
        lines.append(f"- {r['kind']} {r['voice']}: {r['text']!r} -> {r['heard']!r}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
