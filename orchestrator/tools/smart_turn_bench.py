"""Smart Turn v3.2 bench (CPU only; no GPU, no model call): how often it calls a finished question unfinished, and an
unfinished one finished. Machine ground truth only: `say` speech whose text is known to be complete or cut off.

A complete question judged INCOMPLETE waits for turn.smart_turn.stop_secs (3 s) before the turn ends: on 2026-10-05
14:07 "What does a heat pump do?" was judged incomplete twice in a row and its answer started 4.3 s after the person
stopped. A fragment judged COMPLETE cuts the person off ("...a lighthouse keeper and his cat," in an earlier run).

Each clip is scored the way Pipecat 1.12 scores a turn when the VAD stops: the speech plus the VAD's 0.2 s of trailing
silence, through LocalSmartTurnAnalyzerV3._predict_endpoint (the bundled ONNX model; COMPLETE above 0.5).

Usage:  .venv/bin/python tools/smart_turn_bench.py ../state/smart-turn/<stamp>
Writes rows.jsonl and summary.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RATE = 16000
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Ralph"]
COMPLETE = ["What is the capital of France?", "Say hello in Spanish.", "What color is the sky on a clear day?",
            "Name a fruit that is yellow.", "What is the opposite of cold?", "Which animal says moo?",
            "What does a heat pump do?", "Does it work in winter too?", "How can I sleep better?",
            "Why is the sky blue?", "What should I pack for a day hike?", "Tell me a fun fact about octopuses.",
            "What does my knowledge base say about the hold gate?", "Stop. What is the capital of Italy?",
            "Set a timer for ten minutes.", "What's on my calendar tomorrow?", "Thanks, that's all.",
            "Can you explain how compound interest works?", "Read me the first line of the readme.",
            "Go to my journal."]
FRAGMENTS = ["What does a heat pump", "Can you tell me how", "I was thinking that maybe we could",
             "So the thing is", "Tell me about the", "What is the capital of", "Could you look up the",
             "I want to know whether", "My question is about the", "Remind me to call", "And then after that",
             "The reason I ask is", "If it's not too much trouble, could you", "Let me think, um",
             "What's the difference between"]


def say_clip(text: str, voice: str) -> np.ndarray:
    from local_voice.client import say_pcm
    return np.frombuffer(say_pcm(text, voice=voice), "<i2").astype(np.float32) / 32768


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--trailing-s", type=float, default=0.2, help="silence after the speech (the VAD's stop_secs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    st = LocalSmartTurnAnalyzerV3()
    st.set_sample_rate(RATE)
    rows = []
    for kind, texts in (("complete", COMPLETE), ("fragment", FRAGMENTS)):
        for v in VOICES:
            for t in texts:
                a = say_clip(t, v)
                loud = np.flatnonzero(np.abs(a) > 0.01)
                a = a[max(0, loud[0] - 1600):loud[-1] + 1] if loud.size else a
                x = np.concatenate([a, np.zeros(int(args.trailing_s * RATE), np.float32)])
                r = st._predict_endpoint(x)
                rows.append({"kind": kind, "voice": v, "text": t, "probability": round(float(r["probability"]), 3),
                             "complete": bool(r["prediction"])})
    (args.out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines = ["# Smart Turn v3.2 bench", "",
             f"{len(COMPLETE)} complete requests and {len(FRAGMENTS)} fragments, each in {len(VOICES)} voices, with "
             f"{args.trailing_s} s of silence after the speech.", "",
             "| voice | complete judged INCOMPLETE (waits the 3 s fallback) | fragments judged COMPLETE (cut off) |",
             "|---|---|---|"]
    for v in [*VOICES, "all"]:
        sel = [r for r in rows if v == "all" or r["voice"] == v]
        c = [r for r in sel if r["kind"] == "complete"]
        f = [r for r in sel if r["kind"] == "fragment"]
        lines.append(f"| {v} | {sum(not r['complete'] for r in c)}/{len(c)} | {sum(r['complete'] for r in f)}/{len(f)} |")
    lines += ["", "Complete requests judged INCOMPLETE:", ""]
    lines += [f"- {r['voice']}: {r['text']!r} (p={r['probability']})" for r in rows
              if r["kind"] == "complete" and not r["complete"]]
    lines += ["", "Fragments judged COMPLETE:", ""]
    lines += [f"- {r['voice']}: {r['text']!r} (p={r['probability']})" for r in rows
              if r["kind"] == "fragment" and r["complete"]]
    c_all = [r["probability"] for r in rows if r["kind"] == "complete"]
    lines += ["", f"Median probability: complete {statistics.median(c_all):.2f}, fragments "
              f"{statistics.median(r['probability'] for r in rows if r['kind'] == 'fragment'):.2f}."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
