"""Turn-end bench: when does a user turn end, for finished requests and for requests with a pause inside? Machine
ground truth only (`say` speech whose words and pauses are known; a project rule: no human listening or labelling).

Cases, each `say` speech with 0.5 s of silence before and 4 s after, in 5 voices:
- complete: the 20 finished requests of tools/smart_turn_bench.py. Measured: end of speech to the turn's end.
- paused: 16 sentences spoken whole with 0.8 s or 1.5 s of silence spliced in at a word boundary, so the speech before
  the silence keeps the intonation of a sentence that goes on. Cutting a text off and reading it alone gives it final
  intonation instead (smart_turn_bench.py's fragments), and so does `say`'s own [[slnc N]] (see PAUSED).
  Measured: whether the turn ends before the speech after the silence starts (the person cut off), and for the
  first-word cases ("Wait, ... does it work in winter?") whether the first word reaches the transcript.

Events per case, as the pipeline produces them (config.yaml's settings):
- Silero segments: Pipecat 1.12's VAD state machine replayed on the analyzer's own scores (tools/bargein_bench.py).
- Smart Turn v3.2 at each VAD stop, over the turn's audio from the first start (less pre-speech) to the stop, last
  8 s, as BaseSmartTurn scores it; COMPLETE above 0.5.
- The transcript of each segment: `--stt text` stands in with the words `say` spoke, punctuated as written; `--stt
  nemotron` runs the live session as services/stt.py does (the preroll before the VAD start, the segment, the tail
  pad), on an MLX worker: real punctuation, real dropped words. That needs gpu_clear.sh first.

Rules, simulated on the same events:
- smart-turn-3.0: now. COMPLETE ends the turn at the stop; INCOMPLETE waits 3 s of silence (turn.smart_turn.stop_secs).
- smart-turn-1.5: the same with a 1.5 s fallback.
- punct-<s>: smart-turn-3.0, and a segment whose final transcript ends a sentence (local_voice/turn_end.py ends_turn)
  ends the turn <s> after the VAD stop (turn.smart_turn.punctuation_stop_secs).
- `--carry`: a segment that transcribes to nothing is decoded again with the next one (the first-word fix).
- smart-turn-2.0: config.yaml's fallback since 2026-10-05; +hold: a COMPLETE whose final ends on a filler or joining
  word (local_voice/turn_end.py holds) waits for the fallback instead; +hold-q: the same, except a question ending on a
  preposition ("What does it look like?").

`--set fillers` (2026-10-05) measures the hold rule instead: the 20 finished requests and 5 more that
end on a preposition (the rule's cost), 13 sentences with a pause spliced right after a filler or joining word ("Add
milk and [pause] eggs to my shopping list.", "Set a timer for, uh, [pause] ten minutes."), and one rambling
request of the kind early live tests were full of, spoken with a pause at each of its full stops, counted in turns.

Usage:  .venv/bin/python tools/turn_end_bench.py ../state/turn-end/<stamp> [--stt nemotron] [--carry] [--set fillers]
Writes rows.jsonl and summary.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH))
sys.path.insert(0, str(ORCH / "tools"))

from bargein_bench import Case, Trace, pcm16, trace  # noqa: E402
from local_voice.turn_end import ends_turn, goes_on, holds  # noqa: E402
from smart_turn_bench import COMPLETE, VOICES  # noqa: E402

RATE = 16000
LEAD_S, TAIL_S = 0.5, 4.0
FRAME_S = 512 / RATE
# (before, after): the sentence is spoken whole and the silence spliced in at the word boundary, so the speech before
# it keeps the intonation of a sentence that goes on. `say` gives the speech before an embedded [[slnc N]] a falling,
# final intonation (Smart Turn called "What's the difference between [[slnc 1500]]" COMPLETE at 0.97-0.99 in all five
# voices, 2026-10-05), so that form is no measure of cutting someone off either.
PAUSED = [
    ("I was thinking about", "the weather this weekend."),
    ("Can you tell me", "how long it takes to boil an egg?"),
    ("What is the capital of", "Australia?"),
    ("My question is about the,", "the heat pump in the garage."),
    ("So,", "what should I cook tonight?"),
    ("Remind me to call", "my sister tomorrow."),
    ("I want to know whether, um,", "it will rain later."),
    ("Tell me about the", "history of Rome."),
    ("If it's not too much trouble,", "could you read me my notes?"),
    ("What's the difference between", "a heat pump and a furnace?"),
    ("Let me think.", "What did I write yesterday?"),
    ("And then,", "after that, what happens?"),
    ("The reason I ask is", "I'm planning a trip."),
    ("Could you look up the", "opening hours of the library?"),
    ("Okay.", "Now tell me a joke."),
    ("Wait,", "does it work in winter?"),
]
PAUSES_MS = [800, 1500]
FIRST_WORD = {"So, ": "so", "Wait, ": "wait", "Okay. ": "okay", "And then, ": "and"}
# (name, fallback after INCOMPLETE, early end on a finished sentence, wait after a COMPLETE vetoed by the words,
#  veto also when the transcript has no punctuation at its end)
RULES = [("smart-turn-3.0", 3.0, None, None, False), ("smart-turn-1.5", 1.5, None, None, False),
         ("punct-0.6", 3.0, 0.6, None, False), ("punct-0.8", 3.0, 0.8, None, False),
         ("punct-0.8+veto-1.5", 3.0, 0.8, 1.5, False), ("punct-0.8+veto-3.0", 3.0, 0.8, 3.0, False),
         ("punct-0.8+veto-1.5-unpunct", 3.0, 0.8, 1.5, True)]
# --set fillers: (name, fallback, hold: None | "all" | "q")
HOLD_RULES = [("smart-turn-2.0", 2.0, None), ("smart-turn-2.0+hold", 2.0, "all"), ("smart-turn-2.0+hold-q", 2.0, "q")]
# finished requests that end on a word the hold rule waits on: what the rule costs
COMPLETE_DANGLING = ["What does a heat pump look like?", "Who should I give it to?", "What is it made of?",
                     "Which one are you thinking about?", "Where did that come from?"]
# a pause spliced right after a filler or joining word, the sentence spoken whole (see PAUSED)
FILLERS = [
    ("I'd like you to go to my journal and, um,", "add a note that says hello."),
    ("Can you make a, uh,", "journal entry for today?"),
    ("We could go for a walk, like,", "after lunch."),
    ("Add milk and", "eggs to my shopping list."),
    ("It looks good but", "it's a bit expensive."),
    ("Read me the", "last note I wrote."),
    ("Okay so", "what's the weather tomorrow?"),
    ("Remind me to buy, um,", "batteries tomorrow."),
    ("Set a timer for, uh,", "ten minutes."),
    ("What's the best way to, like,", "store fresh basil?"),
    ("Open my notes and", "read the last one."),
    ("I need a new, um,", "phone charger."),
    ("I think like a short note that just,", "says call the plumber."),
]
# One rambling request (synthetic), with a pause at each full stop: one request that
# a turn detector may log as several turns.
RAMBLE = ["Could you go to my journal.", "And uh.", "Just put a, um.", "So.", "I think like a short note that just.",
          "Call the plumber on Monday.", "Something like that.", "And let me know when you're done."]
RAMBLE_PAUSES_MS = [800, 1500]


def holds_q(text: str) -> bool:
    """holds, except a question that ends on a preposition ("What does it look like?"): config.yaml's setting."""
    return holds(text, except_questions=True)


@dataclass
class Seg:
    start: float              # VAD start fired (s from the case start)
    stop: float               # VAD stop fired
    p: float = 0.0            # Smart Turn probability at the stop, over the turn so far
    complete: bool = False
    complete_alone: bool = False   # over this segment alone: what the analyzer sees after a veto cleared its buffer
    text: str = ""            # this segment's final transcript ("" = nothing)


@dataclass
class Row:
    name: str
    kind: str                 # complete | paused
    voice: str
    text: str
    pause_ms: int = 0
    speech_end: float = 0.0   # last sample above -40 dBFS
    resume: float | None = None   # paused: onset of the speech after the silence
    segs: list[Seg] = field(default_factory=list)
    transcript: str = ""
    ends: dict = field(default_factory=dict)   # rule -> turn end (s) or None


def say_clip(text: str, voice: str) -> np.ndarray:
    from local_voice.client import say_pcm
    return np.frombuffer(say_pcm(text, voice=voice), "<i2").astype(np.float32) / 32768


def spliced(before: str, after: str, voice: str, pause_ms: int) -> tuple[np.ndarray, float]:
    """The whole sentence with `pause_ms` of silence spliced in after `before`, and where the speech resumes (s).
    The cut goes at the quietest 10 ms within 150 ms of where `before` alone ends (same voice and rate)."""
    whole = say_clip(f"{before} {after}", voice)
    alone = say_clip(before, voice)
    loud = np.flatnonzero(np.abs(alone) > 0.01)
    guess = int(loud[-1]) if loud.size else len(alone)
    win, best, cut = 160, None, guess
    for c in range(max(win, guess - 2400), min(len(whole) - win, guess + 2400), 40):
        e = float(np.mean(whole[c - win // 2: c + win // 2] ** 2))
        if best is None or e < best:
            best, cut = e, c
    gap = np.zeros(int(pause_ms * RATE / 1000), np.float32)
    out = np.concatenate([whole[:cut], gap, whole[cut:]])
    rest = np.flatnonzero(np.abs(whole[cut:]) > 0.01)
    return out, (cut + len(gap) + (int(rest[0]) if rest.size else 0)) / RATE


def segments(tr: Trace, conf: float, start_secs: float, min_volume: float, stop_secs: float) -> list[tuple[float, float]]:
    """Pipecat 1.12 VADAnalyzer state machine (as bargein_bench.silero_fires), returning (start, stop) pairs."""
    start_n, stop_n = round(start_secs / FRAME_S), round(stop_secs / FRAME_S)
    state, starting, stopping, out, cur = "QUIET", 0, 0, [], None
    i = 0
    while i < len(tr.t):
        j = i
        while j < len(tr.t) and tr.t[j] == tr.t[i]:
            speaking = tr.conf[j] >= conf and tr.vol[j] >= min_volume
            if speaking:
                if state == "QUIET":
                    state, starting = "STARTING", 1
                elif state == "STARTING":
                    starting += 1
                elif state == "STOPPING":
                    state, stopping = "SPEAKING", 0
            else:
                if state == "STARTING":
                    state, starting = "QUIET", 0
                elif state == "SPEAKING":
                    state, stopping = "STOPPING", 1
                elif state == "STOPPING":
                    stopping += 1
            j += 1
        if state == "STARTING" and starting >= start_n:
            state, starting, cur = "SPEAKING", 0, tr.t[i]
        if state == "STOPPING" and stopping >= stop_n:
            state, stopping = "QUIET", 0
            out.append((cur, tr.t[i]))
        i = j
    return out


def smart_turn_scores(st, audio: np.ndarray, segs: list[Seg], start_secs: float) -> None:
    """BaseSmartTurn._process_speech_segment at each stop: from the first start less pre-speech, last 8 s."""
    first = segs[0].start - start_secs - 0.5
    for s in segs:
        a = audio[max(0, int(first * RATE)): int(s.stop * RATE)]
        r = st._predict_endpoint(a[-8 * RATE:])
        s.p, s.complete = float(r["probability"]), r["prediction"] == 1
        alone = audio[max(0, int((s.start - start_secs - 0.5) * RATE)): int(s.stop * RATE)]
        s.complete_alone = st._predict_endpoint(alone[-8 * RATE:])["prediction"] == 1


class Recogniser:
    def __init__(self, kind: str, preroll_s: float, tail_pad_s: float):
        self.kind, self.preroll_s, self.tail_pad_s = kind, preroll_s, tail_pad_s
        if kind == "nemotron":
            from local_voice.config import load_config
            from local_voice.mlx_worker import MLXWorker

            cfg = load_config()
            self.worker = MLXWorker(name="bench-mlx")
            self.engine = cfg.adapter("stt", "nemotron").build()
            self.loop = asyncio.new_event_loop()
            self.loop.run_until_complete(self.worker.run(self.engine.load, guarded=False))

    def text(self, audio: np.ndarray, start: float, stop: float, words: str) -> str:
        if self.kind == "text":
            return words
        from local_voice.services.stt import transcribe_buffer
        a = audio[max(0, int((start - self.preroll_s) * RATE)): int(stop * RATE)]
        return self.loop.run_until_complete(transcribe_buffer(self.worker, self.engine, pcm16(a), guarded=False,
                                                              tail_pad_s=self.tail_pad_s))


def stand_in_words(before: str | None, after: str, segs: list[Seg], resume: float | None) -> list[str]:
    """--stt text: the words before the silence go to the last segment that ends before the speech resumes; the rest
    to the last segment."""
    if resume is None or before is None:
        return [after if k == len(segs) - 1 else "" for k in range(len(segs))]
    out = []
    for k, s in enumerate(segs):
        last_before = s.stop < resume and (k + 1 == len(segs) or segs[k + 1].start >= resume)
        out.append(before if last_before else (after if k == len(segs) - 1 else ""))
    return out


def simulate(segs: list[Seg], fallback: float, punct: float | None, veto: float | None = None,
             unpunct: bool = False, hold: str | None = None) -> float | None:
    """When the turn ends (seconds from the case start), or None if it never does within the clip."""
    vetoed = False
    for k, s in enumerate(segs):
        nxt = segs[k + 1].start if k + 1 < len(segs) else float("inf")
        said = any(x.text for x in segs[:k + 1])   # the strategy needs some final text in the turn, from any segment
        if not said:
            continue
        if s.complete_alone if vetoed else s.complete:
            wait = None
            if veto is not None and s.text and goes_on(s.text, unpunctuated=unpunct):
                wait = veto
            if hold and s.text and (holds(s.text) if hold == "all" else holds_q(s.text)):
                wait = max(wait or 0.0, fallback)          # wait for the silence fallback, as if Smart Turn were unsure
            if wait is not None:
                t_end, vetoed = s.stop + wait, True
            else:
                return s.stop
        else:
            t_end = s.stop + fallback
            if punct is not None and s.text and ends_turn(s.text):
                t_end = min(t_end, s.stop + punct)
        if t_end < nxt:
            return t_end
    return None


def turns_of(segs: list[Seg], fallback: float, hold: str | None) -> list[list[int]]:
    """The utterance cut into turns: each turn ends where simulate() says, and the next begins at the next VAD start."""
    out, k = [], 0
    while k < len(segs):
        e = simulate(segs[k:], fallback, None, None, False, hold)
        n = len(segs) - k if e is None else sum(1 for s in segs[k:] if s.start < e)
        out.append(list(range(k, k + max(1, n))))
        k += max(1, n)
    return out


def spliced_many(parts: list[str], voice: str, pause_ms: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """The parts spoken as one utterance with `pause_ms` of silence spliced in after each part but the last (cut as in
    spliced()), and each part's span in it (s)."""
    whole = say_clip(" ".join(parts), voice)
    cuts = []
    for i in range(1, len(parts)):
        alone = say_clip(" ".join(parts[:i]), voice)
        loud = np.flatnonzero(np.abs(alone) > 0.01)
        guess = int(loud[-1]) if loud.size else len(alone)
        win, best, cut = 160, None, guess
        for c in range(max(win, guess - 2400), min(len(whole) - win, guess + 2400), 40):
            e = float(np.mean(whole[c - win // 2: c + win // 2] ** 2))
            if best is None or e < best:
                best, cut = e, c
        cuts.append(max(cut, cuts[-1] + 1) if cuts else cut)
    gap = np.zeros(int(pause_ms * RATE / 1000), np.float32)
    pieces, spans, prev, shift = [], [], 0, 0
    for c in cuts + [len(whole)]:
        pieces.append(whole[prev:c])
        spans.append(((prev + shift) / RATE, (c + shift) / RATE))
        if c != len(whole):
            pieces.append(gap)
            shift += len(gap)
        prev = c
    return np.concatenate(pieces), spans


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


async def build_traces(cases: list[tuple[Case, Row]]) -> list[Trace]:
    rng = np.random.default_rng(0)
    return [await trace(c, "silence", rng) for c, _ in cases]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stt", choices=["text", "nemotron"], default="text")
    ap.add_argument("--carry", action="store_true", help="decode a segment that gave no words again with the next")
    ap.add_argument("--voices", default=",".join(VOICES))
    ap.add_argument("--set", choices=["pauses", "fillers"], default="pauses",
                    help="pauses: the original cases and rules; fillers: the hold rule on fillers and the rambling request")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from local_voice.config import load_config
    cfg = load_config()
    vad = cfg.vad
    stt_set = cfg.raw["stt"]["adapters"]["nemotron"]
    rec = Recogniser(args.stt, float(stt_set.get("preroll_s", 1.0)), float(stt_set.get("tail_pad_s", 0.3)))
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    st = LocalSmartTurnAnalyzerV3()
    st.set_sample_rate(RATE)

    import bargein_bench
    bargein_bench.LEAD_S, bargein_bench.TAIL_S = LEAD_S, TAIL_S   # trace() pads each case with these
    if args.set == "fillers":
        return run_fillers(args, vad, rec, st)
    cases: list[tuple[Case, Row]] = []
    for v in args.voices.split(","):
        for t in COMPLETE:
            a = say_clip(t, v)
            cases.append((Case(f"complete:{v}:{t}", "complete", a), Row(f"complete:{v}:{t}", "complete", v, t)))
        for before, after in PAUSED:
            for p in PAUSES_MS:
                a, resume = spliced(before, after, v, p)
                tt = f"{before} [{p} ms] {after}"
                cases.append((Case(f"paused:{v}:{p}:{before}", "paused", a),
                              Row(f"paused:{v}:{p}:{before}", "paused", v, tt, pause_ms=p, resume=resume)))
    traces = asyncio.run(build_traces(cases))

    rows = []
    for (case, row), tr in zip(cases, traces):
        loud = np.flatnonzero(np.abs(case.audio) > 0.01)
        row.speech_end = float((loud[-1] + 1) / RATE) if loud.size else 0.0
        audio = np.concatenate([np.zeros(int(LEAD_S * RATE), np.float32), case.audio,
                                np.zeros(int(TAIL_S * RATE), np.float32)])
        shift = LEAD_S   # trace times are from the case start; audio here starts LEAD_S earlier
        pairs = segments(tr, vad["confidence"], vad["start_secs"], vad["min_volume"], vad["stop_secs"])
        row.segs = [Seg(a, b) for a, b in pairs]
        if not row.segs:
            rows.append(row)
            continue
        for s in row.segs:
            s.start += shift
            s.stop += shift
        smart_turn_scores(st, audio, row.segs, vad["start_secs"])
        before, _, after = row.text.partition(f" [{row.pause_ms} ms] ") if row.kind == "paused" else (None, "", row.text)
        words = stand_in_words(before, after, row.segs, None if row.resume is None else row.resume + shift)
        carry_from = None
        for k, s in enumerate(row.segs):
            begin = s.start if carry_from is None else carry_from
            s.text = rec.text(audio, begin, s.stop, words[k])
            carry_from = (begin if (args.carry and not s.text) else None)
        row.transcript = " ".join(s.text for s in row.segs if s.text)
        for s in row.segs:
            s.start, s.stop = round(s.start - shift, 3), round(s.stop - shift, 3)
        for name, fb, punct, veto, unp in RULES:
            e = simulate(row.segs, fb, punct, veto, unp)
            row.ends[name] = None if e is None else round(e, 3)
        rows.append(row)

    with (args.out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
    lines = [f"# Turn-end bench (stt: {args.stt}{', carry' if args.carry else ''})", "",
             f"VAD {vad}; voices {args.voices}. Complete: end of speech to the turn's end. Paused: cut off = the turn "
             "ended before the speech after the planted silence began.", "",
             "| rule | complete: median / p90 / max (s) | complete waiting > 1 s | paused cut off (0.8 s / 1.5 s) |",
             "|---|---|---|---|"]
    comp = [r for r in rows if r.kind == "complete"]
    paus = [r for r in rows if r.kind == "paused"]
    for name, *_ in RULES:
        d = [r.ends[name] - r.speech_end for r in comp if r.ends.get(name) is not None]
        never = sum(1 for r in comp if r.ends.get(name) is None)
        cut = {p: sum(1 for r in paus if r.pause_ms == p and r.ends.get(name) is not None and r.resume is not None
                      and r.ends[name] < r.resume) for p in PAUSES_MS}
        n = {p: sum(1 for r in paus if r.pause_ms == p) for p in PAUSES_MS}
        lines.append(f"| {name} | {statistics.median(d):.2f} / {pct(d, 90):.2f} / {max(d):.2f}"
                     f"{f' ({never} never)' if never else ''} | {sum(1 for x in d if x > 1.0)}/{len(comp)} | "
                     f"{cut[800]}/{n[800]} / {cut[1500]}/{n[1500]} |")
    lines += ["", "## Paused cases cut off by the punct-0.8 rule", ""]
    for r in paus:
        e = r.ends.get("punct-0.8")
        if e is not None and r.resume is not None and e < r.resume:
            lines.append(f"- {r.voice} {r.pause_ms} ms: {r.text!r} -> {[s.text for s in r.segs]}")
    lines += ["", "## First words", ""]
    for r in paus:
        m = next((k for k in FIRST_WORD if r.text.startswith(k)), None)
        if m:
            got = r.transcript.lower().lstrip()
            ok = got.startswith(FIRST_WORD[m])
            lines.append(f"- {'kept' if ok else 'LOST'}: {r.voice} {r.pause_ms} ms: {r.transcript!r}")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return 0


def _events(case: Case, row: Row, tr: Trace, vad: dict, st) -> np.ndarray:
    """VAD segments and Smart Turn's verdicts for one case (times from the case start, as in main()); the padded audio."""
    loud = np.flatnonzero(np.abs(case.audio) > 0.01)
    row.speech_end = float((loud[-1] + 1) / RATE) if loud.size else 0.0
    audio = np.concatenate([np.zeros(int(LEAD_S * RATE), np.float32), case.audio, np.zeros(int(TAIL_S * RATE), np.float32)])
    pairs = segments(tr, vad["confidence"], vad["start_secs"], vad["min_volume"], vad["stop_secs"])
    row.segs = [Seg(a + LEAD_S, b + LEAD_S) for a, b in pairs]
    if row.segs:
        smart_turn_scores(st, audio, row.segs, vad["start_secs"])
    return audio


def _assign(parts: list[str], spans: list[tuple[float, float]], segs: list[Seg], lead: float) -> list[str]:
    """--stt text for several parts: each part's words go to the segment its speech overlaps most."""
    out = [[] for _ in segs]
    for text, (a, b) in zip(parts, spans):
        a, b = a + lead, b + lead
        best = max(range(len(segs)), key=lambda k: (min(b, segs[k].stop) - max(a, segs[k].start),
                                                    -abs((a + b) / 2 - segs[k].stop)))
        out[best].append(text)
    return [" ".join(x) for x in out]


def run_fillers(args, vad: dict, rec: Recogniser, st) -> int:
    cases: list[tuple[Case, Row, list[str] | None, list[tuple[float, float]] | None]] = []
    for v in args.voices.split(","):
        for t in COMPLETE + COMPLETE_DANGLING:
            kind = "complete" if t in COMPLETE else "dangling"
            cases.append((Case(f"{kind}:{v}:{t}", kind, say_clip(t, v)), Row(f"{kind}:{v}:{t}", kind, v, t), None, None))
        for before, after in FILLERS:
            for p in PAUSES_MS:
                a, resume = spliced(before, after, v, p)
                tt = f"{before} [{p} ms] {after}"
                cases.append((Case(f"filler:{v}:{p}:{before}", "filler", a),
                              Row(f"filler:{v}:{p}:{before}", "filler", v, tt, pause_ms=p, resume=resume), None, None))
        for p in RAMBLE_PAUSES_MS:
            a, spans = spliced_many(RAMBLE, v, p)
            cases.append((Case(f"ramble:{v}:{p}", "ramble", a),
                          Row(f"ramble:{v}:{p}", "ramble", v, f" [{p} ms] ".join(RAMBLE), pause_ms=p), RAMBLE, spans))
    traces = asyncio.run(build_traces([(c, r) for c, r, _, _ in cases]))
    rows, ramble_turns = [], []
    for (case, row, parts, spans), tr in zip(cases, traces):
        audio = _events(case, row, tr, vad, st)
        if not row.segs:
            rows.append(row)
            continue
        if parts is not None:
            words = _assign(parts, spans, row.segs, LEAD_S)
        else:
            before, _, after = row.text.partition(f" [{row.pause_ms} ms] ") if row.kind == "filler" else (None, "", row.text)
            words = stand_in_words(before, after, row.segs, None if row.resume is None else row.resume + LEAD_S)
        for k, sg in enumerate(row.segs):
            sg.text = rec.text(audio, sg.start, sg.stop, words[k])
        row.transcript = " ".join(sg.text for sg in row.segs if sg.text)
        for sg in row.segs:
            sg.start, sg.stop = round(sg.start - LEAD_S, 3), round(sg.stop - LEAD_S, 3)
        for name, fb, hold in HOLD_RULES:
            e = simulate(row.segs, fb, None, None, False, hold)
            row.ends[name] = None if e is None else round(e, 3)
            if row.kind == "ramble":
                turns = turns_of(row.segs, fb, hold)
                ramble_turns.append((row.voice, row.pause_ms, name,
                                    [" ".join(row.segs[i].text for i in t if row.segs[i].text) for t in turns]))
        rows.append(row)
    with (args.out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
    lines = [f"# Turn-end bench, fillers (stt: {args.stt})", "",
             f"VAD {vad}; voices {args.voices}. Finished: end of speech to the turn's end (20 requests, and 5 that end "
             "on a preposition). Fillers: cut off = the turn ended before the speech after the planted silence began. "
             "The rambling request: how many turns the one request became.", "",
             "| rule | finished: median / p90 / max (s) | ending on a preposition: median / max (s) | "
             "fillers cut off (0.8 s / 1.5 s) | the rambling request: turns (0.8 s / 1.5 s pauses) |", "|---|---|---|---|---|"]
    for name, *_ in HOLD_RULES:
        def lag(kind):
            return [r.ends[name] - r.speech_end for r in rows if r.kind == kind and r.ends.get(name) is not None]
        d, dd = lag("complete"), lag("dangling")
        fill = [r for r in rows if r.kind == "filler"]
        cut = {p: sum(1 for r in fill if r.pause_ms == p and r.ends.get(name) is not None and r.ends[name] < r.resume)
               for p in PAUSES_MS}
        n = {p: sum(1 for r in fill if r.pause_ms == p) for p in PAUSES_MS}
        ot = {p: [len(t) for v, pm, nm, t in ramble_turns if nm == name and pm == p] for p in RAMBLE_PAUSES_MS}
        lines.append(f"| {name} | {statistics.median(d):.2f} / {pct(d, 90):.2f} / {max(d):.2f} | "
                     f"{statistics.median(dd):.2f} / {max(dd):.2f} | {cut[800]}/{n[800]} / {cut[1500]}/{n[1500]} | "
                     f"{' '.join(map(str, ot[800]))} / {' '.join(map(str, ot[1500]))} |")
    lines += ["", "## The rambling request, turn by turn", ""]
    for v, pm, nm, t in ramble_turns:
        lines.append(f"- {nm}, {v}, {pm} ms: " + " | ".join(t))
    lines += ["", "## Fillers cut off by smart-turn-2.0", ""]
    for r in rows:
        e = r.ends.get("smart-turn-2.0")
        if r.kind == "filler" and e is not None and r.resume is not None and e < r.resume:
            lines.append(f"- {r.voice} {r.pause_ms} ms: {r.text!r}")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
