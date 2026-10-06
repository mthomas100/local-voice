"""The voice-quality bench: which TTS setting fixes what was heard in early live tests, by numbers (2026-10-05).

In an early live test (the browser page; Qwen3-TTS 1.7B CustomVoice "Ryan" through mlx-audio 0.5.7, streaming_interval
0.32 s, temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05, one generation per sentence, no instruct) the
voice at times laughed, swung up and down in pitch and slurred. The same test generated "Hey!" alone, split a path
mid-word ("I run from ~/." then the rest of the path), and gave two first generations 0.94 and 1.07 s of leading
silence: the model's silence ran past the trim's 0.8 s scan, so services/tts.py played it untrimmed. Quality is
measured by machine, never by asking anyone to listen, rate or label (a project rule). This bench renders a fixed text set under each
setting several times, scores every clip, and says which setting fixes what within the latency budget.

Stages (each resumable: a clip whose output exists is skipped; a stop leaves everything already written usable):
  render  GPU. Each setting loads the adapter class config.yaml names (`impl`) with the setting's merged settings,
          load(), warm(), then says every text R times on the main thread as production would: the reply's
          sentences (tts_bench_texts.yaml; SimpleTextAggregator's split where an item gives none), grouped by
          speech_text.group_sentences, filtered by speech_text.speakable, one engine.stream() per group, the first
          one's leading silence trimmed exactly as services/tts.py's run_tts does (Pipecat's detect_speech_onset,
          30 ms kept, scan up to 0.8 s), the generations joined. Per generation it records the chunk boundaries, the
          time to the first chunk and to the first audio out of the trim, the total, engine.last (frames,
          max_tokens, hit_cap) and the leading silence before the trim; per setting the load average. A clip is a
          24 kHz mono 16-bit WAV plus a JSON sidecar: ../state/tts-bench/<stamp>/<setting>/<id>-<r>.wav/.json.
          Seeds: mx.random.seed(crc32("<id>:<r>")) before each reply, the same across settings (common random
          numbers: a setting's effect is not drowned by sampling luck). A clip whose engine settings and generation
          texts equal an earlier setting's (a one-sentence reply under a grouping setting) is copied from it
          ("reused_from"), not rendered again: the same inputs give the same distribution.
  asr     GPU. Every clip through the project's Parakeet adapter (local_voice.engines.parakeet:ParakeetEngine,
          mlx-community/parakeet-tdt-0.6b-v3) at 16 kHz. mlx-audio 0.5.7's AlignedResult carries every token's start
          and duration (stt/models/nemo/alignment.py), so the words keep their times. WER against the text said,
          normalised (lower case, no punctuation, digits written as words on both sides), with the substituted,
          deleted and inserted words. Slurring shows as errors.
  score   CPU, in the scorer venv: tools/tts_bench_scorers.py (pitch with Praat, an AudioSet tagger for laughter and
          other non-speech sounds, UTMOS22 for naturalness, spectral flux at the chunk seams, pauses, pace). Two
          downloads, network only, 2026-10-05 (its docstring has the detail): MIT/ast-finetuned-audioset-10-10-0.4593
          from Hugging Face, 346 MB (the AudioSet tagger with every laughter and breath class, one safetensors file,
          plain transformers), in ~/.cache/huggingface/hub; UTMOS22 strong from tarepan/SpeechMOS v1.2.0 (GitHub
          release v1.0.0), 411 MB plus 25 KB of code (one checkpoint, no fairseq; UTMOSv2 is over 1 GB), in
          ~/.cache/torch/hub.
  report  CPU. results.jsonl (a row per clip, every number) and report.md in the run dir: per setting the medians,
          p90 and worst of each metric, the failure counts (WER > 0.15, laughter > 0.5, a pitch jump > 8 st,
          MOS < 3.0, a runaway, leading silence > 0.3 s before the trim), the first-chunk and first-audio latency,
          a ranking that names the best setting whose first audio comes on average within 0.2 s of the baseline's
          on the 17 reply items (text for text: a conversation's mix, 13 of them several sentences long, so a
          grouping setting's wait counts), and a model-or-setting table (each failure under the baseline, the best
          1.7B setting, the 0.6B and Kokoro).
  all     render, asr, score, report.        plan: the clips and the GPU time, nothing runs.
  texts   the text set as it splits and groups; --check-log compares the reply items with a session log.

The grid: one factor at a time around production (config.yaml tts.adapters.qwen3): temperature 0.7 and 0.5;
grouping (min_words 4; max_sentences 2; max_sentences 3); streaming_interval 0.64, 1.0 and none (mlx-audio's
stream=False: Qwen3TTSEngine hard-codes stream=True in _kwargs, so the bench subclasses only that); instruct (a calm,
natural style line); the 0.6B CustomVoice; Kokoro with am_michael (a steady control). `--combo NAME=a+b` adds a
combination (the overrides of a and b merged); `--settings` picks some (baseline always runs, first).

GPU: render and asr refuse to start unless ../measure/bench/gpu_clear.sh exits 0, check it again
before each setting and every --recheck-s seconds (60), and stop cleanly when it says BUSY; run the same command again
to resume. Nothing here calls the LLM or takes a hold. The neural scorers load only after the same check.

    cd orchestrator
    .venv/bin/python tools/tts_quality_bench.py plan
    .venv/bin/python tools/tts_quality_bench.py all                       # a new run dir under ../state/tts-bench/
    .venv/bin/python tools/tts_quality_bench.py all --run-dir ../state/tts-bench/<stamp>   # resume it
    .venv/bin/python tools/tts_quality_bench.py report --latest

GPU time (plan, 2026-10-05): the default grid is 38 texts x 3 renders x 12 settings = 1,368 clips, 264 of them copies
(one-sentence replies under the grouping settings), about 35 min of rendering plus 1 min of ASR on an idle Mac at the
speeds measured in measure/09-phase1-report.md §3 (SPEED below). Kokoro is deterministic, so its three renders agree;
they cost 0.7 min. --reps 2: about 24 min. The score stage is CPU only but long: about 50 min for the default grid
(the AudioSet tagger, 0.19 s per window in the 2026-10-05 CPU smoke), about 25 min with --tag-hop 1.0.

The first audio of a grouping setting also waits for the LLM to write the sentences it groups: each clip records the
reply characters needed before its first generation could start, and the report turns the extra characters into
seconds at --llm-cps (200 chars/s: a measured live reply, 167 characters in 0.85 s from the first answer token to the
end of the reply).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import inspect
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import wave
import zlib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TOOLS = Path(__file__).resolve().parent
ORCH = TOOLS.parent
REPO = ORCH.parent
for _p in (str(ORCH), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tts_bench_scorers as scorers  # noqa: E402
from local_voice.config import deep_merge, load_impl  # noqa: E402
from local_voice.engines.base import float_to_pcm16  # noqa: E402
from local_voice.speech_text import group_sentences, speakable  # noqa: E402

CONFIG = ORCH / "config.yaml"
TEXTS = TOOLS / "tts_bench_texts.yaml"
RUNS = REPO / "state" / "tts-bench"
GPU_CLEAR = str(REPO / "measure" / "bench" / "gpu_clear.sh")
SCORER_PY = TOOLS / ".venv-bench" / "bin" / "python"
RATE = 24000
BENCH_VERSION = 1
EXIT_OK, EXIT_ERR, EXIT_BUSY = 0, 1, 3
KINDS = ("reply", "short", "list", "question", "dash", "path")
LLM_CPS = 200.0
BUDGET_S = 0.2

STYLE = "Speak in a calm, natural, steady voice."
# `ls ~/.cache/huggingface/hub | grep -i qwen3`, 2026-10-05; its config lists the speaker "ryan" too
QWEN_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
# an American male voice of the cached mlx-community/Kokoro-82M-bf16: target quality B, grade C+ in its VOICES.md, with
# am_fenrir and am_puck the best graded of the nine am_ voices
KOKORO_VOICE = "am_michael"

GRID: dict[str, dict[str, Any]] = {
    "baseline": {},
    "temp0.7": {"settings": {"temperature": 0.7}},
    "temp0.5": {"settings": {"temperature": 0.5}},
    "minwords4": {"grouping": {"min_words": 4}},
    "maxsent2": {"grouping": {"max_sentences": 2}},
    "maxsent3": {"grouping": {"max_sentences": 3}},
    "interval0.64": {"settings": {"streaming_interval": 0.64}},
    "interval1.0": {"settings": {"streaming_interval": 1.0}},
    "nostream": {"stream": False},
    "instruct": {"settings": {"instruct": STYLE}},
    "qwen0.6b": {"settings": {"model": QWEN_06B}},
    "kokoro": {"adapter": "kokoro", "settings": {"voice": KOKORO_VOICE}},
}


# ---------------------------------------------------------------------------------------------------- config, grid

def read_config(path: Path = CONFIG) -> dict:
    """config.yaml with config.local.yaml merged over it, as local_voice.config.load_config merges them (the bench
    needs only the tts and stt sections, not the whole checked Config)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    local = Path(path).with_name("config.local.yaml")
    if local.exists():
        raw = deep_merge(raw, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return raw


@dataclass
class Setting:
    name: str
    adapter: str
    impl: str
    settings: dict[str, Any]
    stream: bool = True
    min_words: int = 0
    max_sentences: int = 1
    override: dict[str, Any] = field(default_factory=dict)

    def engine_key(self) -> str:
        return json.dumps({"impl": self.impl, "settings": self.settings, "stream": self.stream}, sort_keys=True)

    def to_json(self) -> dict:
        return asdict(self)


ONE_PER_SENTENCE = Setting("one-per-sentence", "", "", {})


def make_setting(name: str, spec: dict, tts: dict) -> Setting:
    adapter = spec.get("adapter", tts["adapter"])
    if adapter not in tts["adapters"]:
        raise ValueError(f"setting {name}: no adapter {adapter!r} in config.yaml tts.adapters")
    conf = copy.deepcopy(tts["adapters"][adapter])
    impl = conf.pop("impl")
    conf = deep_merge(conf, spec.get("settings") or {})
    g = spec.get("grouping") or {}
    unknown = set(spec) - {"adapter", "settings", "grouping", "stream"}
    if unknown or set(g) - {"min_words", "max_sentences"}:
        raise ValueError(f"setting {name}: unknown keys {sorted(unknown | (set(g) - {'min_words', 'max_sentences'}))}")
    return Setting(name=name, adapter=adapter, impl=impl, settings=conf, stream=bool(spec.get("stream", True)),
                   min_words=int(g.get("min_words", 0)), max_sentences=int(g.get("max_sentences", 1)),
                   override=copy.deepcopy(spec))


def expand_grid(tts: dict, names: Iterable[str] | None = None, combos: Iterable[str] = ()) -> list[Setting]:
    """The named settings (all of GRID by default, plus any combos), baseline first. A combo is "NAME=a+b": the
    overrides of a and b merged (one adapter at most)."""
    grid = dict(GRID)
    combo_names = []
    for c in combos:
        name, sep, parts = c.partition("=")
        if not sep or not name or not parts:
            raise ValueError(f"--combo {c!r}: write it NAME=a+b")
        if name in grid:
            raise ValueError(f"--combo {name}: the name is taken")
        spec: dict = {}
        for part in parts.split("+"):
            if part not in grid:
                raise ValueError(f"--combo {name}: no setting {part!r}")
            if spec.get("adapter") and grid[part].get("adapter") and spec["adapter"] != grid[part]["adapter"]:
                raise ValueError(f"--combo {name}: two adapters ({spec['adapter']}, {grid[part]['adapter']})")
            spec = deep_merge(spec, grid[part])
        grid[name] = spec
        combo_names.append(name)
    chosen = list(names) if names else list(GRID) + combo_names
    unknown = [n for n in chosen if n not in grid]
    if unknown:
        raise ValueError(f"unknown settings {unknown}; known: {', '.join(grid)}")
    chosen = ["baseline"] + [n for n in dict.fromkeys(chosen) if n != "baseline"]
    return [make_setting(n, grid[n], tts) for n in chosen]


def no_stream(cls: type) -> type:
    """The adapter with mlx-audio's stream=False: the sentence is generated whole and decoded at once (one chunk).
    Only _kwargs changes, so the sampling, max_tokens and the cleanup in stream() stay the adapter's own."""
    base = cls._kwargs

    def _kwargs(self, text):
        kw = base(self, text)
        kw["stream"] = False
        kw.pop("streaming_interval", None)
        return kw

    return type(f"{cls.__name__}NoStream", (cls,), {"_kwargs": _kwargs})


def build_engine(s: Setting):
    cls = load_impl(s.impl)
    if not s.stream:
        if not hasattr(cls, "_kwargs"):
            raise ValueError(f"setting {s.name}: stream: false needs an adapter whose _kwargs builds generate()'s "
                             f"arguments (Qwen3TTSEngine); {s.impl} has none")
        cls = no_stream(cls)
    return cls(copy.deepcopy(s.settings))


# ---------------------------------------------------------------------------------------------------- texts

ITEM_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass
class Item:
    id: str
    kind: str
    text: str
    sentences: list[str]
    logged: bool = False
    turn: int | None = None


def aggregator_split(text: str, chunk: int = 4) -> list[str]:
    """Sentences as Pipecat 1.12's SimpleTextAggregator emits them from a stream: fed `chunk` characters at a time
    (about an LLM token), then flushed at the end of the response. Production's PatternPairAggregator (code fences)
    splits plain text through the same _check_sentence_with_lookahead."""
    from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

    async def run() -> list[str]:
        agg, out = SimpleTextAggregator(), []
        for i in range(0, len(text), chunk):
            async for a in agg.aggregate(text[i:i + chunk]):
                if a.text.strip():
                    out.append(a.text.strip())
        last = await agg.flush()
        if last is not None and last.text.strip():
            out.append(last.text.strip())
        return out

    return asyncio.run(run())


def load_texts(path: Path = TEXTS) -> list[Item]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if raw.get("version") != 1:
        raise ValueError(f"{path}: version must be 1")
    items: list[Item] = []
    for d in raw.get("items") or []:
        iid, kind, text = str(d.get("id", "")), str(d.get("kind", "")), str(d.get("text", "")).strip()
        if not ITEM_ID.match(iid) or any(i.id == iid for i in items):
            raise ValueError(f"{path}: bad or repeated id {iid!r}")
        if kind not in KINDS or not text:
            raise ValueError(f"{path}: {iid}: kind must be one of {KINDS} and text non-empty")
        logged = bool(d.get("sentences"))
        sents = [str(s).strip() for s in d["sentences"]] if logged else aggregator_split(text)
        items.append(Item(id=iid, kind=kind, text=text, sentences=[s for s in sents if s], logged=logged,
                          turn=d.get("turn")))
    return items


def select_items(items: list[Item], ids: str | None = None, kinds: str | None = None) -> list[Item]:
    if ids:
        want = [x.strip() for x in ids.split(",") if x.strip()]
        missing = [w for w in want if w not in {i.id for i in items}]
        if missing:
            raise ValueError(f"no items {missing}")
        items = [i for i in items if i.id in want]
    if kinds:
        items = [i for i in items if i.kind in {k.strip() for k in kinds.split(",")}]
    return items


def generations(item: Item, s: Setting) -> list[str]:
    """What the voice generates for this reply, one text per engine.stream() call: the production grouping rule
    (speech_text.group_sentences), then the production text filter on each (speech_text.speakable, which
    services/tts.py installs as Pipecat's text filter). A text left empty is not said."""
    groups = group_sentences(item.sentences, min_words=s.min_words, max_sentences=s.max_sentences)
    return [t for t in (speakable(g) for g in groups) if t]


def llm_chars(item: Item, s: Setting) -> int:
    """Characters of the reply that had to exist before the first generation could start: the first group's
    sentences, plus the space and the next sentence's first character when one follows (the aggregator's lookahead)."""
    sents = [x.strip() for x in item.sentences if x.strip()]
    groups = group_sentences(sents, min_words=s.min_words, max_sentences=s.max_sentences)
    if not groups:
        return 0
    k = next((k for k in range(1, len(sents) + 1) if " ".join(sents[:k]) == groups[0]), len(sents))
    return len(groups[0]) + (2 if k < len(sents) else 0)


def session_replies(log: Path, turns: Path, window: tuple[str, str]) -> list[dict]:
    """A session's replies as they reached the TTS: every "Generating TTS [...]" line of the orchestrator log in the
    window, given to the last turn of the turn log whose t_start it follows (the turn log's times are local, like the
    log's)."""
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) .*Generating TTS \[(.*)\]\s*$")
    t0, t1 = (datetime.strptime(w, "%Y-%m-%d %H:%M:%S") for w in window)
    starts = []
    for line in Path(turns).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        ts = datetime.fromisoformat(d["t_start"]).replace(tzinfo=None)
        if t0 <= ts <= t1:
            starts.append((ts, int(d["turn"])))
    starts.sort()
    out: dict[int, list[str]] = {}
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        if not (t0 <= ts <= t1):
            continue
        turn = next((n for st, n in reversed(starts) if st <= ts), None)
        if turn is not None:
            out.setdefault(turn, []).append(m.group(2))
    return [{"turn": t, "sentences": s} for t, s in sorted(out.items())]


# ---------------------------------------------------------------------------------------------------- render

@dataclass
class Trim:
    enabled: bool = True
    keep_ms: float = 30.0
    max_scan_s: float = 0.8


def production_trim(tts: dict) -> Trim:
    """services/tts.py's trim as production builds it: the switch and the kept milliseconds from config.yaml
    (runtime.make_tts passes both), the scan window from MLXTTSService's own default."""
    from local_voice.services.tts import MLXTTSService

    p = inspect.signature(MLXTTSService.__init__).parameters
    return Trim(enabled=bool(tts.get("trim_leading_silence", p["trim_leading_silence"].default)),
                keep_ms=float(tts.get("keep_before_onset_ms", p["keep_before_onset_ms"].default)),
                max_scan_s=float(p["max_lead_scan_secs"].default))


def onset_s(audio: np.ndarray, rate: int) -> float | None:
    """Pipecat's detect_speech_onset (the TTFA metric's and the trim's detector) over a whole buffer, in seconds."""
    from pipecat.audio.utils import detect_speech_onset

    o = detect_speech_onset(float_to_pcm16(np.asarray(audio, dtype=np.float32)), rate)
    return None if o is None else o / rate


def service_trim(chunks: list[np.ndarray], rate: int, trim: Trim, first_of_reply: bool) -> dict:
    """services/tts.py run_tts's leading-silence trim replayed on a generation's chunks: only the reply's first
    generation is scanned; the chunks are buffered until Pipecat's detector confirms an onset or the buffer reaches
    the scan window, then the buffer goes out from `keep_ms` before the onset (or whole, when none was found); a
    generation that ends inside the window without an onset is played as it is after its last chunk.
    Returns the samples cut from the front, the chunk at which the first audio leaves (len(chunks) = after the end),
    whether an onset was found, and how much was scanned."""
    from pipecat.audio.utils import detect_speech_onset

    if not (trim.enabled and first_of_reply):
        first = next((i for i, c in enumerate(chunks) if c.size), None)
        return {"start": 0, "out_chunk": first, "onset_found": None, "scanned_s": 0.0}
    lead = bytearray()
    for i, c in enumerate(chunks):
        lead.extend(float_to_pcm16(c))
        onset = detect_speech_onset(bytes(lead), rate)
        if onset is None and len(lead) < trim.max_scan_s * rate * 2:
            continue
        start = 0 if onset is None else max(0, onset - int(trim.keep_ms / 1000 * rate))
        return {"start": start, "out_chunk": i, "onset_found": onset is not None, "scanned_s": len(lead) / 2 / rate}
    return {"start": 0, "out_chunk": len(chunks) if lead else None, "onset_found": False,
            "scanned_s": len(lead) / 2 / rate}


def run_stream(engine, text: str) -> tuple[list[np.ndarray], list[float], float]:
    """One generation: its chunks, when each arrived (s after the stream() call) and the total time."""
    chunks, arrivals = [], []
    t0 = time.perf_counter()
    gen = engine.stream(text)
    try:
        for c in gen:
            arrivals.append(time.perf_counter() - t0)
            chunks.append(np.asarray(c, dtype=np.float32).reshape(-1))
    finally:
        getattr(gen, "close", lambda: None)()
    return chunks, arrivals, time.perf_counter() - t0


def stable_seed(item_id: str, r: int) -> int:
    return zlib.crc32(f"{item_id}:{r}".encode()) & 0x7FFFFFFF


def seed_mlx(seed: int) -> bool:
    """Seed MLX's global generator when an adapter has loaded MLX (the fakes never do)."""
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return False
    mx.random.seed(seed)
    return True


def control_points(gens: list[dict]) -> list[int]:
    """Mid-chunk points to compare the seams with: the middle of every stretch between a generation's start, its
    seams and its end; for a generation with no seams, the middles of a 0.32 s grid from its start."""
    out = []
    for g in gens:
        a, b = g["start"], g["start"] + g["samples"]
        if g["seams"]:
            bounds = [a] + g["seams"] + [b]
            out += [(x + y) // 2 for x, y in zip(bounds, bounds[1:])]
        else:
            step = int(scorers.CONTROL_GRID_S * RATE)
            out += list(range(a + step // 2, b, step))
    return out


def _to_rate(clip: np.ndarray, gens: list[dict], rate: int) -> tuple[np.ndarray, list[dict]]:
    f = RATE / rate
    for g in gens:
        g["start"], g["samples"] = int(round(g["start"] * f)), int(round(g["samples"] * f))
        g["seams"] = [int(round(b * f)) for b in g["seams"]]
        g["chunk_samples"] = [int(round(n * f)) for n in g["chunk_samples"]]
    return scorers.resample(clip, rate, RATE), gens


def render_clip(engine, item: Item, s: Setting, r: int, trim: Trim) -> tuple[np.ndarray, dict]:
    """One reply under one setting: every generation in order, the first trimmed as production trims it, joined."""
    texts = generations(item, s)
    seed = stable_seed(item.id, r)
    seeded = seed_mlx(seed)
    rate = int(engine.sample_rate)
    parts, gens, pos = [], [], 0
    for gi, text in enumerate(texts):
        chunks, arrivals, total = run_stream(engine, text)
        sizes = [int(c.size) for c in chunks]
        raw = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        t = service_trim(chunks, rate, trim, first_of_reply=gi == 0)
        audio = raw[t["start"]:]
        cuts = np.cumsum(sizes)[:-1] if len(sizes) > 1 else []
        seams = [int(c) - t["start"] for c in cuts if 0 < int(c) - t["start"] < audio.size]
        lead = onset_s(raw, rate)
        out_at = t["out_chunk"]
        last = dict(getattr(engine, "last", None) or {})
        gens.append({"text": text, "start": pos, "samples": int(audio.size), "chunk_samples": sizes,
                     "seams": [pos + b for b in seams],
                     "first_chunk_s": arrivals[0] if arrivals else None,
                     "first_out_s": None if out_at is None else (arrivals[out_at] if out_at < len(arrivals) else total),
                     "total_s": total, "lead_raw_s": lead if lead is not None else raw.size / rate,
                     "speech_found": lead is not None, "trimmed_s": t["start"] / rate,
                     "trim_onset_found": t["onset_found"], "trim_scanned_s": t["scanned_s"],
                     **{k: last.get(k) for k in ("frames", "max_tokens", "hit_cap", "finished")}})
        parts.append(audio)
        pos += int(audio.size)
    clip = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if rate != RATE:
        clip, gens = _to_rate(clip, gens, rate)
    heard = onset_s(clip, RATE)
    said = " ".join(texts)
    g0 = gens[0] if gens else {}
    side = {"bench_version": BENCH_VERSION, "item": item.id, "kind": item.kind, "r": r, "setting": s.name,
            "seed": seed, "seeded": seeded, "sample_rate": RATE, "samples": int(clip.size),
            "duration_s": clip.size / RATE, "text": said, "n_words": len(normalise(said)), "n_chars": len(said),
            "generations_n": len(gens), "llm_chars_first": llm_chars(item, s),
            "llm_chars_first_one_per_sentence": llm_chars(item, ONE_PER_SENTENCE),
            "first_chunk_s": g0.get("first_chunk_s"), "first_out_s": g0.get("first_out_s"),
            "lead_raw_s": g0.get("lead_raw_s"), "trim_onset_found": g0.get("trim_onset_found"),
            "lead_heard_s": heard if heard is not None else clip.size / RATE,
            "max_lead_raw_s": max((g["lead_raw_s"] for g in gens), default=None),
            "hit_cap": any(bool(g.get("hit_cap")) for g in gens),
            "seams": [b for g in gens for b in g["seams"]], "controls": control_points(gens),
            "gen_starts": [g["start"] for g in gens], "generations": gens}
    return clip, side


def write_wav(path: Path, audio: np.ndarray, rate: int = RATE) -> None:
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(float_to_pcm16(np.asarray(audio, dtype=np.float32)))
    tmp.replace(path)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    tmp.replace(path)


def clip_path(run_dir: Path, setting: str, item_id: str, r: int, suffix: str = ".json") -> Path:
    return Path(run_dir) / setting / f"{item_id}-{r}{suffix}"


def gpu_check(cmd: str | None) -> tuple[bool, str]:
    return scorers.gpu_check(cmd)


def write_manifest(run_dir: Path, settings: list[Setting], items: list[Item], reps: int, trim: Trim) -> dict:
    """run.json: what this run renders. A resumed run may add settings and items, never change one: the report would
    otherwise mix clips of two definitions under one name."""
    path = Path(run_dir) / "run.json"
    now = datetime.now().isoformat(timespec="seconds")
    new = {"bench_version": BENCH_VERSION, "reps": reps, "trim": asdict(trim),
           "settings": [s.to_json() for s in settings], "items": [asdict(i) for i in items]}
    if path.exists():
        old = json.loads(path.read_text())
        for key, label in (("settings", "name"), ("items", "id")):
            have = {d[label]: d for d in old[key]}
            for d in new[key]:
                if d[label] in have and have[d[label]] != d:
                    raise ValueError(f"{key[:-1]} {d[label]} changed since {path} was written; use a new run dir")
            new[key] = old[key] + [d for d in new[key] if d[label] not in have]
        new["reps"] = max(int(old.get("reps", 0)), reps)
        new["created"] = old.get("created", now)
    else:
        new["created"] = now
    new["updated"] = now
    write_json(path, new)
    return new


def render_key(item: Item, s: Setting) -> str:
    return json.dumps([s.engine_key(), generations(item, s)])


def reuse_source(run_dir: Path, settings: list[Setting], s: Setting, item: Item, r: int) -> str | None:
    key = render_key(item, s)
    for o in settings:
        if o.name == s.name:
            break
        if render_key(item, o) == key and clip_path(run_dir, o.name, item.id, r).exists():
            return o.name
    return None


def _release() -> None:
    """After the caller has dropped its last reference to an engine: collect it and hand MLX's cached buffers back
    (this Mac shares the GPU's memory with the LLM rig), so the next setting's model never loads beside the last."""
    gc.collect()
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        mx.clear_cache()


def render(run_dir: Path, settings: list[Setting], items: list[Item], reps: int, *, trim: Trim,
           gpu_clear: str | None = GPU_CLEAR, recheck_s: float = 60.0, reuse: bool = True,
           log: Callable[[str], None] = print) -> int:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, settings, items, reps, trim)
    ok, line = gpu_check(gpu_clear)
    log(f"gpu_clear: {line}")
    if not ok:
        log("render: refusing to start: gpu_clear.sh did not exit 0")
        return EXIT_BUSY
    for s in settings:
        sdir = run_dir / s.name
        sdir.mkdir(exist_ok=True)
        todo, reused = [], 0
        for item in items:
            for r in range(reps):
                if clip_path(run_dir, s.name, item.id, r).exists():
                    continue
                src = reuse_source(run_dir, settings, s, item, r) if reuse else None
                if src:
                    shutil.copyfile(clip_path(run_dir, src, item.id, r, ".wav"),
                                    clip_path(run_dir, s.name, item.id, r, ".wav"))
                    side = json.loads(clip_path(run_dir, src, item.id, r).read_text())
                    side.update(setting=s.name, reused_from=src)
                    write_json(clip_path(run_dir, s.name, item.id, r), side)
                    reused += 1
                else:
                    todo.append((item, r))
        if not todo:
            log(f"render {s.name}: nothing to render ({reused} reused)")
            continue
        ok, line = gpu_check(gpu_clear)
        if not ok:
            log(f"render: stopped before {s.name}: {line}; run the same command again to resume")
            return EXIT_BUSY
        meta_path = sdir / "setting.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
        run = {"started": datetime.now().isoformat(timespec="seconds"), "loadavg_start": os.getloadavg(),
               "todo": len(todo), "reused": reused, "rendered": 0, "errors": []}
        meta.update(setting=s.to_json())
        meta["runs"].append(run)
        engine = None
        try:
            t = time.monotonic()
            engine = build_engine(s)
            engine.load()
            run["load_s"] = round(time.monotonic() - t, 3)
            t = time.monotonic()
            engine.warm()
            run["warm_s"] = round(time.monotonic() - t, 3)
            run["engine"] = engine.describe()
        except Exception as e:  # noqa: BLE001 - one setting that cannot load must not end the run
            run["errors"].append(f"load: {e!r}")
            write_json(meta_path, meta)
            log(f"render {s.name}: could not load: {e!r}")
            engine = None
            _release()
            continue
        log(f"render {s.name}: {len(todo)} clips ({reused} reused), loaded in {run['load_s']} s")
        last_check = time.monotonic()
        stopped = False
        for item, r in todo:
            if time.monotonic() - last_check >= recheck_s:
                ok, line = gpu_check(gpu_clear)
                last_check = time.monotonic()
                if not ok:
                    log(f"render: stopped in {s.name}: {line}; run the same command again to resume")
                    stopped = True
                    break
            try:
                audio, side = render_clip(engine, item, s, r, trim)
            except Exception as e:  # noqa: BLE001 - a failed clip is recorded and retried on resume
                run["errors"].append(f"{item.id}-{r}: {e!r}")
                log(f"render {s.name} {item.id}-{r}: {e!r}")
                continue
            side["engine"] = run.get("engine")
            write_wav(clip_path(run_dir, s.name, item.id, r, ".wav"), audio)
            write_json(clip_path(run_dir, s.name, item.id, r), side)
            run["rendered"] += 1
        run["loadavg_end"] = os.getloadavg()
        run["finished"] = datetime.now().isoformat(timespec="seconds")
        run["stopped"] = stopped
        write_json(meta_path, meta)
        engine = None
        _release()
        if stopped:
            return EXIT_BUSY
    return EXIT_OK


# ---------------------------------------------------------------------------------------------------- asr, WER

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    for value, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= value:
            head, rest = divmod(n, value)
            return number_words(head) + " " + name + ("" if rest == 0 else " " + number_words(rest))
    return str(n)


_SAME = {"ok": "okay"}


def normalise(text: str) -> list[str]:
    """Words for the WER: lower case; punctuation gone, and a dash, slash, dot or underscore between characters
    splits words ("scout-and-plan", "atlas/journal", "README.md"); digits written out as words on both sides ("6" and
    "six" match, "27b" is "twenty seven b"); apostrophes kept inside words ("didn't"); "ok" is "okay"."""
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = t.replace("%", " percent ").replace("&", " and ")
    t = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", t)
    t = re.sub(r"[^a-z0-9']+", " ", t)
    out: list[str] = []
    for w in t.split():
        w = w.strip("'")
        if not w:
            continue
        if w.isdigit():
            out.extend(number_words(int(w)).split())
        else:
            out.append(_SAME.get(w, w))
    return out


def wer(ref: list[str], hyp: list[str]) -> dict:
    """Word error rate by Levenshtein alignment, with the edit script: substitutions [said in the text, heard],
    deletions (in the text, not heard) and insertions (heard, not in the text)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    sub, dele, ins = [], [], []
    i, j = n, m
    # ties: a match, then an insertion or deletion, then a substitution ("folder" -> "holder, please", not "please")
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif j and d[i, j] == d[i, j - 1] + 1:
            ins.append(hyp[j - 1])
            j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            dele.append(ref[i - 1])
            i -= 1
        else:
            sub.append([ref[i - 1], hyp[j - 1]])
            i, j = i - 1, j - 1
    errors = len(sub) + len(dele) + len(ins)
    return {"wer": errors / n if n else float(m > 0), "ref_words": n, "errors": errors,
            "sub": sub[::-1], "del": dele[::-1], "ins": ins[::-1]}


def tokens_to_words(tokens: list[tuple[str, float, float]]) -> list[dict]:
    """Parakeet's subword tokens (text, start, end) joined into words with times: a token whose decoded text starts
    with a space starts a word (mlx-audio keeps SentencePiece's word marker as a leading space)."""
    words: list[dict] = []
    new = True
    for text, start, end in tokens:
        piece = text.strip()
        if text[:1].isspace() or not words:
            new = True
        if not piece:
            new = True
            continue
        if new or not words:
            words.append({"w": piece, "start": float(start), "end": float(end)})
            new = False
        else:
            words[-1]["w"] += piece
            words[-1]["end"] = float(end)
    return words


def aligned_transcribe(engine, audio16: np.ndarray) -> tuple[str, list[dict]]:
    """ParakeetEngine.transcribe, keeping the token times the adapter drops: mlx-audio's AlignedResult has
    sentences[].tokens[] with start and duration in seconds from the start of the audio (0.5.7)."""
    import mlx.core as mx

    res = engine.model.generate(mx.array(np.ascontiguousarray(audio16, dtype=np.float32)))
    mx.clear_cache()
    text = " ".join(str(getattr(res, "text", "") or "").split())
    toks = [(str(t.text), float(t.start), float(t.end))
            for snt in (getattr(res, "sentences", None) or []) for t in snt.tokens]
    return text, tokens_to_words(toks)


def score_asr(ref_text: str, hyp_text: str, words: list[dict] | None) -> dict:
    ref, hyp = normalise(ref_text), normalise(hyp_text)
    return {"hyp": hyp_text, "words": words, "ref_norm": ref, "hyp_norm": hyp, **wer(ref, hyp)}


def asr(run_dir: Path, stt: dict, *, gpu_clear: str | None = GPU_CLEAR, recheck_s: float = 60.0,
        log: Callable[[str], None] = print, transcribe: Callable | None = None) -> int:
    """Transcribe every rendered clip that has no .asr.json yet. `transcribe` (tests) replaces the model call."""
    sides = scorers.clip_sidecars(run_dir)
    todo = [p for p in sides if not p.with_suffix(".asr.json").exists()]
    if not todo:
        log(f"asr: every clip in {run_dir} is transcribed")
        return EXIT_OK
    engine = None
    if transcribe is None:
        ok, line = gpu_check(gpu_clear)
        log(f"gpu_clear: {line}")
        if not ok:
            log("asr: refusing to start: gpu_clear.sh did not exit 0")
            return EXIT_BUSY
        conf = copy.deepcopy(stt["adapters"]["parakeet"])
        engine = load_impl(conf.pop("impl"))(conf)
        t = time.monotonic()
        engine.load()
        engine.warm()
        log(f"asr: {conf['model']} loaded and warmed in {time.monotonic() - t:.1f} s; {len(todo)} clips")
        transcribe = lambda a16: aligned_transcribe(engine, a16)   # noqa: E731
    last_check = time.monotonic()
    for k, p in enumerate(todo, 1):
        if engine is not None and time.monotonic() - last_check >= recheck_s:
            ok, line = gpu_check(gpu_clear)
            last_check = time.monotonic()
            if not ok:
                log(f"asr: stopped: {line}; run the same command again to resume")
                transcribe = engine = None
                _release()
                return EXIT_BUSY
        side = json.loads(p.read_text())
        x, rate = scorers.read_wav(p.with_suffix(".wav"))
        t = time.monotonic()
        text, words = transcribe(scorers.resample(x, rate, 16000))
        res = score_asr(side["text"], text, words)
        res["asr_s"] = round(time.monotonic() - t, 4)
        write_json(p.with_suffix(".asr.json"), res)
        if k % 50 == 0 or k == len(todo):
            log(f"asr: {k}/{len(todo)}")
    transcribe = engine = None
    _release()
    return EXIT_OK


def score(run_dir: Path, *, neural: bool = True, gpu_clear: str | None = GPU_CLEAR, force: bool = False,
          tag_hop: float | None = None, log: Callable[[str], None] = print) -> int:
    """The CPU scorers run in their own venv (torch and transformers stay out of this one)."""
    if not SCORER_PY.exists():
        log(f"score: no scorer venv at {SCORER_PY}; build it (tools/tts_bench_requirements.txt says how)")
        return EXIT_ERR
    cmd = [str(SCORER_PY), str(TOOLS / "tts_bench_scorers.py"), "score", str(run_dir), "--gpu-clear", gpu_clear or ""]
    cmd += ["--no-neural"] if not neural else []
    cmd += ["--force"] if force else []
    cmd += ["--tag-hop", str(tag_hop)] if tag_hop else []
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------------------------------- report

# flag: (metric, comparison, threshold, label)
FAILS: dict[str, tuple[str, str, Any, str]] = {
    "wer": ("wer", ">", 0.15, "WER>0.15"),
    "laugh": ("laugh_max", ">", 0.5, "laugh>0.5"),
    "jump": ("jump_max_st", ">", 8.0, "jump>8st"),
    "mos": ("mos", "<", 3.0, "MOS<3"),
    "runaway": ("runaway", "==", True, "runaway"),
    "lead": ("lead_raw_s", ">", 0.3, "lead>0.3s"),
}
# metric: (lower is better, label)
METRICS: dict[str, tuple[bool, str]] = {
    "wer": (True, "WER"),
    "laugh_max": (True, "laugh p"),
    "nonspeech_share": (True, "non-speech share"),
    "uncovered_voiced_s": (True, "voiced, no word (s)"),
    "mos": (False, "MOS"),
    "f0_range_st": (True, "F0 p5-p95 (st)"),
    "f0_sd_st": (True, "F0 sd (st)"),
    "jump_max_st": (True, "max jump (st)"),
    "jumps_over6": (True, "jumps>6st"),
    "seam_ratio_median": (True, "seam flux ratio"),
    "control_ratio_median": (True, "mid-chunk ratio"),
    "seams_over_p99": (True, "seams>p99"),
    "s_per_word": (True, "s/word"),
    "lead_raw_s": (True, "lead before trim (s)"),
    "lead_heard_s": (True, "lead heard (s)"),
    "max_lead_raw_s": (True, "any generation's lead (s)"),
    "longest_pause_s": (True, "longest pause (s)"),
    "pause_total_s": (True, "pauses (s)"),
    "first_chunk_s": (True, "first chunk (s)"),
    "first_audio_s": (True, "first audio (s)"),
}


def _flag(row: dict, flag: str) -> bool | None:
    metric, op, thr, _ = FAILS[flag]
    v = row.get(metric)
    if v is None:
        return None
    return bool(v == thr) if op == "==" else bool(v > thr) if op == ">" else bool(v < thr)


def gather(run_dir: Path, llm_cps: float = LLM_CPS) -> list[dict]:
    """A row per clip: its sidecar's numbers, the asr and score results when present, the runaway check (hit_cap, or
    longer than 2.5x the median of the same text across every setting) and the failure flags."""
    rows = []
    for p in scorers.clip_sidecars(run_dir):
        side = json.loads(p.read_text())
        row = {k: v for k, v in side.items() if k not in ("generations", "seams", "controls", "gen_starts", "engine")}
        row["wav"] = str(p.with_suffix(".wav"))
        a = p.with_suffix(".asr.json")
        if a.exists():
            d = json.loads(a.read_text())
            row.update(hyp=d.get("hyp"), wer=d.get("wer"), wer_sub=d.get("sub"), wer_del=d.get("del"),
                       wer_ins=d.get("ins"))
        sc = p.with_suffix(".score.json")
        if sc.exists():
            row.update({k: v for k, v in json.loads(sc.read_text()).items() if k not in row or row[k] is None})
        extra = max(0, int(side.get("llm_chars_first") or 0) - int(side.get("llm_chars_first_one_per_sentence") or 0))
        row["llm_wait_s"] = extra / llm_cps if llm_cps else 0.0
        if side.get("first_out_s") is not None:
            row["first_audio_s"] = side["first_out_s"] + (side.get("lead_heard_s") or 0.0) + row["llm_wait_s"]
        rows.append(row)
    by_item: dict[str, list[float]] = {}
    for row in rows:
        by_item.setdefault(row["item"], []).append(row["duration_s"])
    for row in rows:
        med = statistics.median(by_item[row["item"]])
        long = med > 0 and row["duration_s"] > 2.5 * med
        row["runaway"] = bool(row.get("hit_cap")) or long
        row["runaway_why"] = "hit_cap" if row.get("hit_cap") else (f"{row['duration_s']:.1f} s vs median {med:.1f}"
                                                                    if long else None)
        row["fails"] = [f for f in FAILS if _flag(row, f)]
    return rows


def _stats(vals: list[float], lower_better: bool) -> dict:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(v), "median": float(np.median(a)), "tail": float(np.percentile(a, 90 if lower_better else 10)),
            "worst": float(a.max() if lower_better else a.min()), "mean": float(a.mean())}


def paired_delta(rows: list[dict], setting: str, metric: str = "first_audio_s", baseline: str = "baseline",
                 items: set[str] | None = None) -> dict:
    """`metric` under `setting` minus the baseline's for the same text and render index (only `items`, when given):
    a setting's latency is compared text for text, so a grouping setting's wait on multi-sentence replies is not lost
    in the median of the one-sentence texts."""
    base = {(r["item"], r["r"]): r.get(metric) for r in rows if r["setting"] == baseline}
    d = [r[metric] - base[(r["item"], r["r"])] for r in rows if r["setting"] == setting
         and (items is None or r["item"] in items)
         and r.get(metric) is not None and base.get((r["item"], r["r"])) is not None]
    if not d:
        return {"n": 0}
    a = np.asarray(d)
    return {"n": len(d), "mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def summarise(rows: list[dict], settings: list[str], budget_items: set[str] | None = None) -> dict[str, dict]:
    """Per setting: failure counts, distributions, failures by kind, and the paired first-audio delta against the
    baseline over `budget_items` (the reply items: a conversation's mix of one- and multi-sentence replies; all texts
    when None)."""
    out = {}
    for s in settings:
        rs = [r for r in rows if r["setting"] == s]
        agg: dict[str, Any] = {"n": len(rs), "reused": sum(1 for r in rs if r.get("reused_from")),
                               "trim_missed": sum(1 for r in rs if r.get("trim_onset_found") is False),
                               "delta_first_audio": paired_delta(rows, s, items=budget_items),
                               "delta_first_audio_all": paired_delta(rows, s)}
        for f in FAILS:
            vals = [_flag(r, f) for r in rs]
            agg[f"fail_{f}"] = sum(1 for v in vals if v)
            agg[f"measured_{f}"] = sum(1 for v in vals if v is not None)
        agg["fail_any"] = sum(1 for r in rs if r["fails"])
        for m, (low, _) in METRICS.items():
            agg[m] = _stats([r.get(m) for r in rs], low)
        agg["kinds"] = {k: (sum(1 for r in rs if r["kind"] == k and r["fails"]), sum(1 for r in rs if r["kind"] == k))
                        for k in KINDS}
        out[s] = agg
    return out


def rank(summary: dict[str, dict], budget_s: float = BUDGET_S) -> list[dict]:
    """Settings ordered by the share of clips failing any check, then MOS (higher first), then WER; a setting is in
    the budget when its first audio comes on average no more than budget_s after the baseline's, text for text."""
    out = []
    for name, a in summary.items():
        if not a["n"]:
            continue
        d = a["delta_first_audio"]
        within = not d.get("n") or d["mean"] <= budget_s
        out.append({"setting": name, "within_budget": within, "fail_share": a["fail_any"] / a["n"],
                    "fail_any": a["fail_any"], "n": a["n"], "first_audio_median": a["first_audio_s"].get("median"),
                    "delta_first_audio": d.get("mean"), "delta_first_audio_p90": d.get("p90"),
                    "mos_median": a["mos"].get("median"), "wer_median": a["wer"].get("median")})
    out.sort(key=lambda d: (not d["within_budget"], d["fail_share"], -(d["mos_median"] or 0.0),
                            d["wer_median"] if d["wer_median"] is not None else 1.0))
    return out


def _f(v: Any, nd: int = 2) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _cell(st: dict, nd: int = 2) -> str:
    if not st.get("n"):
        return "–"
    return f"{st['median']:.{nd}f} / {st['tail']:.{nd}f} / {st['worst']:.{nd}f}"


def same_model(settings_json: dict[str, dict]) -> list[str]:
    """The settings that run the baseline's adapter and model (only sampling, grouping, streaming or instruct differ)."""
    base = settings_json.get("baseline", {})
    return [n for n, d in settings_json.items() if d.get("adapter") == base.get("adapter")
            and d.get("settings", {}).get("model") == base.get("settings", {}).get("model")]


def verdicts(summary: dict[str, dict], settings_json: dict[str, dict]) -> list[dict]:
    """Per check: the failing share under the baseline, under the best setting of the baseline's model, and under
    each other model, and what that says: a setting of the model halves it -> a setting; none does but another model
    does -> the model; nothing measured halves it -> neither."""
    mine = [n for n in same_model(settings_json) if n in summary]
    others = [n for n in summary if n not in mine]
    rows = []
    for f, (_, _, _, label) in FAILS.items():
        def rate(n: str) -> float | None:
            a = summary.get(n)
            m = a.get(f"measured_{f}") if a else 0
            return a[f"fail_{f}"] / m if a and m else None
        b = rate("baseline")
        cands = sorted((rate(n), n) for n in mine if n != "baseline" and rate(n) is not None)
        best = cands[0] if cands else (None, None)
        other = {n: rate(n) for n in others}
        lower = [n for n, v in other.items() if v is not None and best[0] is not None and v < best[0]]
        if b is None:
            why = "not measured"
        elif b == 0:
            why = "the baseline passes"
        elif best[0] is not None and best[0] <= 0.5 * b:
            why = f"a setting: {best[1]} cuts it to {best[0]:.0%}" + (
                f"; lower still with {', '.join(lower)}" if lower else "")
        elif any(v is not None and v <= 0.5 * b for v in other.values()):
            good = [n for n, v in other.items() if v is not None and v <= 0.5 * b]
            why = f"the model: no setting of it halves it; {', '.join(good)} does"
        else:
            why = "neither: no setting or model measured halves it"
        rows.append({"flag": label, "baseline": b, "best": best, "others": other, "why": why})
    return rows


def write_report(run_dir: Path, *, llm_cps: float = LLM_CPS, budget_s: float = BUDGET_S) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    settings_json = {s["name"]: s for s in manifest["settings"]}
    names = [s["name"] for s in manifest["settings"]]
    rows = gather(run_dir, llm_cps)
    with (run_dir / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    session = {i["id"] for i in manifest["items"] if i.get("logged")}
    summary = summarise(rows, names, session or None)
    ranking = rank(summary, budget_s=budget_s)
    budget_on = f"the {len(session)} reply items" if session else "every text"
    n_stage = {"asr": sum(1 for r in rows if r.get("wer") is not None),
               "score": sum(1 for r in rows if r.get("scorer_version") is not None),
               "neural": sum(1 for r in rows if r.get("mos") is not None)}
    L: list[str] = []
    L.append(f"# TTS quality bench: {run_dir.name}")
    L.append("")
    L.append(f"{len(rows)} clips: {len(manifest['items'])} texts x {manifest['reps']} renders x {len(names)} settings "
             f"(created {manifest.get('created')}). Transcribed: {n_stage['asr']}; scored: {n_stage['score']}; "
             f"neural scores: {n_stage['neural']}. Every number is a machine measurement (tools/tts_quality_bench.py).")
    L.append("")
    L.append("## Failures per setting")
    L.append("")
    L.append("Clips failing each check, of the clips measured. \"trim missed\": first generations whose leading silence "
             "outlasted the trim's scan, so all of it was heard (0.94 and 1.07 s in an early live test). First audio is the "
             "TTS's first audible sound once the text is complete (the trim's first output plus the silence heard), "
             f"plus, for a grouping setting, the extra characters the LLM must write first at {llm_cps:.0f} chars/s. "
             f"Budget: on {budget_on}, first audio on average at most {budget_s:.1f} s after the baseline's, text for "
             "text (\"vs baseline\": the mean and p90 of that paired difference).")
    L.append("")
    head = ["setting", "clips", "any"] + [FAILS[f][3] for f in FAILS] + [
        "trim missed", "first chunk med/p90 (s)", "first audio med/p90 (s)", "vs baseline mean/p90 (s)", "in budget"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for n in names:
        a = summary[n]
        fc, fa, d = a["first_chunk_s"], a["first_audio_s"], a["delta_first_audio"]
        within = "–" if not d.get("n") else ("yes" if d["mean"] <= budget_s else "NO")
        cells = [n, f"{a['n']}" + (f" ({a['reused']} reused)" if a["reused"] else ""), f"{a['fail_any']}"]
        cells += [f"{a[f'fail_{f}']}/{a[f'measured_{f}']}" if a[f"measured_{f}"] else "–" for f in FAILS]
        cells += [str(a["trim_missed"]),
                  f"{_f(fc.get('median'))} / {_f(fc.get('tail'))}" if fc.get("n") else "–",
                  f"{_f(fa.get('median'))} / {_f(fa.get('tail'))}" if fa.get("n") else "–",
                  f"{d['mean']:+.2f} / {d['p90']:+.2f}" if d.get("n") else "–", within]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Is it the model or a setting?")
    L.append("")
    L.append("The share of clips failing each check under the baseline, under the best setting of the baseline's model, "
             "and under the other models; the last column reads it off (a setting of the model halves it: a setting; "
             "none does but another model does: the model; nothing measured halves it: neither).")
    L.append("")
    vrows = verdicts(summary, settings_json)
    others = [n for n in names if n not in same_model(settings_json)]
    head = ["check", "baseline", "best setting, same model"] + others + ["reading"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    pct = lambda x: "–" if x is None else f"{x:.0%}"   # noqa: E731
    for v in vrows:
        best = "–" if v["best"][0] is None else f"{v['best'][1]} {v['best'][0]:.0%}"
        L.append("| " + " | ".join([v["flag"], pct(v["baseline"]), best] + [pct(v["others"].get(o)) for o in others]
                                   + [v["why"]]) + " |")
    L.append("")
    L.append("## Ranking")
    L.append("")
    best = next((r for r in ranking if r["within_budget"]), None)
    if best:
        L.append(f"**Best within the budget: {best['setting']}** ({best['fail_any']}/{best['n']} clips failing any "
                 f"check, MOS median {_f(best['mos_median'])}, WER median {_f(best['wer_median'], 3)}, first audio "
                 f"median {_f(best['first_audio_median'])} s"
                 + ("" if best["delta_first_audio"] is None else
                    f", {best['delta_first_audio']:+.2f} s on average vs the baseline") + ").")
        L.append("")
    L.append("| # | setting | failing any | MOS median | WER median | first audio median (s) | "
             "vs baseline mean / p90 (s) | in budget |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranking, 1):
        delta = "–" if r["delta_first_audio"] is None else \
            f"{r['delta_first_audio']:+.2f} / {r['delta_first_audio_p90']:+.2f}"
        L.append(f"| {i} | {r['setting']} | {r['fail_any']}/{r['n']} ({r['fail_share']:.0%}) | {_f(r['mos_median'])} "
                 f"| {_f(r['wer_median'], 3)} | {_f(r['first_audio_median'])} | {delta} | "
                 f"{'yes' if r['within_budget'] else 'NO'} |")
    L.append("")
    L.append("## Distributions (median / p90 / worst; MOS: median / p10 / worst)")
    L.append("")
    groups = [("WER, laughter, naturalness", ["wer", "laugh_max", "nonspeech_share", "uncovered_voiced_s", "mos"]),
              ("pitch", ["f0_range_st", "f0_sd_st", "jump_max_st", "jumps_over6"]),
              ("seams (flux at a streamed chunk boundary over the clip's median voiced flux, against mid-chunk points "
               "in the same clips; nostream has no seams, so its 0.32 s grid points are the control a seam is compared "
               "with; Kokoro's joins are clause pieces, not streamed chunks)",
               ["seam_ratio_median", "control_ratio_median", "seams_over_p99"]),
              ("timing", ["s_per_word", "lead_raw_s", "lead_heard_s", "max_lead_raw_s", "longest_pause_s",
                          "pause_total_s", "first_chunk_s", "first_audio_s"])]
    for title, ms in groups:
        L.append(f"**{title}**")
        L.append("")
        L.append("| setting | " + " | ".join(METRICS[m][1] for m in ms) + " |")
        L.append("|---|" + "---|" * len(ms))
        for n in names:
            L.append(f"| {n} | " + " | ".join(_cell(summary[n][m], 3 if m in ("wer", "laugh_max") else 2)
                                              for m in ms) + " |")
        L.append("")
    L.append("## Failing clips by kind (any check)")
    L.append("")
    L.append("| setting | " + " | ".join(KINDS) + " |")
    L.append("|---|" + "---|" * len(KINDS))
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{a}/{b}" if b else "–" for a, b in
                                         (summary[n]["kinds"][k] for k in KINDS)) + " |")
    L.append("")
    L.append("## Per text: failing renders per setting")
    L.append("")
    letters = {"wer": "W", "laugh": "L", "jump": "J", "mos": "M", "runaway": "R", "lead": "S"}
    L.append("Each cell: renders failing any check (of R), then which: W WER, L laughter, J pitch jump, M MOS, "
             "R runaway, S leading silence.")
    L.append("")
    L.append("| text | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for item in manifest["items"]:
        cells = []
        for n in names:
            rs = [r for r in rows if r["setting"] == n and r["item"] == item["id"]]
            if not rs:
                cells.append("–")
                continue
            bad = [r for r in rs if r["fails"]]
            kinds = "".join(sorted({letters[f] for r in bad for f in r["fails"]}))
            cells.append(f"{len(bad)}{' ' + kinds if kinds else ''}")
        L.append(f"| {item['id']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Worst clips")
    L.append("")
    L.append("Rendered clips only (a copy reused from an earlier setting is listed under that setting).")
    L.append("")
    for metric, label, low in (("wer", "WER", True), ("laugh_max", "laughter probability", True),
                               ("jump_max_st", "largest pitch jump (st)", True), ("mos", "MOS", False),
                               ("lead_raw_s", "leading silence before the trim (s)", True)):
        rs = [r for r in rows if r.get(metric) is not None and not r.get("reused_from")]
        if not rs:
            continue
        rs.sort(key=lambda r: r[metric], reverse=low)
        L.append(f"**{label}**")
        L.append("")
        for r in rs[:8]:
            extra = ""
            if metric == "wer":
                extra = (f" — sub {r.get('wer_sub')}, del {r.get('wer_del')}, ins {r.get('wer_ins')}; "
                         f"heard: {r.get('hyp')!r}")
            elif metric == "laugh_max":
                extra = (f" — {r.get('laugh_label')} at {_f(r.get('laugh_at_s'))} s; "
                         f"top non-speech {r.get('nonspeech_top')}")
            L.append(f"- {_f(r[metric], 3)} {r['setting']}/{r['item']}-{r['r']}{extra}")
        L.append("")
    L.append("## Settings")
    L.append("")
    L.append("| setting | override of the baseline | load s | warm s | load average at start / end | errors |")
    L.append("|---|---|---|---|---|---|")
    for n in names:
        meta_path = run_dir / n / "setting.json"
        runs = json.loads(meta_path.read_text()).get("runs", []) if meta_path.exists() else []
        load = "; ".join(f"{_f(r.get('loadavg_start', [None])[0])} / {_f((r.get('loadavg_end') or [None])[0])}"
                         for r in runs) or "–"
        errs = sum(len(r.get("errors", [])) for r in runs)
        L.append(f"| {n} | `{json.dumps(settings_json[n].get('override') or {})}` | "
                 f"{'; '.join(_f(r.get('load_s')) for r in runs) or '–'} | "
                 f"{'; '.join(_f(r.get('warm_s')) for r in runs) or '–'} | {load} | {errs} |")
    L.append("")
    L.append("## How the numbers are made")
    L.append("")
    L.append(f"- Render: tools/tts_quality_bench.py (docstring). Trim as services/tts.py: {manifest.get('trim')}. "
             "Clips copied from an earlier setting with the same engine settings and generation texts say "
             "`reused_from` in their sidecars.")
    L.append("- Runaway: the engine hit its max_tokens, or the clip is over 2.5x the median length of the same text "
             "across every setting.")
    L.append("- WER: Parakeet TDT 0.6B v3 on the clip at 16 kHz, both texts normalised (lower case, no punctuation, "
             "digits as words). Scorers: tools/tts_bench_scorers.py (docstring: models, sizes, why).")
    path = run_dir / "report.md"
    path.write_text("\n".join(L) + "\n")
    return path, run_dir / "results.jsonl"


# ---------------------------------------------------------------------------------------------------- plan

# Per model: seconds of speech per character, real-time factor (compute / audio), load plus first-call warm-up, and the
# silence it generates around each generation, as measured on this Mac (measure/raw/clean_tts_*.jsonl, 2026-10-04 22:01,
# interval 0.32; 09-phase1-report.md §3): the 257- and 59-character texts took 1.7B 14.9-20.1 s and 3.7-5.0 s (lead
# 0.08 s, tail 0.02-0.04, RTF 0.26-0.28), 0.6B 16.6-18.8 s and 4.8-6.6 s (lead 0.42, tail 0.44-0.48, RTF 0.20-0.22),
# Kokoro 15.5 s and 3.9 s (lead 0.29, tail 0.45-0.48, RTF 0.02, first call 4.3 s).
SPEED = {"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": (0.064, 0.27, 2.5, 0.15), QWEN_06B: (0.070, 0.215, 2.5, 0.9),
         "mlx-community/Kokoro-82M-bf16": (0.056, 0.02, 4.5, 0.75)}
GEN_PREFILL_S, ASR_PER_CLIP_S = 0.12, 0.05


def plan(settings: list[Setting], items: list[Item], reps: int, reuse: bool = True) -> list[dict]:
    """The clips each setting renders and an estimate of its GPU time (nothing runs)."""
    out = []
    for s in settings:
        spc, rtf, warm, edge = SPEED.get(s.settings.get("model"), (0.064, 0.27, 2.5, 0.15))
        new = reused = 0
        audio = gpu = 0.0
        for item in items:
            same = reuse and any(render_key(item, o) == render_key(item, s)
                                 for o in settings[:settings.index(s)])
            if same:
                reused += reps
                continue
            gens = generations(item, s)
            a = sum(len(t) * spc + edge for t in gens)
            new += reps
            audio += reps * a
            gpu += reps * (a * rtf + len(gens) * GEN_PREFILL_S)
        out.append({"setting": s.name, "renders": new, "reused": reused, "audio_s": audio,
                    "gpu_s": gpu + (warm if new else 0.0)})
    return out


# ---------------------------------------------------------------------------------------------------- CLI

def _latest() -> Path:
    runs = sorted(p for p in RUNS.glob("*") if (p / "run.json").exists())
    if not runs:
        raise SystemExit(f"no runs under {RUNS}")
    return runs[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["render", "asr", "score", "report", "all", "plan", "texts"])
    ap.add_argument("--run-dir", type=Path, help="a run to create or resume (default: a new stamp for render/all)")
    ap.add_argument("--latest", action="store_true", help="the newest run under ../state/tts-bench")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--texts", type=Path, default=TEXTS)
    ap.add_argument("--settings", help="comma-separated setting names (baseline always runs)")
    ap.add_argument("--combo", action="append", default=[], help="NAME=a+b: a combination of two or more settings")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--items", help="comma-separated item ids")
    ap.add_argument("--kinds", help="comma-separated kinds")
    ap.add_argument("--gpu-clear", default=GPU_CLEAR, help="the check that must exit 0 before any GPU work")
    ap.add_argument("--recheck-s", type=float, default=60.0)
    ap.add_argument("--no-reuse", action="store_true")
    ap.add_argument("--no-neural", action="store_true", help="score: Praat, flux and pauses only")
    ap.add_argument("--force", action="store_true", help="score: rescore every clip")
    ap.add_argument("--tag-hop", type=float, help="score: seconds between the AudioSet windows (default 0.5)")
    ap.add_argument("--llm-cps", type=float, default=LLM_CPS)
    ap.add_argument("--check-log", action="store_true", help="texts: compare the reply items with --log/--turns")
    ap.add_argument("--log", type=Path, help="texts --check-log: an orchestrator log (state/orchestrator/logs/...)")
    ap.add_argument("--turns", type=Path, help="texts --check-log: that day's turn log (state/turns/<day>.jsonl)")
    ap.add_argument("--window", nargs=2, metavar=("START", "END"), help='texts --check-log: "YYYY-MM-DD HH:MM:SS" twice')
    a = ap.parse_args(argv)
    raw = read_config(a.config)
    items = select_items(load_texts(a.texts), a.items, a.kinds)
    settings = expand_grid(raw["tts"], a.settings.split(",") if a.settings else None, a.combo)

    if a.stage == "texts":
        for it in items:
            print(f"{it.id} [{it.kind}]{' (logged)' if it.logged else ''}")
            for s in settings:
                if s.name in ("baseline", "minwords4", "maxsent2", "maxsent3"):
                    print(f"  {s.name:10} {generations(it, s)}  (first after {llm_chars(it, s)} chars)")
        if a.check_log:
            logged = {it.turn: it.sentences for it in items if it.logged}
            if not (a.log and a.turns and a.window):
                ap.error("--check-log needs --log, --turns and --window")
            replies = session_replies(a.log, a.turns, tuple(a.window))
            bad = [d for d in replies if logged.get(d["turn"]) != d["sentences"]]
            print(f"session items vs the log: {len(logged)} items, {len(bad)} differ" + (f": {bad}" if bad else ""))
            return EXIT_ERR if bad else EXIT_OK
        return EXIT_OK
    if a.stage == "plan":
        rows = plan(settings, items, a.reps, reuse=not a.no_reuse)
        total = sum(r["gpu_s"] for r in rows)
        clips = sum(r["renders"] + r["reused"] for r in rows)
        asr_s = clips * ASR_PER_CLIP_S + 3.0
        for r in rows:
            print(f"{r['setting']:13} {r['renders']:4} renders {r['reused']:4} reused  audio {r['audio_s']:7.0f} s  "
                  f"GPU ~{r['gpu_s'] / 60:5.1f} min")
        print(f"{len(items)} texts x {a.reps} reps x {len(settings)} settings = {clips} clips; render ~{total / 60:.1f} "
              f"min + asr ~{asr_s / 60:.1f} min = ~{(total + asr_s) / 60:.0f} min of GPU on an idle Mac")
        return EXIT_OK

    if a.run_dir:
        run_dir = a.run_dir
    elif a.latest or a.stage not in ("render", "all"):
        run_dir = _latest()
    else:
        run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"run dir: {run_dir}")
    trim = production_trim(raw["tts"])
    stages = ["render", "asr", "score", "report"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "render":
            rc = render(run_dir, settings, items, a.reps, trim=trim, gpu_clear=a.gpu_clear, recheck_s=a.recheck_s,
                        reuse=not a.no_reuse)
        elif st == "asr":
            rc = asr(run_dir, raw["stt"], gpu_clear=a.gpu_clear, recheck_s=a.recheck_s)
        elif st == "score":
            rc = score(run_dir, neural=not a.no_neural, gpu_clear=a.gpu_clear, force=a.force, tag_hop=a.tag_hop)
        else:
            path, res = write_report(run_dir, llm_cps=a.llm_cps)
            print(f"report: {path}\nresults: {res}")
            rc = EXIT_OK
        if rc != EXIT_OK:
            print(f"{st}: exit {rc}; resume with: "
                  f".venv/bin/python tools/tts_quality_bench.py {a.stage} --run-dir {run_dir}")
            return rc
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
