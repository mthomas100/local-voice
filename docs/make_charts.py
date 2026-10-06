#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Rebuild the README charts in docs/media/ from the measurement files in this repo (no models are run).

    uv run docs/make_charts.py
"""
from __future__ import annotations

import csv
import json
import statistics
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "measure" / "raw"
P2 = RAW / "phase2-20261005-084528"
DATA = ROOT / "docs" / "data"
OUT = ROOT / "docs" / "media"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def med(path: Path, text: str, mode: str, key: str) -> float:
    return statistics.median(r[key] for r in rows(path)
                             if r["event"] == "gen" and r["text_name"] == text and r["mode"] == mode)


def table(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    return list(csv.DictReader(lines))


def save(fig, name: str, source: str) -> None:
    fig.text(0.01, -0.03, textwrap.fill(source, 150), fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def tts_engines() -> None:
    # (label, file, streaming mode for first chunk); Kokoro has no streaming, so its first output is the whole sentence.
    engines = [
        ("Pocket TTS", P2 / "tts_pocket_warm.jsonl", "stream_0.32"),
        ("VibeVoice-Realtime 0.5B", P2 / "tts_vibevoice.jsonl", "stream_0.32"),
        ("Kokoro-82M (whole sentence)", RAW / "clean_tts_kokoro.jsonl", "nonstream"),
        ("Qwen3-TTS 0.6B", RAW / "clean_tts_qwen3_0.6b.jsonl", "stream_0.32"),
        ("Qwen3-TTS 1.7B (default)", RAW / "clean_tts_qwen3_1.7b.jsonl", "stream_0.32"),
        ("Marvis 250M", P2 / "tts_marvis.jsonl", "stream_0.32"),
    ]
    labels = [e[0] for e in engines][::-1]
    ttfa = [med(f, "short", m, "ttfa_s") * 1000 for _, f, m in engines][::-1]
    rtf = [med(f, "long", "nonstream", "rtf") for _, f, _ in engines][::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    colors = [S1 if "default" in lab else "#9fb8d9" for lab in labels]
    a.barh(labels, ttfa, color=colors, height=0.6)
    for y, v in enumerate(ttfa):
        a.text(v + 2, y, f"{v:.0f} ms", va="center", fontsize=9, color=INK)
    a.set_title("First audio chunk, 11-word sentence", loc="left")
    a.set_xlabel("ms from generate() to first chunk (interval 0.32 s)")
    a.set_xlim(0, max(ttfa) * 1.25)
    a.grid(axis="y", visible=False)
    b.barh(labels, rtf, color=colors, height=0.6)
    for y, v in enumerate(rtf):
        b.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=9, color=INK)
    b.set_title("Real-time factor, 46-word text", loc="left")
    b.set_xlabel("generation time / audio duration")
    b.set_xlim(0, max(rtf) * 1.25)
    b.grid(axis="y", visible=False)
    save(fig, "chart-tts-engines.png",
         "Data: measure/raw/clean_tts_*.jsonl (2026-10-04 22:01) and measure/raw/phase2-20261005-084528/tts_*.jsonl "
         "(2026-10-05 08:45), mlx-audio on Apple M5 Max, nothing else on the GPU. First chunk is not first sound: "
         "see leading silence in measure/09b-phase2-report.md.")


def coresidency() -> None:
    pairs = [("Parakeet + Kokoro", "coresident_kokoro.jsonl"), ("Parakeet + Qwen3-TTS 0.6B", "coresident_qwen06.jsonl")]
    conds = [("LLM loaded, idle (10-04 21:50)", RAW, S1), ("nothing else loaded (10-05 08:47)", P2, S2)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"wspace": 0.3})
    w = 0.36
    for i, (cname, d, col) in enumerate(conds):
        warm = [next(r for r in rows(d / f) if r["event"] == "round" and r["run"] == 2) for _, f in pairs]
        done = [next(r for r in rows(d / f) if r["event"] == "done") for _, f in pairs]
        xs = [x + (i - 0.5) * w for x in range(len(pairs))]
        tt = [r["tts_ttfa_s"] * 1000 for r in warm]
        pk = [r["peak_gib"] for r in done]
        a.bar(xs, tt, w - 0.03, color=col, label=cname)
        b.bar(xs, pk, w - 0.03, color=col, label=cname)
        for x, v in zip(xs, tt):
            a.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=8.5)
        for x, v in zip(xs, pk):
            b.text(x, v + 0.08, f"{v:.2f}", ha="center", fontsize=8.5)
    for ax in (a, b):
        ax.set_xticks(range(len(pairs)), [p for p, _ in pairs])
        ax.grid(axis="x", visible=False)
    a.set_title("Warm TTS first audio, STT + TTS in one process", loc="left")
    a.set_ylabel("ms")
    a.set_ylim(0, 140)
    b.set_title("MLX peak memory of the pair", loc="left")
    b.set_ylabel("GiB")
    b.set_ylim(0, 6.5)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "chart-coresidency.png",
         "Data: measure/raw/coresident_*.jsonl and measure/raw/phase2-20261005-084528/coresident_*.jsonl "
         "(round 2 = warm). The LLM was a separate server process; the speech pair leaves >100 GiB of the GPU working set.")


def waterfall() -> None:
    t = list(csv.DictReader((DATA / "m1_e2e_latency_2026-10-05.csv").open()))
    warm = [r for r in t if r["warm"] == "True"]
    segs = [("endpointing_wait_ms", "VAD stop wait", "#c9c7c0"), ("transcription_ms", "STT final", S3),
            ("turn_detection_ms", "Smart Turn", S4), ("llm_first_token_ms", "LLM first token", S1),
            ("tts_first_audio_ms", "TTS to first audio", S2)]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    labels = [f'"{r["prompt_spoken_by_macos_say"]}"' for r in warm][::-1]
    left = [0.0] * len(warm)
    for key, lab, col in segs:
        vals = [float(r[key]) for r in warm][::-1]
        ax.barh(labels, vals, left=left, color=col, height=0.6, label=lab, edgecolor=SURFACE, linewidth=1.5)
        left = [l + v for l, v in zip(left, vals)]
    for y, r in enumerate(warm[::-1]):
        ax.text(left[y] + 8, y, f'{float(r["server_eos_to_first_audio_ms"]):.0f} ms server / '
                f'{r["client_eos_to_first_audio_ms"]} ms client', va="center", fontsize=8.5)
    ax.set_xlim(0, 1550)
    ax.set_xlabel("ms after the end of speech")
    ax.set_title("Where a warm turn's ~0.9 s goes (M1 run, 5 warm spoken turns)", loc="left", pad=30)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, "chart-latency-waterfall.png",
         "Data: docs/data/m1_e2e_latency_2026-10-05.csv (e2e test, 2026-10-05 14:24, prompts spoken by macOS `say` "
         "and streamed at real-time pace). Cold first turn: 3.9 s (LLM prefilled 2.2k tokens from zero).")


def runs() -> None:
    t = table(DATA / "m1_runs_2026-10-05.csv")
    fig, ax = plt.subplots(figsize=(10, 3.0))
    xs = range(len(t))
    ys = [int(r["median_ms"]) for r in t]
    cols = [S2 if i < 4 else S1 for i in xs]
    ax.plot(list(xs), ys, color=GRID, linewidth=2, zorder=1)
    ax.scatter(list(xs), ys, s=60, color=cols, zorder=2, edgecolor=SURFACE, linewidth=2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 230, f"{y:,}", ha="center", fontsize=8.5)
    ax.axhline(1500, color=INK2, linewidth=1, linestyle="--")
    ax.text(len(t) - 0.6, 1580, "1.5 s target", fontsize=8.5, color=INK2, ha="right")
    ax.axvline(3.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(3.55, 4900, "LLM server: Background -> Interactive QoS,\n250 ms GPU-queue keepalive", fontsize=8.5,
            color=INK2, va="top")
    ax.set_xticks(list(xs), [r["run_start"] for r in t])
    ax.set_ylim(0, 5600)
    ax.set_ylabel("median ms, end of speech\nto first audio")
    ax.set_title("Full e2e runs through 2026-10-05", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "chart-e2e-runs.png",
         "Data: docs/data/m1_runs_2026-10-05.csv (warm turns, client-measured). The 12:01 run ran under heavy CPU load.")


def turn_end() -> None:
    t = table(DATA / "turn_end_rules_2026-10-05.csv")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    marks = ["o", "s", "D", "^"]
    cols = [S1, S2, S3, S5]
    for r, m, c in zip(t, marks, cols):
        x = float(r["finished_p90_s"])
        ax.scatter([x], [int(r["cut_off_at_0.8s_pause_of_80"])], s=80, marker=m, color=c, edgecolor=SURFACE,
                   linewidth=2, zorder=3, label=r["rule"])
        ax.scatter([x], [int(r["cut_off_at_1.5s_pause_of_80"])], s=80, marker=m, facecolor="none", edgecolor=c,
                   linewidth=1.8, zorder=3)
    ax.set_xlabel("p90 wait after a finished sentence (s)")
    ax.set_ylabel("mid-sentence pauses cut off (of 80)")
    ax.set_xlim(0.5, 3.6)
    ax.set_ylim(0, 42)
    ax.set_title("Turn-end rules: waiting on finished speech vs cutting people off (down-left is better)", loc="left")
    ax.text(1.07, 25.6, "1.5 s pause", fontsize=8, color=INK2)
    ax.text(1.07, 15.0, "0.8 and 1.5 s pause", fontsize=8, color=INK2)
    ax.text(1.07, 34.4, "0.8 and 1.5 s pause", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="filled = 0.8 s pause, hollow = 1.5 s",
              title_fontsize=8)
    save(fig, "chart-turn-end.png",
         "Data: docs/data/turn_end_rules_2026-10-05.csv (tools/turn_end_bench.py; 100 finished + 80 paused `say` sentences, "
         "stand-in transcripts). With real streaming-ASR transcripts the early-end rule cut off more (37/44).")


def bargein() -> None:
    t = table(DATA / "bargein_detectors_2026-10-05.csv")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for r, c in zip(t, [S1, S2, S3]):
        x, y = int(r["interrupt_median_ms"]), int(r["esc50_false_interrupts_of_260"])
        ax.errorbar([x], [y], xerr=[[0], [int(r["interrupt_p90_ms"]) - x]], fmt="o", color=c, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=0, elinewidth=2, label=r["detector"])
    ax.axvline(300, color=INK2, linewidth=1, linestyle="--")
    ax.text(303, 16, "300 ms goal", fontsize=8.5, color=INK2)
    ax.set_xlim(150, 420)
    ax.set_ylim(15, 48)
    ax.set_xlabel("interrupt latency after speech onset, ms (dot = median, line to p90)")
    ax.set_ylabel("ESC-50 sounds that interrupt\n(of 260: coughs, sneezes...)")
    ax.set_title("Barge-in: faster detection costs false interruptions", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    save(fig, "chart-bargein.png",
         "Data: docs/data/bargein_detectors_2026-10-05.csv (tools/bargein_bench.py, 110 `say` interruptions, ESC-50). "
         "Shipped: pause the reply on the first speech frame (median 72 ms), interrupt at start 0.15 s.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tts_engines()
    coresidency()
    waterfall()
    runs()
    turn_end()
    bargein()
    print("charts written to", OUT)
