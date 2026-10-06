"""The voice-quality bench's scorers: every rendered clip measured by machine, on the CPU (2026-10-05).

In early live tests the voice at times laughed, swung up and down in pitch and slurred; the rule is that quality is
measured by machine, never by asking anyone to listen, rate or label. This
module turns each clip of a bench run (tools/tts_quality_bench.py render/asr) into numbers:

- pitch, with Praat (Parselmouth): F0 range p5-p95 and sd in semitones, the largest and p95 jump between consecutive
  voiced 10 ms frames, jumps over 6 semitones, the voiced share of the speech span. Floor 60 Hz, ceiling 500 Hz and the
  10 ms step are tone/local_voice_tone/pitch.py's (the project's Praat settings). "Up and down" shows as range and sd,
  a crack as a jump.
- non-speech and laughter: an AudioSet tagger over 1 s windows (0.5 s hop): the largest Laughter/Giggle/Snicker/
  Belly laugh/Chuckle probability, and the share of windows with sound in them where a non-speech vocal class
  (laughter, breathing, gasp, pant, sigh, cough, throat clearing, humming, singing) beats the speech classes. With the
  asr stage's word times, also the voiced time no recognised word covers (babble, a laugh or a hum the recogniser
  does not write down).
- naturalness: UTMOS22 strong, a MOS predictor (1-5).
- seams: spectral flux (log magnitude, half-wave rectified, 21 ms Hann window, 5 ms hop) at every streamed chunk
  boundary inside a generation, against the clip's median flux in voiced frames; how many seams exceed the clip's
  p99 flux; and the same at control points in the middle of chunks (or on a 0.32 s grid when nothing was streamed),
  so a seam is compared with the same clip's own speech.
- duration and pauses: speech span per word and per character; leading and trailing silence, the longest internal
  pause and the total, by tools/speech_gaps.py's rule (20 ms frames, speech within 40 dB of the loudest frame, a pause
  is 0.35 s or more).

Why these neural models (downloads, network only, 2026-10-05; sizes in bytes as served):
- MIT/ast-finetuned-audioset-10-10-0.4593 (Hugging Face, revision f826b80d…, model.safetensors 346,404,948 bytes plus
  two small JSON files), the Audio Spectrogram Transformer fine-tuned on AudioSet (mAP 0.459): the strongest AudioSet
  tagger that loads with plain transformers, it has every laughter and breath class by name, and it is a single
  safetensors file. PANNs CNN14 (mAP 0.431) is faster on short windows but needs the panns_inference package and a
  Zenodo download. AST pads each 1 s window to its 10.24 s input; measured speed is in the bench's docstring.
- UTMOS22 strong learner via tarepan/SpeechMOS v1.2.0 (torch.hub; checkpoint utmos22_strong_step7459_v1.pt from the
  v1.0.0 release, 411,179,119 bytes; code from the v1.2.0 tag): the VoiceMOS 2022 winner's strong learner as one
  checkpoint with no fairseq dependency. UTMOSv2 needs its own package and several fold checkpoints, over the 1 GB line.
Both live in the user's caches: ~/.cache/huggingface/hub and ~/.cache/torch/hub (checkpoints/ and
tarepan_SpeechMOS_v1.2.0/).

CPU smoke (2026-10-05, right after gpu_clear.sh said CLEAR; torch 2.14.1 on the CPU, 6 threads): AST loaded
from the cache in 3.07 s, UTMOS22 in 0.19 s. A 3 s synthetic vowel (a harmonic source through three formants): the
tagger's top classes Insect 0.38, Fly 0.31, Mosquito 0.13, laughter 0.00, MOS 1.34; 3 s of synthetic breathy bursts:
Grunt 0.10, Sound effect 0.09, laughter 0.02, a non-speech vocal class over speech in 3 of 5 windows, MOS 1.31. Neither
is speech, and neither model took it for speech. Speed: AST 0.19 s per 1 s window (1.71 s for the first five, with its
warm-up), UTMOS 0.07 s per 3 s clip. Scoring the default grid (about 7,650 s of rendered audio; the 264 copied clips
copy their source's score) costs about 45 min of CPU at the default 0.5 s hop (--tag-hop 1.0: about 21 min), plus
about 3 min of UTMOS and 1 min of Praat, flux and pauses.

Runs in its own venv (tools/.venv-bench, tools/tts_bench_requirements.txt) so torch never enters the orchestrator's
venv. The numpy half (wav reading, pauses, flux and seams, pitch statistics from an F0 track, word coverage) imports
nothing else, so the orchestrator's tests use it directly; parselmouth, torch and transformers are imported only
inside the functions that need them. torch runs on the CPU only (never "mps"), and the neural models load only after
gpu_clear.sh exits 0 (a project rule).

    tools/.venv-bench/bin/python tools/tts_bench_scorers.py score ../state/tts-bench/<stamp> [--no-neural]
    tools/.venv-bench/bin/python tools/tts_bench_scorers.py download      # fetch the two models; loads nothing
    tools/.venv-bench/bin/python tools/tts_bench_scorers.py smoke         # both models on two synthetic clips
    tools/.venv-bench/bin/python tools/tts_bench_scorers.py selftest-pitch
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
GPU_CLEAR = str(REPO / "measure" / "bench" / "gpu_clear.sh")
SCORER_VERSION = 1

# tone/local_voice_tone/pitch.py
STEP_S, FLOOR_HZ, CEILING_HZ = 0.01, 60.0, 500.0
# tools/speech_gaps.py
FRAME_S, GAP_S, GAP_DB = 0.02, 0.35, 40.0
# spectral flux at 24 kHz: 512-sample (21.3 ms) Hann window, 120-sample (5 ms) hop, magnitudes floored at -100 dB
FLUX_WIN, FLUX_HOP, FLUX_FLOOR = 512, 120, 1e-5
# a voiced median below this (log10 units per bin) is no speech at all, only a steady tone or silence
FLUX_MIN_MEDIAN = 1e-4
SEAM_OVER = 3.0
CONTROL_GRID_S = 0.32

AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_REV = "f826b80d28226b62986cc218e5cec390b1096902"
UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_URL = "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"
UTMOS_ZIP = "https://github.com/tarepan/SpeechMOS/archive/refs/tags/v1.2.0.zip"
LAUGH = ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle")
VOCAL_NONSPEECH = LAUGH + ("Breathing", "Gasp", "Pant", "Sigh", "Cough", "Throat clearing", "Humming", "Singing")
SPEECH = ("Speech", "Male speech, man speaking", "Female speech, woman speaking", "Narration, monologue",
          "Conversation")
TAG_WIN_S, TAG_HOP_S, TAG_BATCH = 1.0, 0.5, 8
TAG_ACTIVE = 0.3     # a window counts toward the non-speech share when this much of it is sound (20 ms frames, 40 dB)


# ---------------------------------------------------------------------------------------------------- numpy half

def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """A 16-bit mono WAV as float32 in [-1, 1) and its rate (the bench writes nothing else)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono")
        rate = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return a, rate


def pauses(x: np.ndarray, rate: int, min_gap: float = GAP_S, db: float = GAP_DB) -> dict:
    """tools/speech_gaps.py's gaps() on an array, unrounded: lead, tail, every inner silence of at least min_gap, the
    speech span. `speech` is False when nothing is louder than silence."""
    n = int(FRAME_S * rate)
    frames = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else np.zeros((0, n), np.float32)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) if len(frames) else np.zeros(0)
    length = len(x) / rate
    if not len(rms) or rms.max() <= 0:
        return {"speech": False, "length_s": length, "lead_s": length, "tail_s": 0.0, "gaps": [], "gaps_s": 0.0,
                "longest_s": 0.0, "speech_span_s": 0.0, "first_s": 0.0, "last_s": 0.0}
    loud = rms >= rms.max() * 10 ** (-db / 20)
    idx = np.flatnonzero(loud)
    first, last = int(idx[0]), int(idx[-1])
    inner, run = [], 0
    for k in range(first, last + 1):
        if loud[k]:
            if run * FRAME_S >= min_gap:
                inner.append(run * FRAME_S)
            run = 0
        else:
            run += 1
    return {"speech": True, "length_s": length, "lead_s": first * FRAME_S, "tail_s": length - (last + 1) * FRAME_S,
            "gaps": inner, "gaps_s": float(sum(inner)), "longest_s": float(max(inner, default=0.0)),
            "speech_span_s": (last + 1 - first) * FRAME_S, "first_s": first * FRAME_S, "last_s": (last + 1) * FRAME_S}


def active_frames(x: np.ndarray, rate: int, db: float = GAP_DB) -> np.ndarray:
    """Per 20 ms frame: within `db` of the loudest frame (speech_gaps.py's speech rule)."""
    n = int(FRAME_S * rate)
    frames = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else np.zeros((0, n), np.float32)
    if not len(frames):
        return np.zeros(0, bool)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return rms >= rms.max() * 10 ** (-db / 20) if rms.max() > 0 else np.zeros(len(rms), bool)


def spectral_flux(x: np.ndarray, rate: int, win: int = FLUX_WIN, hop: int = FLUX_HOP) -> tuple[np.ndarray, np.ndarray]:
    """(flux per frame, frame centres in samples). Flux is the mean over bins of the rise in log10 magnitude from the
    previous frame (half-wave rectified), so a click, which raises every bin at once, stands far above speech."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < win + hop:
        return np.zeros(0), np.zeros(0)
    w = np.hanning(win)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * w
    mag = np.abs(np.fft.rfft(frames, axis=1)) / w.sum()
    logm = np.log10(np.maximum(mag, FLUX_FLOOR))
    flux = np.maximum(0.0, np.diff(logm, axis=0)).mean(axis=1)
    centres = np.arange(1, len(frames)) * hop + win / 2
    return flux, centres


def seam_stats(x: np.ndarray, rate: int, seams: list[int], controls: list[int],
               voiced_s: np.ndarray | None = None) -> dict:
    """Flux at each seam (the largest within half a window plus a hop of it) over the clip's median flux in voiced
    frames; seams above the clip's p99 flux (and at least SEAM_OVER x its median voiced flux: a steady tone's ripple,
    1.7x its median on a clean 220 Hz sine, is no jump); the same at the control points. `voiced_s` are the centres (s)
    of Praat's voiced 10 ms frames; without them the 40 dB energy rule decides what counts as voiced. A quarter-period
    phase break in a sine measures 583x (tests/test_tts_quality_bench.py)."""
    flux, centres = spectral_flux(x, rate)
    if not len(flux):
        return {"seams_n": len(seams), "control_n": len(controls)}
    if voiced_s is not None and len(voiced_s):
        near = np.searchsorted(np.asarray(voiced_s), centres / rate)
        lo = np.clip(near - 1, 0, len(voiced_s) - 1)
        hi = np.clip(near, 0, len(voiced_s) - 1)
        dist = np.minimum(np.abs(voiced_s[lo] - centres / rate), np.abs(voiced_s[hi] - centres / rate))
        voiced = dist <= STEP_S
    else:
        act = active_frames(x, rate)
        k = np.clip((centres / rate / FRAME_S).astype(int), 0, max(0, len(act) - 1))
        voiced = act[k] if len(act) else np.zeros(len(flux), bool)
    med = float(np.median(flux[voiced])) if voiced.any() else float(np.median(flux))
    den = max(med, FLUX_MIN_MEDIAN)
    p99 = float(np.percentile(flux, 99))
    over = max(p99, SEAM_OVER * den)
    reach = FLUX_WIN / 2 + FLUX_HOP

    def at(pos: int) -> float:
        sel = np.abs(centres - pos) <= reach
        return float(flux[sel].max()) if sel.any() else 0.0

    sv = [at(p) for p in seams]
    cv = [at(p) for p in controls]
    out = {"flux_median": med, "flux_p99": p99, "seams_n": len(sv), "control_n": len(cv)}
    if sv:
        out.update(seam_ratio_median=float(np.median(sv)) / den, seam_ratio_max=max(sv) / den,
                   seams_over_p99=int(sum(v > over for v in sv)))
    if cv:
        out.update(control_ratio_median=float(np.median(cv)) / den, control_ratio_max=max(cv) / den,
                   control_over_p99=int(sum(v > over for v in cv)))
    return out


def semitones(hz: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 100.0)


def f0_metrics(track_hz: np.ndarray, times_s: np.ndarray | None = None, span: tuple[float, float] | None = None
               ) -> dict:
    """Pitch statistics from a 10 ms F0 track (0 = unvoiced): range p5-p95 and sd in semitones, jumps between
    consecutive voiced frames (largest, p95, how many over 6 st), and the voiced share of `span` (first to last
    speech, seconds) or of the whole track. NaN-free: too little voicing gives None values."""
    f = np.asarray(track_hz, dtype=np.float64)
    v = f > 0
    if span is not None and times_s is not None and len(times_s):
        inside = (np.asarray(times_s) >= span[0]) & (np.asarray(times_s) <= span[1])
        n_span = int(inside.sum())
    else:
        n_span = len(f)
    out = {"voiced_s": float(v.sum() * STEP_S), "voiced_share": float(v.sum() / n_span) if n_span else None}
    if v.sum() < 10:
        out.update(f0_median_st=None, f0_range_st=None, f0_sd_st=None, jump_max_st=None, jump_p95_st=None,
                   jumps_over6=0, jump_pairs=0)
        return out
    st = semitones(f[v])
    st_all = np.zeros(len(f))
    st_all[v] = st
    pair = v[:-1] & v[1:]
    jumps = np.abs(np.diff(st_all))[pair]
    out.update(f0_median_st=float(np.median(st)), f0_range_st=float(np.percentile(st, 95) - np.percentile(st, 5)),
               f0_sd_st=float(np.std(st)), jump_pairs=int(pair.sum()),
               jump_max_st=float(jumps.max()) if len(jumps) else 0.0,
               jump_p95_st=float(np.percentile(jumps, 95)) if len(jumps) else 0.0,
               jumps_over6=int((jumps > 6.0).sum()))
    return out


def uncovered_voiced_s(voiced_s: np.ndarray, words: list[dict], pad: float = 0.1) -> float:
    """Seconds of voiced 10 ms frames outside every recognised word (each widened by `pad`): a laugh, a hum or babble
    the recogniser did not write down."""
    if not len(voiced_s):
        return 0.0
    covered = np.zeros(len(voiced_s), bool)
    for w in words:
        covered |= (voiced_s >= float(w["start"]) - pad) & (voiced_s <= float(w["end"]) + pad)
    return float((~covered).sum() * STEP_S)


def resample(x: np.ndarray, rate: int, to: int) -> np.ndarray:
    if rate == to:
        return np.asarray(x, dtype=np.float32)
    import soxr
    return soxr.resample(np.asarray(x, dtype=np.float32), rate, to, quality="HQ").astype(np.float32)


# ---------------------------------------------------------------------------------------------------- Praat

def f0_track_praat(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """(F0 in Hz per 10 ms frame with 0 for unvoiced, frame centres in s): Praat's autocorrelation tracker with the
    tone package's floor and ceiling."""
    import parselmouth
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=rate)
    p = snd.to_pitch_ac(time_step=STEP_S, pitch_floor=FLOOR_HZ, pitch_ceiling=CEILING_HZ)
    return np.asarray(p.selected_array["frequency"], dtype=np.float64), np.asarray(p.xs(), dtype=np.float64)


# ---------------------------------------------------------------------------------------------------- neural

def _cpu_only():
    import torch
    torch.set_grad_enabled(False)
    return torch


class AudioSetTagger:
    """AST on AudioSet over 1 s windows every `hop` seconds, CPU only."""

    def __init__(self, model_id: str = AST_ID, revision: str = AST_REV, hop: float = TAG_HOP_S):
        torch = _cpu_only()
        self.hop = float(hop)
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        kw = dict(revision=revision)
        try:
            self.fe = ASTFeatureExtractor.from_pretrained(model_id, local_files_only=True, **kw)
            self.model = ASTForAudioClassification.from_pretrained(model_id, use_safetensors=True,
                                                                   local_files_only=True, **kw)
        except OSError:   # not cached yet: the download is allowed (network only), then it loads on the CPU
            self.fe = ASTFeatureExtractor.from_pretrained(model_id, **kw)
            self.model = ASTForAudioClassification.from_pretrained(model_id, use_safetensors=True, **kw)
        self.model.eval()
        if any(p.device.type != "cpu" for p in self.model.parameters()):
            raise RuntimeError("AST must run on the CPU")
        names = {v: int(k) for k, v in self.model.config.id2label.items()}
        missing = [n for n in VOCAL_NONSPEECH + SPEECH if n not in names]
        if missing:
            raise RuntimeError(f"AudioSet labels not in {model_id}: {missing}")
        self.labels = self.model.config.id2label
        self.laugh = [names[n] for n in LAUGH]
        self.vocal = [names[n] for n in VOCAL_NONSPEECH]
        self.speech = [names[n] for n in SPEECH]

    def probs(self, x: np.ndarray, rate: int) -> tuple[np.ndarray, list[float]]:
        """Sigmoid probabilities per window (windows x 527) and the window starts (s)."""
        x16 = resample(x, rate, 16000)
        n, h = int(TAG_WIN_S * 16000), int(self.hop * 16000)
        starts = list(range(0, max(1, len(x16) - n + h), h)) if len(x16) > n else [0]
        wins = [x16[s:s + n] for s in starts]
        out = []
        for i in range(0, len(wins), TAG_BATCH):
            feats = self.fe(wins[i:i + TAG_BATCH], sampling_rate=16000, return_tensors="pt")
            logits = self.model(**feats).logits
            out.append(self.torch.sigmoid(logits).numpy())
        return np.concatenate(out), [s / 16000 for s in starts]

    def tag(self, x: np.ndarray, rate: int) -> dict:
        p, starts = self.probs(x, rate)
        laugh = p[:, self.laugh].max(axis=1)
        vocal = p[:, self.vocal].max(axis=1)
        speech = p[:, self.speech].max(axis=1)
        act = active_frames(x, rate)
        span = int(round(TAG_WIN_S / FRAME_S))
        parts = [act[int(round(s / FRAME_S)):int(round(s / FRAME_S)) + span] for s in starts]
        active = np.array([len(a) > 0 and float(a.mean()) >= TAG_ACTIVE for a in parts])
        beats = (vocal > speech) & active
        i = int(laugh.argmax())
        j = int(p[:, self.vocal].max(axis=0).argmax())
        return {"laugh_max": float(laugh.max()), "laugh_at_s": float(starts[i]),
                "laugh_label": self.labels[self.laugh[int(p[i, self.laugh].argmax())]],
                "nonspeech_share": float(beats.sum() / active.sum()) if active.any() else 0.0,
                "nonspeech_windows": int(beats.sum()), "tag_windows": int(active.sum()),
                "nonspeech_top": self.labels[self.vocal[j]], "nonspeech_top_p": float(p[:, self.vocal[j]].max())}


class MOSPredictor:
    """UTMOS22 strong (SpeechMOS), CPU only. The clip is resampled to 16 kHz here, so the model's own torchaudio
    resampling is a no-op."""

    def __init__(self):
        torch = _cpu_only()
        self.torch = torch
        self.model = torch.hub.load(UTMOS_REPO, "utmos22_strong", trust_repo=True, verbose=False).eval()
        if any(p.device.type != "cpu" for p in self.model.parameters()):
            raise RuntimeError("UTMOS must run on the CPU")

    def mos(self, x: np.ndarray, rate: int) -> float:
        x16 = resample(x, rate, 16000)
        return float(self.model(self.torch.from_numpy(x16).unsqueeze(0), 16000)[0])


# ---------------------------------------------------------------------------------------------------- per clip

def score_clip(x: np.ndarray, rate: int, side: dict, asr: dict | None = None, tagger: AudioSetTagger | None = None,
               mos: MOSPredictor | None = None, praat: bool = True) -> dict:
    """Every CPU metric of one clip; the neural ones when their models are given."""
    out: dict = {"scorer_version": SCORER_VERSION, "neural": bool(tagger or mos)}
    p = pauses(x, rate)
    out.update(length_s=p["length_s"], lead_energy_s=p["lead_s"], tail_s=p["tail_s"], longest_pause_s=p["longest_s"],
               pause_total_s=p["gaps_s"], n_pauses=len(p["gaps"]), speech_span_s=p["speech_span_s"])
    words, chars = int(side.get("n_words") or 0), int(side.get("n_chars") or 0)
    out["s_per_word"] = p["speech_span_s"] / words if words and p["speech"] else None
    out["s_per_char"] = p["speech_span_s"] / chars if chars and p["speech"] else None
    voiced_s = None
    if praat:
        track, times = f0_track_praat(x, rate)
        out.update(f0_metrics(track, times, (p["first_s"], p["last_s"]) if p["speech"] else None))
        voiced_s = times[track > 0]
        if asr and asr.get("words") is not None:
            out["uncovered_voiced_s"] = uncovered_voiced_s(voiced_s, asr["words"])
    out.update(seam_stats(x, rate, list(side.get("seams") or []), list(side.get("controls") or []), voiced_s))
    if tagger is not None:
        out.update(tagger.tag(x, rate))
    if mos is not None:
        out["mos"] = mos.mos(x, rate)
    return out


CLIP = re.compile(r"^[a-z0-9][a-z0-9-]*-\d+\.json$")


def clip_sidecars(run_dir: Path) -> list[Path]:
    return sorted(p for p in Path(run_dir).glob("*/*.json") if CLIP.match(p.name))


def gpu_check(cmd: str | None) -> tuple[bool, str]:
    if not cmd:
        return True, "no check"
    try:
        p = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"gpu check did not run: {e}"
    lines = p.stdout.strip().splitlines()
    return p.returncode == 0, (lines[-1] if lines else f"exit {p.returncode}")


def score_run(run_dir: Path, *, neural: bool = True, gpu_clear: str | None = GPU_CLEAR, force: bool = False,
              limit: int | None = None, tag_hop: float = TAG_HOP_S, log=print) -> int:
    """Score every clip of a run that has no score yet (or one without the neural metrics when they are wanted). A
    clip the render stage copied from an earlier setting ("reused_from": the same audio) gets a copy of that clip's
    score once it exists; rendered clips are scored first."""
    todo = []
    for side_path in clip_sidecars(run_dir):
        out = side_path.with_suffix(".score.json")
        if force or not out.exists():
            todo.append(side_path)
        elif neural and not json.loads(out.read_text()).get("neural"):
            todo.append(side_path)
    todo.sort(key=lambda p: bool(json.loads(p.read_text()).get("reused_from")))
    todo = todo[:limit] if limit else todo
    if not todo:
        log(f"score: every clip in {run_dir} is scored")
        return 0
    tagger = mos = None
    if neural:
        ok, line = gpu_check(gpu_clear)
        log(f"gpu_clear: {line}")
        if not ok:
            log("score: the neural scorers load only after gpu_clear.sh exits 0; rerun later or pass --no-neural")
            return 3
        t = time.monotonic()
        tagger, mos = AudioSetTagger(hop=tag_hop), MOSPredictor()
        log(f"score: AST and UTMOS22 loaded on the CPU in {time.monotonic() - t:.1f} s")
    t0 = time.monotonic()
    copied = 0
    for k, side_path in enumerate(todo, 1):
        side = json.loads(side_path.read_text())
        src = side.get("reused_from")
        src_score = side_path.parent.parent / str(src) / side_path.with_suffix(".score.json").name if src else None
        if src_score is not None and src_score.exists():
            res = json.loads(src_score.read_text())
            if res.get("neural") == neural or not neural:
                res["copied_from"] = src
                tmp = side_path.with_suffix(".score.json.tmp")
                tmp.write_text(json.dumps(res, indent=1))
                tmp.replace(side_path.with_suffix(".score.json"))
                copied += 1
                continue
        x, rate = read_wav(side_path.with_suffix(".wav"))
        asr_path = side_path.with_suffix(".asr.json")
        asr = json.loads(asr_path.read_text()) if asr_path.exists() else None
        t = time.monotonic()
        res = score_clip(x, rate, side, asr, tagger, mos)
        res["score_s"] = round(time.monotonic() - t, 3)
        tmp = side_path.with_suffix(".score.json.tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(side_path.with_suffix(".score.json"))
        if k % 25 == 0 or k == len(todo):
            log(f"score: {k}/{len(todo)} clips ({copied} copied), {time.monotonic() - t0:.0f} s")
    return 0


# ---------------------------------------------------------------------------------------------------- synthetic

def tone(f0_hz: np.ndarray, rate: int, harmonics: int = 12, amp: float = 0.3) -> np.ndarray:
    """A harmonic tone following an F0 contour (one value per sample), phase-continuous, 1/k harmonic amplitudes."""
    phase = 2 * np.pi * np.cumsum(f0_hz) / rate
    out = sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))
    return (amp * out / np.abs(out).max()).astype(np.float32)


def pitch_signals(rate: int = 24000, seconds: float = 2.0) -> dict[str, np.ndarray]:
    """Three planted pitch cases: steady 120 Hz, a glide 100 -> 200 Hz (12 st, smooth), and 110/185 Hz alternating
    every 0.25 s (9.0 st jumps, seven of them)."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    steady = np.full(n, 120.0)
    glide = 100.0 * 2.0 ** (t / seconds)
    jumps = np.where((t // 0.25).astype(int) % 2 == 0, 110.0, 185.0)
    return {k: tone(v, rate) for k, v in {"steady": steady, "glide": glide, "jumps": jumps}.items()}


def selftest_pitch() -> dict:
    out = {}
    for name, x in pitch_signals().items():
        track, times = f0_track_praat(x, 24000)
        out[name] = f0_metrics(track, times)
    return out


def _resonate(x: np.ndarray, rate: int, f: float, bw: float) -> np.ndarray:
    r = np.exp(-np.pi * bw / rate)
    a1, a2 = -2 * r * np.cos(2 * np.pi * f / rate), r * r
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = x[i] - a1 * (y[i - 1] if i else 0.0) - a2 * (y[i - 2] if i > 1 else 0.0)
    return y


def smoke_clips(rate: int = 24000) -> dict[str, np.ndarray]:
    """Two synthetic clips for the CPU smoke (no speech model may run): a vowel-like harmonic source through three
    formants with a 4 Hz syllable envelope, and the same source in 5 Hz breathy bursts at a higher pitch."""
    rng = np.random.default_rng(5)
    n = rate * 3
    t = np.arange(n) / rate
    src = tone(120 * (1 + 0.04 * np.sin(2 * np.pi * 0.7 * t)), rate)
    vowel = sum(_resonate(src.astype(np.float64), rate, f, bw) for f, bw in ((700, 90), (1200, 110), (2600, 160)))
    vowel *= 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 4 * t), 0, None)
    burst_src = tone(230 * (1 + 0.1 * np.sin(2 * np.pi * 5 * t)), rate) + 0.3 * rng.standard_normal(n)
    bursts = sum(_resonate(burst_src.astype(np.float64), rate, f, bw) for f, bw in ((800, 150), (1300, 200)))
    bursts *= np.clip(np.sin(2 * np.pi * 5 * t), 0, None) ** 2
    norm = lambda y: (0.3 * y / np.abs(y).max()).astype(np.float32)   # noqa: E731
    return {"vowel": norm(vowel), "bursts": norm(bursts)}


# ---------------------------------------------------------------------------------------------------- CLI

def download(log=print) -> int:
    """Fetch the two models into the user's caches. Network only: nothing is loaded or run."""
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    import torch.hub
    from huggingface_hub import snapshot_download

    t = time.monotonic()
    path = snapshot_download(AST_ID, revision=AST_REV, allow_patterns=["*.json", "model.safetensors"])
    size = sum(f.stat().st_size for f in Path(path).iterdir() if f.is_file())
    log(f"AST: {path} ({size:,} bytes, {time.monotonic() - t:.1f} s)")
    hub = Path(torch.hub.get_dir())
    ckpt = hub / "checkpoints" / UTMOS_URL.rsplit("/", 1)[1]
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        t = time.monotonic()
        torch.hub.download_url_to_file(UTMOS_URL, str(ckpt), progress=False)
        log(f"UTMOS22 checkpoint: {ckpt} ({ckpt.stat().st_size:,} bytes, {time.monotonic() - t:.1f} s)")
    else:
        log(f"UTMOS22 checkpoint: {ckpt} ({ckpt.stat().st_size:,} bytes, cached)")
    owner, _, rest = UTMOS_REPO.partition("/")
    name, _, ref = rest.partition(":")
    repo_dir = hub / f"{owner}_{name}_{ref}"     # torch.hub's own layout for "owner/name:ref"
    if not repo_dir.exists():
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "repo.zip"
            urllib.request.urlretrieve(UTMOS_ZIP, z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
                top = Path(d) / zf.namelist()[0].split("/")[0]
            shutil.move(str(top), repo_dir)
        log(f"SpeechMOS code: {repo_dir} ({sum(f.stat().st_size for f in repo_dir.rglob('*') if f.is_file()):,} bytes)")
    else:
        log(f"SpeechMOS code: {repo_dir} (cached)")
    return 0


def smoke(gpu_clear: str | None, log=print) -> int:
    """Both neural scorers on the two synthetic clips, on the CPU, right after gpu_clear.sh says CLEAR."""
    ok, line = gpu_check(gpu_clear)
    log(f"gpu_clear: {line}")
    if not ok:
        log("smoke: not run (BUSY)")
        return 3
    import torch
    log(f"torch {torch.__version__}, threads {torch.get_num_threads()}, device cpu")
    t = time.monotonic()
    tagger = AudioSetTagger()
    log(f"AST loaded in {time.monotonic() - t:.2f} s")
    t = time.monotonic()
    mos = MOSPredictor()
    log(f"UTMOS22 strong loaded in {time.monotonic() - t:.2f} s")
    for name, x in smoke_clips().items():
        t = time.monotonic()
        tag = tagger.tag(x, 24000)
        t_tag = time.monotonic() - t
        t = time.monotonic()
        m = mos.mos(x, 24000)
        t_mos = time.monotonic() - t
        p, _ = tagger.probs(x, 24000)
        top = [(tagger.labels[int(i)], round(float(p.max(axis=0)[i]), 3)) for i in np.argsort(-p.max(axis=0))[:3]]
        log(json.dumps({"clip": name, "seconds": len(x) / 24000, "tag_s": round(t_tag, 2), "mos_s": round(t_mos, 2),
                        "mos": round(m, 3), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in tag.items()},
                        "top3": top}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score every clip of a run")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--no-neural", action="store_true", help="Praat, flux and pauses only (no model loads)")
    s.add_argument("--gpu-clear", default=GPU_CLEAR, help="the check that must exit 0 before the neural models load")
    s.add_argument("--force", action="store_true")
    s.add_argument("--limit", type=int)
    s.add_argument("--tag-hop", type=float, default=TAG_HOP_S,
                   help="seconds between the AudioSet tagger's 1 s windows (1.0 halves its CPU time)")
    sub.add_parser("download", help="fetch AST and UTMOS22 (network only)")
    sm = sub.add_parser("smoke", help="the neural scorers on two synthetic clips, CPU")
    sm.add_argument("--gpu-clear", default=GPU_CLEAR)
    sub.add_parser("selftest-pitch", help="Praat metrics of the planted pitch signals, as JSON")
    a = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.cmd == "score":
        return score_run(a.run_dir, neural=not a.no_neural, gpu_clear=a.gpu_clear, force=a.force, limit=a.limit,
                         tag_hop=a.tag_hop)
    if a.cmd == "download":
        return download()
    if a.cmd == "smoke":
        return smoke(a.gpu_clear)
    print(json.dumps(selftest_pitch()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
