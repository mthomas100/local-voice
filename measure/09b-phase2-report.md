# 09b — voice components measured on this Mac, phase 2 (final)

**Status: complete for the agreed plan, except VoiceChat 11B 8-bit.** That checkpoint is not downloaded, and the
instruction was not to download its 13.9 GB now. Phase-2 numbers come from two windows, both with nothing else
resident on the GPU:

- **2026-10-04, 22:01–22:05:** Kokoro, Qwen3-TTS 0.6B and 1.7B, Pocket TTS cold, Parakeet. Per-run detail is in
  the `raw/clean_*.jsonl` files.
- **2026-10-05, 08:45:36–08:48:14,** inside one exclusive hold taken by `bench/run_remaining.sh`: Pocket TTS warm,
  VibeVoice-Realtime, Marvis, Silero VAD, Smart Turn v3, Nemotron streaming ASR, the two co-residency pairs, and
  VoiceChat 11B 4-bit.

- **Machine:** Apple M5 Max, 128 GB (112 GiB Metal working set), macOS 27.
- **Runtimes:** the mlx-audio clone at `~/repos/mlx-audio` (0.5.3 plus 21 commits, `ee0c65d`) with MLX 0.31.2, for
  everything except VoiceChat. VoiceChat ran on mlx-vlm 0.7.4 with MLX 0.32.3 and mlx-audio 0.5.7 (`bench/.venv-vlm`),
  the only runtime that loads its checkpoint (report 09c, section 8).
- **What was resident on 2026-10-05:** nothing. The pre-flight gate at 08:44:30 saw the hold gate open, no GPU job,
  and qwen38 loaded but idle since 01:09:29. The hold (`hd5dae0`, kind `bench`) unloaded qwen38 in 7.7 s and was
  granted at 08:45:36. All 11 per-phase gate checks under it said GO. llama-swap received no POST from 08:45:28 on.
  The hold was released at 08:48:16, when the runner exited; qwen38 stays unloaded until its next request.
- **Evidence:** JSON in `raw/phase2-20261005-084528/`. The full command log (setup output, every gate block) was kept
  locally and is not published. Full outputs and WAVs are written to `bench/results/<run>/`, which is gitignored.
- **Protocol:** unchanged from phase 1. One model per process, warm-up first, three runs per text and mode, medians
  with all runs, `mx.reset_peak_memory()` before each run, and `/usr/bin/time -l` for the process. TTFA is the time
  from calling `generate()` to the first chunk materialised as NumPy. Every download (the models, plus the auxiliary
  files setup.sh fetched at 08:45) finished before the hold, and the run had `HF_HUB_OFFLINE=1`, so no load included
  a download.
- **Texts:** short, 11 words, "The kettle is on, and the rain has finally stopped outside." Long, 46 words, "Before
  we start the meeting, let me summarize where things stand. The new build passed every test last night, the design
  review is scheduled for Thursday morning, and two customers have already asked when they can try the beta. I think
  we are in good shape."
- **Clips** (`say -v Samantha`, then `afconvert -f WAVE -d LEI16@16000 -c 1`): clip3 is 2.48 s, "Could you check
  the weather for tomorrow afternoon?"; clip8 is 9.27 s, "I was thinking we could move the planning meeting to
  Thursday afternoon, since two people are travelling on Wednesday. Does that work for you, or would Friday morning
  be better?"

## 1. Headline: what each component costs

| component | first output | compute | MLX memory | caveat |
|---|---|---|---|---|
| Pocket TTS (warm) | **18 ms** first chunk (interval 0.32) | RTF 0.04 | 0.22 GiB weights, 0.89 GiB peak | leading silence 0.04–0.26 s |
| VibeVoice-Realtime 0.5B 8-bit | **31–35 ms** first chunk (0.32) | RTF 0.12–0.15 | 1.05 GiB active, 1.36–1.42 GiB peak streaming | leading silence 0.16–0.86 s |
| Kokoro-82M (22:01) | 74–79 ms for a whole sentence (no streaming) | RTF 0.02 | 0.30 GiB, 1.39–2.45 GiB peak | leading silence 0.29 s (phase 1) |
| Qwen3-TTS 0.6B CustomVoice (22:01) | **78 ms** first chunk (0.32) | RTF 0.20–0.22 | 2.39 GiB, 2.87 GiB peak streaming | leading silence 0.42 s (phase 1) |
| Qwen3-TTS 1.7B CustomVoice (22:01) | 103 ms first chunk (0.32) | RTF 0.26–0.28 | 4.27 GiB, 4.75 GiB peak streaming | leading silence 0.08 s (phase 1) |
| Marvis TTS 250M v0.2 8-bit | 117 ms first chunk (0.32) | RTF 0.25 | 1.05 GiB, 2.35 GiB peak | **no leading silence** in any first-run WAV |
| Parakeet TDT 0.6B v3 (22:03) | 23 ms for the whole 2.48 s clip | about 100× real time | 2.38 GiB, 2.44 GiB peak | batch only, no live session |
| Nemotron 3.5 ASR streaming 0.6B 8-bit | final transcript **53–54 ms** after the audio ends; partials every 0.96–1.28 s of audio | 45× real time unpaced | 0.74 GiB, 0.93 GiB peak | 1.12 s native chunk |
| Silero VAD | 0.4 ms per 32 ms chunk (p50; p95 0.7–0.9 ms) | | 3 MB | |
| Smart Turn v3 | 1.3–1.7 ms per call (p50; p95 1.4–4.7 ms) | | 0.03 GiB, 0.09 GiB peak | judged the 9.3 s question incomplete (0.22) |
| Parakeet + Kokoro, one process | STT 28 ms, TTS 65 ms (warm) | | 2.68 GiB active, **3.63 GiB peak** | |
| Parakeet + Qwen3-TTS 0.6B, one process | STT 22 ms, TTS 103 ms (warm, interval 0.5) | | 4.76 GiB active, **5.36 GiB peak** | |
| **VoiceChat 11B 4-bit** (mlx-vlm) | first assistant audio 0.24 s after the first clip ended (0.39 s after its last speech) | **59.8 ms p50, 61.0 ms p95 per 80 ms frame; RTF 0.77** | 8.58 GiB after load, **10.27 GiB peak** | the first two frames cost 0.92 s and 0.51 s |

## 2. TTS

### 2a. New on 2026-10-05: per-run tables

Pocket TTS, warm rerun (`tts_pocket_warm.jsonl`): load 0.867 s, MLX active after load 0.22 GiB, warm-up 0.141 s,
MLX peak 0.89 GiB. Max RSS 0.47 GB, peak footprint 2.55 GB, 7.6 s wall for the process.

| text | mode | TTFA s median (runs) | total s median (runs) | audio s (runs) | RTF | first chunk audio s |
|---|---|---|---|---|---|---|
| short | non-stream | 0.143 (0.136, 0.160, 0.143) | 0.143 (0.136, 0.160, 0.143) | 3.44 (3.28, 3.84, 3.44) | 0.042 | 3.28 |
| short | stream 0.32 | **0.018** (0.018, 0.020, 0.018) | 0.150 (0.150, 0.175, 0.143) | 3.52 (3.52, 4.00, 3.44) | 0.043 | 0.32 |
| short | stream 0.5 | 0.027 (0.027, 0.028, 0.027) | 0.152 (0.152, 0.167, 0.143) | 3.68 (3.68, 4.00, 3.44) | 0.041 | 0.56 |
| long | non-stream | 0.517 (0.516, 0.517, 0.534) | 0.517 (0.516, 0.517, 0.534) | 12.88 (12.72, 12.88, 13.20) | 0.040 | 12.72 |
| long | stream 0.32 | **0.018** (0.018, 0.019, 0.018) | 0.582 (0.601, 0.502, 0.582) | 13.04 (14.08, 12.16, 13.04) | 0.043 | 0.32 |
| long | stream 0.5 | 0.028 (0.028, 0.028, 0.028) | 0.539 (0.537, 0.539, 0.597) | 13.12 (13.12, 13.12, 14.32) | 0.041 | 0.56 |

The warm rerun replaces the cold pass's erratic long-text numbers. Those were 1.5–3.3 s for non-stream, while three
downloads were writing to disk; warm, it is 0.52 s for 12.9 s of audio. Load fell from 17.5 s to 0.87 s and warm-up
from 15.7 s to 0.14 s, now that the tokenizer and voice embedding are cached.

VibeVoice-Realtime 0.5B 8-bit, voice `en-Carter_man` (`tts_vibevoice.jsonl`): load 0.574 s, MLX active 1.05 GiB,
warm-up 1.48 s. Max RSS 1.40 GB, 22.2 s wall. The `time -l` peak footprint read 22.98 GB, against a measured MLX peak
of 3.80 GiB; that figure counts transient Metal allocations and is not what stays resident.

| text | mode | TTFA s median (runs) | total s median (runs) | audio s (runs) | RTF | first chunk audio s | MLX peak GiB |
|---|---|---|---|---|---|---|---|
| short | non-stream | 0.329 (0.329, 0.437, 0.299) | 0.329 (0.329, 0.437, 0.300) | 2.80 (2.80, 4.00, 2.80) | 0.109 | 2.80 | 3.12 |
| short | stream 0.32 | **0.031** (0.043, 0.031, 0.031) | 0.430 (0.430, 0.424, 0.465) | 3.47 (3.47, 3.47, 3.87) | 0.122 | 0.267 | 1.35 |
| short | stream 0.5 | 0.045 (0.045, 0.045, 0.051) | 0.457 (0.367, 0.457, 0.477) | 3.47 (3.20, 3.47, 3.73) | 0.128 | 0.40 | 1.41 |
| long | non-stream | 1.651 (1.842, 1.651, 1.458) | 1.651 (1.842, 1.651, 1.459) | 14.00 (14.13, 14.00, 12.93) | 0.118 | 14.13 | 3.80 |
| long | stream 0.32 | **0.035** (0.036, 0.034, 0.035) | 1.978 (1.978, 1.972, 2.050) | 14.00 (14.27, 13.47, 14.00) | 0.146 | 0.267 | 1.36 |
| long | stream 0.5 | 0.048 (0.050, 0.048, 0.046) | 1.721 (1.789, 1.721, 1.691) | 13.73 (13.73, 13.87, 12.67) | 0.130 | 0.40 | 1.42 |

VibeVoice's first chunk holds 0.267 s of audio at interval 0.32 and 0.40 s at 0.5, so its chunking unit is not the
12.5 Hz frame the other models use.

Marvis TTS 250M v0.2 8-bit through mlx-audio's sesame loader, default voice prompt `conversational_a`
(`tts_marvis.jsonl`): load 1.019 s, MLX active 1.05 GiB, warm-up 1.48 s. Max RSS 1.42 GB, peak footprint 6.11 GB,
40.9 s wall. It is runnable through mlx-audio today, once the Mimi codec, the tokenizer and the voice prompt are
cached.

| text | mode | TTFA s median (runs) | total s median (runs) | audio s (runs) | RTF | first chunk audio s | MLX peak GiB |
|---|---|---|---|---|---|---|---|
| short | non-stream | 1.006 (1.006, 1.068, 0.905) | 1.018 (1.018, 1.080, 0.917) | 3.92 (3.92, 4.16, 3.52) | 0.260 | 3.92 | 2.37 |
| short | stream 0.32 | **0.117** (0.117, 0.115, 0.119) | 0.818 (0.795, 0.949, 0.818) | 3.12 (3.04, 3.68, 3.12) | 0.261 | 0.32 | 2.35 |
| short | stream 0.5 | 0.156 (0.154, 0.156, 0.156) | 0.957 (0.909, 0.957, 1.260) | 3.68 (3.52, 3.68, 4.96) | 0.258 | 0.48 | 2.35 |
| long | non-stream | 3.325 (3.183, 3.338, 3.325) | 3.363 (3.219, 3.376, 3.363) | 13.28 (12.64, 13.36, 13.28) | 0.253 | 12.64 | 2.58 |
| long | stream 0.32 | **0.118** (0.120, 0.118, 0.117) | 3.285 (3.389, 3.285, 3.119) | 13.28 (13.76, 13.28, 12.64) | 0.247 | 0.32 | 2.35 |
| long | stream 0.5 | 0.157 (0.157, 0.158, 0.156) | 3.229 (3.124, 3.329, 3.229) | 13.20 (12.72, 13.52, 13.20) | 0.246 | 0.48 | 2.35 |

### 2b. Leading silence and first audible sound

Leading silence is the first 20 ms frame whose peak exceeds 0.01, the phase-1 method. It was measured on the first
run of each text and mode only, because only those WAVs are saved:

| model | leading silence, first runs | TTFA at 0.32 | first audible sound, about |
|---|---|---|---|
| Marvis | 0.00 s in all six WAVs | 0.117 s | 0.12 s |
| Pocket TTS | 0.04 (short, non-stream), 0.00 (long, non-stream), 0.20 / 0.10 (stream 0.32), 0.26 / 0.18 (stream 0.5) | 0.018 s | 0.05–0.28 s |
| Qwen3-TTS 1.7B (phase 1) | 0.08 s | 0.103 s | 0.18 s |
| Kokoro (phase 1) | 0.29 s | 0.074 s (whole sentence) | 0.37 s |
| Qwen3-TTS 0.6B (phase 1) | 0.42 s | 0.078 s | 0.50 s |
| VibeVoice | 0.16 / 0.86 (non-stream), 0.52 / 0.74 (stream 0.32), 0.52 / 0.52 (stream 0.5) | 0.031–0.035 s | 0.55–0.78 s |

Pairs are short / long text. For Qwen3-TTS, report 09c section 3 explains where the silence comes from: the talker
generates it, and no API knob controls it. Energy-gating the first chunks removes it on any of these models.

### 2c. Notes

- At interval 0.32 the first chunk orders as Pocket (18 ms), VibeVoice (31–35 ms), Qwen3-TTS 0.6B (78 ms), Qwen3-TTS
  1.7B (103 ms), Marvis (117 ms). Counting leading silence, Marvis and Pocket make the first sound soonest.
- Every streaming model's first chunk arrives in 18–118 ms, far inside a 300 ms budget for the TTS share. RTF is
  0.04–0.28, so all of them generate several times faster than playback.
- None of the new models loads in more than 1.1 s warm, and none needs more than 2.6 GiB of MLX memory at peak.

## 3. STT

**Parakeet TDT 0.6B v3** (2026-10-04 22:03, `raw/clean_stt_parakeet.jsonl`): 0.023 s on the 2.48 s clip from a
path (0.023, 0.023, 0.024) and 0.025 s from a pre-loaded array (0.024, 0.026, 0.025). Transcript exact. Load 1.07 s,
MLX peak 2.44 GiB, max RSS 2.64 GB. In the co-residency rerun below it took 0.022–0.028 s warm.

**Nemotron 3.5 ASR streaming 0.6B 8-bit** through mlx-audio's live `StreamingSession` (`nemotron_stream.jsonl`).
Load 0.51 s, MLX active 0.74 GiB, peak 0.93 GiB, max RSS 0.89 GB. The session reports 16 kHz input, a native encoder
chunk of 112 mel frames × hop 160 = 1.12 s, and attention context [56, 13]. The bench fed 320 ms chunks:

| run | clip | when partials arrived (wall s, with audio fed so far) | final transcript | compute per 320 ms chunk |
|---|---|---|---|---|
| paced (real time) | clip3, 2.48 s, first clip in the process | "Could you check the weather" at 1.600 (1.28 s fed); "for tomorrow afternoon?" at 2.526–2.537 (2.483 s, after close) | complete at 2.537 s, **54 ms** after the audio ended; exact | p50 1.7 ms, p95 230 ms, max 326 ms (first encoder chunks include kernel compilation) |
| paced (real time) | clip8, 9.27 s | bursts at 1.31, 2.59, 3.55, 4.83, 5.79, 7.09, 8.04 s (audio fed 1.28, 2.56, 3.52, 4.80, 5.76, 7.04, 8.00 s), the rest after close | complete at 9.326 s, **53 ms** after the audio ended | p50 1.7 ms, p95 48 ms, max 56 ms (sum 0.37 s) |
| unpaced (compute only) | clip8 | 73 deltas in 0.207 s | 0.207 s for 9.27 s of audio, **45× real time** | p50 0.2 ms, p95 31 ms, max 37 ms |

The clip8 transcript: "I was thinking we could move the planning meeting to Thursday afternoon since two people are
traveling on Wednesday.  Does that work for you, or would Friday morning be better?" It differs from the input only
in the missing comma before "since", the US spelling "traveling" and a double space.

Partials arrive in bursts each time a 1.12 s encoder chunk completes. With 320 ms feeding that is every 0.96 or
1.28 s of audio, and on clip8 each burst lands 28–46 ms after the chunk that completed it. A word therefore appears roughly
0.3–1.3 s after it is spoken, depending on where it falls in the chunk. The final transcript needs only the last
partial chunk, about 53 ms after the audio ends. The deltas are subword pieces such as "we", "a", "ther"; the
session's output shape is described in 09c section 5.

## 4. VAD and turn detection

Silero VAD and Smart Turn v3 ran in one process per clip (`vad.jsonl`). Max RSS 0.15 GB; MLX active 3 MB for Silero
and 33 MB with Smart Turn added; peak 0.09 GiB.

| measure | clip8 (9.27 s) | clip3 (2.48 s) |
|---|---|---|
| Silero load | 0.48 s | 0.47 s |
| Silero `feed()` per 512-sample (32 ms) chunk | p50 0.4 ms, p95 0.7 ms, max 1.5 ms (289 chunks) | p50 0.4 ms, p95 0.9 ms, max 1.1 ms (77 chunks) |
| Silero speech chunks / first / last speech | 264 of 289 / 0.032 s / 9.088 s | 71 of 77 / 0.032 s / 2.304 s |
| Silero whole clip (`predict_proba`, 5 runs) | p50 26.5 ms | p50 7.6 ms |
| Silero `get_speech_timestamps` | 26.4 ms: segments 0.00–3.42, 3.55–6.01, 6.15–7.29, 7.43–9.15 s | 7.3 ms: segment 0.00–2.33 s |
| Smart Turn `predict_endpoint`, full clip, 10 calls | p50 1.7 ms, p95 4.7 ms: **incomplete**, probability 0.215 | p50 1.3 ms, p95 1.4 ms: **complete**, probability 0.989 |
| Smart Turn, clip cut at 45%, 5 calls | cut at 4.17 s: incomplete, 0.0075; p50 1.5 ms | cut at 1.12 s: incomplete, 0.010; p50 1.3 ms |

Smart Turn got three of four right. It scored the full 9.27 s clip as incomplete (0.215), although it ends on a
finished question ("…or would Friday morning be better?") with 0.12 s of audio after the speech. The short question
scored 0.989. Smart Turn sees only the last 8 s of a call (09c section 7). At the default threshold of 0.5, a turn
like clip8 would have to end by the VAD's silence timeout instead. Four judgments are too few for an error rate; this
is one observed miss.

## 5. Co-residency: Parakeet and a TTS model in one process

Run at 08:47 with nothing resident (`coresident_*.jsonl`). Each pair transcribes clip3 and then speaks a 10-word reply
("Sure, I can check the weather for tomorrow afternoon."), twice.

| pair | STT load s | TTS load s | MLX active after both | MLX peak | max RSS / footprint | round 1 (cold): STT s / TTS TTFA s / TTS total s | round 2 (warm) |
|---|---|---|---|---|---|---|---|
| Parakeet + Kokoro (`.venv-kokoro`) | 0.647 | 0.047 | **2.68 GiB** | **3.63 GiB** | 3.15 GB / 5.37 GB | 0.485 / 0.826 / 0.834 (1 chunk) | **0.028 / 0.065 / 0.070** |
| Parakeet + Qwen3-TTS 0.6B, stream 0.5 | 0.486 | 0.648 | **4.76 GiB** | **5.36 GiB** | 5.27 GB / 6.54 GB | 0.041 / 0.120 / 0.813 (8 chunks) | **0.022 / 0.103 / 0.877** (9 chunks) |

Memory is identical to phase 1, which ran beside an idle resident LLM. This time Qwen3-TTS's reply was a normal 8–9
chunks; phase 1 saw a 29-chunk over-generation. The first Kokoro call in a process costs 0.83 s, for kernels plus
G2P initialisation, so warm it at startup.

## 6. VoiceChat 11B 4-bit, full duplex (mlx-vlm 0.7.4)

The model was `mlx-community/NemotronLabs-VoiceChat-11B-4bit`, snapshot `dffd203`, run with
`bench/voicechat_bench.py --runtime vlm` (`voicechat.jsonl`), system prompt "You are a helpful assistant. Be
concise and answer in one sentence.", seed 0, profiling on. The input was 67.27 s of synthetic conversation: clip3
and clip8 alternating as six user utterances, separated by 4–7 s of silence. User speech ran 0.00–2.48, 6.48–15.75,
21.76–24.24, 29.24–38.51, 45.51–47.99 and 51.99–61.26 s. The bench fed it frame by frame as fast as possible, so the
per-frame time is pure compute.

| measure | value |
|---|---|
| load (`mlx_vlm.load` plus eval) | 7.17 s; MLX active 8.58 GiB |
| session creation (system prompt prefill, TTS warm-up) | 3.15 s; active 8.86 GiB |
| frames | 840 × 80 ms (1280 samples at 16 kHz) |
| **compute per frame, all 840** | **p50 59.8 ms, p95 61.0 ms**, mean 61.5 ms, max 915 ms |
| compute per frame after the first 10 | p50 59.8 ms, p95 61.0 ms, max 78.0 ms |
| frames over 80 ms | 2: frame 0 at 915 ms and frame 1 at 510 ms, both warm-up. Frame 31, the first with assistant text, took 78 ms |
| wall for 67.27 s of input | 51.65 s: **real-time factor 0.77** |
| per stage after the first 10 frames (runtime profiler), p50 / p95 | perception 14.6 / 14.8 ms; user-transcript RNNT 0.45 / 1.30 ms; language (Nemotron-H) 13.8 / 14.1 ms; TTS 27.5 / 28.1 ms; codec 3.2 / 3.3 ms; total 59.8 / 60.9 ms |
| **MLX peak during the session** | **10.27 GiB**; active at the end 9.27 GiB; max RSS 9.81 GB; peak footprint 15.28 GB |
| output audio | 67.2 s at 22.05 kHz; leading silence 2.76 s; voiced fraction 0.41 (20 ms frames above 0.01); RMS 0.012 |
| **speech or not** (machine check) | **speech**: Parakeet transcribed 133 words of English from the output WAV, in 1.40 s |
| first assistant text, on the session timeline | 2.48 s, the frame in which the first user utterance ended |
| first assistant audio | 2.72 s: **0.24 s after the first clip ended**, and 0.39 s after its last speech (Silero puts clip3's speech end at 2.33 s) |
| later replies | each final reply's text began 0.06–0.11 s before its clip ended: 15.68 vs 15.75, 24.16 vs 24.24, 38.40 vs 38.51, 47.92 vs 47.99, 61.20 vs 61.26 s. Each clip ends with 0.12–0.15 s of silence, so the text started about as the speech ended |
| speaking into pauses | during each 9.27 s utterance, which pauses after "afternoon," and "Wednesday.", it emitted 2–3 short text fragments before the final reply |
| function channel | **1 event in 67 s**: frame 733 (58.6 s), token 2168, text " would". No tools were declared in the prompt; no tool-call structure appeared |
| VoiceChat 8-bit | **not measured**: not downloaded (`~/.cache/huggingface/hub` has only the 4-bit) |

Assistant text channel, verbatim:

> I do not have real-time access to weather data, but I can suggest ways to find it. You could check a weather app,
> a website like weather dot COM comma or contact a local weather service. Would you like help with any of those
> options?Sure, what is there you are lookingSince Wednesday is not work for you, would?Great, Wednesday would you
> prefer another day?Since Friday morning works for you, I will adjust the meeting to Thursday afternoon. Would you
> like me to send out the updated schedule to the group?I do not have real-time access to weather data, but I can
> suggest ways to find it. You could check a weather app, a website like weather dot COM comma or contact a local
> weather service. Would you like help with any of those options?Sure, since you are bookingSince Wednesday does not
> work for you, would you forGreat, or would you prefer another daySince Friday morning works for you, I will adjust
> the meeting to Thursday afternoon. Would you like me to send out the updated schedule to the group?I do not have
> real-time access to weather data, but I can suggest ways to find it. You could check a weather app, a website like
> weather dot COM comma or contact a local weather service. Would you like help with any of those options?Sure, since
> you are bookingSince Wednesday does not work for you, would work forGreat, or would would you prefer another Friday
> morning morning?Since Friday morning works for you, I will adjust the meeting to Thursday afternoon. Would you like
> me to send out the updated schedule to the group?

Parakeet's transcript of the output audio, verbatim:

> I do not have real-time access to weather data, but I can suggest ways. Sure. Since Wednesday Great Wednesday.
> Since Friday morning works for you, I will adjust the meeting to Thursday afternoon. Would you like me to send out
> the updated? I do not have real-time access to weather data, but I can suggest ways to find it. Sure. Since
> Wednesday, great. Since Friday morning works for you, I will adjust the meeting to Thursday afternoon. Would you like
> me to send out the updated schedule to the group? I do not have real-time access to weather data, but I can suggest
> ways to since Wednesday. Great. Or would Since Friday morning works for you, I will adjust the meeting to Thursday
> afternoon. Would you like me to send out the updated

User-transcript channel (the model's own RNNT), verbatim: "Could you check the weather for tomorrow afternoon I was
thinking we could move the planning meeting to Thursday afternoon since two people are traveling on Wednesday does
that work for you or would Friday morning be better the weather for tomorrow afternoon I was thinking …". It is
accurate except that "Could you check" is missing from the second clip3.

The spoken audio is shorter than the text channel: Parakeet's transcript lacks some text-channel words, such as the
end of the first weather answer. That is consistent with the speech being cut when the user starts talking again,
though this run did not test it directly.

## 7. What this means for the real-time budget

- **Cascade.** End of speech to first audible TTS sound can now be budgeted from measured parts. Smart Turn takes
  about 2 ms per decision. The final transcript takes 22–28 ms with Parakeet on the utterance, or about 53 ms after
  the audio ends with Nemotron streaming. The LLM's time to first token is not measured here. TTS first sound takes
  0.12 s (Marvis), 0.05–0.28 s (Pocket), 0.18 s (Qwen3-TTS 1.7B), 0.37 s (Kokoro), 0.5 s (Qwen3-TTS 0.6B, unless its
  silence is gated) or 0.55–0.78 s (VibeVoice). On an idle GPU the speech components add about 0.15–0.35 s plus the
  LLM with Marvis, Pocket or Qwen3-TTS 1.7B, and up to about 0.85 s with VibeVoice.
- **Memory.** The full speech side, Parakeet with Qwen3-TTS 0.6B plus Silero and Smart Turn, peaks at about
  5.5 GiB. Nemotron instead of Parakeet is 0.9 GiB. Beside qwen38's about 80 GiB that leaves over 25 GiB of the
  112 GiB working set.
- **Full duplex.** VoiceChat 11B 4-bit keeps up with real time on this Mac when nothing else uses the GPU: about
  60 ms of compute per 80 ms frame, 25% headroom, 10.3 GiB peak. It begins replies within 0.1 s of the end of the
  user's speech on its own timeline, with first audio about 0.4 s after the speech ends. Its margin is thin. In phase 1 a render slowed Parakeet about 10×, so VoiceChat
  is unlikely to stay real time beside a decoding LLM or a render; that was not measured.
- **Turn-taking caveat.** Smart Turn missed the two-sentence question (0.22). VoiceChat spoke into the pauses inside
  the long utterance. Both are endpointing behaviours to test with real speech before choosing.

## 8. Not measured, and caveats

- **VoiceChat 11B 8-bit:** not downloaded (13.9 GB); left for a later window, as instructed.
- **No contention:** every 2026-10-05 number has nothing else on the GPU. TTS and VoiceChat beside a decoding LLM are
  still unmeasured.
- **VoiceChat was fed faster than real time.** A live session paces frames at 80 ms. Per-frame compute is what decides
  whether it keeps up, and the timeline latencies (0.24 s, and −0.11 to −0.06 s relative to speech end) exclude
  capture, transport and playback buffering. They are measured from clip boundaries, not from the end of speech. Its first two frames cost 1.4 s together, so a live session should warm
  up, for example by feeding silence, before the user speaks.
- **VoiceChat's MLX is 0.32.3**; everything else used 0.31.2.
- **Load times are warm-disk loads,** with the files written within the last day. A cold read after reboot was not
  measured.
- **Leading silence** comes from one run per mode; durations vary run to run (Qwen3-TTS especially).
- **Smart Turn's** sample is four judgments.
- **A stray run, not used:** at 00:47:58–00:48:17 on 2026-10-05 a leftover background watcher ran an old runner
  without a hold, between stages of another GPU job. It ran VAD (crashed on the bug fixed since), the two co-residency
  pairs and the Pocket rerun, and stopped when the other job took the hold at 00:48. Its numbers are not used. No
  watcher process remains.
- **Footprints:** the `/usr/bin/time -l` "peak memory footprint" (VibeVoice 23 GB, VoiceChat 15 GB) counts transient
  Metal allocations. MLX peak and max RSS are the resident figures.

## 9. Timeline of the 2026-10-05 run

| time | event |
|---|---|
| 08:41 | `gpu_clear.sh` CLEAR: hold gate open, no GPU job, qwen38 idle since 01:09:29 |
| 08:43–08:44 | `gate.sh` own-hold logic dry-checked on canned hold states |
| 08:44:30 | pre-flight gate GO |
| 08:44:30–08:45:28 | `setup.sh`: built `.venv-kokoro` and `.venv-vlm`; fetched the Mimi codec (0.385 GB, 43.5 s), the Marvis tokenizer and voice prompt, and the Qwen2.5 tokenizer; every phase READY in an offline check |
| 08:45:28 | hold `hd5dae0` requested; qwen38 unloaded in 7.7 s |
| 08:45:36 | hold granted; the phases ran in order, each behind a GO gate |
| 08:48:14 | runner done, all seven phases OK; hold released 08:48:16 |
