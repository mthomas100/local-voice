# 09 — Local TTS / STT / audio-native measurements (Apple M5 Max, 128 GB, macOS 27.0), 2026-10-04

Evidence: JSON results in `raw/*.jsonl`; scripts in `bench/`; generated audio in `audio/` (gitignored). Numbers below
are copied from those files; the full command log was kept locally and is not published.

## 0. What could and could not be measured, and why

GPU safety protocol followed: before each phase `curl 127.0.0.1:8090/running`, the render-job `pgrep`,
and `hold status`. Timeline:

- 20:59–21:05: another GPU job (a video render, RSS 20 GB) was running.
  Nothing was run on the GPU; code reading, cache inspection, the test clip, the scratch venv and the Kokoro
  download happened in this window.
- 21:05: the render finished. 21:05:43: **another agent session loaded the LLM** (qwen38 =
  Qwen3.8 Flash Next, ds4-server, RSS 12.9 GB, llama-swap ttl 0) and started sending long chat completions
  (57 s and more). `/running` was therefore not `[]`. Treated as a live GPU user; waited.
- 21:33: a manual hold (`hold status` → "held: manual") unloaded the LLM; 21:34:03 `/running` = `[]`.
  **Kokoro (21:35:52), Qwen3-TTS 0.6B (21:36:24) and 1.7B (21:37:35) ran in this clean window** (no render, LLM
  unloaded).
- 21:38:38: another GPU job (a short 480p video render, done 21:40) started. The first Parakeet
  run (21:38:58) overlapped it (my check printed the job but the chain did not stop; fixed afterwards with
  `bench/gate.sh`, which refuses to run when a render job exists). That run is reported as contended and was redone.
- 21:41:44: the hold was released and the other session reloaded the LLM (last request 21:41:55), then went idle.
  From 21:50 the LLM was resident but idle for 9+ minutes, the hold gate open, no render: **the Parakeet rerun,
  both co-residency runs, both server runs and the sampling check ran in that state (21:50–21:57)**. This deviates
  from the plan's literal "running must be []" (an agreed fallback); the llama-swap log
  shows no LLM request between 21:41:55 and the end of the measurements, so nothing contended, and the GPU gate output
  before each phase was logged.

Cache reality (checked with `du` on `blobs/`, not on the snapshot symlinks):

| Hub id | On disk | Status |
|---|---|---|
| mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 | 4.3 MB | metadata only, **no weights** (hub: 4.52 GB) → not downloaded (over the 2 GB line) |
| Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign, -1.7B-Base | 4.3 MB each | metadata only, no weights |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-8bit | 3.1 MB | config/tokenizer only, **no weights** (hub: 38.78 GB, 8 shards) → not downloaded |
| mlx-community/IndexTTS | 468 KB | metadata only |
| Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | 2.3 GB | weights present (PyTorch checkpoint; mlx-audio converts on load) |
| Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | 4.2 GB | weights present (same) |
| mlx-community/parakeet-tdt-0.6b-v3 | 2.3 GB | weights present |
| mlx-community/Kokoro-82M-bf16 | 372 MB | **downloaded during this task** (32.5 s), 108 voice packs |
| mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-bf16 | — | does not exist on the hub (404) |

So the measured TTS set is Qwen3-TTS 1.7B **CustomVoice** (voice Ryan), Qwen3-TTS 0.6B CustomVoice, and
Kokoro-82M-bf16 (voice af_heart). VoiceDesign 1.7B is the same talker architecture with a different
conditioning head; speed and memory should be within a few percent of CustomVoice 1.7B, but that is an
inference, not a measurement.

## 1. Environment and baseline

| Item | Value |
|---|---|
| Machine | Apple M5 Max (Mac17,6), `hw.memsize` 137438953472 B = 128 GiB, macOS 27.0 (26A5425a) |
| `mx.device_info()` | max_recommended_working_set_size 120259084288 B (112 GiB); max_buffer_length 86586540032 B; arch applegpu_g17s |
| mlx-audio | clone `~/repos/mlx-audio` @ ee0c65d (2026-09-11), version 0.5.3; venv python 3.11.15, mlx 0.31.2, mlx-lm 0.31.3, transformers 5.14.1 |
| vm_stat 21:01 (during the render) | pages free 36856, active 3478132, inactive 3908401, wired 1178740 (16 KiB pages) |
| vm_stat 21:34 (render done, LLM unloaded) | free 1075431, active 1717664, inactive 5331910 |
| vm_stat 21:54 (after all phases; LLM resident) | free 722534 (11.0 GiB), active 1063872, inactive 1595995, wired 5083729 (77.6 GiB, the ds4-server LLM with a 262K context, not mine) |

Kokoro needs the `misaki` G2P package, which is not in the mlx-audio venv. To avoid installing into the
user's venv, a scratch venv was created under `bench/venv-kokoro` (same 3.11.15 interpreter) with
`misaki num2words spacy` (the `misaki[en]` extra would have pulled torch + spacy-curated-transformers, which
`KokoroPipeline` does not use: it calls `en.G2P(trf=False)`), plus a `.pth` that adds the clone and the
clone venv's site-packages. Nothing system-wide was changed.

## 2. What mlx-audio actually offers for streaming (from source)

- **TTS generate API.** `mlx_audio.tts.generate.generate_audio(..., stream=False, streaming_interval=2.0)`
  forwards both into `model.generate(**kwargs)` and iterates the generator; `GenerationResult` carries
  `is_streaming_chunk` / `is_final_chunk`. CLI: `python -m mlx_audio.tts.generate --stream --streaming_interval S`.
- **Qwen3-TTS streams for real** (`tts/models/qwen3_tts/qwen3_tts.py:1126`): with `stream=True` the
  talker loop decodes every `max(1, int(streaming_interval*12.5))` new codec tokens (codec rate 12.5 Hz)
  through `speech_tokenizer.decoder.streaming_step()` (incremental conv buffers + KV cache) and yields a
  chunk; a final flush yields the tail. So TTFA = prefill + first `streaming_interval` seconds' worth of
  codec tokens + one incremental decode. `batch_generate()` also accepts `stream`.
- **Kokoro does not stream inside a segment** (`tts/models/kokoro/kokoro.py:293`): `generate()` yields one
  result per text segment from `split_pattern` (default `\n+`). Sentence-level streaming = pass
  `split_pattern=r"(?<=[.!?])\s+"`. Each segment is one full forward pass (≤510 phonemes).
- **STT.** `stt/streaming.py` defines a live-input `StreamingSession` protocol (`feed` / `close` / `step`),
  used by the server's `/v1/audio/transcriptions/realtime` websocket via `model.create_streaming_session()`.
  Only `nemotron_asr` implements it (`NemotronStreamingSession`). **Parakeet has no live session**: its
  `generate(..., stream=True)` → `stream_generate(audio, chunk_duration=5.0, overlap_duration=1.0)` chops an
  already-complete array into chunks and yields `StreamingResult` per chunk. For a voice agent that means
  Parakeet is run per utterance (VAD-gated), not fed frame by frame; with 0.3 s per utterance that is fine.
- **Server** (`mlx_audio/server.py`): routes `GET /`, `GET|POST|DELETE /v1/models`, `POST /v1/audio/speech`
  (OpenAI-shaped `SpeechRequest`: `stream`, `streaming_interval`, `response_format` default `mp3`),
  `GET /v1/audio/voices`, `POST /v1/audio/transcriptions`, `POST /v1/audio/separations`, websockets
  `/v1/audio/transcriptions/realtime` and `/v1/realtime`. `/v1/audio/speech` always returns a
  `StreamingResponse(media_type=audio/<format>)`; in stream mode each generated chunk is encoded separately
  with `audio_write(buffer, audio, sr, format=response_format)` and emitted, i.e. **each HTTP chunk is a
  self-contained encoded file** (for `wav`, each chunk has its own RIFF header). Flags: `--host --port
  --realtime --realtime-model --vad-model --tts-max-batch-size`.
- **Pi's tts skill** (`~/.pi/agent/skills/tts/scripts/engines/mlx_audio_tts.py`) loads via
  `mlx_audio.tts.utils.load_model`, warms up once, and calls `batch_generate`/`generate` **without
  `stream=True`**; it gets latency by splitting replies into sentences (first chunk ≤60 chars) and batching
  (`max_batch 6`). Its config comments record 2026-09-13 numbers: 0.6B load 3.8 s, RTF 0.25 (one sentence),
  2.6 GB; 1.7B load 2.2 s, RTF 0.30, 4.6 GB.

## 3. TTS measurements

Protocol: one model per process (`bench/tts_bench.py`), `/usr/bin/time -l` around it for max RSS, `mx.get_peak_memory()`
inside (reset before each run), 3 runs per cell, median first then all three runs. Texts (exact):

- short (11 words): `The kettle is on, and the rain has finally stopped outside.`
- long (46 words): `Before we start the meeting, let me summarize where things stand. The new build passed every test last night, the design review is scheduled for Thursday morning, and two customers have already asked when they can try the beta. I think we are in good shape.`
- warm-up (not counted): `Hello there, this is a warm-up sentence.`

TTFA = time from calling `model.generate()` to the first chunk materialised as a NumPy array. "audio s" is the
duration the model chose to produce (Qwen3-TTS samples, so it varies). RTF = total generation time / audio duration.
Raw: `raw/tts_kokoro.jsonl`, `raw/tts_qwen3_0.6b.jsonl`, `raw/tts_qwen3_1.7b.jsonl`; WAVs in `audio/`.

### 3a. Kokoro-82M-bf16 (`mlx-community/Kokoro-82M-bf16`, voice af_heart, lang a, scratch venv, EspeakFallback disabled)

Command: `/usr/bin/time -l bench/venv-kokoro/bin/python bench/tts_bench.py --model mlx-community/Kokoro-82M-bf16 --voice af_heart --lang a --kokoro --no-espeak-fallback --runs 3` (21:35:52, `/running` was `[]`, LLM unloaded under a manual hold)

| Item | Value |
|---|---|
| Load (`load_model` + eval) | 0.173 s (other loads in this session: 0.157, 0.189, 0.228) |
| MLX active after load | 0.30 GiB |
| Warm-up (first generate: Metal kernels + spaCy/misaki G2P init) | 4.28 s |
| Max RSS (`time -l`) / peak footprint | 644 MB / 5.53 GB |

| text | mode | TTFA s median (runs) | total s median (runs) | audio s | RTF | first chunk audio s | MLX peak GiB |
|---|---|---|---|---|---|---|---|
| short (11 w) | whole text (no streaming exists) | 0.077 (0.084, 0.077, 0.076) | 0.083 (0.090, 0.083, 0.082) | 3.90 | 0.021 | 3.90 | 1.39 |
| short (11 w) | sentence split | 0.077 (0.078, 0.076, 0.077) | 0.083 (0.083, 0.082, 0.083) | 3.90 | 0.021 | 3.90 | 1.39 |
| long (46 w) | whole text | 0.288 (0.510, 0.288, 0.288) | 0.299 (0.519, 0.299, 0.298) | 15.47 | 0.019 | 15.47 | 2.45 |
| long (46 w) | sentence split (`split_pattern=r"(?<=[.!?])\s+"`, 3 segments) | 0.078 (0.083, 0.078, 0.078) | 0.326 (0.421, 0.326, 0.325) | 15.72 | 0.021 | 3.93 | 2.38 |

Kokoro is deterministic (identical durations every run). Its output has 0.28–0.30 s of leading silence and
0.45–0.48 s trailing silence (20 ms frames, |x| > 0.01), so first audible sound ≈ TTFA + 0.3 s unless trimmed.
Parakeet transcribed the server-generated Kokoro paragraph back verbatim (section 5), so disabling the espeak fallback
did not drop words from these texts.

### 3b. Qwen3-TTS 0.6B CustomVoice (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, voice Ryan, lang english; mlx-audio converts the PyTorch checkpoint on load)

Command: `/usr/bin/time -l ~/repos/mlx-audio/.venv/bin/python bench/tts_bench.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --voice Ryan --lang english --stream-intervals 2.0,0.5 --runs 3` (21:36:24, `/running` `[]`)

| Item | Value |
|---|---|
| Load | 1.442 s (0.944 s on a second load in the co-residency run; Pi's skill measured 3.8 s cold on 2026-09-13) |
| MLX active after load | 2.39 GiB |
| Warm-up | 1.80 s |
| Max RSS / peak footprint | 2.75 GB / 15.75 GB |

| text | mode | TTFA s median (runs) | total s median (runs) | audio s median (runs) | RTF | first chunk audio s | MLX peak GiB |
|---|---|---|---|---|---|---|---|
| short | non-stream | 1.189 (1.189, 1.286, 1.056) | 1.193 (1.193, 1.292, 1.062) | 5.76 (5.76, 6.24, 5.20) | 0.207 | 5.76 | 5.71 |
| short | stream, interval 2.0 s (25 codec tokens) | 0.408 (0.404, 0.408, 0.430) | 1.630 (1.803, 1.630, 1.324) | 8.08 (9.04, 8.08, 6.24) | 0.202 | 2.00 | 4.08 |
| short | stream, interval 0.5 s (6 codec tokens) | **0.112** (0.112, 0.110, 0.119) | 1.150 (1.250, 1.107, 1.150) | 5.28 (5.84, 5.12, 5.28) | 0.216 | 0.48 | 2.99 |
| long | non-stream | 3.978 (3.969, 4.508, 3.978) | 3.994 (3.977, 4.522, 3.994) | 19.68 (18.88, 22.16, 19.68) | 0.204 | 18.88 | 7.89 |
| long | stream 2.0 | 0.429 (0.429, 0.452, 0.414) | 3.665 (3.991, 3.665, 3.491) | 18.16 (19.36, 18.16, 17.44) | 0.202 | 2.00 | 4.08 |
| long | stream 0.5 | **0.115** (0.116, 0.113, 0.115) | 3.982 (3.860, 3.982, 4.346) | 19.20 (18.64, 19.20, 20.72) | 0.207 | 0.48 | 3.04 |

Leading silence 0.42–0.44 s, trailing 0.44–0.48 s in every 0.6B file: the first 0.48 s streaming chunk is
almost entirely silence, so first audible sound ≈ 0.11 + 0.42 ≈ 0.55 s unless the orchestrator trims it.
Parakeet transcribed the long non-stream output back verbatim (one comma rendered as a sentence break).

### 3c. Qwen3-TTS 1.7B CustomVoice (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, voice Ryan, lang english)

Command: same with `--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (21:37:35, `/running` `[]`)

| Item | Value |
|---|---|
| Load | 1.322 s |
| MLX active after load | 4.27 GiB |
| Warm-up | 1.04 s |
| Max RSS / peak footprint | 4.77 GB / 14.58 GB |

| text | mode | TTFA s median (runs) | total s median (runs) | audio s median (runs) | RTF | first chunk audio s | MLX peak GiB |
|---|---|---|---|---|---|---|---|
| short | non-stream | 1.198 (1.241, 1.134, 1.198) | 1.203 (1.245, 1.139, 1.203) | 4.72 (4.96, 4.48, 4.72) | 0.254 | 4.96 | 7.08 |
| short | stream 2.0 | 0.504 (0.511, 0.502, 0.504) | 1.054 (0.947, 1.054, 1.368) | 4.16 (3.68, 4.16, 5.44) | 0.253 | 2.00 | 5.96 |
| short | stream 0.5 | **0.132** (0.139, 0.132, 0.130) | 1.014 (0.961, 1.097, 1.014) | 3.92 (3.68, 4.24, 3.92) | 0.259 | 0.48 | 4.87 |
| long | non-stream | 3.783 (3.549, 3.783, 3.827) | 3.793 (3.564, 3.793, 3.839) | 15.28 (14.40, 15.28, 15.52) | 0.248 | 14.40 | 8.65 |
| long | stream 2.0 | 0.514 (0.520, 0.507, 0.514) | 3.552 (3.552, 4.024, 3.465) | 14.40 (14.40, 16.40, 14.08) | 0.246 | 2.00 | 5.97 |
| long | stream 0.5 | **0.138** (0.138, 0.140, 0.138) | 4.984 (4.984, 6.395, 4.574) | 19.76 (19.76, 25.04, 17.92) | 0.255 | 0.48 | 4.92 |

The 1.7B has only 0.08 s leading and 0.02–0.04 s trailing silence, so first audible sound ≈ 0.14 + 0.08 ≈ 0.22 s.
Its speech is faster (46 words in 14–15.5 s vs 18–22 s for the 0.6B), so despite RTF 0.25 vs 0.21 it finishes the
paragraph in about the same wall time. One stream-0.5 run produced 25.0 s for the 46 words (sampling at the default
temperature 0.9); the saved 19.8 s run transcribes back verbatim.

### 3d. Qwen3-TTS VoiceDesign 1.7B — not measured
`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` has no weights in the cache (4.52 GB on the hub). Same talker
and speech tokenizer as CustomVoice 1.7B, so expect the 3c numbers within a few percent plus the instruct-prompt
prefill; unverified.

## 4. STT: Parakeet TDT 0.6B v3 on a 2.48 s utterance

Clip: `say -v Samantha` "Could you check the weather for tomorrow afternoon?" → `afconvert -f WAVE -d LEI16@16000 -c 1`
→ `audio/clip3_16k.wav` (16 kHz mono int16, 2.483 s). Command: `/usr/bin/time -l ~/repos/mlx-audio/.venv/bin/python
bench/stt_bench.py --clip audio/clip3_16k.wav --runs 3` (`bench/stt_bench.py`).

Two runs exist. The first (21:38:58) **overlapped a 480p video render** that another job had just started
(21:38:38–21:40); the rerun (21:50:53) had no render running (LLM resident but idle 9 min). Only the
rerun is a clean number; both are reported.

| run | load s | warm-up s | latency from file path, median (runs) | latency from pre-loaded array, median (runs) | transcript | MLX active / peak GiB | max RSS |
|---|---|---|---|---|---|---|---|
| clean rerun 21:50 | 1.063 | 0.213 | **0.021** (0.021, 0.022, 0.021) | **0.020** (0.020, 0.021, 0.020) | exact: "Could you check the weather for tomorrow afternoon?" | 2.38 / 2.44 | 2.64 GB |
| contended 21:38 (render) | 1.018 | 0.367 | 0.195 (0.138, 0.208, 0.195) | 0.187 (0.189, 0.187, 0.110) | exact | 2.38 / 2.44 | 2.64 GB |

So a 2.5 s utterance decodes in ~20 ms once warm (≈120× real time), and GPU contention from a render costs ~10×.
`stream_generate()` over the finished clip (chunk 5 s / overlap 1 s): one final result at 0.022 s. With chunk 1.0 s /
overlap 0.5 s: 4 steps at 0.023 / 0.043 / 0.062 / 0.083 s yielding "Could you check the weather?" then
" Tomorrow afternoon?" — it works as chunked decoding of a complete buffer, but there is no feed-as-you-record
session for Parakeet (section 2), so a live agent will run it once per VAD-delimited utterance.

## 5. Co-residency (STT + TTS in one process, `bench/coresident.py`)

| pair | STT load s | TTS load s | MLX active after both | MLX peak during use | max RSS | time -l footprint | round 2 (warm): STT s / TTS TTFA s / TTS total s |
|---|---|---|---|---|---|---|---|
| Parakeet + Kokoro (scratch venv) | 0.70 | 0.19 | **2.68 GiB** | 3.63 GiB | 3.16 GB | 5.33 GB | 0.035 / 0.064 / 0.070 (1 chunk, 10-word reply) |
| Parakeet + Qwen3-TTS 0.6B (stream 0.5) | 0.73 | 0.94 | **4.76 GiB** | 5.36 GiB | 5.27 GB | 6.46 GB | 0.025 / 0.108 / 2.909 (29 chunks ≈ 13.9 s of audio for a 10-word reply; round 1: 0.038 / 0.124 / 1.072, 11 chunks) |

Round 1 includes warm-up (Kokoro pair: STT 0.497 s, TTS 1.18 s). Both pairs leave >100 GiB of the 112 GiB
recommended working set for the LLM. The 29-chunk round is another instance of Qwen3-TTS over-generating.

## 6. `mlx_audio.server` on 127.0.0.1:18800 (`bench/server_test.sh`)

Both runs: `python -m mlx_audio.server --host 127.0.0.1 --port 18800`; up in 2 s; `GET /v1/models` → `{"object":"list","data":[]}`
until the first request loads the model named in the request; `POST /v1/audio/speech` with `{}` → 422 (route exists,
validation error). Server stopped by pid afterwards; `pgrep -f mlx_audio.server` → 0 left (checked after each run).
curl `-w %{time_starttransfer}` is **not** a first-audio measure here: the route returns a `StreamingResponse` whose
headers go out in 1–3 ms (0.8–1.2 s on the very first request, which blocks on the model load); real chunk timing
comes from the Python reader (`read1`, one HTTP chunk per read).

Qwen3-TTS 0.6B (Ryan, english, `response_format: wav`):

| request | total s (3 runs) | bytes (3 runs) → audio s |
|---|---|---|
| first request (includes load), short, non-stream | 2.90 | 272,684 → 5.7 s |
| short non-stream | 1.47, 1.35, 1.49 | 238k, 219k, 250k → 5.0, 4.6, 5.2 s |
| short stream 2.0 | 1.07, 1.30, 2.71 | 246k, 292k, 642k → 5.1, 6.1, 13.4 s |
| short stream 0.5 | 0.95, 1.28, 1.25 | 212k, 285k, 277k → 4.4, 5.9, 5.8 s |
| long non-stream (server default sampling) | **11.87, 9.45, 5.49** | 1,909k, 1,513k, 891k → **39.8, 31.5, 18.6 s** |
| long stream 2.0 | 3.20, 4.23, 3.53 | 745k, 999k, 818k → 15.5, 20.8, 17.0 s |
| long stream 0.5 | 4.36, 4.61, 4.05 | 977k, 1,035k, 912k → 20.4, 21.6, 19.0 s |

Chunk trace (long, stream, interval 0.5): headers at +0.003 s; **first audio chunk at +0.121 s** (23,084 bytes = a
complete 0.48 s WAV, arriving as an 8,186-byte HTTP chunk with the RIFF header plus a 14,898-byte one); then one
chunk every ~0.100 s; 54 RIFF headers in a 1,227,336-byte stream (25.6 s of audio in 5.47 s). **Each streamed chunk
is a self-contained WAV file** (`_emit_audio` runs `audio_write(format=response_format)` per chunk), so a client must
parse chunk by chunk; `afinfo` on a saved stream reports only the first chunk (0.48 s). `response_format: pcm` is not
offered by the WAV writer path I exercised; mp3 is the default and would be re-encoded per chunk the same way.

**Runaway generation with the server's default sampling (checked 21:56).** Three
more long non-stream requests with the server defaults (`temperature 0.7, top_p 0.95, top_k 40, repetition_penalty
1.0, max_tokens 1200`) and two with the model's own defaults passed in the request body
(`temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05`), each transcribed back with Parakeet:

| sampling | wall s | audio s | leading / trailing silence s | words transcribed (of 46) |
|---|---|---|---|---|
| server defaults, run 1 | 33.9 | **96.0** (= the 1200-token cap) | 0.42 / **74.3** | 27, then silence |
| server defaults, run 2 | 7.9 | 25.1 | 0.42 / 0.58 | 46, complete |
| server defaults, run 3 | 30.9 | **96.0** | 0.44 / **64.1** | 11, then silence |
| model defaults, run 1 | 5.1 | 16.7 | 0.46 / 0.46 | 46, complete |
| model defaults, run 2 | 4.8 | 16.2 | 0.46 / 0.50 | 46, complete |

So with `repetition_penalty 1.0` the 0.6B talker sometimes stops speaking mid-text and emits silence tokens until
`max_tokens`; the earlier 39.8 s and 31.5 s server outputs were the same failure caught by shorter texts. Anyone
driving Qwen3-TTS through `mlx_audio.server` must pass the model's sampling parameters explicitly and set a tight
`max_tokens` (12.5 codec tokens per second of expected speech). In streaming mode the client would hear the sentence
stop and then receive silence chunks for up to 90 s unless it gates on energy.

Kokoro (af_heart, lang a, scratch venv with the sitecustomize espeak patch):

| request | total s (3 runs) | bytes → audio s |
|---|---|---|
| first request (includes load), short | 0.80 (headers at 0.29) | 187,244 → 3.9 s |
| short, any of non-stream / stream 2.0 / stream 0.5 | 0.089–0.095 | 187,244 → 3.9 s (identical every run) |
| long, any mode | 0.31–0.33 | 742,844 → 15.47 s |

Chunk trace (long, stream 0.5): everything arrives at +0.326 s in one burst (5 reads, 1 RIFF header). The `stream`
flag has no effect for Kokoro; the server does not sentence-split for it.

## 7. What this means for the real-time budget

- **STT is not the bottleneck.** Parakeet decodes a 2.5 s utterance in ~20 ms warm (0.2 s warm-up, 1.1 s load,
  2.4 GiB). Turn latency on the input side will be dominated by end-of-speech detection (VAD hangover, typically
  0.3–0.7 s, not measured here), not the model. There is no live-input Parakeet session in mlx-audio, so run it per
  utterance; chunked partials every 1 s cost ~20 ms each if you want early text.
- **TTS first-audio, warm, on an otherwise idle GPU:** Kokoro 0.08 s per sentence (sentence-split a paragraph and the
  first sentence is out in 0.08 s, the whole 46-word paragraph in 0.33 s; RTF 0.02). Qwen3-TTS with
  `stream=True, streaming_interval=0.5`: 0.11 s (0.6B) / 0.13–0.14 s (1.7B) to the first 0.48 s chunk; with the
  default interval 2.0 it is 0.41 / 0.51 s. The default interval is wrong for a voice agent; use 0.5 (or 0.25–0.3 s,
  = 3–4 codec tokens, untested).
- **Leading silence is part of the latency:** Qwen3-TTS 0.6B starts every clip with ~0.42 s of silence (Kokoro
  ~0.29 s, Qwen3-TTS 1.7B ~0.08 s). Trim the first chunk with a simple energy gate or the perceived first sound is
  0.55 s (0.6B) / 0.36 s (Kokoro) / 0.22 s (1.7B) after the text arrives.
- **Sustained throughput is fine everywhere:** RTF 0.02 (Kokoro), 0.21 (0.6B), 0.25 (1.7B); after the first chunk,
  playback never starves. Qwen3-TTS at RTF 0.21–0.25 also leaves ~75–80 % of the GPU for a concurrently decoding LLM
  (not measured together; expect both to slow when they overlap, as the render-contended Parakeet run showed ~10×).
- **Memory beside a 20–35 GB LLM:** Parakeet + Kokoro = 2.7 GiB active / 3.6 GiB peak; Parakeet + Qwen3-TTS 0.6B =
  4.8 / 5.4 GiB (streaming); Qwen3-TTS 1.7B alone = 4.3 GiB active, 4.9–6.0 GiB peak streaming, **up to 8.7 GiB** if
  you decode a long paragraph non-streaming (the whole-sequence codec decode is the memory spike; streaming avoids it).
  Budget 10 GiB for STT+TTS and the total stays at 30–45 GB; the `time -l` "peak memory footprint" figures (5–16 GB)
  include Metal's transient allocations and over-state what stays resident.
- **Warm-up matters:** first generate costs Kokoro 4.3 s (kernel compile + spaCy/misaki init), Qwen3-TTS 1.0–1.8 s,
  Parakeet 0.2–0.4 s. Warm every model at startup (Pi's tts daemon already does). Loads are 0.2–1.5 s, so unload /
  reload on idle is viable if the warm-up is repeated after reload.
- **Qwen3-TTS duration variance is a real risk:** the same 46 words came out as 17.4–22.2 s with model defaults,
  once 25.0 s (1.7B), 18.6–39.8 s with the server's default sampling (temperature 0.7, top_p 0.95, top_k 40,
  **repetition_penalty 1.0**), and a 10-word reply once ran 13.9 s. Keep `repetition_penalty` ≥ 1.05, cap
  `max_tokens` per sentence (~12.5 tokens per second of expected speech), and sentence-split so a runaway costs one
  sentence. Kokoro has none of this (deterministic).
- **Rough turn budget (text in → first audible sound):** STT 0.02 s + LLM time-to-first-sentence (not measured here)
  + TTS 0.08–0.14 s + leading silence 0.08–0.42 s (trimmable) ⇒ the voice stack adds ~0.1–0.6 s; the LLM and
  endpointing dominate.


## 8. Audio-native model feasibility: Qwen3-Omni-30B-A3B-Instruct (MLX 8-bit)

Not run: the weights are not on disk (only `config.json`, tokenizer, chat template, index; the 8
safetensors shards total **38.76 GB** per `model.safetensors.index.json`). Downloading is far over the
2 GB line, so this is a paper check.

- **Runtime exists on this Mac.** `~/repos/local-decision/.venv` has mlx-vlm **0.7.4** (mlx 0.32.3,
  transformers 5.18.0, mlx-audio 0.5.7 as a dependency). Its `mlx_vlm/models/qwen3_omni_moe/` package has
  `audio.py` (audio tower, 32-layer `qwen3_omni_moe_audio_encoder`), `thinker.py`, `talker.py`,
  `code2wav.py`, `processing_qwen3_omni_moe.py` (builds `input_features` / `feature_attention_mask` from
  audio), `omni_utils.process_multimodal_info(conversation, use_audio_in_video)`, and the model exposes
  `get_audio_features`, `generate`, `generate_stream`, `enable_talker`/`disable_talker`
  (`enable_audio_output: true` in the config, so speech out is wired). Audio files are read with
  `miniaudio` (`mlx_vlm/utils.py:1821`). The model card itself only shows the image CLI
  (`python -m mlx_vlm.generate --model ... --image ...`); audio goes through the Python API / chat
  template with an `audio` content part.
- **Weights/architecture.** Thinker: 48 layers, hidden 2048, 128 experts, 8 active (A3B); talker + code
  predictor + code2wav for speech output. Weight groups in the index: thinker 2227 tensors, talker 909,
  code2wav 352.
- **Memory budget if downloaded.** 8-bit weights ≈ 38.8 GB resident (thinker ≈ the bulk; talker and
  code2wav can be skipped with `disable_talker` for understanding-only use, saving a few GB). Plus KV cache
  and activations: expect ~42–46 GB active for text-out, more with speech out. Beside a 20–35 GB chat LLM
  that is 62–81 GB, inside the 112 GiB recommended working set, but it leaves little room for anything
  else (video/3D jobs are out while it is resident).
- **Expected speed (not measured).** A3B MoE at 8-bit on an M5 Max should decode text at roughly 40–70
  tok/s after a prefill that includes the audio encoder (3 s of audio ≈ 75 audio tokens after the
  encoder's 8× merge; the encoder pass is the TTFT cost). These are estimates from similar-size MoE models
  on this machine; measure before relying on them.
- **To run it:** `uv pip install mlx-vlm` into a scratch venv , then `snapshot_download("mlx-community/Qwen3-Omni-30B-A3B-Instruct-8bit")` (38.8 GB), then
  `mlx_vlm.load(...)`, build a chat with `{"type":"audio","audio": path}` and ask it to transcribe and
  describe tone. The question asked ("does it say anything about tone?") cannot be answered
  without the weights.

## 9. Clean-up check

21:54:47: no `tts_bench|stt_bench|coresident|mlx_audio.server|server_test`
process left; no other process mentioning `mlx`; `/running` shows only the other session's
qwen38; the server on :18800 was stopped after each run (`pgrep -f mlx_audio.server` → 0, checked 4 times). Every
measurement ran in its own process, so all model memory was released at exit; `vm_stat` free pages 722534 (11 GiB)
afterwards. Nothing was installed system-wide or into the user's venvs; the only additions are under `measure/`
(`bench/venv-kokoro` with misaki, spacy, en_core_web_sm, phonemizer, espeakng-loader, a `.pth` to the clone and a
`sitecustomize.py`) and the Kokoro weights in the Hugging Face cache (372 MB, the one download allowed).

## 10. Could not measure

- **Qwen3-TTS 1.7B VoiceDesign (MLX)**: only metadata cached; 4.52 GB download needed. Measured CustomVoice 1.7B instead.
- **Qwen3-TTS 0.6B in MLX form from mlx-community**: no VoiceDesign-bf16 0.6B exists on the hub (404); the original
  Qwen/ 0.6B CustomVoice checkpoint was measured through mlx-audio's on-load conversion.
- **Qwen3-Omni-30B-A3B-Instruct-8bit**: weights absent (38.8 GB). Runtime (mlx-vlm 0.7.4 with `qwen3_omni_moe`,
  audio tower and talker) is present in `~/repos/local-decision/.venv`. No load time, TTFT, tok/s, memory or
  tone-description result.
- **Live streaming STT**: Parakeet has no feed/step session in mlx-audio; only `nemotron_asr` implements
  `StreamingSession`, and no Nemotron ASR weights are cached. Measured chunked decoding of a finished buffer instead.
- **Strictly idle machine for every phase**: the three TTS phases ran with the LLM unloaded; the rest ran with the
  LLM resident but idle (no requests for 9–15 min), see section 0. The first Parakeet run was contended and is marked.
- **Kokoro with its espeak-ng fallback**: `misaki.espeak.EspeakFallback` aborts the process (espeak-ng data path
  baked in at the wheel's build: "Error processing file '<build-machine>/runner/work/espeakng-loader/.../phontab'"; setting
  `ESPEAK_DATA_PATH` did not help). Disabled it, so out-of-dictionary words would be skipped; the test texts had none
  (transcripts came back verbatim).
- **TTS beside a decoding LLM**: not run (would have meant sending the LLM a chat request, which the plan forbade).
  The 10× slowdown of the render-contended Parakeet run is the only contention data point.
- **Pi's tts daemon end-to-end** (socket → sentence split → batch → afplay): not measured; its engine does not use
  `stream=True` (section 2).
- **`response_format: pcm` / mp3 chunk framing on the server**: only `wav` exercised.
- **Server curl `time_starttransfer`**: measured (1–3 ms) but meaningless as first-audio because headers are committed
  before generation; the Python chunk trace is the first-audio number (0.121 s).
