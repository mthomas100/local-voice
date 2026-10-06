# 09c — mlx-audio 0.5.7: the API facts the voice agent depends on (read from source)

Written 2026-10-05 by the measurement agent. Every fact is read from source, with a file and line. Where the
2026-10-05 08:45 run (report `09b-phase2-report.md`) exercised an API, its measurement is cited; nothing else was
executed for this report.

**What was read.**

- **mlx-audio 0.5.7** came from `pip download mlx-audio==0.5.7 --no-deps` into a scratch folder. The wheel is
  `mlx_audio-0.5.7-py3-none-any.whl`, sha256 `2f2b3c66…c5123`, 827 Python files. It is byte-identical (`diff -rq`)
  to the copy installed in `~/repos/local-decision/.venv`. Paths below are relative to `mlx_audio/`, and line
  numbers are 0.5.7's.
- **The clone** at `~/repos/mlx-audio` is "0.5.3" in this report: version 0.5.3 plus 21 commits (`ee0c65d`,
  2026-09-11). It is what every phase-1 and phase-2 number used. `diff -rq` against 0.5.7, ignoring tests, READMEs
  and the UI, finds 26 changed files and 4 new ones. "Same in the clone" below means the file is byte-identical.
- **mlx-vlm 0.7.4** was read from `~/repos/local-decision/.venv` (`mlx_vlm/models/nemotron_voicechat/`), read-only,
  because the VoiceChat checkpoint loads only there (section 8).

## 0. What changed from the clone to 0.5.7, per component

| component | changed? | what changed |
|---|---|---|
| Qwen3-TTS (`tts/models/qwen3_tts/`) | no | identical; the batch session and its `cancel()` were already in the clone |
| Kokoro (`tts/models/kokoro/`) | no | identical; its inverse STFT output is unchanged too (see `dsp.py` below) |
| Parakeet (`stt/models/parakeet/`) | yes | adds "Parakeet Redux", a ternary 2-bit variant (`redux.py`, `convert.py`, `prepare_config` in `__init__.py`), and two options that default to off (`normalize_valid_frames` in `audio.py:22`, `mask_padding` in `conformer.py:33`); fixes the chunked loop so it stops after the last chunk (`parakeet.py:273`). Standard parakeet-tdt-0.6b-v3 decoding is otherwise unchanged |
| Nemotron ASR (`stt/models/nemotron_asr/`) | yes, not the live session | `session.py` and `streaming.py` identical. New: `create_speaker_streaming_session`, `stream_generate_speakers` and `generate_speakers` (`nemotron_asr.py:112-153`, `speaker_streaming.py`), used with the new `vad/models/nemotron_diarization`. `_prepare_audio` now accepts NumPy (`mx.array(audio, dtype=dtype)`, `nemotron_asr.py:110`). The offline decoder moved into `GreedyDecoderState` (`rnnt.py:14`) |
| Silero VAD, Smart Turn (`vad/models/`) | no | identical |
| VoiceChat in mlx-audio (`sts/models/nemotron_voicechat/`) | no | identical, and it cannot parse the checkpoint we have (section 8) |
| `server.py`, `server_inference.py`, `realtime_vad.py`, `sts/voice_pipeline.py` | no | identical |
| `tts/generate.py` (CLI) | yes | passes `voice=` to `generate()` only when one was given (`generate.py:334`) |
| `dsp.py` | yes | `stft`/`istft` centre-pad a window shorter than `n_fft`, `istft` infers `n_fft` from the frequency axis and trims by it. No effect on Kokoro, whose window equals `n_fft` (20; `istftnet.py:791-795`) |
| `stt/utils.py` | yes | `parakeet_tdt` alias for the shared loader |
| `lm/generate.py`, `tts/models/llama/llama.py`, `tts/models/qwen3/qwen3.py` | yes | not used by any model in this project (Marvis uses `lm/models/llama`, unchanged) |
| `ui/` | removed | the wheel ships no UI |
| packaging | — | 0.5.7 declares `mlx>=0.31.1`, `transformers>=5.14.0`; extras `stt`, `tts`, `sts`, `server`. **misaki is in none of them**, so Kokoro still needs it installed by hand (section 6) |

## 1. Qwen3-TTS `generate()` keyword arguments

`Model.generate()` is at `tts/models/qwen3_tts/qwen3_tts.py:1126-1146` (same in the clone):

| kwarg | default | reaches the CustomVoice path? | notes |
|---|---|---|---|
| `text` | required | yes | |
| `voice` | `None` | yes, as `speaker` | **required** for CustomVoice, which raises without it (`:1205-1209`); checked against `supported_speakers` (`:2116-2119`) |
| `instruct` | `None` | yes | style instruction. The guard meant to drop it for 0.6B models (`:2121-2126`) tests `tts_model_type != "custom_voice"` inside the CustomVoice method, so it never fires: the 0.6B receives `instruct` too. Whether the 0.6B honours it is untested |
| `temperature` | 0.9 | yes | |
| `top_k` | 50 | yes | |
| `top_p` | 1.0 | yes | |
| `repetition_penalty` | 1.05 | yes | applied to the first codebook only (`_sample_token(... generated_tokens=...)`, `:2587-2595`) |
| `max_tokens` | 4096 | yes | codec frames at 12.5 Hz, so 4096 caps one call at 327.7 s of audio. Honoured as given (`:2550`) |
| `stream` | `False` | yes | |
| `streaming_interval` | 2.0 | yes | chunk = `max(1, int(streaming_interval * 12.5))` frames (`:2568`): 0.32 → 4 frames (0.32 s), 0.5 → 6 frames (0.48 s) |
| `lang_code` | `"auto"` | yes, as `language` | selects the codec language-ID prefill (`_prepare_generation_inputs`, `:326`; the prefill lists are at `:410` and `:416`) |
| `split_pattern` | `"\n"` | **no** | used only on the Base-model path (`:1272-1275`). CustomVoice synthesises the whole `text` in one pass, so split sentences yourself |
| `streaming_context_size` | 25 | **no** | used only by `batch_generate` (`:1851`, `:1956`, `:2001`) |
| `speed` | 1.0 | **no** | documented as "not directly supported yet" |
| `ref_audio`, `ref_text` | `None` | no | voice cloning on Base models (ICL path, `:2204`) |
| `verbose` | `False` | yes | tqdm bar only |

**The server's defaults differ from the model's.** `SpeechRequest` (`server.py:168`) defaults to temperature 0.7,
top_p 0.95, top_k 40, repetition_penalty 1.0, streaming_interval 2.0 and max_tokens 1200 (`:179-186`), and
`run_serial` passes every one of them to `generate()` (`:636-653`). That is how phase 1 got a 0.6B run to the
1200-frame cap with a minute of silence. Pass the model's values explicitly when going through the server.

What a chunk is: `GenerationResult` (`tts/models/base.py:72-84`) carries `audio` (mx.array, mono, 24 kHz for
Qwen3-TTS), `samples`, `sample_rate`, `token_count`, `real_time_factor`, `peak_memory_usage` (GB, process-wide),
`is_streaming_chunk`, and `is_final_chunk`, which is true only on the flush after EOS (`:2753`).

## 2. Stopping a generation early (barge-in) and releasing memory

**The mechanism is the generator.** `_generate_with_instruct` (`:2516-2804`) is a plain Python generator. Each
streaming chunk is yielded between talker steps (`:2680`), and the final flush after EOS at `:2728`. There is no cancel flag, no `stop` argument and no
`try`/`finally`. Breaking out of the loop, or calling `gen.close()`, raises `GeneratorExit` at the paused `yield`.
The work stops there, at most one chunk late: about 60 ms of compute for the 0.6B at interval 0.32 (RTF 0.2 × 0.32 s).

**What an early stop leaves behind,** because the cleanup at `:2755-2756` runs only after a normal finish:

- **The decoder's streaming state** stays allocated on the model object: `speech_tokenizer.decoder._transformer_cache`
  plus every conv buffer (`speech_tokenizer.py:882-887`, `:889-931`). The next streaming call resets it first
  (`:2571`), so it does not leak into the next utterance. Call
  `model.speech_tokenizer.decoder.reset_streaming_state()` yourself to free it at once.
- **The talker and code-predictor KV caches** (`:2553-2554`) are locals of the generator frame. They are freed when
  the generator object is collected, so drop every reference to it.
- **MLX's buffer cache keeps freed buffers.** Call `mx.clear_cache()` to return them; the library's own paths do that
  after each generation (`:2756`, `:2760`, `:2804`).

So: `gen.close(); model.speech_tokenizer.decoder.reset_streaming_state(); mx.clear_cache()`. The weights stay
resident until the model object itself is dropped.

**Non-streaming calls cannot be interrupted.** The whole utterance is generated before the single `yield`. The same
holds for one Kokoro segment (section 6).

**Reference implementations in the same package,** all the same in the clone:

- `server.py:655-659` checks `request.cancel_event.is_set()` after every yielded chunk. When set, it calls
  `mx.clear_cache()`, emits done and returns. The event is set by `InferenceHandle.cancel()`
  (`server_inference.py:61-62`), for example when the HTTP client disconnects (`server.py:868-869`). Barge-in through
  the server is therefore "close the HTTP stream".
- `sts/voice_pipeline.py:1291-1330` (`_speak_response`) pulls one chunk at a time with `next(generator)` on the MLX
  worker thread and checks an `asyncio.Event` before and after each chunk. `_handle_barge_in` (`:1214-1225`) sets that
  event, cancels the TTS task, clears the queued output audio and flushes the audio device.
- `Qwen3TTSBatchSession.cancel(sequence_id)` (`continuous_batching.py:73-80`) drops one request at a step boundary.
  Its docstring says the batch session is non-streaming, so it is not a barge-in path for streamed speech.

**One stream per model object.** All streaming generators share the decoder state on `model.speech_tokenizer`, and
every new stream resets it (`:2571`). Two interleaved streaming generators on one model corrupt each other, even
pulled alternately from one thread. To overlap sentence N+1 with sentence N, generate N+1 after N's generator ends, or
load a second model instance.

## 3. Where Qwen3-TTS's leading silence comes from

Phase 1 measured leading silence of about 0.42 s for the 0.6B and 0.08 s for the 1.7B. Both use the same 12 Hz speech
tokenizer and decoder, so the difference is not a decoder delay. The decoder is causal, and its transposed
convolutions trim on the right only (`speech_tokenizer.py:634-655`). The prompt carries no silence: the prefill is
the language-ID or think tokens, then the speaker embedding, then the text (`_prepare_generation_inputs`,
`qwen3_tts.py:326`; the prefill lists are at `:410` and `:416`). The silence is therefore codec frames that the talker generates at the
start.

**Nothing in the API controls it.** No `generate()` argument trims or suppresses leading frames. The non-streaming
path trims only the padded tail (`:2767-2773`), and the streaming path yields every frame. The server does no
trimming either. `lang_code` changes the prefill and is the only input that plausibly shifts it; that is untested.
Mitigations stay on the client: energy-gate or trim the first chunks (as `09-phase1-report.md` recommends), or use
the 1.7B, whose silence is 80 ms.

## 4. Parakeet STT on an in-memory array

`stt/models/parakeet/parakeet.py`. Changed in 0.5.7, but not in any of the lines below.

- `generate(audio, *, dtype=mx.bfloat16, chunk_duration=None, overlap_duration=None, chunk_callback=None,
  stream=False, **kwargs)` is at `:164`. A `str`/`Path` is loaded and resampled to `preprocessor_config.sample_rate`
  (16 kHz). **An array is used as is**: `audio.astype(dtype)` (`:214`), so pass an `mx.array` that is already mono
  float at 16 kHz. A NumPy array fails at that `astype(mx.bfloat16)` call; wrap it with `mx.array(...)`. Nothing
  resamples an array.
- With `chunk_duration=None` it runs `decode_chunk` (`:155-162`): log-mel, then greedy TDT decode. It returns an
  `AlignedResult` with `.text` and `.sentences` (token timestamps). `max_tokens` and `generation_stream` are popped
  and ignored (`:197-198`).
- `stream=True` / `stream_generate` (`:293`) is chunked re-decoding of a complete buffer: 5 s chunks with 1 s overlap
  by default. It yields `StreamingResult(text, tokens, is_final, start_time, end_time, progress, audio_position,
  audio_duration, language)` (`:86-111`). It is not a live feed; Parakeet has no `create_streaming_session`.
- Measured on this Mac: 0.023 s from a path and 0.025 s from a pre-loaded array for a 2.48 s clip (09b section 3).

## 5. Nemotron ASR streaming session

`model.create_streaming_session(*, temperature=0.0, language=None)` (`stt/models/nemotron_asr/nemotron_asr.py:95-103`,
same line in the clone) returns a `NemotronStreamingSession` (`session.py`, identical in the clone):

| member | line | behaviour |
|---|---|---|
| constructor | `:22-30` | greedy only: any `temperature != 0` raises; `language` defaults to the model's; `input_sample_rate` = the model's (16 kHz) |
| `feed(samples)` | `:55-66` | mono finite float PCM at `input_sample_rate`, any length; copied into a queue under a lock. **Thread-safe**: may run on a producer thread. More than 30 s queued raises `BufferError`. Feeding after `close()` raises |
| `close()` | `:68-71` | end of input; `step()` then drains and flushes exactly once |
| `step(*, max_decode_tokens=4)` | `:108-152` | must run on one consumer thread. Ingests at most one native encoder chunk of audio per call (`_ingest`, `:80-106`: `chunk_mel × hop_length` samples), then runs up to `max_decode_tokens` joint evaluations, blanks included |
| `done` | `:51-53` | true once the flushed input is fully decoded; `step()` then returns `[]` |
| `cancel()` | `:73-78` | discards the session without a final, on the decoder thread |
| `reset()` | `:32-49` | fresh state, same session object |

**The partial result is a `list[str]` of text deltas,** append-only, one entry per non-blank token. There are no
timestamps, no stability flags and no final marker; completion is `done`. The pieces are clean text: the tokenizer
drops `<unk>`, `<pad>`, `<s>`, `</s>` and language tags such as `<en-US>` (`tokenizer.py:33-46`). The first delta is
`lstrip`ped (`session.py:135-136`). An empty list does not mean the session is finished.

The shared contract is the `StreamingSession` protocol (`stt/streaming.py`): "`feed` and `close` may run on a
producer thread; `step` must run on a single model executor. Construction may also perform model/MLX work." The
realtime server drives it with `session.step(max_decode_tokens=8)` under one asyncio lock on the event-loop thread
(`server.py:1665-1672`, `REALTIME_INFERENCE_LOCK` at `:213`).

Measured on 2026-10-05 (09b section 3), feeding 320 ms chunks in real time: deltas arrive in bursts each time a
1.12 s encoder chunk completes (`encoder_chunk_mel` 112 × hop 160), every 0.96 or 1.28 s of audio. After `close()`,
the last deltas arrive 53–54 ms after the audio ends. `step(max_decode_tokens=8)` costs 0.2–2 ms when no encoder
chunk completes and 31–56 ms when one does.

New in 0.5.7: `create_speaker_streaming_session(diarization_model, ...)` (`nemotron_asr.py:112-121`). Its `feed(pcm)`
returns speaker-tagged token deltas, and `feed([], final=True)` flushes. It pairs with the new
`vad/models/nemotron_diarization`.

## 6. Kokoro: `split_pattern`, misaki and espeak

- `generate(text, voice=None, speed=1.0, lang_code="a", split_pattern=r"\n+", **kwargs)` (`tts/models/kokoro/kokoro.py:293-313`).
  The default voice is `af_heart`.
- `split_pattern` goes to `KokoroPipeline.__call__` (`pipeline.py:425-470`), which does
  `re.split(split_pattern, text.strip())` (`:438`). Each segment is converted to phonemes, then cut into pieces of at
  most 510 phonemes at punctuation, preferring `!.?…`, then `:;`, then `,—` (`waterfall_last`, `:237`;
  `en_tokenize`, `:266`). Every piece is one `GenerationResult`, so `split_pattern=r"(?<=[.!?])\s+"` gives one yield
  per sentence. There is no streaming inside a piece.
- **misaki is imported lazily and is not a declared dependency** of 0.5.7 (`pipeline.py:21-58`;
  `_import_misaki_submodule` raises "Kokoro requires the optional 'misaki' package"). English (`lang_code` `a`/`b`)
  imports `misaki.en` and `misaki.espeak`. In misaki 0.9.4, `misaki.espeak` imports `phonemizer` and
  `espeakng_loader` and sets the espeak library and data path when it is imported (`misaki/espeak.py:3-10`).
- **The espeak fallback.** `KokoroPipeline.__init__` builds `espeak.EspeakFallback(...)` inside `try`/`except`; on an
  exception it warns "EspeakFallback not Enabled: OOD words will be skipped" and continues with `fallback=None`
  (`pipeline.py:146-151`). On this Mac the constructor does not raise. espeakng-loader's wheel has a build-machine data
  path baked in (`<build-machine>/runner/work/...`), and the process aborts later. The `sitecustomize.py` in
  `bench/.venv-kokoro` makes the constructor raise, so the library's own except branch runs.
- **The spaCy model.** `misaki.en.G2P` loads `en_core_web_sm` and, if it is missing, calls `spacy.cli.download`
  (`misaki/en.py:499-503`). spaCy 3.8 falls back to a bare `uv pip install` when pip is absent
  (`spacy/cli/download.py:195-196`), and that fails in a uv venv ("No virtual environment found"). Install the model
  wheel directly; `bench/requirements-kokoro.txt` does. `trf=False` is the pipeline's default (`:122`), so
  `misaki[en]`'s torch and spacy-curated-transformers are not needed.

## 7. Silero VAD and Smart Turn wrappers

Load either with `mlx_audio.vad.utils.load_model(repo)` (`vad/utils.py`). Both are identical in the clone.

**Silero** (`vad/models/silero_vad/silero_vad.py`):

- `feed(chunk, state=None, sample_rate=16000) -> (probability, state)` (`:162-196`). The chunk must be exactly
  `config.branch_16k.chunk_size` = 512 samples (32 ms) at 16 kHz; 256 at 8 kHz (`config.py:9-18`, `:43-52`). Pass
  `state=None` on the first call; the 64-sample context is carried in the returned `SileroVADState`. The probability
  is an `mx.array` of shape `[batch, 1]`.
- `ModelConfig` has **no** `chunk_size` attribute. The bench used `vad.config.chunk_size` until 2026-10-05.
- `predict_proba(audio, sample_rate=None)` (`:201-207`) scores a whole buffer.
  `get_speech_timestamps(audio, sample_rate=None, threshold=None, min_speech_duration_ms=None,
  min_silence_duration_ms=None, speech_pad_ms=None, return_seconds=False)` (`:209-241`) returns a list of
  `{"start", "end"}`, in samples unless `return_seconds`. The defaults are threshold 0.5, minimum speech 250 ms,
  minimum silence 100 ms and padding 30 ms (`config.py:22-31`).
- `realtime_vad.StreamingVad(vad_model, ServerVadConfig)` (`realtime_vad.py:151-195`) buffers arbitrary 16 kHz input
  into 512-sample frames (`:28-30`) and drives a `TurnDetector` (`:95`) that emits speech started and stopped events.
  The defaults are threshold 0.5, `prefix_padding_ms` 300 and `silence_duration_ms` 500 (`:41-43`). Its docstring says
  `process()` "runs MLX work, so call it from a worker thread".

**Smart Turn v3** (`vad/models/smart_turn/smart_turn.py`):

- `predict_endpoint(audio, sample_rate=None, threshold=None) -> EndpointOutput(prediction: int, probability: float)`
  (`:231-246`, `:15-18`). `prediction` is 1 when `probability > threshold`; the default threshold is 0.5
  (`config.py:32`).
- The input is any length. `_prepare_audio_array` (`:158-201`) resamples, keeps the **last** 8 s
  (`max_audio_seconds`, `config.py:27`), left-pads shorter input with zeros, and z-normalises if `normalize_audio`.
  It then computes a Whisper log-mel (`prepare_input_features`, `:203-229`). One call scores the end of the turn as it
  stands; there is no streaming state.
- Measured on 2026-10-05 (09b section 4): Silero `feed` takes 0.4 ms per chunk (p50), and Smart Turn 1.3–1.7 ms per
  call. Smart Turn scored the 9.27 s two-sentence question incomplete (0.215) and the 2.48 s question complete (0.989).

## 8. VoiceChat's duplex session API

**Which runtime loads our checkpoint.** `mlx-community/NemotronLabs-VoiceChat-11B-4bit` (snapshot `dffd203`) has
`mlx_runtime_config_version: 2` and top-level `text_config`, `audio_config`, `tts_config` and `codec_config`. Its
model card says `pip install -U mlx-vlm` and `mlx_vlm.load`.

- **mlx-audio** (`sts/models/nemotron_voicechat/`, identical in the clone and 0.5.7) parses the NeMo layout:
  `model.stt.model.perception`, `_rnnt_merge_info`, `model.speech_generation` (`config.py:107-128`). Its LLM config
  comes from `config["mlx_audio"]["llm_config"]` or else `PretrainedConfig.get_config_dict(pretrained_llm)` from the
  hub, default `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (`:92-100`, `:128`). `post_load_hook` fetches that repo's tokenizer
  (`model.py:226-236`). None of those keys exists in our checkpoint, and "mlx_runtime_config_version" appears nowhere in
  mlx-audio.
- **mlx-vlm 0.7.4** has the parser for it (`mlx_vlm/models/nemotron_voicechat/config.py`) and loads the tokenizer from
  the snapshot.

**API, mlx-vlm 0.7.4** (what the bench uses; on 2026-10-05 it loaded the 4-bit checkpoint in 7.2 s and ran 59.8 ms p50
and 61.0 ms p95 per 80 ms frame, 09b section 6):

```python
from mlx_vlm import load
model, processor = load("mlx-community/NemotronLabs-VoiceChat-11B-4bit")
vc = model.create_session(processor)                      # model.py:92-105
s = vc.create_streaming_session(system_prompt=..., seed=0, max_streaming_seconds=None,
                                use_language_cache=True, use_perception_cache=True, profile=False)  # session.py:133-155
events = s.push_audio(pcm_f32_16k, sample_rate=16000)     # streaming.py:518-547
events = s.flush(pad_partial=True)                        # :549-567, ends with kind="done"
events = s.cancel()                                       # :569-575, kind="cancelled"
```

- `push_audio` accepts any chunk length. It runs one model step per complete 80 ms frame (`frame_samples` = 1280 at
  16 kHz) and returns that frame's events, synchronously, on the calling thread. Any other sample rate raises.
- An event is `VoiceChatEvent(kind, frame_index, token_id, delta, text, samples, sample_rate, audio_codes)`
  (`streaming.py:29-39`). `kind` is one of `assistant_text_delta`, `function_delta`, `user_transcript_delta`,
  `audio`, `done` or `cancelled`. On text kinds `text` is cumulative and `delta` incremental. `audio` events carry
  22.05 kHz `samples`.
- `profile=True` records a `VoiceChatFrameTiming` per frame: perception, rnnt, language, tts, codec and total, in ms
  (`:43-53`). `s.profile.summary(drop_first=n)` returns mean, p50, p95 and max per stage, plus a realtime factor
  (`:73-107`). Each stage ends in `mx.eval` (`:353`, `:389`, `:415`, `:454`), so the stage times are compute, not graph
  building.
- `max_streaming_seconds` bounds the context; past it, `push_audio` raises `VoiceChatContextLimitError` (`:116`,
  `:474-477`).
- Creating a session calls `mx.random.seed(seed)` (`:271`), which reseeds MLX's global generator for the whole process,
  including any TTS sampling running beside it.

**API, mlx-audio** (same shape, for a NeMo-layout checkpoint): `model = mlx_audio.sts.load(path)`;
`s = model.create_duplex_session(**kw)`, which is `create_session().create_streaming_session(**kw)`
(`model.py:245-252`); the same `push_audio` / `flush` / `cancel` (`streaming.py:372`, `:397`, `:415`) and the same
event fields (`:16-37`). It has **no** `profile` option and reads `source_sample_rate`/`target_sample_rate` from its
config.

## 9. Thread safety: MLX streams belong to a thread

The package says the same thing in three places (all the same in the clone):

- `server_inference.py:196-201`: "MLX default streams are thread-local. The inference broker owns all model execution
  on this worker, so create its CPU/GPU streams here rather than inheriting streams from the ASGI or model-load
  threads." It calls `mx.new_stream(mx.cpu)`, `mx.new_stream(mx.gpu)` and `mx.set_default_stream(...)` inside the
  worker thread. One worker serialises all GPU work (`:217-219`).
- `sts/voice_pipeline.py:104-138` (`MLXWorkScheduler`): "Recent MLX streams are thread-local, so model load, cached
  state creation, and inference must stay on the same worker thread. A generic `asyncio.to_thread` call can hop
  between pool threads and later fail when evaluating arrays that reference a stream created in another thread." It
  uses a `ThreadPoolExecutor(max_workers=1)`, creates its GPU stream on first use and wraps every call in
  `with mx.stream(...)`.
- `server.py:1665-1672`: "MLX streams are thread-bound, so all realtime MLX work — the transcription step and the VAD —
  must share one thread, otherwise you hit 'no Stream(gpu, N) in current thread'."

**Rules for the orchestrator that follow from the source:**

1. Load every model and run every MLX call (STT, VAD, Smart Turn, TTS, VoiceChat) on **one** dedicated thread that
   created its own stream. A `ThreadPoolExecutor(max_workers=1)` is the library's pattern; `asyncio.to_thread` is not
   safe.
2. Only Nemotron's `feed`/`close` are documented as safe from another thread (`session.py:15`). Everything else,
   including `step`, Silero `feed`, Smart Turn, `generate()` and VoiceChat `push_audio`, belongs on the MLX thread.
3. Barge-in is a flag checked between chunks on the MLX thread. Then come `gen.close()`, the decoder reset and
   `mx.clear_cache()` (section 2). Nothing can pre-empt a chunk that is already computing.
4. One streaming TTS generator per model object at a time (section 2).
5. A VoiceChat session reseeds the global RNG (section 8).

## 10. What this means for the build

- Qwen3-TTS streaming barge-in works today by closing the generator between chunks. At interval 0.32 the stop lands
  within one chunk of compute, about 60 ms for the 0.6B. Add the decoder reset and `mx.clear_cache()` after an abort.
- Sentence-split before Qwen3-TTS: CustomVoice ignores `split_pattern`, and per-sentence calls bound both runaway
  generation (`max_tokens` per sentence) and stop latency for non-streamed fallbacks.
- Use the model's sampling values explicitly if `mlx_audio.server` is in the path; its request defaults differ.
- Parakeet takes an `mx.array` at 16 kHz with no resampling, and has no live session. For live partials the package
  offers Nemotron's `StreamingSession`, whose deltas are plain strings without timestamps.
- Smart Turn scores the last 8 s on each call, and Silero wants exact 512-sample frames. `realtime_vad.StreamingVad`
  already does the framing and the turn events.
- VoiceChat needs mlx-vlm 0.7.4 (MLX 0.32.3) for the checkpoint we have. A cascade on mlx-audio and a VoiceChat
  experiment on mlx-vlm would run on different MLX versions unless one is moved.
- Moving the clone (`ee0c65d`) to 0.5.7 changes nothing in the pieces measured. The changes are the Parakeet Redux
  additions, the chunk-loop fix, Nemotron speaker streaming and NumPy input, and the `dsp.py` window centring.
