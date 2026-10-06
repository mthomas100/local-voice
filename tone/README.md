# tone: hearing how it was said, Tier 1 (milestone M5)

A small CPU library the orchestrator can call once per user turn. It measures how the person spoke (pace, pitch,
volume, pauses, fillers), compares that with their own recent usual, and, only when something stands well out,
returns one bracketed line for the model to read before their words:

```
[delivery vs usual, automatic and uncertain: pace faster (+5.0 sd)]
[delivery vs usual, automatic and uncertain: pausing more (39% of the time, usual 0%); a 1.6 s pause (usual longest 0.0 s)]
[delivery vs usual, automatic and uncertain: fillers more (4 in 19 words, usual none); pace slower (-2.7 sd)]
```

It never names an emotion. The background research behind the design: categorical emotion labels from speech are unreliable,
valence is mostly in the words, and what holds up is arousal-like measures, pace, pauses and fillers against a
personal baseline, handed to the model as an uncertain hint it should ask about rather than assert. The line is built
from fixed words (faster, slower, higher, lower, louder, quieter, more, less), numbers and units; a test holds every
emitted line to that vocabulary.

**Off by default.** With `mode` unset or `off`, `analyze` returns `None` at once and nothing is computed, read or
written. Nothing here loads a model, so it needs no GPU and no `gpu_clear.sh`.

## The call (orchestrator)

```python
from local_voice_tone import ToneHook, HINT_INSTRUCTION

tone = ToneHook(cfg.raw.get("tone"), state_dir=cfg.state_dir / "tone")    # a dict from config.yaml; None = off

# once per user turn, after the final transcript, with the utterance audio the recogniser got
# (PCM int16 LE mono 16 kHz, bytes or a NumPy int16 array). Plain CPU work: asyncio.to_thread is fine here
# (the one-thread rule is for MLX); about 3 ms for a 4 s utterance.
result = await asyncio.to_thread(tone.analyze, utterance_pcm, transcript, session=session_id, turn=turn_no,
                                 channel=client_kind,                 # "iphone", "mac", "browser": one baseline each
                                 speech_segments=silero_segments)     # optional [(start_s, end_s)], else energy-based

prompt = f"{result.hint}\n{transcript}" if result and result.hint else transcript
turn_record["tone"] = {"hint": result.due, "shown": bool(result.hint)} if result and result.due else None
```

- When `mode` is `on`, append `HINT_INSTRUCTION` to the persona (`--system-prompt`). It tells the model the line is an
  uncertain measurement, never to name an emotion from it, and to ask rather than assert. Add it only in `on` mode,
  so the persona, and the prompt cache, stay the same otherwise.
- **Never** let the hint reach the person's words: not the Atlas capture (SPACES.md: delivery cues never enter their
  words), not `user_text` in the turn log (brain/TURN_LOG.md), not the captions. It goes only into what Pi receives.
- Never speak it, and never show it as if it were what they said.

## Modes and settings

`ToneHook(config_dict, state_dir=...)`. Every key is optional; an unknown key is an error.

| Key | Default | Meaning |
|---|---|---|
| `mode` | `off` | `off`; `log` measures, compares, logs and grows the baseline but never returns a hint (the week of measuring the design asks for); `on` also returns the line when one is due |
| `z_threshold` | 1.5 | a feature is named in the line when its z passes this (the design value) |
| `trigger_z` | 2.0 | a line is due only when one feature's z passes this (see Calibration) |
| `min_samples` | 20 | a feature stays silent until the baseline holds this many of its values |
| `window` | 200 | the rolling baseline: the last N eligible utterances per channel |
| `cooldown_turns` | 3 | after a line, the next one is due no sooner than this many turns later in the same session |
| `ab_strip_fraction` | 0.5 | in `on` mode, the share of due lines that are logged but not given to the model, so the conversations with and without hints can be compared (a stable hash of session and turn picks the arm) |
| `f0` | `praat` | the pitch tracker: `praat` (Parselmouth, about 2 ms) or `pyin` (librosa, the `pyin` extra) |
| `min_speech_s`, `min_words` | 1.0, 3 | shorter turns ("yes") teach and trigger nothing |
| `min_pause_s`, `long_pause_s` | 0.25, 1.5 | what counts as a pause; the longest pause is named only from 1.5 s |
| `max_items` | 3 | at most this many features in one line, largest first |

## What is measured

Per utterance, all logged raw to `state/tone/hints-<day>.jsonl` with the z-scores and the arm (note 08: keep the raw
values so they can be re-interpreted): speech and pause structure (energy-based, or Silero's segments when given;
a pause is 250 ms or more inside the speech), median pitch and its p90 − p10 range in semitones (Praat
autocorrelation, octave jumps trimmed), median speech level in dBFS, articulation rate in syllables per second of
speech (a vowel-group syllable estimate; CMUdict was tried and was no steadier), words per minute, and fillers,
repetitions, cut-offs and discourse markers from the transcript.

The baseline is per channel (a phone mic and the Mac's record different levels), robust (median and MAD, with a floor
under each scale), kept in `state/tone/baseline-<channel>.json` under a file lock so several processes can share it,
and updated only after the comparison, so an utterance never moves its own yardstick. Fillers are counted against
the usual rate per word with a Poisson z, because a count of two in ten words is not a continuous measure.

## Calibration

`scripts/calibrate.py` (machine ground truth, no listener): a baseline of 30 ordinary `say` utterances with the spread
a real day has (rate 160-192 wpm, pitch base ±1 step, level ±1.5 dB); 20 held-out ordinary utterances, where a line
is a false positive; and 36 planted everyday-size changes (35% faster, 25% slower by time-stretch, ±3.6 semitones,
−10 and +6 dB, two 0.7 s silences, one 1.8 s pause, four fillers). Measured 2026-10-05:

| A line is due when one z passes | False positives (of 20 ordinary) | Planted changes named (of 36) |
|---|---|---|
| 1.5 (the design value) | 4 | 36 |
| **2.0 (default trigger)** | **1** | **33** |
| 3.0 | 0 | 32 |

With six features each allowed past 1.5 by chance, about one ordinary turn in five gets a line, which is the
over-reading the background research warns about; the 2.0 trigger keeps the design's 1.5 for naming features once a line is
due. The three misses are pace changes on particular sentences; pace is the noisiest measure. Real speech varies more
than `say`, so the real baseline's own spread will set most scales: the week of `log` mode is the real calibration.

## Not built (yet)

- SenseVoice-Small's free emotion and event tags (Tier 1 of the design): it is a model (GGUF, 253 MB), so it waits for a
  model adapter and the GPU rules; laughter and sighs come with it.
- Tiers 2 and 3 (dimensional models, an audio LLM as perceiver): only if Tier 1 changes anything.
- The A/B verdict: `scripts/summary.py` joins the hint log with the turn log and reports, per arm, how often the reply
  was interrupted, how long it was and whether it asked a question. A week of real use comes first.

## Files and commands

| Path | What |
|---|---|
| `local_voice_tone/hook.py` | `ToneHook`, `ToneConfig`, `ToneResult`, `HINT_INSTRUCTION` |
| `local_voice_tone/features.py`, `audio.py`, `pitch.py`, `text.py` | the measures |
| `local_voice_tone/baseline.py`, `hint.py` | the rolling baseline; deviations and the line |
| `scripts/calibrate.py`, `scripts/summary.py` | calibration on planted speech; the log summary and A/B readout |
| `tests/` | 37 tests: measures on tones and silences of known size, the hook's modes, the calibration bounds |

```bash
uv run --project tone pytest tone -q                 # about 7 s once the say corpus is cached (first run ~1 min)
uv run --project tone python tone/scripts/calibrate.py
uv run --project tone python tone/scripts/summary.py --days 7
```

Parselmouth (GPL-3.0) is a dependency, installed separately, not vendored; `f0 = "pyin"` (librosa, ISC) avoids it.
