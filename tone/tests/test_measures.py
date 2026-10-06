"""Each measure against a signal whose answer is known by construction: tones of known pitch, silences of known
length, words with known syllables. No listener, no model."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import SR, tone_pcm

from local_voice_tone import audio, pitch, text
from local_voice_tone.features import extract


@pytest.mark.parametrize("word,n", [("the", 1), ("build", 1), ("station", 2), ("configuration", 5), ("tomorrow", 3),
                                    ("vegetables", 4), ("finally", 3), ("8090", 4), ("th-", 1), ("", 0)])
def test_syllables(word, n):
    assert text.syllables(word) == n


def test_text_counts_fillers_repetitions_cutoffs_and_markers():
    c = text.count("Um, so I I think, uh, the th- the thing is, you know, kind of hard. Hmm.")
    assert (c.fillers, c.repetitions, c.cutoffs, c.discourse) == (3, 1, 1, 2)
    assert c.words == 17
    assert text.count("").words == 0


@pytest.mark.parametrize("f0", ["praat", "pyin"])
@pytest.mark.parametrize("hz", [90.0, 150.0, 240.0])
def test_pitch_level_of_a_known_tone(f0, hz):
    x = audio.pcm16_to_float(tone_pcm(hz))
    med, rng, voiced = pitch.summarise(pitch.BACKENDS[f0](x, SR))
    assert abs(med - pitch.semitones(np.array([hz]))[0]) < 0.3      # within a third of a semitone
    assert rng < 0.5 and voiced > 1.5


def test_pitch_range_of_a_glide():
    glide = tone_pcm(lambda t: 120.0 * 2 ** (t / 2.0 * 7 / 12), seconds=2.0)  # rises 7 semitones over 2 s
    flat = tone_pcm(150.0)
    rng_glide = pitch.summarise(pitch.f0_praat(audio.pcm16_to_float(glide), SR))[1]
    rng_flat = pitch.summarise(pitch.f0_praat(audio.pcm16_to_float(flat), SR))[1]
    assert 5.0 < rng_glide < 7.0 and rng_flat < 0.3                   # p90 - p10 of a linear 7 st glide is 5.6 st


def test_pauses_from_energy_are_the_planted_silences():
    burst = np.frombuffer(tone_pcm(160.0, seconds=0.6), dtype="<i2")
    gap = lambda s: np.zeros(int(s * SR), dtype="<i2")                  # noqa: E731
    pcm = np.concatenate([gap(0.4), burst, gap(0.8), burst, gap(0.15), burst, gap(1.6), burst, gap(0.5)])
    segs = audio.energy_segments(audio.pcm16_to_float(pcm), SR)
    p = audio.pauses(segs)
    assert p.count == 2                                                 # the 0.15 s gap is inside speech, not a pause
    assert abs(p.longest_s - 1.6) < 0.05 and abs(p.total_s - 2.4) < 0.08
    assert abs(p.region_s - (0.6 * 4 + 0.8 + 0.15 + 1.6)) < 0.08         # leading and trailing silence excluded


def test_pauses_with_background_noise():
    rng = np.random.default_rng(1)
    burst = np.frombuffer(tone_pcm(160.0, seconds=0.7), dtype="<i2").astype(float)
    sil = np.zeros(int(0.9 * SR))
    x = np.concatenate([sil[:4000], burst, sil, burst, sil[:4000]])
    x = x + rng.normal(0, 30, len(x))                                   # a noise floor about 40 dB under the speech
    p = audio.pauses(audio.energy_segments(x / 32768.0, SR))
    assert p.count == 1 and abs(p.longest_s - 0.9) < 0.06


def test_given_speech_segments_are_used_as_they_are():
    f = extract(tone_pcm(150.0, seconds=3.0), "one two three four five six", speech_segments=[(0.2, 1.0), (1.6, 2.8)])
    assert f.pause_count == 1 and abs(f.pause_s - 0.6) < 1e-9 and abs(f.speech_s - 2.0) < 1e-9


def test_level_tracks_gain():
    loud = extract(tone_pcm(150.0, amp=0.5), "a b c")
    quiet = extract(tone_pcm(150.0, amp=0.05), "a b c")
    assert abs((loud.level_db - quiet.level_db) - 20.0) < 0.5


def test_silence_and_empty_input_do_not_crash():
    f = extract(b"\x00\x00" * SR, "")
    assert f.speech_s == 0 and np.isnan(f.f0_median_st) and f.words == 0
    f = extract(b"", "hello")
    assert f.speech_s == 0 and f.words == 1
