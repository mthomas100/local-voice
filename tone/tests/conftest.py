"""Synthetic speech from macOS `say` (Samantha, 16 kHz PCM), cached under tone/.cache/say/ (gitignored).

Machine ground truth, never a listener: every deviation is planted with `say`'s own controls,
measured on this Mac on 2026-10-05: `-r` words per minute; `[[pbas N]]` pitch base, about 0.73 semitones per step
(45 is Samantha's default, 173 Hz median); `[[slnc MS]]` an exact silence; and a gain applied to the samples
afterwards. `[[pmod]]` changes nothing for Samantha, so pitch range is tested on synthetic tones instead.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

TONE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TONE))
CACHE = TONE / ".cache" / "say"
SR = 16000

BASELINE = [
    "I think the build went fine today, and the tests passed on the first try.",
    "The weather was mild this morning, so I walked to the station instead of taking the bus.",
    "Can you check whether the new release changed anything in the configuration file?",
    "We should probably move the meeting to Thursday, because Wednesday is already full.",
    "The library on the corner closes early on Sundays, which I always forget.",
    "Let me know when the download finishes, and then we can run the comparison again.",
    "I read most of the report last night, and the second half was more useful than the first.",
    "The coffee machine on the third floor has been broken for about a week now.",
    "Please remind me to send the invoice before the end of the month.",
    "The garden needs watering again, since it has not rained for a few days.",
    "I would like the summary to be shorter, maybe two or three sentences at most.",
    "The train was a little late, but I still made it in time for the start.",
    "Could you find the email from last Tuesday about the shipping address?",
    "Most of the photos came out well, although a few of them are slightly blurry.",
    "The model seems to answer faster when the prompt stays the same between turns.",
    "We ran out of milk, so I picked some up on the way back from the gym.",
    "I need to renew my passport sometime before the trip in the spring.",
    "The new keyboard feels better than the old one, especially for long sessions.",
    "Let's keep the meeting short and focus on the two decisions we still need.",
    "The printer finally works again after I reinstalled the driver this afternoon.",
    "I am going to try the other recipe tonight, the one with the roasted vegetables.",
    "There is a small scratch on the screen, but it does not affect anything.",
    "Could you read me the first paragraph of the document I opened earlier?",
    "The neighbours are painting their fence, so the street smells a bit like varnish.",
    "I usually go for a walk after lunch when the weather is good enough.",
    "The battery lasted the whole day, which is better than I expected.",
    "We could compare both versions side by side and then pick the clearer one.",
    "The package arrived this morning, a day earlier than the tracking page said.",
    "I want to spend the weekend tidying up the spare room and the garage.",
    "The concert starts at eight, so we should leave the house around seven.",
]
# Held-out ordinary utterances: new sentences with the baseline's spread. A hint on one of these is a false positive.
HELD_OUT = [
    "I left the charger at the office, so I will pick it up tomorrow morning.",
    "The soup needs a bit more salt, but otherwise it tastes really good.",
    "We might need a bigger table if everyone comes for dinner on Saturday.",
    "The update took longer than usual because the network was slow today.",
    "My sister called to say the flight lands an hour later than planned.",
    "Could you look up the opening hours of the hardware store near the bridge?",
    "The cat has been sleeping on the windowsill all afternoon again.",
    "I moved the old files into a separate folder so the desktop looks cleaner.",
    "The bakery started selling a rye bread that is actually quite good.",
    "Let me finish this paragraph first, and then we can go through the list.",
    "The heating comes on at six, which is a bit early for the weekend.",
    "I found the receipt in the drawer under a pile of old letters.",
    "The bus driver waited for us even though we were a minute late.",
    "Can you tell me how much space is left on the external drive?",
    "We planted the tomatoes too close together, so they are fighting for light.",
    "The meeting notes are in the shared folder under this week's date.",
    "I prefer the darker colour for the kitchen, but the lighter one is fine too.",
    "The museum is free on the first Sunday of every month.",
    "I should probably call the dentist before they close for the day.",
    "The new route to work takes about ten minutes less than the old one.",
]


def _say(text: str, rate: int, pbas: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    spoken = f"[[pbas {pbas}]] {text}"
    key = hashlib.sha1(f"Samantha|{rate}|{spoken}".encode()).hexdigest()[:16]
    out = CACHE / f"{key}.wav"
    if not out.exists():
        tmp = out.with_suffix(".tmp.wav")
        subprocess.run(["say", "-v", "Samantha", "-r", str(rate), "-o", str(tmp), "--file-format=WAVE",
                        "--data-format=LEI16@16000", spoken], check=True)
        tmp.rename(out)
    return out


def say_pcm(text: str, *, rate: int = 175, pbas: int = 45, gain_db: float = 0.0) -> tuple[bytes, str]:
    """PCM int16 LE mono 16 kHz of `text` (with [[slnc]] commands allowed) and its transcript without commands."""
    with wave.open(str(_say(text, rate, pbas))) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    x = np.clip(x * 10 ** (gain_db / 20), -32768, 32767).astype("<i2")
    return x.tobytes(), " ".join(re.sub(r"\[\[[^\]]*\]\]", " ", text).split())


def ordinary(sentences: list[str], offset: int = 0):
    """Ordinary utterances with the spread a real day has: rate 160-192 wpm, pitch base 44-46, level ±1.5 dB."""
    out = []
    for j, s in enumerate(sentences):
        i = j + offset
        out.append(say_pcm(s, rate=160 + (i * 11) % 33, pbas=44 + i % 3, gain_db=((i * 7) % 7 - 3) / 2))
    return out


def baseline_utterances():
    return ordinary(BASELINE)


def held_out_utterances():
    return ordinary(HELD_OUT, offset=5)


def stretch(pcm: bytes, factor: float) -> bytes:
    """Slower speech at the same pitch (librosa's phase vocoder): Samantha's own rate stops at about -r 145."""
    import librosa
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    y = librosa.effects.time_stretch(x, rate=factor)
    return (np.clip(y, -1, 0.99997) * 32768).astype("<i2").tobytes()


@pytest.fixture(scope="session")
def say_baseline():
    if subprocess.run(["which", "say"], capture_output=True).returncode:
        pytest.skip("macOS say is not available")
    return baseline_utterances()


def tone_pcm(f0_hz, seconds: float = 2.0, sr: int = SR, amp: float = 0.3) -> bytes:
    """A harmonic tone with a known pitch track (a constant or a callable of time), int16 PCM."""
    t = np.arange(int(seconds * sr)) / sr
    f = np.full_like(t, f0_hz) if not callable(f0_hz) else f0_hz(t)
    phase = 2 * np.pi * np.cumsum(f) / sr
    x = sum(amp / k * np.sin(k * phase) for k in range(1, 6))
    return (x / np.max(np.abs(x)) * amp * 32767).astype("<i2").tobytes()
