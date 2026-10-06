"""The planted-echo mixer: what a protocol v1 client's speaker plays, put back into its own microphone the way a room
would, in pure numpy, so every property the echo bench relies on is measured by its tests instead of trusted
(tools/echo_bench.py, tests/test_echo_bench.py; 2026-10-05).

Why: in an early live test (the browser page in Chrome on the laptop's speakers) 5 of
19 voice turns were the agent's own words: replays of one reply's last ~10 s that came back 6 to 59 s after the reply
ended, two of them cutting a reply, while during the replies themselves Chrome's echo canceller held the live echo
(Silero started no turn in about 150 s of the agent's speech; local_voice/echo_guard.py). The server now has a guard
(echo_guard.py, config `echo:`). Whether it works is measured by machine, never by anyone listening, so the test client
needs a room. One clock for everything, the client's (`V1Client.now()`), with t0 the moment its microphone starts:

- the speaker: local_voice/client.py's simulated player. Each reply's audio plays from its arrival, a pause in the
  server's sending (bargein.py) leaves a gap (`Reply.segments`), an `interrupt` flushes it (`Reply.interrupted_at`).
  Its output is a 24 kHz timeline, sample j at t0 + j / 24000.
- the microphone: 16 kHz, sample k at t0 + k / 16000 (the same origin, so speaker sample 3m is mic sample 2m), sent
  in 20 ms frames, each once its last sample is "captured": a real microphone cannot send its audio any sooner.
- the echo: the speaker output resampled to 16 kHz (windowed-sinc polyphase, 2/3, zero phase), delayed by
  `delay_ms`, scaled by `level_db` relative to the reply's own level, optionally convolved with a synthetic room
  response (Polack's model: Gaussian noise under an exponential decay, RT60 0.3 s, fixed seed, unit energy so a
  broadband echo keeps its level; a pure tone does not, a room's response varies with frequency).
- the room's own sound: optionally pink noise, -60 dBFS by default; a room is never digital silence, and the bench
  wants the VAD to see a floor, not zeros.
- planted clips: the person's speech (`say` renders) or a replay of the agent, added from a mic sample on.

Every part is a pure function of the mic sample index (the noise comes in seeded blocks, the dry echo is cached in
order), so rendering frame by frame equals rendering at once, and the echo of mic sample k needs the speaker only up to
t_k - delay + 2 ms: in the past when the frame is sent, so later audio can never change an echo already sent.

Levels: dB is 20 log10 of an RMS ratio; dBFS takes full scale as 1.0, so a full-scale sine is -3.0 dBFS.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

SPK_RATE = 24000                 # protocol v1 reply audio (PROTOCOL.md)
MIC_RATE = 16000                 # protocol v1 microphone audio
FRAME = 320                      # 20 ms at 16 kHz: local_voice.client.V1Client's frame
# Resampler: 2 x 48 taps per phase at 24 kHz (2 ms each side), cutoff 7.5 kHz, Kaiser beta 5.65 (about 60 dB down):
# with this window the transition is about 0.9 kHz wide, so what would alias (8-12 kHz) is in the stopband.
HALF_TAPS = 48
CUTOFF_HZ = 7500.0
KAISER_BETA = 5.65
DEFAULT_SEED = 20261005          # fixed: the room is the same in every run
NOISE_BLOCK = 1 << 15            # pink noise is shaped per 2.05 s block; a block seam at -60 dBFS is far below any VAD
AUDIBLE = 0.01                   # -40 dBFS: where a clip's speech starts and ends (client.speech_end_s's threshold)


def db(ratio: float) -> float:
    return 20 * float(np.log10(max(ratio, 1e-12)))


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def pcm16_to_float(pcm: bytes | bytearray) -> np.ndarray:
    return np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# ------------------------------------------------------------------------------------------------ resampling

def _kernel(u: np.ndarray) -> np.ndarray:
    """The interpolation kernel at offsets u (in 24 kHz samples): a sinc with its cutoff at CUTOFF_HZ under a Kaiser
    window HALF_TAPS samples wide on each side."""
    fc = CUTOFF_HZ / SPK_RATE
    w = np.zeros_like(u, dtype=np.float64)
    inside = np.abs(u) < HALF_TAPS
    w[inside] = np.i0(KAISER_BETA * np.sqrt(1.0 - (u[inside] / HALF_TAPS) ** 2)) / np.i0(KAISER_BETA)
    return 2 * fc * np.sinc(2 * fc * u) * w


_OFFS = np.arange(-HALF_TAPS + 1, HALF_TAPS + 1)          # speaker sample j - floor(1.5 k)
# Mic sample k sits at speaker position 1.5 k: an integer for even k, half way between two samples for odd k. One tap
# set per case, each normalised to a DC gain of exactly 1.
_PHASES = []
for _frac in (0.0, 0.5):
    _h = _kernel(_frac - _OFFS.astype(np.float64))
    _PHASES.append((_h / _h.sum()).astype(np.float32))


def span_24k(k0: int, n: int) -> tuple[int, int]:
    """The speaker samples [a, b) that mic samples k0 .. k0+n-1 are made from."""
    return (3 * k0) // 2 - HALF_TAPS + 1, (3 * (k0 + n - 1)) // 2 + HALF_TAPS + 1


def resample_window(x: np.ndarray, j_start: int, k0: int, n: int) -> np.ndarray:
    """Mic samples k0 .. k0+n-1 (16 kHz) of the speaker signal `x`, whose first element is speaker sample `j_start`
    and which covers at least span_24k(k0, n)."""
    if n <= 0:
        return np.zeros(0, np.float32)
    k = np.arange(k0, k0 + n)
    idx = ((3 * k) // 2 - j_start)[:, None] + _OFFS[None, :]
    taps = np.where((k % 2 == 0)[:, None], _PHASES[0][None, :], _PHASES[1][None, :])
    return (np.asarray(x, np.float32)[idx] * taps).sum(axis=1, dtype=np.float32)


def resample_24k_to_16k(x: np.ndarray) -> np.ndarray:
    """A whole 24 kHz clip at 16 kHz, on the same time origin (speaker sample 3m is mic sample 2m)."""
    x = np.asarray(x, np.float32)
    n = -(-2 * len(x) // 3)
    if n == 0:
        return np.zeros(0, np.float32)
    a, b = span_24k(0, n)
    pad = np.zeros(b - a, np.float32)
    pad[-a:-a + len(x)] = x
    return resample_window(pad, a, 0, n)


# ------------------------------------------------------------------------------------------------ the room

def room_response(rt60_s: float = 0.3, rate: int = MIC_RATE, seed: int = DEFAULT_SEED) -> np.ndarray:
    """A synthetic room impulse response (Polack's statistical model): Gaussian noise under an envelope that falls 60 dB
    in `rt60_s`, cut there, normalised to unit energy. Deterministic for a seed."""
    n = max(1, int(round(rt60_s * rate)))
    t = np.arange(n) / rate
    h = np.random.default_rng(seed).standard_normal(n) * 10.0 ** (-3.0 * t / rt60_s)
    return (h / np.sqrt(np.sum(h * h))).astype(np.float32)


def decay_rt60(h: np.ndarray, rate: int = MIC_RATE, top_db: float = -5.0, bottom_db: float = -35.0) -> float:
    """RT60 measured from an impulse response: Schroeder's backward integral, a line fitted from -5 to -35 dB (T30),
    extended to 60 dB. The tests measure room_response with this rather than trust its envelope."""
    e = np.cumsum((np.asarray(h, np.float64) ** 2)[::-1])[::-1]
    edc = 10 * np.log10(np.maximum(e / e[0], 1e-30))
    sel = (edc <= top_db) & (edc >= bottom_db)
    t = np.arange(len(h))[sel] / rate
    slope = np.polyfit(t, edc[sel], 1)[0]           # dB per second (negative)
    return float(-60.0 / slope)


class PinkNoise:
    """Pink (1/f power) noise at `dbfs`: white noise shaped by 1/sqrt(f) in blocks of NOISE_BLOCK samples, block b from
    the seed (seed, b) and scaled to the exact RMS, so any stretch can be rendered on its own and equals the same
    stretch of a longer render."""

    def __init__(self, dbfs: float = -60.0, seed: int = DEFAULT_SEED, block: int = NOISE_BLOCK):
        self.rms = 10 ** (dbfs / 20)
        self.seed = seed
        self.block = block
        self._cache: dict[int, np.ndarray] = {}

    def _get(self, b: int) -> np.ndarray:
        x = self._cache.get(b)
        if x is None:
            spec = np.fft.rfft(np.random.default_rng([self.seed, b]).standard_normal(self.block))
            spec[0] = 0.0
            spec[1:] /= np.sqrt(np.arange(1, len(spec)))
            x = np.fft.irfft(spec, n=self.block)
            x = (x * (self.rms / rms(x))).astype(np.float32)
            if len(self._cache) > 4:
                self._cache.pop(min(self._cache))
            self._cache[b] = x
        return x

    def take(self, k0: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        k = max(k0, 0)
        while k < k0 + n:
            b, off = divmod(k, self.block)
            m = min(self.block - off, k0 + n - k)
            out[k - k0:k - k0 + m] = self._get(b)[off:off + m]
            k += m
        return out


# ------------------------------------------------------------------------------------------------ the speaker

def played(replies: Iterable, t0: float | None, j0: int, n: int) -> np.ndarray:
    """What local_voice.client's simulated player played over speaker samples [j0, j0+n) (24 kHz, sample j at
    t0 + j / 24000): every reply's audio where its segments put it (a frame plays from its arrival, or right after the
    one before; a paused send leaves a gap), nothing of a reply from its `interrupted_at` on (the client flushed it).

    `replies` are local_voice.client.Reply objects (or anything with pcm, segments and interrupted_at)."""
    out = np.zeros(max(n, 0), np.float32)
    if t0 is None or n <= 0:
        return out
    for r in replies:
        segs = list(getattr(r, "segments", None) or [])
        if not segs:
            continue
        pcm = r.pcm
        total = len(pcm) // 2
        cut = None if r.interrupted_at is None else int(round((r.interrupted_at - t0) * SPK_RATE))
        pos = 0                                       # samples into this reply's audio
        for s, e in segs:
            length = int(round((e - s) * SPK_RATE))
            js = int(round((s - t0) * SPK_RATE))
            je = js + length if cut is None else min(js + length, cut)
            a, b = max(js, j0), min(je, j0 + n)
            if a < b:
                src_a, src_b = pos + (a - js), min(pos + (b - js), total)
                if src_a < src_b:
                    # a copy of just this stretch: a numpy view would pin the bytearray the reader still appends to
                    seg = pcm16_to_float(pcm[2 * src_a:2 * src_b])
                    out[a - j0:a - j0 + len(seg)] += seg
            pos += length
    return out


# ------------------------------------------------------------------------------------------------ the mixer

@dataclass
class EchoSettings:
    """One room. level_db None means no live echo (Chrome's canceller holding it, as in early live tests)."""
    level_db: float | None = None     # the echo's level relative to the reply's own
    delay_ms: float = 150.0           # from the speaker playing a sample to the microphone hearing it
    reverb: bool = False
    rt60_s: float = 0.3
    noise_dbfs: float | None = -60.0  # the pink-noise floor; None: digital silence between sounds
    seed: int = DEFAULT_SEED

    def label(self) -> str:
        if self.level_db is None:
            return "none"
        return f"{self.level_db:+.0f} dB, {self.delay_ms:.0f} ms, {'room' if self.reverb else 'dry'}"


@dataclass
class Planted:
    """A clip added to the microphone: the person's speech (`kind` speech) or the agent's own audio played back
    (`kind` echo, the delayed replay). Positions are mic samples."""
    label: str
    kind: str
    text: str
    start: int
    samples: np.ndarray = field(repr=False)
    gain_db: float = 0.0
    onset: int = 0                    # first sample above -40 dBFS
    end: int = 0                      # one past the last sample above -40 dBFS

    def to_json(self, mic_t0: float) -> dict:
        t = lambda k: round(mic_t0 + k / MIC_RATE, 4)   # noqa: E731
        return {"label": self.label, "kind": self.kind, "text": self.text, "gain_db": self.gain_db,
                "start_t": t(self.start), "onset_t": t(self.onset), "end_t": t(self.end),
                "clip_s": round(len(self.samples) / MIC_RATE, 3)}


def audible_bounds(x: np.ndarray, threshold: float = AUDIBLE) -> tuple[int, int]:
    """(first, one past last) sample above `threshold`; (0, 0) for a silent clip."""
    loud = np.flatnonzero(np.abs(np.asarray(x)) > threshold)
    return (int(loud[0]), int(loud[-1]) + 1) if loud.size else (0, 0)


class EchoMixer:
    """The microphone of one connection: render(k0, n) gives mic samples k0 .. k0+n-1 as floats, frame(i) the i-th
    20 ms frame as PCM16 bytes. `speaker(j0, n)` must return the 24 kHz speaker output over [j0, j0+n) (played())."""

    def __init__(self, settings: EchoSettings, speaker: Callable[[int, int], np.ndarray]):
        self.settings = settings
        self.speaker = speaker
        self.gain = None if settings.level_db is None else 10 ** (settings.level_db / 20)
        self.delay = int(round(settings.delay_ms * MIC_RATE / 1000))
        if self.gain is not None and self.delay < FRAME:
            # the echo of what plays while a frame is captured must not be needed before that frame is sent
            raise ValueError(f"delay_ms must be at least {1000 * FRAME / MIC_RATE:.0f} ms, got {settings.delay_ms}")
        self.rir = room_response(settings.rt60_s, seed=settings.seed) if settings.reverb else None
        self.noise = PinkNoise(settings.noise_dbfs, settings.seed) if settings.noise_dbfs is not None else None
        self.planted: list[Planted] = []
        self._dry = np.zeros(0, np.float32)            # the dry echo of mic samples [0, len), final once computed

    def plant(self, samples: np.ndarray, start: int, *, label: str, kind: str, text: str = "",
              gain_db: float = 0.0) -> Planted:
        x = np.asarray(samples, np.float32) * np.float32(10 ** (gain_db / 20))
        a, b = audible_bounds(x)
        p = Planted(label=label, kind=kind, text=text, start=int(start), samples=x, gain_db=gain_db,
                    onset=int(start) + a, end=int(start) + b)
        self.planted.append(p)
        return p

    def _extend_dry(self, upto: int) -> None:
        have = len(self._dry)
        if upto <= have:
            return
        out = np.zeros(upto - have, np.float32)
        # mic sample k echoes the speaker at mic position k - delay; positions before the speaker's origin are silent
        k0 = max(have - self.delay, 0)
        k1 = upto - self.delay
        if k1 > k0:
            a, b = span_24k(k0, k1 - k0)
            y = resample_window(self.speaker(a, b - a), a, k0, k1 - k0)
            out[k0 + self.delay - have:] = self.gain * y
        self._dry = np.concatenate([self._dry, out])

    def dry_echo(self, k0: int, n: int) -> np.ndarray:
        """The delayed, scaled, resampled speaker output for mic samples [k0, k0+n); zero before mic sample 0."""
        out = np.zeros(n, np.float32)
        if self.gain is None or n <= 0 or k0 + n <= 0:
            return out
        self._extend_dry(k0 + n)
        a = max(k0, 0)
        out[a - k0:] = self._dry[a:k0 + n]
        return out

    def echo(self, k0: int, n: int) -> np.ndarray:
        if self.gain is None:
            return np.zeros(n, np.float32)
        if self.rir is None:
            return self.dry_echo(k0, n)
        L = len(self.rir)
        d = self.dry_echo(k0 - L + 1, n + L - 1)
        return np.convolve(d, self.rir, mode="valid").astype(np.float32)

    def render(self, k0: int, n: int) -> np.ndarray:
        x = self.noise.take(k0, n) if self.noise is not None else np.zeros(n, np.float32)
        for p in self.planted:
            a, b = max(k0, p.start), min(k0 + n, p.start + len(p.samples))
            if a < b:
                x[a - k0:b - k0] += p.samples[a - p.start:b - p.start]
        if self.gain is not None:
            x += self.echo(k0, n)
        return x

    def frame(self, i: int) -> bytes:
        return float_to_pcm16(self.render(i * FRAME, FRAME))
