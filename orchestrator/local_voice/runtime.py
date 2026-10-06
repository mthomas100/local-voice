"""Process-wide speech runtime: the MLX worker, the loaded STT and TTS adapters, and the rendered notices.

Models are loaded and warmed once at startup on the MLX thread and stay resident (reloading per reply is "how local
voice ends up feeling like a demo"); every pipeline (connection) shares them. Startup refuses while the
hold gate is draining or held (milestone M1), and the busy notice is rendered then, because no model
may run during a hold.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .config import Config
from .engines.base import StreamingTranscriber, Synthesizer, Transcriber, float_to_pcm16
from .hold import HoldMonitor
from .mlx_worker import MLXWorker
from .services.stt import TAIL_PAD_S, SegmentedMLXSTTService, StreamingMLXSTTService, transcribe_buffer
from .services.tts import MLXTTSService
from .session import Notice


class HoldNotOpen(RuntimeError):
    """Startup refused: the hold gate is draining or held."""


@dataclass
class SpeechRuntime:
    cfg: Config
    hold: HoldMonitor
    use_mlx: bool = True
    worker: MLXWorker = field(init=False)
    stt_engine: Transcriber | StreamingTranscriber | None = None
    tts_engine: Synthesizer | None = None
    busy_notice: Notice | None = None
    timings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.worker = MLXWorker(guard=self.hold.gpu_refusal, use_mlx=self.use_mlx)

    async def start(self, *, state_dir: Path | None = None) -> dict[str, Any]:
        """Check the gate, load and warm both adapters, render the busy notice. Returns the timings."""
        st = await self.hold.check()
        if not st.allows_gpu:
            raise HoldNotOpen(f"the hold gate is {st.phase} ({st.holder or st.why}); not loading speech models now")
        t0 = time.monotonic()
        self.stt_engine = self.cfg.stt.build()
        self.tts_engine = self.cfg.tts.build()
        for name, eng in (("stt", self.stt_engine), ("tts", self.tts_engine)):
            t = time.monotonic()
            await self.worker.run(eng.load)
            self.timings[f"{name}_load_s"] = round(time.monotonic() - t, 3)
            t = time.monotonic()
            await self.worker.run(eng.warm)
            self.timings[f"{name}_warm_s"] = round(time.monotonic() - t, 3)
            logger.info(f"{name}: {type(eng).__name__} {eng.settings.get('model')} loaded in "
                        f"{self.timings[f'{name}_load_s']} s, warmed in {self.timings[f'{name}_warm_s']} s")
        t = time.monotonic()
        self.busy_notice = await self.render(self.cfg.busy_notice)
        self.timings["notice_render_s"] = round(time.monotonic() - t, 3)
        self.timings["startup_s"] = round(time.monotonic() - t0, 3)
        if state_dir is not None and self.busy_notice is not None:
            _save_wav(state_dir / "notices" / "busy.wav", self.busy_notice)
        return self.timings

    async def render(self, text: str) -> Notice:
        """Synthesize a whole notice now (GPU allowed), trimmed of leading silence, as PCM16."""
        eng = self.tts_engine
        assert eng is not None

        def run() -> np.ndarray:
            parts = [np.asarray(c, dtype=np.float32).reshape(-1) for c in eng.stream(text)]
            return np.concatenate(parts) if parts else np.zeros(0, np.float32)

        audio = await self.worker.run(run)
        loud = np.flatnonzero(np.abs(audio) > 0.01)
        if loud.size:
            audio = audio[max(0, loud[0] - int(0.03 * eng.sample_rate)):]
        return Notice(text=text, pcm=float_to_pcm16(audio), rate=eng.sample_rate)

    # -- per-pipeline services sharing the loaded engines

    def make_stt(self, *, on_held_audio=None):
        eng = self.stt_engine
        s = self.cfg.stt.settings
        if isinstance(eng, StreamingTranscriber):
            return StreamingMLXSTTService(engine=eng, worker=self.worker, on_held_audio=on_held_audio,
                                          preroll_s=float(s.get("preroll_s", 0.5)),
                                          tail_pad_s=float(s.get("tail_pad_s", TAIL_PAD_S)),
                                          ttfs_p99_latency=float(s.get("ttfs_p99_s", 0.25)),
                                          carry_s=float(s.get("carry_s", 0.0)))
        return SegmentedMLXSTTService(engine=eng, worker=self.worker, on_held_audio=on_held_audio,
                                      trailing_silence_secs=float(s.get("trailing_silence_s", 0.3)),
                                      ttfs_p99_latency=float(s.get("ttfs_p99_s", 0.25)))

    def make_tts(self, *, on_held=None, sample_rate: int = 24000):
        return MLXTTSService(engine=self.tts_engine, worker=self.worker, sample_rate=sample_rate,
                             trim_leading_silence=self.cfg.trim_leading_silence,
                             keep_before_onset_ms=self.cfg.keep_before_onset_ms, on_held=on_held,
                             skip_aggregator_types=["code"], group_min_words=self.cfg.tts_group.get("min_words", 0),
                             group_hold_s=self.cfg.tts_group.get("hold_s", 0.5))

    async def transcribe(self, pcm: bytes) -> str:
        return await transcribe_buffer(self.worker, self.stt_engine, pcm)

    def describe(self) -> dict[str, Any]:
        return {"stt": self.stt_engine.describe() if self.stt_engine else None,
                "tts": self.tts_engine.describe() if self.tts_engine else None,
                "mlx_calls": self.worker.calls, "mlx_refused": self.worker.refused, "timings": self.timings}

    def shutdown(self) -> None:
        self.worker.shutdown()


def _save_wav(path: Path, notice: Notice) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(notice.rate)
        w.writeframes(notice.pcm)
