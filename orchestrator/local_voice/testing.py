"""Model-free stand-ins for the turn-taking models, used by the unit and integration tests.

The real stack is Silero and Smart Turn v3.2 (CPU ONNX, loaded by Pipecat); the tests run the same pipeline with an
energy VAD and a timer instead, so they load no model at all and behave deterministically on synthetic audio.
"""
from __future__ import annotations

import numpy as np

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

from .bargein import SpeechCueMixin
from .pipeline import TurnSetup


class EnergyVADAnalyzer(VADAnalyzer):
    """Voice = RMS above a threshold, in 512-sample (32 ms) windows like Silero's."""

    def __init__(self, *, threshold: float = 0.02, params: VADParams | None = None, **kwargs):
        super().__init__(params=params or VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.2, min_volume=0.0),
                         **kwargs)
        self._threshold = threshold

    def num_frames_required(self) -> int:
        return 512

    def voice_confidence(self, buffer: bytes) -> float:
        a = np.frombuffer(buffer, dtype="<i2").astype(np.float32) / 32768.0
        return 1.0 if a.size and float(np.sqrt(np.mean(a * a))) >= self._threshold else 0.0


class CueEnergyVADAnalyzer(SpeechCueMixin, EnergyVADAnalyzer):
    """The energy VAD reporting each frame's score, as the pipeline's Silero does (bargein.py)."""


def model_free_turn(mic: str, *, speech_timeout: float = 0.4) -> TurnSetup:
    if mic == "ptt":
        return TurnSetup(vad=None, stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)], stop_timeout_s=3.0)
    return TurnSetup(vad=CueEnergyVADAnalyzer(), stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=speech_timeout)],
                     stop_timeout_s=3.0)
