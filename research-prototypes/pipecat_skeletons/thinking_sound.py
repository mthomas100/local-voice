"""The "working" loop while a tool runs, on Pipecat 1.12.0's SoundfileMixer.

Facts (pipecat-ai 1.12.0):
- SoundfileMixer(sound_files={name: path}, default_sound, volume=0.4, mixing=True, loop=True)
  (audio/mixers/soundfile_mixer.py:40-69). Files must be MONO at the output transport's sample rate; a file
  at another rate is skipped with only a warning, so the loop is silently absent (soundfile_mixer.py:146-161).
- Control at runtime with frames: MixerEnableFrame(enable) and MixerUpdateSettingsFrame(settings={"sound",
  "volume", "loop"}) (frames.py:2518-2543, soundfile_mixer.py:88-124). The output transport applies them the
  moment they arrive, not in step with queued audio (base_output.py:395-396, 680-687). Both are
  ControlFrames, so they are processed in order and dropped by an interruption.
- No ducking: mix() adds sound * volume to whatever audio is going out (soundfile_mixer.py:163-188). Speech
  over the loop is a sum. Turn it off when the answer starts (the bridge does, on the first text delta) or
  lower "volume" while the bot speaks.
- With a mixer attached, the output transport never goes quiet: when no audio is queued it emits
  mixer-only frames continuously (base_output.py:907-937), and the WebSocket transports pace them at real
  time. For protocol v1 that means a constant 24 kHz stream to the phone (~384 kbit/s) even when nothing is
  said, and audio flow no longer marks a reply. Mixer-only frames are plain OutputAudioRawFrame, so they do
  not trigger BotStartedSpeakingFrame (base_output.py:852-861).

Recommendation: use the mixer only on the browser (SmallWebRTC) path, whose audio track is continuous anyway
(audio_out_auto_silence, smallwebrtc/transport.py:538-545). Native clients get {"t":"tool","phase":"start"}
and play their own loop until {"t":"tool","phase":"end"} or audio_start; their audio engine can duck it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer
from pipecat.frames.frames import MixerEnableFrame, MixerUpdateSettingsFrame


def write_working_loop(path: Path, *, sample_rate: int = 24000, seconds: float = 2.0) -> Path:
    """Synthesize a soft two-note pulse that loops cleanly, as mono PCM16 at the transport's rate.

    Generated at startup because the repo ignores *.wav (.gitignore). Swap in a designed sound later.
    """
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    pulse = np.zeros_like(t)
    for start, freq in ((0.0, 523.25), (0.18, 659.25)):  # C5 then E5, 120 ms each with a soft envelope
        idx = (t >= start) & (t < start + 0.12)
        env = np.sin(np.pi * (t[idx] - start) / 0.12) ** 2
        pulse[idx] += 0.25 * env * np.sin(2 * np.pi * freq * t[idx])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pulse.astype(np.float32), sample_rate, subtype="PCM_16")
    return path


def make_working_mixer(path: Path, *, volume: float = 0.15) -> SoundfileMixer:
    """A mixer that starts silent (mixing=False) and loops the working sound once enabled."""
    return SoundfileMixer(
        sound_files={"working": str(path)}, default_sound="working", volume=volume, mixing=False, loop=True
    )


def start_working() -> MixerEnableFrame:
    """Frame that starts the loop."""
    return MixerEnableFrame(enable=True)


def stop_working() -> MixerEnableFrame:
    """Frame that stops the loop."""
    return MixerEnableFrame(enable=False)


def duck(volume: float = 0.05) -> MixerUpdateSettingsFrame:
    """Frame that lowers the loop under speech."""
    return MixerUpdateSettingsFrame(settings={"volume": volume})
