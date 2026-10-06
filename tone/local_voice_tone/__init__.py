"""local-voice's Tier-1 delivery hints (M5): how something was said, measured against the person's own usual.

    from local_voice_tone import ToneHook, HINT_INSTRUCTION
    hook = ToneHook(config_dict, state_dir=state / "tone")      # mode "off" unless the config says otherwise
    result = hook.analyze(pcm16, transcript, session=sid, turn=n, channel=client)
    if result and result.hint: prompt = result.hint + "\\n" + transcript

See tone/README.md for the call the orchestrator makes and what it must never do with the line.
"""
from .features import Features, extract
from .hook import HINT_INSTRUCTION, ToneConfig, ToneHook, ToneResult

__all__ = ["Features", "HINT_INSTRUCTION", "ToneConfig", "ToneHook", "ToneResult", "extract"]
