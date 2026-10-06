"""The tone hook (../tone/README.md, M5), wired in off by default.

The tone package is imported from its own folder only when config.yaml's tone.mode is `log` or `on`, so the
orchestrator neither needs its dependencies (Parselmouth) nor loads it otherwise; a failed import leaves the hook off
with an error in the log instead of stopping the server. It is CPU work, about 3 ms for a 4 s utterance, so
asyncio.to_thread is fine (the one-thread rule is for MLX).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .config import Config


def make_tone_hook(cfg: Config):
    """A ToneHook, or None when tone is off or cannot be loaded."""
    t = dict(cfg.tone or {})
    if t.get("mode", "off") == "off":
        return None
    pkg = Path(t.pop("package_dir"))
    try:
        if str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))
        from local_voice_tone import ToneHook   # noqa: PLC0415 - optional, see the module docstring
        return ToneHook(t, state_dir=cfg.state_dir / "tone")
    except Exception as e:  # noqa: BLE001 - tone is optional: the conversation goes on without it
        logger.error(f"tone hook not loaded ({e}); delivery hints stay off")
        return None


def hint_instruction(cfg: Config) -> str | None:
    """HINT_INSTRUCTION for the persona, only in `on` mode, so the persona and the prompt cache stay the same otherwise."""
    if (cfg.tone or {}).get("mode") != "on":
        return None
    try:
        if cfg.tone["package_dir"] not in sys.path:
            sys.path.insert(0, cfg.tone["package_dir"])
        from local_voice_tone import HINT_INSTRUCTION   # noqa: PLC0415
        return HINT_INSTRUCTION
    except Exception:  # noqa: BLE001
        return None


async def analyze(hook, pcm: bytes, text: str, *, session: str, turn: int, channel: str) -> Any:
    if hook is None or not pcm:
        return None
    try:
        return await asyncio.to_thread(hook.analyze, pcm, text, session=session, turn=turn, channel=channel or "unknown")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"tone analysis failed: {e}")
        return None
