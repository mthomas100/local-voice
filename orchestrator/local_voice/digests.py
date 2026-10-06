"""The brain's spoken digests (brain/TURN_LOG.md §2): what the background reflection left to be said aloud.

The brain cannot speak (it never loads a TTS model); it queues `<brain state>/outbox/<day>.json`. The orchestrator
speaks the pending ones, oldest first and once, at the first idle moment after the first completed turn of a new
session, while the hold gate is open, as a reply of their own kept out of the context; then moves each file to
`outbox/delivered/`. The brain's own `outbox.py` (standard library only) is imported by path, not copied, so the two
sides cannot drift.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from loguru import logger

from .turnlog import now_local


def load_outbox_module(path: Path) -> ModuleType | None:
    if not path.exists():
        logger.warning(f"brain outbox module {path} not found; spoken digests are off")
        return None
    name = "local_voice_brain_outbox"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # registered before it runs: its dataclasses resolve their (postponed) annotations through sys.modules
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


class DigestOutbox:
    """Process-wide: two sessions finishing their first turns at once must not both speak the same digest."""

    def __init__(self, state_dir: Path | None, module_path: Path | None):
        self.state_dir = state_dir
        self.mod = load_outbox_module(module_path) if (state_dir and module_path) else None
        self._lock = asyncio.Lock()
        self._claimed: set[str] = set()
        self.spoken: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.mod is not None and self.state_dir is not None

    def pending(self) -> list[Any]:
        if not self.enabled:
            return []
        try:
            return [d for d in self.mod.pending(self.state_dir, now_local()) if d.id not in self._claimed]
        except Exception as e:  # noqa: BLE001 - a broken outbox must not break a conversation
            logger.warning(f"brain outbox unreadable: {e}")
            return []

    async def claim(self) -> list[Any]:
        async with self._lock:
            out = self.pending()
            self._claimed.update(d.id for d in out)
            return out

    def release(self, digest: Any) -> None:
        self._claimed.discard(digest.id)

    def delivered(self, digest: Any) -> None:
        try:
            self.mod.mark_delivered(digest.path, now_local())
            self.spoken.append(digest.id)
            logger.info(f"brain digest {digest.id} spoken and moved to delivered/")
        except Exception as e:  # noqa: BLE001
            logger.error(f"could not mark brain digest {digest.id} delivered: {e}")
