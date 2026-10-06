"""Pipecat services over this project's engines (stt.py, tts.py) and the agent (../agent.py)."""
from __future__ import annotations

from dataclasses import fields
from typing import TypeVar

from pipecat.services.settings import ServiceSettings
from pipecat.utils.types import is_given

S = TypeVar("S", bound=ServiceSettings)


def store_settings(settings: S) -> S:
    """A settings object in Pipecat's "store mode": every field we do not set is None (unsupported), not NOT_GIVEN.
    Pipecat 1.12 logs an ERROR at every pipeline start for each NOT_GIVEN field (`validate_complete`): two lines per
    connection in every serve.log (a real-server run, 2026-10-05), noise that could hide a real error."""
    for f in fields(settings):
        if f.name != "extra" and not is_given(getattr(settings, f.name)):
            setattr(settings, f.name, None)
    return settings
