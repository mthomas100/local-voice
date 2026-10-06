"""Load brain/config.toml (plus an optional config.local.toml merged over it) into a checked Config.

Nothing is filled in silently: every key the job uses must be in the file, and a wrong type fails with the key's name.
Paths expand `~` and resolve relative to the local-voice repo root.
"""
from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BRAIN = Path(__file__).resolve().parent.parent          # brain/
REPO = BRAIN.parent                                       # local-voice/
DEFAULT_CONFIG = BRAIN / "config.toml"


class ConfigError(ValueError):
    """config.toml is wrong; the message names the key."""


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def expand(p: str | Path, base: Path = REPO) -> Path:
    path = Path(str(p)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _get(d: dict, dotted: str, typ: type | tuple[type, ...]) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"missing key: {dotted}")
        cur = cur[part]
    if typ is float and isinstance(cur, int) and not isinstance(cur, bool):
        cur = float(cur)
    if not isinstance(cur, typ) or (typ in (int, float) and isinstance(cur, bool)):
        raise ConfigError(f"{dotted} must be {getattr(typ, '__name__', typ)}, got {type(cur).__name__}")
    return cur


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    turns_dir: Path
    state_dir: Path
    gpu_clear: Path
    gpu_wait_s: float
    gpu_poll_s: float
    quiet_minutes: float
    user_idle_minutes: float
    orchestrator_status: str
    busy_states: list[str]
    runtime: str
    pi_bin: Path
    agent_dir: Path | None
    session_dir: Path
    model: str
    fallback_model: str
    restore_model: str
    timeout_s: float
    max_prompt_chars: int
    kb_enabled: bool
    kb_home: Path
    atlas_enabled: bool
    atlas_root: Path
    atlas_space: str
    atlas_by: str
    max_facts: int
    max_reflections: int
    spoken_max_words: int
    outbox_expiry_days: float
    max_attempts: int
    max_backlog_days: int

    @property
    def kb_bin(self) -> Path:
        return self.kb_home / "bin" / "kb"

    @property
    def kb_pi_adapter(self) -> Path:
        return self.kb_home / "adapters" / "pi" / "kb.ts"


def from_dict(raw: dict[str, Any], path: Path = DEFAULT_CONFIG, base: Path = REPO) -> Config:
    if _get(raw, "llm.runtime", str) != "pi":
        raise ConfigError("llm.runtime: only 'pi' is implemented")
    agent_dir = _get(raw, "llm.agent_dir", str)
    by = _get(raw, "atlas.by", str).strip()
    if not by or by.lower() == "me":
        raise ConfigError("atlas.by must name an agent, never 'me' (Atlas treats `by: me` as the person's words)")
    return Config(
        raw=raw, path=path,
        turns_dir=expand(_get(raw, "paths.turns_dir", str), base),
        state_dir=expand(_get(raw, "paths.state_dir", str), base),
        gpu_clear=expand(_get(raw, "gates.gpu_clear", str), base),
        gpu_wait_s=_get(raw, "gates.gpu_wait_s", float),
        gpu_poll_s=_get(raw, "gates.gpu_poll_s", float),
        quiet_minutes=_get(raw, "gates.quiet_minutes", float),
        user_idle_minutes=_get(raw, "gates.user_idle_minutes", float),
        orchestrator_status=_get(raw, "gates.orchestrator_status", str),
        busy_states=[str(s) for s in _get(raw, "gates.busy_states", list)],
        runtime="pi",
        pi_bin=expand(_get(raw, "llm.pi_bin", str), base),
        agent_dir=expand(agent_dir, base) if agent_dir else None,
        session_dir=expand(_get(raw, "llm.session_dir", str), base),
        model=_get(raw, "llm.model", str),
        fallback_model=_get(raw, "llm.fallback_model", str),
        restore_model=_get(raw, "llm.restore_model", str),
        timeout_s=_get(raw, "llm.timeout_s", float),
        max_prompt_chars=_get(raw, "llm.max_prompt_chars", int),
        kb_enabled=_get(raw, "kb.enabled", bool),
        kb_home=expand(_get(raw, "kb.home", str), base),
        atlas_enabled=_get(raw, "atlas.enabled", bool),
        atlas_root=expand(_get(raw, "atlas.root", str), base),
        atlas_space=_get(raw, "atlas.space", str),
        atlas_by=by,
        max_facts=_get(raw, "digest.max_facts", int),
        max_reflections=_get(raw, "digest.max_reflections", int),
        spoken_max_words=_get(raw, "digest.spoken_max_words", int),
        outbox_expiry_days=_get(raw, "digest.outbox_expiry_days", float),
        max_attempts=_get(raw, "digest.max_attempts", int),
        max_backlog_days=_get(raw, "digest.max_backlog_days", int),
    )


def load(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """config.toml, then config.local.toml beside it if present, then `overrides` (tests)."""
    p = Path(path) if path else DEFAULT_CONFIG
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"cannot read {p}: {e}") from e
    local = p.with_name(p.stem + ".local.toml")
    if local.exists():
        raw = deep_merge(raw, tomllib.loads(local.read_text(encoding="utf-8")))
    if overrides:
        raw = deep_merge(raw, overrides)
    return from_dict(raw, p)
