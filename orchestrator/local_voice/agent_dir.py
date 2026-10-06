"""The Pi agent dir the voice children run with.

"derived" (the default) writes one under the state dir at every start, from your Pi models.json (read only):
the `local` provider with just the models spaces.yaml names, each capped at pi.max_tokens per call (a voice-only
bound on runaway replies, 05d), packages off, and `skills` pointing at your global Pi skills so `skills: auto`
spaces discover what an interactive pi would. Nothing is written under ~/.pi/agent; Pi's sessions go to the state
dir through --session-dir anyway. The 05d brain test ran the real rig through exactly this kind of throwaway dir.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Config, ConfigError
from .spaces import Spaces

GLOBAL_SKILLS = Path("~/.pi/agent/skills").expanduser()


def models_in_use(spaces: Spaces) -> set[tuple[str, str]]:
    out = set()
    for ref in [spaces.defaults.get("model"), spaces.defaults.get("deep_model"), *[s.model for s in spaces.spaces.values()]]:
        if ref and "/" in str(ref):
            prov, _, mid = str(ref).partition("/")
            out.add((prov, mid))
    return out


def derive_agent_dir(cfg: Config, spaces: Spaces, dest: Path, *, base_url: str | None = None) -> Path:
    """Write <dest>/models.json and settings.json; returns dest. base_url overrides the provider's (tests: a stub)."""
    try:
        src = json.loads(cfg.pi_models_source.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"pi.models_source {cfg.pi_models_source} does not exist") from e
    wanted = models_in_use(spaces)
    providers: dict[str, dict] = {}
    for prov, mid in sorted(wanted):
        p = (src.get("providers") or {}).get(prov)
        if p is None:
            raise ConfigError(f"spaces.yaml names {prov}/{mid}, but {cfg.pi_models_source} has no provider {prov!r}")
        entry = providers.setdefault(prov, {**{k: v for k, v in p.items() if k != "models"}, "models": []})
        if base_url:
            entry["baseUrl"] = base_url
        m = next((x for x in p.get("models") or [] if x.get("id") == mid), None)
        if m is None:
            raise ConfigError(f"spaces.yaml names {prov}/{mid}, which {cfg.pi_models_source} does not list")
        m = dict(m)
        m["maxTokens"] = min(int(m.get("maxTokens") or cfg.pi_max_tokens), cfg.pi_max_tokens)
        entry["models"].append(m)
    default = str(spaces.defaults.get("model", "local/qwen38")).partition("/")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "models.json").write_text(json.dumps({"providers": providers}, indent=1))
    (dest / "settings.json").write_text(json.dumps({"defaultProvider": default[0], "defaultModel": default[2],
                                                    "packages": []}, indent=1))
    link = dest / "skills"
    if GLOBAL_SKILLS.is_dir() and not link.exists() and not link.is_symlink():
        os.symlink(GLOBAL_SKILLS, link)
    return dest


def agent_dir_for(cfg: Config, spaces: Spaces, state_dir: Path, *, base_url: str | None = None) -> Path:
    if cfg.pi_agent_dir == "derived":
        return derive_agent_dir(cfg, spaces, state_dir / "pi-agent", base_url=base_url)
    p = Path(cfg.pi_agent_dir).expanduser()
    if not (p / "models.json").exists():
        raise ConfigError(f"pi.agent_dir {p} has no models.json")
    return p
