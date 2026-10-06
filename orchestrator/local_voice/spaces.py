"""spaces.yaml (SPACES.md): load, check, and derive each space's Pi child from it.

The file is the single source the orchestrator derives the Pi child per space (cwd, --tools, --skill, -e, the
voice gate's environment), the trigger router and the dashboard from. A bad file fails at start with a message that
names the space and the key. Nothing here starts a process.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .approvals import settings as approval_settings
from .config import Config
from .pi_rpc import SpaceConfig

TIERS = ("readonly", "ask", "trusted")
MODES = ("conversation", "act")
READ_TOOLS = ("read", "grep", "find", "ls")
SPACE_KEYS = {"root", "description", "triggers", "skills", "extensions", "tools", "act_tools", "tier", "bash_allow",
              "model", "thinking", "voice", "rules", "gpu_exclusive"}
DEFAULT_KEYS = {"model", "deep_model", "thinking", "mode", "tier", "voice"}


class SpacesError(ValueError):
    """spaces.yaml is wrong; the message names the space and the key."""


@dataclass
class Space:
    name: str
    root: Path
    description: str
    triggers: list[str]
    skills: str | list[str]
    skill_dirs: list[Path]
    extensions: list[Path]
    tools: list[str]
    act_tools: list[str]
    tier: str
    bash_allow: list[list[str]]
    model: str
    thinking: str
    voice: str
    rules: list[str]
    gpu_exclusive: bool

    def dashboard(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "root": str(self.root), "tier": self.tier,
                "model": self.model, "tools": self.tools, "act_tools": self.act_tools,
                "skills": self.skills if isinstance(self.skills, str) else list(self.skills)}


@dataclass
class Spaces:
    path: Path
    defaults: dict[str, Any]
    spaces: dict[str, Space] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Space:
        return self.spaces[name]

    def __contains__(self, name: str) -> bool:
        return name in self.spaces

    @property
    def default_mode(self) -> str:
        return str(self.defaults.get("mode", "conversation"))


def _thinking(v: Any) -> Any:
    """YAML 1.1 reads an unquoted `off` as False (SPACES.md writes `thinking: off`): take it as the word it was."""
    return {False: "off", True: "on"}.get(v, v) if isinstance(v, bool) else v


def _list_of_str(v: Any, where: str) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
        raise SpacesError(f"{where} must be a list of non-empty strings")
    return [x.strip() for x in v]


def _find_skill(name: str, root: Path, skill_dirs: list[Path]) -> Path | None:
    for base in [root / ".pi" / "skills", root / ".agents" / "skills", *skill_dirs]:
        d = base / name
        if (d / "SKILL.md").exists():
            return d
    return None


def load_spaces(cfg: Config, path: Path | None = None, *, check_paths: bool = True) -> Spaces:
    path = Path(path or cfg.spaces_file)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        raise SpacesError(f"no spaces file at {path}") from e
    except yaml.YAMLError as e:
        raise SpacesError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise SpacesError(f"{path}: version must be 1")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SpacesError("defaults must be a mapping")
    if "thinking" in defaults:
        defaults["thinking"] = _thinking(defaults["thinking"])
    unknown = set(defaults) - DEFAULT_KEYS
    if unknown:
        raise SpacesError(f"defaults: unknown key(s) {', '.join(sorted(unknown))}")
    for k in ("model", "thinking", "mode", "tier"):
        if not isinstance(defaults.get(k), str):
            raise SpacesError(f"defaults.{k} is required (a string)")
    if defaults["tier"] not in TIERS:
        raise SpacesError(f"defaults.tier must be one of {', '.join(TIERS)}")
    if defaults["mode"] not in MODES:
        raise SpacesError(f"defaults.mode must be one of {', '.join(MODES)}")
    entries = raw.get("spaces")
    if not isinstance(entries, dict) or not entries:
        raise SpacesError("spaces must be a non-empty mapping of name -> space")

    out = Spaces(path=path, defaults=defaults)
    seen_triggers: dict[str, str] = {}
    for name, s in entries.items():
        w = f"spaces.{name}"
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", str(name)):
            raise SpacesError(f"{w}: a space name is lowercase letters, digits, - and _")
        if not isinstance(s, dict):
            raise SpacesError(f"{w} must be a mapping")
        unknown = set(s) - SPACE_KEYS
        if unknown:
            raise SpacesError(f"{w}: unknown key(s) {', '.join(sorted(unknown))}")
        for req in ("root", "description", "tools"):
            if req not in s:
                raise SpacesError(f"{w}.{req} is required")
        # YAML reads a bare `~` as null (SPACES.md writes `root: ~` for home): null means the home directory.
        root = Path.home() if s["root"] is None else Path(str(s["root"])).expanduser()
        if check_paths and not root.is_dir():
            raise SpacesError(f"{w}.root {root} is not a directory")
        if not isinstance(s["description"], str) or not s["description"].strip():
            raise SpacesError(f"{w}.description must be a short noun phrase")
        tools = _list_of_str(s["tools"], f"{w}.tools")
        if not tools:
            raise SpacesError(f"{w}.tools must list at least one tool")
        act_tools = _list_of_str(s.get("act_tools"), f"{w}.act_tools")
        tier = s.get("tier", defaults["tier"])
        if tier not in TIERS:
            raise SpacesError(f"{w}.tier must be one of {', '.join(TIERS)}, got {tier!r}")
        triggers = [t.lower() for t in _list_of_str(s.get("triggers"), f"{w}.triggers")]
        for t in triggers:
            if t in seen_triggers:
                raise SpacesError(f"{w}.triggers: {t!r} is also a trigger of {seen_triggers[t]} (the router would have to guess)")
            seen_triggers[t] = name
        skills_raw = s.get("skills", "auto")
        skill_dirs: list[Path] = []
        if skills_raw == "auto":
            skills: str | list[str] = "auto"
        else:
            skills = _list_of_str(skills_raw, f"{w}.skills")
            for sk in skills:
                d = _find_skill(sk, root, cfg.pi_skill_dirs)
                if d is None and check_paths:
                    raise SpacesError(f"{w}.skills: no skill {sk!r} under {root}/.pi/skills, {root}/.agents/skills or "
                                      + ", ".join(str(p) for p in cfg.pi_skill_dirs))
                if d is not None:
                    skill_dirs.append(d)
        exts = [Path(str(e)).expanduser() for e in _list_of_str(s.get("extensions"), f"{w}.extensions")]
        for e in exts:
            if "film-rig" in e.name:
                raise SpacesError(f"{w}.extensions: never film-rig.ts (a video rig's extension: a voice turn would hang mute during a render, 05c)")
            if check_paths and not e.exists():
                raise SpacesError(f"{w}.extensions: {e} does not exist")
        bash_allow_raw = s.get("bash_allow") or []
        if not isinstance(bash_allow_raw, list) or not all(isinstance(p, list) and p and all(isinstance(x, str) and x for x in p)
                                                         for p in bash_allow_raw):
            raise SpacesError(f"{w}.bash_allow must be a list of argv prefixes, each a non-empty list of strings")
        rules = _list_of_str(s.get("rules"), f"{w}.rules")
        gpu_exclusive = s.get("gpu_exclusive", False)
        if not isinstance(gpu_exclusive, bool):
            raise SpacesError(f"{w}.gpu_exclusive must be true or false")
        out.spaces[name] = Space(
            name=name, root=root, description=s["description"].strip(), triggers=triggers, skills=skills,
            skill_dirs=skill_dirs, extensions=exts, tools=tools, act_tools=act_tools, tier=tier,
            bash_allow=[list(p) for p in bash_allow_raw], model=str(s.get("model", defaults["model"])),
            thinking=str(_thinking(s.get("thinking", defaults["thinking"]))),
            voice=str(s.get("voice", defaults.get("voice", ""))),
            rules=rules, gpu_exclusive=gpu_exclusive)
    if cfg.default_space not in out.spaces:
        raise SpacesError(f"config agent.default_space is {cfg.default_space!r}, which spaces.yaml does not define")
    return out


def bash_globs(prefixes: list[list[str]]) -> list[str]:
    """argv prefixes (SPACES.md) as the gate's globs: the prefix exactly, or the prefix then a space and anything.
    voice_gate.ts applies them only to a command that is one simple command (no ;, &&, |, $(), redirections)."""
    globs: list[str] = []
    for p in prefixes:
        head = " ".join(p)
        globs += [head, head + " *"]
    return globs


def persona_text(cfg: Config, space: Space) -> str:
    from .tone_hook import hint_instruction

    base = cfg.persona_file.read_text(encoding="utf-8").strip()
    if space.rules:
        base += "\n\n" + "\n".join(space.rules)
    hint = hint_instruction(cfg)   # tone `on` only (../tone/README.md), so the prompt cache is unchanged otherwise
    if hint:
        base += "\n\n" + hint
    return base


SPACE_SWITCH = "space_switch"     # voice_mode.ts's tool: the model's switch, for words the router does not take


def pi_space_config(cfg: Config, space: Space, *, agent_dir: Path | None, state_dir: Path,
                    extra_env: dict[str, str] | None = None, others: list[Space] | None = None) -> SpaceConfig:
    """The Pi child for a space, exactly as SPACES.md describes it (one per space, started lazily by the pool).
    others: the other spaces, which the space_switch tool can switch to (registered and active in both modes when any)."""
    ext = cfg.pi_extensions
    others = [o for o in (others or []) if o.name != space.name]
    conversation = space.tools + ([SPACE_SWITCH] if others else [])
    registered = conversation + [t for t in space.act_tools if t not in conversation]
    extensions = [ext["voice_gate"], ext["voice_mode"], ext["hold"], ext["today"]]
    if "kb" in registered and "kb" in ext:
        extensions.append(ext["kb"])
    extensions += [e for e in space.extensions if e not in extensions]
    # the orchestrator answers every approval itself (a choice, or `timeout` once its own wait is over); the gate's
    # timeout is only a backstop behind the longest wait, in case the orchestrator never answers
    backstop_ms = max(cfg.pi_confirm_timeout_ms, int((approval_settings(cfg).longest_wait_s + 15) * 1000))
    env = {
        "VOICE_TIER": space.tier,
        "VOICE_TOOLS": ",".join(conversation),           # voice_mode.ts narrows to these at session start
        "VOICE_SPACES": json.dumps([{"name": o.name, "description": o.description} for o in others]),
        "VOICE_SPACE": space.name,                       # voice_gate.ts: the approval's action.space
        "VOICE_SPACE_DESC": space.description,           # ... and the space's root as said aloud
        "VOICE_CONFIRM_TIMEOUT_MS": str(backstop_ms),
        "VOICE_BASH_ALLOW": json.dumps(bash_globs(space.bash_allow)),
        "VOICE_READ_TOOLS": ",".join(READ_TOOLS),
        "VOICE_TOOL_BUDGET_CALLS": str(cfg.tool_budget_calls),   # conversation mode's budget (voice_gate.ts)
        "VOICE_TOOL_BUDGET_S": str(cfg.tool_budget_s),
        "HOLD_GATE": cfg.gate,
    }
    if agent_dir is not None:
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env.update(cfg.pi_env)
    env.update(extra_env or {})
    return SpaceConfig(
        name=space.name, root=space.root, model=space.model, thinking=space.thinking, tools=registered,
        skills=space.skills if space.skills == "auto" else [str(d) for d in space.skill_dirs],
        extensions=[str(e) for e in extensions], discover_extensions=False,
        session_dir=state_dir / "sessions" / space.name, session_name=f"voice-{space.name}",
        env=env, pi_bin=cfg.pi_bin, extra_args=["--system-prompt", persona_text(cfg, space)])
