"""A scratch copy of the orchestrator's configuration, so a client or an agent can run against a real server without
touching real state (2026-10-05).

A test run against the production config would put its turns into the real turn log the brain reflects on, let
kb.ts write session digests into the real knowledge base, and give the spaces the real journal (2026-10-05). The copy
written here keeps everything a server writes under one folder:

    DIR/config.yaml      config.yaml (with config.local.yaml merged) with every path absolute, the state dir, the turn
                         log and the brain's state under DIR, KB_HOME on a fresh `git clone` of the kb (searched with
                         rg, since a clone has no qmd index, and UV_OFFLINE so the kb CLI never resolves packages), and
                         the server bound to 127.0.0.1 only (servers under test listen on loopback)
    DIR/spaces.yaml      spaces.yaml with each space's root moved: a git work tree is cloned under DIR/spaces/<name>
                         (the kb space shares DIR/kb), the home folder becomes the empty DIR/home, anything else is
                         refused (SPACES.md "Test rule": tests never touch a real space)
    DIR/state, DIR/turns, DIR/brain, DIR/recordings (record.dir, when recording), DIR/kb, DIR/spaces/..., DIR/home

Clones are made once; a second run over the same DIR rewrites the two YAML files from the current sources and keeps
the clones (and whatever the last run wrote). Run it as `./run.sh --scratch DIR [--port N --browser-port N]`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from .config import DEFAULT_CONFIG, ConfigError, deep_merge, expand

KB_SOURCE = Path("~/kb").expanduser()
# Keys of config.yaml that hold paths, relative to the config file's folder (config.expand's rule).
PATH_KEYS = ("agent.spaces_file", "agent.persona_file", "pi.models_source", "brain.outbox_module")


def _git_root(p: Path) -> bool:
    return (p / ".git").exists()


def _clone(src: Path, dest: Path) -> Path:
    """A plain `git clone` (committed state only; the source is only read). Kept if it exists already."""
    if dest.exists():
        if not _git_root(dest):
            raise ConfigError(f"scratch: {dest} exists and is not a git clone; remove it or pick another folder")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "clone", "--quiet", str(src), str(dest)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ConfigError(f"scratch: git clone {src} failed: {r.stderr.strip()}")
    return dest


def _get(d: dict, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _set(d: dict, dotted: str, value: Any) -> None:
    *head, last = dotted.split(".")
    for part in head:
        d = d.setdefault(part, {})
    d[last] = value


def scratch_spaces(spaces_path: Path, dest: Path, kb_clone: Path, kb_source: Path = KB_SOURCE) -> Path:
    """DIR/spaces.yaml: spaces.yaml with every root moved off real data (module docstring)."""
    raw = yaml.safe_load(spaces_path.read_text(encoding="utf-8")) or {}
    home = Path.home().resolve()
    for name, s in (raw.get("spaces") or {}).items():
        if not isinstance(s, dict) or "root" not in s:
            continue   # load_spaces names the problem when the server starts
        # YAML reads SPACES.md's `root: ~` as null: the home folder
        root = home if s["root"] is None else Path(str(s["root"])).expanduser().resolve()
        if root == home:
            new = dest / "home"
            new.mkdir(parents=True, exist_ok=True)
        elif root == kb_source.resolve():
            new = kb_clone
        elif _git_root(root):
            new = _clone(root, dest / "spaces" / name)
        else:
            raise ConfigError(f"scratch: space {name}'s root {root} is neither the home folder nor a git work tree, so "
                              "it cannot be cloned; a scratch server never runs on a real space (SPACES.md)")
        s["root"] = str(new)
    out = dest / "spaces.yaml"
    out.write_text("# scratch copy of " + str(spaces_path) + ", written by local_voice/scratch.py; roots on clones\n"
                   + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return out


def make_scratch(dest: str | Path, *, config: str | Path | None = None, kb_source: Path = KB_SOURCE,
                 port: int | None = None, browser_port: int | None = None) -> Path:
    """Write DIR/config.yaml and DIR/spaces.yaml (module docstring) and return the config's path."""
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(config).resolve() if config else DEFAULT_CONFIG
    base = src.parent
    raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    local = src.with_name("config.local.yaml")
    if local.exists():
        raw = deep_merge(raw, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    for key in PATH_KEYS:
        v = _get(raw, key)
        if v:
            _set(raw, key, str(expand(v, base)))
    exts = _get(raw, "pi.extensions") or {}
    for k, v in exts.items():
        exts[k] = str(expand(v, base))
    _set(raw, "pi.skill_dirs", [str(expand(p, base)) for p in _get(raw, "pi.skill_dirs") or []])
    kb = _clone(kb_source, dest / "kb")
    raw["state_dir"] = str(dest / "state")
    _set(raw, "brain.turn_log_dir", str(dest / "turns"))
    _set(raw, "brain.state_dir", str(dest / "brain"))
    _set(raw, "record.dir", str(dest / "recordings"))     # relative, it would land beside DIR, not in it
    _set(raw, "server.hosts", ["127.0.0.1"])
    if port is not None:
        _set(raw, "server.port", int(port))
    if browser_port is not None:
        _set(raw, "server.browser.port", int(browser_port))
    env = dict(_get(raw, "pi.env") or {})
    env.update(KB_HOME=str(kb), KB_SEARCH_BACKEND="rg", UV_OFFLINE="1")
    _set(raw, "pi.env", env)
    spaces_src = Path(_get(raw, "agent.spaces_file") or base / "spaces.yaml")
    _set(raw, "agent.spaces_file", str(scratch_spaces(spaces_src, dest, kb, kb_source)))
    out = dest / "config.yaml"
    out.write_text(f"# scratch copy of {src}, written by local_voice/scratch.py: state, turn log, brain state, "
                   "KB_HOME and space roots under this folder; loopback only\n"
                   + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return out
