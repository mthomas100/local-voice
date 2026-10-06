"""The scratch copy (local_voice/scratch.py): everything a server writes lands under one folder, on clones."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from local_voice.config import DEFAULT_CONFIG, ConfigError, load_config
from local_voice.scratch import make_scratch
from local_voice.spaces import load_spaces

HERE = Path(__file__).resolve().parent.parent


def git_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", *cmd], cwd=path, check=True, capture_output=True)
    return path


@pytest.fixture
def sources(tmp_path: Path):
    kb = git_repo(tmp_path / "src-kb", {"AGENTS.md": "kb\n", "wiki/a.md": "page\n"})
    atlas = git_repo(tmp_path / "src-atlas", {"atlas.py": "print('atlas')\n", "journal/2026-10-05.md": "words\n"})
    spaces = {"version": 1,
              "defaults": {"model": "local/qwen38", "thinking": "off", "mode": "conversation", "tier": "ask"},
              "spaces": {"home": {"root": None, "description": "your Mac", "triggers": ["home"], "skills": [],
                                  "tools": ["read", "ls"], "act_tools": ["bash", "write"], "tier": "ask"},
                         "atlas": {"root": str(atlas), "description": "your atlas journal", "triggers": ["atlas"],
                                   "skills": "auto", "tools": ["read", "bash"], "tier": "trusted"},
                         "kb": {"root": str(kb), "description": "the knowledge base", "triggers": ["the kb"],
                                "skills": [], "tools": ["read"], "tier": "readonly"}}}
    sp = tmp_path / "src-spaces.yaml"
    sp.write_text(yaml.safe_dump(spaces))
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["agent"]["spaces_file"] = str(sp)
    raw["agent"]["persona_file"] = str(HERE / "persona.md")
    raw["pi"]["extensions"] = {k: str((HERE / v).resolve()) if not str(v).startswith("~") else v
                               for k, v in raw["pi"]["extensions"].items()}
    cfg = tmp_path / "src-config.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    return {"kb": kb, "atlas": atlas, "spaces": sp, "config": cfg}


def test_everything_written_lands_under_the_scratch_folder_on_clones(tmp_path: Path, sources):
    dest = tmp_path / "scratch"
    out = make_scratch(dest, config=sources["config"], kb_source=sources["kb"], port=18770, browser_port=17860)
    cfg = load_config(out)
    assert out == dest.resolve() / "config.yaml"
    d = dest.resolve()
    assert cfg.state_dir == d / "state"
    assert cfg.turn_log_dir == d / "turns" and cfg.brain_state_dir == d / "brain"
    assert cfg.hosts == ["127.0.0.1"], "a server under test listens on loopback only"
    assert (cfg.port, cfg.browser_port) == (18770, 17860)
    assert cfg.pi_env["KB_HOME"] == str(d / "kb") and (d / "kb" / ".git").exists()
    assert (d / "kb" / "wiki" / "a.md").read_text() == "page\n"
    assert cfg.pi_env["KB_SEARCH_BACKEND"] == "rg" and cfg.pi_env["UV_OFFLINE"] == "1"
    for p in [cfg.spaces_file, cfg.persona_file, *cfg.pi_extensions.values(), cfg.brain_outbox_module]:
        assert p.is_absolute()
    spaces = load_spaces(cfg)
    assert spaces["home"].root == d / "home" and spaces["home"].root.is_dir()
    assert spaces["kb"].root == d / "kb", "the kb space shares the KB_HOME clone"
    assert spaces["atlas"].root == d / "spaces" / "atlas"
    assert (d / "spaces" / "atlas" / "journal" / "2026-10-05.md").read_text() == "words\n"
    # the sources were only read
    assert subprocess.run(["git", "status", "--porcelain"], cwd=sources["atlas"], capture_output=True,
                          text=True).stdout == ""


def test_a_second_run_keeps_the_clones_and_rewrites_the_config(tmp_path: Path, sources):
    dest = tmp_path / "scratch"
    make_scratch(dest, config=sources["config"], kb_source=sources["kb"])
    marker = dest / "spaces" / "atlas" / "journal" / "2026-10-05.md"
    marker.write_text("captured in the last run\n")
    out = make_scratch(dest, config=sources["config"], kb_source=sources["kb"], port=18771)
    assert marker.read_text() == "captured in the last run\n"
    assert load_config(out).port == 18771


def test_a_space_that_cannot_be_cloned_is_refused(tmp_path: Path, sources):
    plain = tmp_path / "plain"
    plain.mkdir()
    raw = yaml.safe_load(sources["spaces"].read_text())
    raw["spaces"]["notes"] = {"root": str(plain), "description": "notes", "tools": ["read"]}
    sources["spaces"].write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="neither the home folder nor a git work tree"):
        make_scratch(tmp_path / "scratch", config=sources["config"], kb_source=sources["kb"])


def test_the_server_cli_offers_the_scratch_options():
    out = subprocess.run([sys.executable, "-m", "local_voice.server", "--help"], cwd=HERE, capture_output=True,
                         text=True).stdout
    for opt in ("--scratch", "--port", "--browser-port"):
        assert opt in out
