"""M1 DoD 1, the refusal half: run.sh exits 75 without loading anything while the hold gate is held or draining."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
import yaml

from fake_gate import FakeGate

HERE = Path(__file__).resolve().parent.parent


def temp_config(tmp_path: Path, gate_url: str) -> Path:
    raw = yaml.safe_load((HERE / "config.yaml").read_text())
    raw["state_dir"] = str(tmp_path / "state")
    raw["hold"]["gate"] = gate_url
    raw["agent"]["spaces_file"] = str(HERE / "spaces.yaml")
    raw["agent"]["persona_file"] = str(HERE / "persona.md")
    raw["pi"]["extensions"]["voice_gate"] = str(HERE / "pi/voice_gate.ts")
    raw["pi"]["extensions"]["voice_mode"] = str(HERE / "pi/voice_mode.ts")
    raw["brain"]["turn_log_dir"] = str(tmp_path / "turns")
    raw["brain"]["state_dir"] = str(tmp_path / "brain")
    raw["brain"]["outbox_module"] = str(HERE.parent / "brain/local_voice_brain/outbox.py")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.parametrize("phase", ["held", "draining"])
def test_run_sh_refuses_while_the_gate_is_not_open(tmp_path, phase):
    gate = FakeGate().start()
    try:
        gate.set(phase, "render refusal-test")
        t0 = time.monotonic()
        r = subprocess.run([str(HERE / "run.sh"), "--config", str(temp_config(tmp_path, gate.url))],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 75, r.stdout + r.stderr
        assert f"the hold gate is {phase}" in r.stderr and "render refusal-test" in r.stderr
        assert "loaded in" not in r.stderr                    # no adapter was loaded
        assert time.monotonic() - t0 < 60
    finally:
        gate.stop()
