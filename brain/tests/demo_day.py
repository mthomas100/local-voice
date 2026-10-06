"""Run one whole heartbeat through the CLI on a synthetic Sunday and print everything it produced.

    uv run --project brain python brain/tests/demo_day.py [--keep]

The world is temporary: the stub LLM (scripted with the replies in synth.py), real pi in JSON mode, a git clone of
~/kb, a git clone of Atlas, and a fake gpu_clear.sh that says CLEAR. Nothing touches the real kb, Atlas, ~/.pi/agent
or :8090. --keep leaves the temporary directory for inspection.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conftest  # noqa: E402
import synth  # noqa: E402
from test_job import _write_cfg  # noqa: E402


def main() -> int:
    keep = "--keep" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="brain-demo-"))
    stub = conftest.StubLLM(0, tmp / "stub").start()
    try:
        agent = conftest.write_agent_dir(tmp / "pi-agent", stub.base_url)
        kb = conftest.git_clone(conftest.REAL_KB, tmp / "kb")
        atlas = conftest.git_clone(conftest.REAL_ATLAS, tmp / "atlas")
        synth.capture_into_atlas(atlas)
        cfg = conftest.make_config(tmp, agent_dir=agent, kb=kb, atlas=atlas)
        synth.write_day(cfg.turns_dir, atlas)
        stub.script([{"text": json.dumps(synth.TECH_REPLY)}, {"text": json.dumps(synth.LIFE_REPLY)}],
                    default={"text": "OK"})
        cmd = [sys.executable, "-m", "local_voice_brain", "--config", str(_write_cfg(cfg)),
               "--now", "2026-10-05T06:30:00-07:00", "heartbeat"]
        r = subprocess.run(cmd, cwd=conftest.BRAIN, capture_output=True, text=True)
        print(f"$ run.sh heartbeat   (exit {r.returncode})\n{r.stderr}{r.stdout}")
        day = cfg.state_dir / "days" / synth.DAY
        sections = [("the written digest (state/brain/days/2026-10-04/report.md)", day / "report.md")]
        sections += [(f"the kb digest ({p.relative_to(kb)})", p)
                     for p in (kb / ".sessions" / "digests").glob("*/*/pi-voice-brain-*.md")]
        sections += [(f"the Atlas inbox page (inbox/{p.name})", p) for p in (atlas / "inbox").glob("voice-*.md")]
        sections += [("the spoken digest in the outbox", p) for p in (cfg.state_dir / "outbox").glob("*.json")]
        for title, p in sections:
            print(f"\n=========== {title}\n{p.read_text()}")
        log = subprocess.run(["git", "-C", str(atlas), "log", "--stat", "-1", "--format=%h %s"], capture_output=True,
                             text=True).stdout
        print(f"=========== the Atlas commit\n{log}")
        print(f"=========== model requests: {[q['body']['model'] for q in stub.requests()]}")
        return r.returncode
    finally:
        stub.stop()
        if keep:
            print(f"\nkept {tmp}")
        else:
            subprocess.run(["rm", "-rf", str(tmp)])


if __name__ == "__main__":
    sys.exit(main())
