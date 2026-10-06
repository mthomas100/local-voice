"""The whole heartbeat, end to end, against the stub LLM through real Pi, a git clone of a kb repo and a git clone of
Atlas: the M4 definition of done as tests.

- it refuses to think unless gpu_clear.sh passes (no request reaches the model, nothing is written);
- it digests a finished day: facts become `kb session note` lines (kb verbs only), reflections an Atlas inbox page
  (exact quotes, a commit holding only that page), and the spoken digest is queued, or nothing for NO_REPLY;
- life content never reaches the kb; a failed write is retried later without asking the model again.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import synth
from conftest import gpu_calls, make_config, set_gpu, write_agent_dir

from local_voice_brain import job, outbox, state
from local_voice_brain.__main__ import main as cli

MORNING = datetime.fromisoformat("2026-10-05T06:30:00-07:00")

pytestmark = [pytest.mark.needs_pi, pytest.mark.needs_kb, pytest.mark.needs_atlas]


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout


def user_text(req: dict) -> str:
    c = req["body"]["messages"][-1]["content"]
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c)


@pytest.fixture
def world(tmp_path, stub, agent_dir, kb_clone, atlas_clone):
    """A finished Sunday of voice sessions, the clones as the orchestrator left them, and a config pointing at all."""
    synth.capture_into_atlas(atlas_clone)
    cfg = make_config(tmp_path, agent_dir=agent_dir, kb=kb_clone, atlas=atlas_clone)
    synth.write_day(cfg.turns_dir, atlas_clone)
    return {"cfg": cfg, "stub": stub, "kb": kb_clone, "atlas": atlas_clone, "tmp": tmp_path,
            "atlas_head": git(atlas_clone, "rev-parse", "HEAD").strip()}


def script_day(stub, tech=synth.TECH_REPLY, life=synth.LIFE_REPLY):
    stub.script([{"text": json.dumps(tech), "chunk_words": 5}, {"text": json.dumps(life), "chunk_words": 5}],
                default={"text": "OK"})


def test_refuses_to_think_unless_gpu_clear_passes(world):
    cfg, stub = world["cfg"], world["stub"]
    set_gpu(cfg.gpu_clear, False)
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert (out.kind, out.code, out.day) == ("deferred", 75, synth.DAY) and "gpu_clear" in out.detail
    assert stub.requests() == []                                         # no model call
    assert git(world["kb"], "status", "--porcelain") == ""                # nothing in the kb
    assert git(world["atlas"], "rev-parse", "HEAD").strip() == world["atlas_head"]
    assert outbox.pending(cfg.state_dir, MORNING) == []
    # --force skips the idle gates but never this one
    out = job.heartbeat(cfg, now=MORNING, day=synth.DAY, force=True, log=lambda s: None)
    assert out.kind == "deferred" and stub.requests() == []
    # and a missing gpu_clear.sh is not a pass either
    cfg.gpu_clear.unlink()
    assert job.heartbeat(cfg, now=MORNING, log=lambda s: None).kind == "deferred" and stub.requests() == []
    assert (cfg.state_dir / "heartbeat.log").read_text().count("deferred") == 3


def test_idle_gates_defer_before_touching_the_gpu(world):
    cfg, stub = world["cfg"], world["stub"]
    synth.write_next_day_turn(cfg.turns_dir, "06:20")                    # a conversation ten minutes ago
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "deferred" and "quiet" in out.detail
    assert gpu_calls(cfg.gpu_clear) == 0 and stub.requests() == []


def test_a_finished_day_becomes_kb_notes_an_atlas_page_and_a_spoken_digest(world):
    cfg, stub, kb, atlas = world["cfg"], world["stub"], world["kb"], world["atlas"]
    script_day(stub)
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert (out.kind, out.day, out.code) == ("done", synth.DAY, 0), out.detail
    reqs = stub.requests()
    assert len(reqs) == 2 and all(r["body"]["model"] == "qwen38" for r in reqs)   # qwen27 lacks the compat: fallback
    tech_in, life_in = user_text(reqs[0]), user_text(reqs[1])
    # the technical pass never sees the Atlas space; the life pass sees only the person's saved words
    assert "stretched thin" not in tech_in and "river" not in tech_in and "[S1T2] 09:16 · space home" in tech_in
    assert "OWNER: okay let's keep kokoro" in tech_in
    assert "ASSISTANT (interrupted; this is what the owner heard): The VoiceChat 11B benchmark ran for sixty-four " \
           "seconds\n" in tech_in
    assert synth.MUSING_1 in life_in and synth.MUSING_2_SAVED in life_in and "note this:" not in life_in
    assert "sleep" not in life_in and "Saved." not in life_in                   # no question turns, no replies
    assert "Tags: reuse these when one fits: " in reqs[1]["body"]["messages"][0]["content"]

    # kb: three facts as `kb session note` lines on the technical pass's own kb session, nothing else touched
    digests = list((kb / ".sessions" / "digests").glob("*/*/pi-voice-brain-2026-10-04-tech-*.md"))
    assert len(digests) == 1
    text = digests[0].read_text()
    assert "voice 2026-10-04 · decision: Kokoro is the fallback voice and Ryan stays the main voice. (voice session " \
           "s-20261004-0914-a1 turn 2, 09:16, space home) — \"keep kokoro as the fallback voice\"" in text
    assert "· preference: Spoken replies" in text and "· todo: Check why Smart Turn" in text
    assert "two jobs share" not in text and "twice real time" not in text           # the dropped facts
    for word in ("stretched", "river", "quiet", "sleep"):
        assert word not in text                                                       # no life content in the kb
    assert git(kb, "status", "--porcelain").splitlines() == ["?? .sessions/digests/2026/10/"]

    # Atlas: one inbox page, exact quotes, a commit holding only that page
    page = f"inbox/voice-reflections-{synth.DAY}.md"
    assert git(atlas, "show", "--name-only", "--format=", "HEAD").split() == [page]
    assert git(atlas, "rev-parse", "HEAD~1").strip() == world["atlas_head"]
    body = (atlas / page).read_text()
    assert "> I keep saying yes to things before I've even thought about them" in body   # straight apostrophe, theirs
    assert "Burnout" not in body and "I hate my job" not in body and "area/work" in body
    note = (atlas / "journal" / f"{synth.DAY}.md").read_text()
    for line in body.splitlines():
        if line.startswith("> ") and not line.startswith("> —"):
            assert line[2:] in note
    assert git(atlas, "status", "--porcelain") == ""

    # the spoken digest, for the next conversation
    (d,) = outbox.pending(cfg.state_dir, MORNING)
    assert d.text == ("From Sunday's voice conversations: Kokoro stays the fallback voice, and you asked for replies of "
                      "two sentences at most. I left a reflection from your journal talk in your Atlas inbox.")
    assert d.data["atlas_page"] == page and d.data["kb_digest"].startswith(".sessions/digests/2026/10/pi-voice-brain")
    assert {"session": "s-20261004-0914-a1", "turn": 2} in d.data["cites"]

    # the written digest cites the sessions
    report = (cfg.state_dir / "days" / synth.DAY / "report.md").read_text()
    for sid in ("s-20261004-0914-a1", "s-20261004-1330-b2", "s-20261004-2102-c3"):
        assert sid in report
    assert "quote not found in S1T1" in report and "S9T9" in report and "(Burnout): a clinical word" in report
    assert state.Ledger.load(cfg.state_dir).status(synth.DAY) == "done"

    # the next heartbeat has nothing to do and touches nothing
    n = len(stub.requests())
    assert job.heartbeat(cfg, now=MORNING, log=lambda s: None).kind == "idle" and len(stub.requests()) == n


def test_a_quiet_day_ends_in_no_reply(world):
    cfg, stub = world["cfg"], world["stub"]
    script_day(stub, tech={"facts": [], "spoken": "NO_REPLY"}, life={"reflections": []})
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "done" and "NO_REPLY" in out.detail
    assert outbox.pending(cfg.state_dir, MORNING) == []
    assert git(world["atlas"], "rev-parse", "HEAD").strip() == world["atlas_head"]
    (digest,) = (world["kb"] / ".sessions" / "digests").glob("*/*/pi-voice-brain-*.md")
    assert "voice 2026-10-04" not in digest.read_text()                       # registered, but no notes
    assert "NO_REPLY" in (cfg.state_dir / "days" / synth.DAY / "report.md").read_text()


def test_a_reply_that_is_not_json_is_asked_once_more_then_the_day_fails_and_gives_up(world):
    cfg, stub = world["cfg"], world["stub"]
    stub.script([{"text": "Sure, here are the facts: none."}] * 20)
    for attempt in range(1, cfg.max_attempts + 1):
        out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
        assert out.kind == "failed" and out.code == 1
        entry = state.Ledger.load(cfg.state_dir).data["days"][synth.DAY]
        assert entry["attempts"] == attempt
    assert entry["status"] == "failed" and "gave up" in out.detail
    assert len(stub.requests()) == 2 * cfg.max_attempts                       # one retry per attempt
    assert job.heartbeat(cfg, now=MORNING, log=lambda s: None).kind == "idle"  # a failed day is left alone
    assert cli(["--config", str(_write_cfg(cfg)), "retry", synth.DAY]) == 0
    assert state.Ledger.load(cfg.state_dir).status(synth.DAY) == "retry"


def _write_cfg(cfg) -> Path:
    """The test config as a TOML file, for the CLI."""
    def toml_value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ", ".join(toml_value(x) for x in v) + "]"
        return json.dumps(str(v))
    lines = []
    for section, values in cfg.raw.items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {toml_value(v)}" for k, v in values.items()]
    p = cfg.state_dir.parent / "test-config.toml"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_a_failed_write_is_retried_without_asking_the_model_again(world):
    cfg, stub, atlas = world["cfg"], world["stub"], world["atlas"]
    script_day(stub)
    (atlas / ".git" / "index.lock").write_text("")                           # Obsidian's git is mid-commit
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "failed" and "AtlasBusy" in out.detail
    n = len(stub.requests())
    sinks = state.Ledger.load(cfg.state_dir).data["days"][synth.DAY]["sinks"]
    assert sinks["kb"]["status"] == "done" and "atlas" not in sinks
    (atlas / ".git" / "index.lock").unlink()
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "done" and len(stub.requests()) == n                  # the saved proposal was reused
    (digest,) = (world["kb"] / ".sessions" / "digests").glob("*/*/pi-voice-brain-*.md")
    assert digest.read_text().count("voice 2026-10-04 ·") == 3               # no duplicated kb notes
    assert (atlas / f"inbox/voice-reflections-{synth.DAY}.md").exists()
    assert len(outbox.pending(cfg.state_dir, MORNING)) == 1


def test_qwen27_with_its_compat_reflects_and_then_reloads_the_live_brain(world, tmp_path):
    cfg, stub = world["cfg"], world["stub"]
    write_agent_dir(cfg.agent_dir, stub.base_url, qwen27_compat={"thinkingFormat": "qwen-chat-template"})
    script_day(stub)
    calls_before = gpu_calls(cfg.gpu_clear)
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "done", out.detail
    assert [r["body"]["model"] for r in stub.requests()] == ["qwen27-262k", "qwen27-262k", "qwen38"]
    assert gpu_calls(cfg.gpu_clear) - calls_before == 3        # before each of the three calls
    assert "actor: pi/qwen27-262k" in next((world["kb"] / ".sessions" / "digests").glob("*/*/pi-voice-brain-*.md")
                                           ).read_text()


def test_the_gpu_must_clear_again_between_passes(world):
    cfg, stub = world["cfg"], world["stub"]
    script_day(stub)
    calls = {"n": 0}
    real = job.gates.gpu_clear

    def flaky(script, timeout_s=30):
        calls["n"] += 1
        if calls["n"] >= 2:                                    # a render starts after the technical pass
            set_gpu(cfg.gpu_clear, False)
        return real(script, timeout_s)

    job.gates.gpu_clear = flaky
    try:
        out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    finally:
        job.gates.gpu_clear = real
    assert out.kind == "failed" and "did not clear" in out.detail
    assert len(stub.requests()) == 1                           # the life pass never asked
    # half an hour later the render is over: only the missing life pass is asked, the technical one is reused
    set_gpu(cfg.gpu_clear, True)
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "done", out.detail
    reqs = stub.requests()
    assert len(reqs) == 2 and synth.MUSING_1 in user_text(reqs[1])
    assert len(list((world["kb"] / ".sessions" / "digests").glob("*/*/pi-voice-brain-*.md"))) == 1


def test_kb_notes_resume_after_the_last_one_written(world):
    cfg, stub, kb = world["cfg"], world["stub"], world["kb"]
    script_day(stub)
    real = kb / "bin" / "kb"
    keep = kb / "bin" / "kb.real"
    real.rename(keep)
    count = kb / "bin" / "notes.count"
    # a kb whose second `session note` fails once (a disk hiccup, a lock): everything else is the real binary
    real.write_text(f"""#!/bin/zsh
if [ "$1" = session ] && [ "$2" = note ]; then
  n=$(( $(cat {count} 2>/dev/null || echo 0) + 1 )); echo $n > {count}
  [ $n -eq 2 ] && {{ echo "kb: simulated failure" >&2; exit 1; }}
fi
exec {keep} "$@"
""")
    real.chmod(0o755)
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "failed" and "simulated failure" in out.detail
    assert state.Ledger.load(cfg.state_dir).data["days"][synth.DAY]["sinks"]["kb"] == {"status": "partial", "notes": 1}
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "done", out.detail
    (digest,) = (kb / ".sessions" / "digests").glob("*/*/pi-voice-brain-*.md")
    text = digest.read_text()
    assert text.count("voice 2026-10-04 ·") == 3 and text.count("· decision: Kokoro") == 1
    assert len(stub.requests()) == 2                           # the model was asked once per pass, never again


def test_a_page_left_uncommitted_by_a_crash_is_committed_as_it_is(world):
    cfg, stub, atlas = world["cfg"], world["stub"], world["atlas"]
    script_day(stub)
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "done"
    page = f"inbox/voice-reflections-{synth.DAY}.md"
    # rewind to "written but not committed": the page is on disk, the commit is gone, the ledger says not done
    git(atlas, "reset", "-q", "HEAD~1")
    assert git(atlas, "status", "--porcelain").strip() == f"?? {page}"
    led = state.Ledger.load(cfg.state_dir)
    led.data["days"][synth.DAY]["sinks"].pop("atlas")
    led.data["days"][synth.DAY]["status"] = "retry"
    led.save()
    out = job.heartbeat(cfg, now=MORNING, log=lambda s: None)
    assert out.kind == "done"
    assert git(atlas, "show", "--name-only", "--format=", "HEAD").split() == [page]
    assert git(atlas, "status", "--porcelain") == "" and len(stub.requests()) == 2


def test_dry_run_writes_nothing(world):
    cfg, stub, kb, atlas = world["cfg"], world["stub"], world["kb"], world["atlas"]
    script_day(stub)
    out = job.heartbeat(cfg, now=MORNING, day=synth.DAY, dry_run=True, log=lambda s: None)
    assert out.kind == "done", out.detail
    assert git(kb, "status", "--porcelain") == ""             # not even a kb session: kb.ts was not loaded
    assert git(atlas, "rev-parse", "HEAD").strip() == world["atlas_head"] and git(atlas, "status", "--porcelain") == ""
    assert outbox.pending(cfg.state_dir, MORNING) == []
    assert not (cfg.state_dir / "ledger.json").exists() and not (cfg.state_dir / "heartbeat.log").exists()
    rep = (cfg.state_dir / "days" / synth.DAY / "report.dry-run.md").read_text()
    assert "dry run: 3 kb notes not written" in rep and "(dry run)" in rep


def test_old_days_are_skipped_and_today_waits(world):
    cfg, stub = world["cfg"], world["stub"]
    later = datetime.fromisoformat("2026-10-20T06:30:00-07:00")
    out = job.heartbeat(cfg, now=later, log=lambda s: None)
    assert out.kind == "idle" and state.Ledger.load(cfg.state_dir).status(synth.DAY) == "skipped"
    same_day = datetime.fromisoformat("2026-10-04T23:59:00-07:00")
    (cfg.state_dir / "ledger.json").unlink()
    assert job.heartbeat(cfg, now=same_day, log=lambda s: None).kind == "idle"      # the day is not over yet
    assert stub.requests() == []


def test_cli_status_and_heartbeat_exit_codes(world, capsys):
    cfg, stub = world["cfg"], world["stub"]
    path = _write_cfg(cfg)
    assert cli(["--config", str(path), "--now", MORNING.isoformat(), "status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["model"] == "local/qwen38" and any(g.startswith("gpu_clear: ok") for g in st["gates"])
    set_gpu(cfg.gpu_clear, False)
    assert cli(["--config", str(path), "--now", MORNING.isoformat(), "heartbeat"]) == 75
    bad = path.with_name("bad.toml")
    bad.write_text(path.read_text().replace('by = "voice-brain"', 'by = "me"'))
    assert cli(["--config", str(bad), "status"]) == 78
    assert stub.requests() == []
