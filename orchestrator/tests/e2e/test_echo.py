"""The echo guard on real models (Nemotron, Qwen3-TTS, Silero, Smart Turn): tools/echo_bench.py's planted-echo
scenarios with the guard on (`echo.guard: true`, `echo.hold_for_words: all`, so protocol v1 exercises the hold too),
asserting what an early live test needs (2026-10-05: 5 of 19 voice turns were the agent's own words, replays of
one reply's last ~10 s that came back 6-59 s after it, two of them cutting a reply; local_voice/echo_guard.py):

- live echo at -20 dB and below (-30 and -20, 150 ms, dry and through the room, and 300 ms at -20): no echo turn and
  no false barge-in, and the reply still comes;
- a true barge-in ("Stop. What is the capital of Italy?" 2.0 s into the reply, over -20 dB room echo) still interrupts;
- the delayed replay (the reply's last 8 s at full level, 6 s after it ended) never cuts a reply;
- a sentence of the reply read aloud in another voice while the agent is idle goes through as a turn and is answered.

The LLM is the bench's routed stub (deterministic replies, no LLM traffic): `pi.models_source` points at a copy of the
configured models.json aimed at it. Every root is on a clone (e2e_fixtures.orch_spaces's scratch_spaces). The -10 dB
scenarios are measured by the bench, not asserted here: echo that loud is what the guard has to win against, and the
bench's before/after table is where it shows. Results: <run dir>/echo/results.jsonl and report.md.

Run (GPU, about 7 min): ../measure/bench/gpu_clear.sh && .venv/bin/python -m pytest -m e2e tests/e2e/test_echo.py -s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from e2e_fixtures import ORCH, _serve, gpu_still_ours, kb_clone, run_dir  # noqa: F401 - fixtures

sys.path.insert(0, str(ORCH / "tools"))
import echo_bench as bench  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio(loop_scope="module")]

LIVE = ["live-30-dry", "live-30-room", "live-20-dry", "live-20-room", "live-20-room-300ms"]
SCENARIOS = [s for s in bench.SCENARIOS if s.name in {*LIVE, "bargein-20-room", "replay", "read-aloud"}]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def echo_orch(run_dir, kb_clone):
    from local_voice.config import load_config
    from local_voice.scratch import scratch_spaces

    d = run_dir / "echo"
    stub = bench.start_stub(0, d / "stub")
    models = bench.stub_models(load_config().pi_models_source, d / "stub-models.json", stub.base_url)
    spaces = scratch_spaces(ORCH / "spaces.yaml", run_dir / "m3", kb_clone)
    extra = {"state_dir": str(run_dir / "state-test_echo"), "echo": dict(bench.CONFIGS["on"]),
             "agent": {"spaces_file": str(spaces)}, "pi": {"models_source": str(models)}}
    try:
        async for o in _serve(run_dir, kb_clone, extra):
            o.echo_dir = d
            yield o
    finally:
        stub.stop()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def echo_runs(echo_orch) -> dict[str, dict]:
    """Every scenario once, in order, each on its own connection; the rows the tests below read."""
    d: Path = echo_orch.echo_dir
    assert echo_orch.cfg.echo.get("guard") is True, f"the server did not take the guard: {echo_orch.cfg.echo}"
    (d / "run.json").write_text(json.dumps({"stamp": d.parent.name, "llm": "stub (e2e)",
                                           "server": "in-process, tests/e2e/test_echo.py",
                                           "server_echo": {"on": echo_orch.cfg.echo}}, indent=1))
    clips = bench.render_clips()
    rows: dict[str, dict] = {}
    for sc in SCENARIOS:
        why = gpu_still_ours()
        if why:
            pytest.skip(f"stopping: {why}")
        run = await bench.run_scenario(echo_orch.e2e_url, sc, clips, config="on", device=f"e2e-echo-{sc.name}")
        # the turn log is e2e_fixtures' (<run dir>/turns), the log this module's state dir's (setup_logging)
        row = bench.result_row(run, echo_orch.cfg.turn_log_dir, echo_orch.state_dir / "logs")
        with open(d / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows[sc.name] = row
    bench.write_report(d)
    return rows


def brief(m: dict) -> dict:
    """What a failure message needs to show."""
    return {"turns": [(u["kind"], u["text"]) for u in m["user_turns"]], "interrupts": m["interrupts"],
            "main_reply": {k: v for k, v in (m["main_reply"] or {}).items() if k != "text"}, "notes": m["notes"],
            "log": m["log"]}


async def test_no_echo_turn_and_no_false_barge_in_at_minus_20_db_and_below(echo_runs):
    for name in LIVE:
        m = echo_runs[name]["metrics"]
        assert m["main_reply"] is not None, (name, brief(m))
        assert m["n_echo_turns"] == 0 and m["false_barge_ins"] == 0, (name, brief(m))


async def test_the_true_barge_in_still_interrupts(echo_runs):
    m = echo_runs["bargein-20-room"]["metrics"]
    b = m["barge-in"]
    assert b["interrupt_ms"] is not None, brief(m)
    # the hold waits for words (echo.max_hold_s 1.5 s), so this may come later than DoD 4's 300 ms without echo;
    # the reply's audio itself stops at the first speech frame (b["audio_stop_ms"], recorded)
    assert b["interrupt_ms"] <= 3000, (b, brief(m))
    assert m["false_barge_ins"] == 0, brief(m)


async def test_the_delayed_replay_never_cuts_a_reply(echo_runs):
    m = echo_runs["replay"]["metrics"]
    r = m["replay"]
    assert r["replies_cut"] == 0 and r["interrupts"] == 0, (r, brief(m))
    assert not any(u["interrupted"] for u in m["user_turns"] if u["during_replay"]), brief(m)


async def test_the_read_aloud_sentence_goes_through_as_a_turn(echo_runs):
    m = echo_runs["read-aloud"]["metrics"]
    ra = m["read-aloud"]
    assert ra["turn"] and ra["answered"], (ra, brief(m))
