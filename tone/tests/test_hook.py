"""The orchestrator's hook: off by default, log mode never shows a hint, on mode shows one only when due, with a
cooldown and an A/B arm, a baseline that survives restarts and keeps channels apart, and a log of every turn."""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pytest
from conftest import HELD_OUT, say_pcm, tone_pcm

from local_voice_tone import HINT_INSTRUCTION, ToneConfig, ToneHook
from local_voice_tone import hint as hn

pytestmark = pytest.mark.needs_say


def warm(hook, baseline, session="base", channel="default"):
    for i, (pcm, tr) in enumerate(baseline):
        r = hook.analyze(pcm, tr, session=session, turn=i + 1, channel=channel)
        assert r.hint is None
    return r


def test_off_by_default_computes_and_writes_nothing(tmp_path):
    hook = ToneHook(state_dir=tmp_path / "tone")
    assert not hook.enabled and hook.cfg.mode == "off"
    t0 = time.perf_counter()
    assert hook.analyze(b"\x00\x01" * 32000, "hello there friend", session="s", turn=1) is None
    assert time.perf_counter() - t0 < 0.005
    assert not (tmp_path / "tone").exists()
    assert ToneHook(None).analyze(b"", "", session="s", turn=1) is None       # no state dir needed when off


def test_config_is_checked():
    with pytest.raises(ValueError, match="unknown keys"):
        ToneConfig.from_dict({"mode": "on", "treshold": 2})
    with pytest.raises(ValueError, match="mode"):
        ToneConfig.from_dict({"mode": "loud"})
    with pytest.raises(ValueError, match="f0"):
        ToneConfig.from_dict({"f0": "crepe"})
    with pytest.raises(ValueError, match="state_dir"):
        ToneHook({"mode": "log"})
    assert ToneConfig.from_dict({"z_threshold": "2", "min_samples": 5}).z_threshold == 2.0


def test_log_mode_measures_but_never_shows(tmp_path, say_baseline):
    hook = ToneHook({"mode": "log"}, state_dir=tmp_path)
    warm(hook, say_baseline)
    pcm, tr = say_pcm(HELD_OUT[0], rate=285)
    r = hook.analyze(pcm, tr, session="s", turn=1)
    assert r.hint is None and r.due and r.due.startswith(hn.PREFIX) and r.arm == "log"
    (log,) = tmp_path.glob("hints-*.jsonl")
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(rows) == len(say_baseline) + 1
    last = rows[-1]
    assert last["arm"] == "log" and last["hint"] is None and last["due"] == r.due
    assert last["features"]["rate_sps"] > 5.5 and last["deviations"][0]["name"] == "pace"


def test_on_mode_warm_up_hint_cooldown_and_arms(tmp_path, say_baseline):
    hook = ToneHook({"mode": "on", "ab_strip_fraction": 0.0, "cooldown_turns": 3}, state_dir=tmp_path)
    fast = say_pcm(HELD_OUT[0], rate=285)
    # before min_samples there is no yardstick, so no hint even for very fast speech
    early = hook.analyze(*fast, session="early", turn=1)
    assert early.hint is None and early.due is None
    r = warm(hook, say_baseline[:25])
    assert r.baseline_n == 26
    r = hook.analyze(*fast, session="s1", turn=10)
    assert r.hint and "pace faster" in r.hint and r.arm == "show"
    assert hook.analyze(*fast, session="s1", turn=11).arm == "cooldown"
    assert hook.analyze(*fast, session="s1", turn=12).hint is None
    assert hook.analyze(*fast, session="s1", turn=13).hint                      # three turns later
    assert hook.analyze(*fast, session="s2", turn=1).hint                       # cooldowns are per session
    print(f"\nsample hint: {r.hint}")
    strip = ToneHook({"mode": "on", "ab_strip_fraction": 1.0}, state_dir=tmp_path)
    s = strip.analyze(*fast, session="s3", turn=1)
    assert s.hint is None and s.arm == "strip" and s.due and "pace faster" in s.due


def test_ab_arm_is_a_stable_split():
    hook = ToneHook({"mode": "on", "ab_strip_fraction": 0.5}, state_dir="/tmp/unused")
    arms = [hook._arm("s", n) for n in range(2000)]
    assert arms == [hook._arm("s", n) for n in range(2000)]                    # same turn, same arm, every time
    assert 900 < arms.count("strip") < 1100


def test_baseline_survives_restarts_and_channels_stay_apart(tmp_path, say_baseline):
    warm(ToneHook({"mode": "on"}, state_dir=tmp_path), say_baseline, channel="iphone")
    fast = say_pcm(HELD_OUT[1], rate=285)
    again = ToneHook({"mode": "on", "ab_strip_fraction": 0.0}, state_dir=tmp_path)   # a new process, same files
    assert again.analyze(*fast, session="x", turn=1, channel="iphone").hint
    other = again.analyze(*fast, session="x", turn=5, channel="mac")             # the Mac's mic has no history yet
    assert other.hint is None and other.baseline_n == 1
    assert sorted(p.name for p in tmp_path.glob("baseline-*.json")) == ["baseline-iphone.json", "baseline-mac.json"]


def test_short_and_silent_turns_teach_nothing(tmp_path):
    hook = ToneHook({"mode": "log"}, state_dir=tmp_path)
    for i in range(5):
        hook.analyze(*say_pcm("Yes."), session="s", turn=i)
        hook.analyze(b"\x00\x00" * 16000, "", session="s", turn=100 + i)
    data = json.loads((tmp_path / "baseline-default.json").read_text())
    assert data["values"] == {} and data["fillers"] == [] and data["n"] == 10


def test_cost_per_turn_is_small(tmp_path, say_baseline):
    hook = ToneHook({"mode": "on"}, state_dir=tmp_path)
    warm(hook, say_baseline)
    ms = sorted(hook.analyze(*say_pcm(s), session="t", turn=i).ms for i, s in enumerate(HELD_OUT[:10]))
    print(f"\nanalyze per turn: median {ms[len(ms) // 2]:.1f} ms, max {ms[-1]:.1f} ms (praat, 3-5 s utterances)")
    assert ms[len(ms) // 2] < 50


def test_given_silero_segments_feed_the_pauses(tmp_path):
    hook = ToneHook({"mode": "log"}, state_dir=tmp_path)
    r = hook.analyze(tone_pcm(150.0, seconds=4.0), "one two three four five six seven eight", session="s", turn=1,
                     speech_segments=[(0.1, 1.5), (2.5, 3.9)])
    assert r.features["pause_count"] == 1 and abs(r.features["longest_pause_s"] - 1.0) < 1e-6


def test_the_instruction_for_the_model_forbids_naming_emotions():
    assert "Never name an emotion" in HINT_INSTRUCTION and "ask rather than assert" in HINT_INSTRUCTION


def test_summary_joins_the_hint_log_with_the_turn_log(tmp_path, say_baseline, capsys, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import summary
    hook = ToneHook({"mode": "on", "ab_strip_fraction": 0.5, "cooldown_turns": 0}, state_dir=tmp_path / "tone")
    warm(hook, say_baseline)
    turns = tmp_path / "turns"
    turns.mkdir()
    rows = []
    for n in range(1, 9):
        r = hook.analyze(*say_pcm(HELD_OUT[n], rate=285), session="s9", turn=n)
        rows.append({"v": 1, "type": "turn", "session": "s9", "turn": n, "t_start": "2026-10-05T09:00:00-07:00",
                     "space": "home", "user_text": "x", "reply_text": "Is now a good time?" if r.hint else "Sure.",
                     "interrupted": n % 3 == 0})
    (turns / f"{date.today().isoformat()}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    monkeypatch.setattr(sys, "argv", ["summary", "--state", str(tmp_path)])
    assert summary.main() == 0
    out = capsys.readouterr().out
    assert "turns analysed" in out and "show :" in out and "strip:" in out
