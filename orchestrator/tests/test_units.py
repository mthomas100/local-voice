"""Pure parts: spaces.yaml checks and the Pi child derived from it, the derived agent dir, spoken yes/no and
permission wording, the played_ms timeline, the protocol v1 serializer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from local_voice.agent_dir import derive_agent_dir
from local_voice.config import load_config
from local_voice.protocol_v1 import ClientMessageFrame, ProtocolV1Serializer, ReplyTimeline
from local_voice.spaces import SpacesError, bash_globs, load_spaces, pi_space_config
from local_voice.speech_text import spoken_confirm, yes_no
from pipecat.frames.frames import (InputAudioRawFrame, InterruptionFrame, InterruptionWorkerFrame, LLMMessagesAppendFrame,
                                   OutputAudioRawFrame, OutputTransportMessageFrame, OutputTransportMessageUrgentFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)


def spaces_file(tmp_path: Path, spaces: dict, defaults: dict | None = None) -> Path:
    p = tmp_path / "spaces.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "defaults": defaults or {"model": "local/qwen38", "thinking": "off",
                                                                        "mode": "conversation", "tier": "ask"},
                                 "spaces": spaces}))
    return p


def cfg_with(tmp_path: Path, spaces: dict, **kw):
    return load_config(overrides={"agent": {"spaces_file": str(spaces_file(tmp_path, spaces, **kw)),
                                            "default_space": next(iter(spaces))}})


def test_the_repo_spaces_file_loads_and_derives_the_home_child():
    cfg = load_config()
    sp = load_spaces(cfg)
    home = sp["home"]
    assert home.tier == "ask" and home.tools == ["read", "grep", "find", "ls", "kb"] and home.root == Path.home()
    sc = pi_space_config(cfg, home, agent_dir=Path("/tmp/agent"), state_dir=Path("/tmp/state"))
    argv = sc.argv()
    assert argv[argv.index("--tools") + 1] == "read,grep,find,ls,kb,bash,write,edit"   # every tool registered at spawn
    assert sc.env["VOICE_TOOLS"] == "read,grep,find,ls,kb"                             # narrowed by voice_mode.ts
    assert "--no-extensions" in argv and "--no-skills" in argv and argv[argv.index("--thinking") + 1] == "off"
    exts = [argv[i + 1] for i, a in enumerate(argv) if a == "--extension"]
    assert [Path(e).name for e in exts] == ["voice_gate.ts", "voice_mode.ts", "hold.ts", "today.ts", "kb.ts"]
    assert not any("film-rig" in e for e in exts)
    skills = [Path(argv[i + 1]).name for i, a in enumerate(argv) if a == "--skill"]
    assert skills == ["wiki-query", "web-research", "chrome-read", "reddit", "x"]
    prompt = argv[argv.index("--system-prompt") + 1]
    assert prompt.startswith("You are the user's voice assistant") and "--append-system-prompt" not in argv
    assert sc.env["PI_CODING_AGENT_DIR"] == "/tmp/agent" and sc.env["HOLD_GATE"] == cfg.gate
    assert argv[argv.index("--session-dir") + 1] == "/tmp/state/sessions/home"


@pytest.mark.parametrize("space,msg", [
    ({"root": "/nonexistent/x", "description": "x", "tools": ["read"]}, "is not a directory"),
    ({"root": "~", "tools": ["read"]}, "description is required"),
    ({"root": "~", "description": "x", "tools": []}, "at least one tool"),
    ({"root": "~", "description": "x", "tools": ["read"], "tier": "admin"}, "tier must be one of"),
    ({"root": "~", "description": "x", "tools": ["read"], "skills": ["no-such-skill-xyz"]}, "no skill 'no-such-skill-xyz'"),
    ({"root": "~", "description": "x", "tools": ["read"], "extensions": ["~/x/film-rig.ts"]}, "never film-rig.ts"),
    ({"root": "~", "description": "x", "tools": ["read"], "bash_allow": ["python3 atlas.py *"]}, "argv prefixes"),
    ({"root": "~", "description": "x", "tools": ["read"], "colour": "red"}, "unknown key"),
])
def test_a_bad_spaces_file_fails_with_a_clear_message(tmp_path, space, msg):
    with pytest.raises(SpacesError, match=msg):
        cfg = cfg_with(tmp_path, {"home": space})
        load_spaces(cfg)


def test_duplicate_triggers_are_refused(tmp_path):
    cfg = cfg_with(tmp_path, {"home": {"root": "~", "description": "a", "tools": ["read"], "triggers": ["journal"]},
                              "atlas": {"root": "~", "description": "b", "tools": ["read"], "triggers": ["Journal"]}})
    with pytest.raises(SpacesError, match="is also a trigger of"):
        load_spaces(cfg)


def test_bash_allow_prefixes_become_exact_or_prefix_globs():
    assert bash_globs([["python3", "atlas.py", "capture"], ["git", "status"]]) == [
        "python3 atlas.py capture", "python3 atlas.py capture *", "git status", "git status *"]


def test_derived_agent_dir_reads_the_users_models_and_caps_tokens(tmp_path):
    cfg = load_config()
    sp = load_spaces(cfg)
    d = derive_agent_dir(cfg, sp, tmp_path / "agent", base_url="http://127.0.0.1:1/v1")
    m = json.loads((d / "models.json").read_text())["providers"]["local"]
    assert m["baseUrl"] == "http://127.0.0.1:1/v1"
    assert {x["id"] for x in m["models"]} == {"qwen38", "qwen27-262k"}
    assert all(x["maxTokens"] <= cfg.pi_max_tokens for x in m["models"])
    s = json.loads((d / "settings.json").read_text())
    assert s == {"defaultProvider": "local", "defaultModel": "qwen38", "packages": []}


@pytest.mark.parametrize("text,verdict", [
    ("Yes.", True), ("yeah go ahead", True), ("Sure, do it.", True), ("OK", True), ("yes please", True),
    ("No.", False), ("nope", False), ("don't do that", False), ("cancel", False), ("never mind", False),
    ("what does it do?", None), ("", None), ("maybe later", None),
])
def test_spoken_yes_no(text, verdict):
    assert yes_no(text) is verdict


def test_permission_question_is_said_without_shell_noise():
    assert spoken_confirm("May I run a command?", "git push") == "May I run a command? git push. Yes or no?"
    said = spoken_confirm("May I run a command?", "python3 atlas.py capture --via voice <<'ATLAS_END'")
    assert said == "May I run a command? It starts with python3 atlas.py. Yes or no?" and "<<" not in said
    assert spoken_confirm("May I write a file?", "") == "May I write a file? Yes or no?"
    assert spoken_confirm("May I write a file?", "write notes/Hello.txt") == "May I write Hello.txt? Yes or no?"
    assert spoken_confirm("May I edit a file?", "edit /tmp/My Notes.md") == "May I edit My Notes.md? Yes or no?"


def test_timeline_maps_played_ms_to_words():
    t = ReplyTimeline()
    t.open("r1")
    t.audio(24000 * 2, 24000)          # 1 s of audio
    t.sentence("one two three four")
    t.audio(24000 * 2, 24000)          # 1 s more
    t.sentence("five six seven eight")
    assert t.heard("r1", 1000) == "one two three four"
    assert t.heard("r1", 1500) == "one two three four five six"
    assert t.heard("r1", 400) == "one"
    assert t.heard("r1", 5000) == "one two three four five six seven eight"
    assert t.heard("nope", 1000) is None
    t.open("r2")
    assert t.heard("r1", 1000) == "one two three four"   # the last few replies are kept


def test_timeline_counts_the_sentence_still_streaming():
    """A barge-in 2.4 s into a long first sentence used to map to no words (e2e 2026-10-05)."""
    t = ReplyTimeline()
    t.open("r1")
    long = "Once upon a time there was a lighthouse keeper who lived alone with his cat on a rocky island far out at sea."
    t.upcoming(long)
    t.audio(24000 * 2 * 3, 24000)      # 3 s queued, the sentence not finished
    heard = t.heard("r1", 2400).split()
    assert heard == long.split()[: len(heard)] and 4 <= len(heard) <= 8     # about 2.6 words a second
    t.audio(24000 * 2 * 3, 24000)
    t.sentence(long)                    # now finished (6 s in all), the next one starts streaming
    t.upcoming("The end.")
    t.audio(24000 * 2 * 1, 24000)
    words = long.split()
    assert t.heard("r1", 1000).split() == words[: int(len(words) * 1000 / 6000)]   # its real duration is known now
    assert t.heard("r1", 6500).split() == words + ["The"]                         # 0.5 s into "The end." at its rate
    t.open("r2")
    assert t.heard("r1", 2400).split() == words[: int(len(words) * 2400 / 6000)]  # kept with the archived reply


def test_timeline_acknowledgement_is_not_counted_as_heard_words():
    t = ReplyTimeline()
    t.open("r1")
    t.upcoming("Let me look that up.")
    t.audio(24000 * 2, 24000)
    t.skip()                            # the acknowledgement, kept out of Pi's context
    t.upcoming("The answer is four.")
    t.audio(24000 * 2, 24000)
    assert t.heard("r1", 1500).split() == ["The", "answer"][: len(t.heard("r1", 1500).split())]
    assert "Let" not in t.heard("r1", 1500)


async def test_serializer_both_directions():
    s = ProtocolV1Serializer()
    f = await s.deserialize(bytes(640))
    assert isinstance(f, InputAudioRawFrame) and f.sample_rate == 16000 and f.num_frames == 320
    assert isinstance(await s.deserialize('{"t":"start"}'), VADUserStartedSpeakingFrame)
    assert isinstance(await s.deserialize('{"t":"stop"}'), VADUserStoppedSpeakingFrame)
    assert isinstance(await s.deserialize('{"t":"text","text":"hi"}'), LLMMessagesAppendFrame)
    cm = await s.deserialize('{"t":"played_ms","reply_id":"r1","ms":2140}')
    assert isinstance(cm, ClientMessageFrame) and cm.message["ms"] == 2140
    assert await s.deserialize("not json") is None
    assert await s.serialize(InterruptionFrame()) is None                        # nothing playing: not forwarded
    await s.serialize(OutputTransportMessageFrame(message={"t": "audio_start", "reply_id": "r2", "rate": 24000}))
    assert await s.serialize(OutputAudioRawFrame(audio=bytes(1920), sample_rate=24000, num_channels=1)) == bytes(1920)
    assert json.loads(await s.serialize(InterruptionFrame())) == {"t": "interrupt", "reply_id": "r2"}
    await s.serialize(OutputTransportMessageFrame(message={"t": "audio_start", "reply_id": "r3", "rate": 24000}))
    assert isinstance(await s.deserialize('{"t":"interrupt","reply_id":"r3"}'), InterruptionWorkerFrame)
    assert await s.serialize(InterruptionFrame()) is None                        # the client asked for it itself
    rtvi = OutputTransportMessageUrgentFrame(message={"label": "rtvi-ai", "type": "bot-ready"})
    assert await s.serialize(rtvi) is None


# ------------------------------------------------------------------------------------- the reply pause (bargein.py)

async def test_reply_pause_only_while_a_reply_plays_and_resumes_when_unconfirmed():
    import asyncio
    import time

    from local_voice.bargein import PauseSettings, ReplyPause

    p = ReplyPause(settings=PauseSettings(cue_confidence=0.5, cue_frames=2, min_volume=0.5, resume_after_s=0.1,
                                          max_pause_s=0.4))
    p.frame(0.9, 0.9)
    p.frame(0.9, 0.9)
    assert p.paused_at is None                     # nothing playing: nothing to pause
    p.audio_sent()
    p.frame(0.9, 0.2)                              # speech-like but under the VAD's loudness floor
    p.frame(0.9, 0.2)
    p.frame(0.4, 0.9)
    assert p.paused_at is None
    p.frame(0.9, 0.9)
    assert p.paused_at is None                     # one frame of two
    p.frame(0.9, 0.9)
    assert p.paused_at is not None and not p._open.is_set()
    t0 = time.monotonic()
    while p.paused_at is not None and time.monotonic() - t0 < 1:
        p.frame(0.9, 0.9)                          # still sounding like speech: held, up to max_pause_s
        await asyncio.sleep(0.03)
    held = p.events[-1]
    assert held.kind == "resumed" and 380 <= held.held_ms <= 520, held
    assert p._open.is_set()


async def test_reply_pause_an_interruption_confirms_and_bumps_the_generation():
    from local_voice.bargein import PauseSettings, ReplyPause

    p = ReplyPause(settings=PauseSettings(cue_frames=1, resume_after_s=5.0, max_pause_s=5.0))
    p.audio_sent()
    p.frame(0.9, 0.9)
    gen = p.generation
    waiter = __import__("asyncio").ensure_future(p.wait_open())
    p.interrupting()
    await __import__("asyncio").sleep(0.01)
    assert not waiter.done()                       # shut while Pipecat empties the transport's queue
    p.interrupted()
    await waiter                                   # the transport's held write wakes up ...
    assert p.generation == gen + 1                 # ... and drops its chunk: it belongs to the reply that was cut
    assert [e.kind for e in p.events] == ["paused", "confirmed"]
    p.interrupting()                               # every user turn start interrupts; nothing paused, nothing logged
    p.interrupted()
    assert [e.kind for e in p.events] == ["paused", "confirmed"]


async def test_reply_pause_never_after_an_interruption_and_stops_pausing_after_two_false_alarms():
    import asyncio

    from local_voice.bargein import PauseSettings, ReplyPause

    p = ReplyPause(settings=PauseSettings(cue_frames=1, resume_after_s=0.05, max_pause_s=0.05))
    p.audio_sent()
    p.interrupting()                               # the reply was cut off: its queued audio is gone
    p.interrupted()
    p.frame(0.9, 0.9)
    assert p.paused_at is None and p.events == []  # 2026-10-05 13:31: a late frame paused again after the confirm
    for _ in range(2):                             # two pauses the VAD never confirms (a television, say) ...
        p.audio_sent()
        p.frame(0.9, 0.9)
        assert p.paused_at is not None
        await asyncio.sleep(0.12)
        assert p.paused_at is None
    p.audio_sent()
    p.frame(0.9, 0.9)
    assert p.paused_at is None                     # ... and the third is not paused: the VAD alone decides again
    assert [e.kind for e in p.events] == ["paused", "resumed", "paused", "resumed"]
