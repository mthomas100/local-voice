"""The turn start that waits for words while the agent is busy (local_voice/turn_start.py), model-free.

Units: the strategy's rules driven frame by frame, with a fake reply pause and a scripted agent (no pipeline): the
idle and pending-yes/no starts at the VAD start, the hold and what each kind of words does with it, the no-words
timers. Whole turns (needs_pi): the harness orchestrator (energy VAD, fake streaming recogniser at 4 words/s, the stub
LLM, protocol v1 with the reply pause) with echo.hold_for_words all, where the scripted transcript of a tone streamed
over a reply stands for what the microphone heard."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from local_voice.config import ConfigError, load_config
from local_voice.echo_guard import EchoGuard
from local_voice.turn_start import AgentActivity, BusyHoldStartStrategy, hold_applies
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterimTranscriptionFrame,
                                   TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.turns.types import ProcessFrameResult
from pipecat.utils.asyncio.task_manager import TaskManager

SAID = ["Sentence number 1 is here.", "Sentence number 2 is here.", "The capital of France is Paris."]


# ------------------------------------------------------------------------------------------------ settings

def test_hold_for_words_modes_and_where_they_apply():
    assert load_config().echo["hold_for_words"] == "browser"           # the shipped setting
    for given, want in ((True, "all"), (False, "off"), ("all", "all"), ("browser", "browser")):
        assert load_config(overrides={"echo": {"guard": True, "hold_for_words": given}}).echo["hold_for_words"] == want
    assert load_config(overrides={"echo": {"guard": False, "hold_for_words": False}}).echo == {}   # both off
    with pytest.raises(ConfigError, match="hold_for_words"):
        load_config(overrides={"echo": {"hold_for_words": "sometimes"}})
    assert hold_applies("browser", protocol_v1=False, mic="vad") and not hold_applies("browser", protocol_v1=True, mic="vad")
    assert hold_applies("all", protocol_v1=True, mic="vad") and not hold_applies("all", protocol_v1=True, mic="ptt")
    assert not hold_applies("off", protocol_v1=False, mic="vad")


# ------------------------------------------------------------------------------------- the rules, frame by frame

class FakePause:
    def __init__(self):
        self.calls: list[str] = []

    def hold_for_words(self):
        self.calls.append("hold")

    def resume_now(self, *, echo: bool = False):
        self.calls.append("resume_echo" if echo else "resume")

    def stop_holding(self):
        self.calls.append("stop")

    def speech_over(self):
        self.calls.append("over")


async def strategy(run: str = "running", speaking: bool = True, tail_s: float = 0.0, **kw):
    """The strategy with a scripted agent (`state["run"]`), a fake pause and the events Pipecat's controller would
    act on: a turn start (which tells every strategy, as UserTurnController does) and a reset of the aggregation."""
    state = {"run": run}
    activity = AgentActivity(lambda: state["run"], tail_s=tail_s)
    guard = EchoGuard()
    for s in SAID:
        guard.said(s)
    s = BusyHoldStartStrategy(activity=activity, guard=guard, **kw)
    await s.setup(SimpleNamespace(task_manager=TaskManager()))
    s.pause = FakePause()
    events: list[str] = []

    async def started(strat, _params):
        events.append("start")
        await strat.handle_user_turn_started()

    s.add_event_handler("on_user_turn_started", started)
    s.add_event_handler("on_reset_aggregation", lambda _strat: events.append("reset"))
    if speaking:
        await s.process_frame(BotStartedSpeakingFrame())
    return s, state, events


def final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "u", "t", finalized=True)


def interim(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text, "u", "t")


async def test_idle_starts_at_the_vad_start_and_a_pending_yes_no_is_never_held():
    s, state, events = await strategy(run="idle", speaking=False)
    assert await s.process_frame(VADUserStartedSpeakingFrame()) == ProcessFrameResult.STOP
    assert events == ["start"] and s.decisions[-1].reason == "idle"
    await s.handle_user_turn_stopped()
    state["run"] = "confirm"
    await s.process_frame(BotStartedSpeakingFrame())       # the question is being said: busy, but yes/no pending
    assert await s.process_frame(VADUserStartedSpeakingFrame()) == ProcessFrameResult.STOP
    assert events == ["start", "start"] and s.decisions[-1].reason == "confirm" and s.decisions[-1].text == ""


async def test_a_transcript_with_no_turn_starts_one_when_idle():
    s, _, events = await strategy(run="idle", speaking=False)
    assert await s.process_frame(final("hello there")) == ProcessFrameResult.STOP   # the VAD missed it
    assert events == ["start"]


async def test_busy_holds_and_the_persons_words_start_the_turn_on_the_first_interim():
    s, _, events = await strategy()
    await s.process_frame(VADUserStartedSpeakingFrame())
    assert events == [] and s.pause.calls == ["hold"] and s.decisions[-1].action == "hold"
    assert await s.process_frame(interim("what time")) == ProcessFrameResult.STOP
    assert events == ["start"] and s.decisions[-1].reason == "words" and s.decisions[-1].final is False
    assert s.pause.calls[1:3] == ["over", "stop"]       # the hold ended; the interruption releases the pause


async def test_echo_resumes_the_reply_and_keeps_holding_until_its_final():
    s, _, events = await strategy()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.on_echo(False, "")                             # an interim the filter dropped
    assert s.pause.calls == ["hold", "resume_echo"] and s._hold is not None
    await s.on_echo(False, "")                             # the next one: nothing more to do
    assert s.pause.calls == ["hold", "resume_echo"]
    await s.process_frame(VADUserStoppedSpeakingFrame())
    await s.on_echo(True, "")                              # the final, dropped
    assert s._hold is None and events == [] and s.pause.calls[-2:] == ["resume_echo", "over"]
    assert [d.reason for d in s.decisions] == ["busy", "echo", "echo"]


async def test_the_persons_words_after_echo_start_the_turn():
    s, _, events = await strategy()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.on_echo(False, "what does that mean")          # the filter cut the echo out and passes the rest on
    assert events == []
    await s.process_frame(interim("what does that mean"))
    assert events == ["start"]


async def test_a_backchannel_final_resumes_and_leaves_the_aggregation():
    s, _, events = await strategy()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(interim("Yeah"))                 # "Yeah, but ..." may follow: undecided
    assert events == [] and s.decisions[-1].action == "hold"
    await s.process_frame(VADUserStoppedSpeakingFrame())
    await s.process_frame(final("Yeah, yeah."))
    assert events == ["reset"] and s._hold is None and "resume" in s.pause.calls
    assert s.decisions[-1].reason == "backchannel"


async def test_a_fragment_of_the_last_sentences_is_echo_cut_short_but_other_short_words_are_not():
    s, _, events = await strategy()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame())
    await s.process_frame(final("Paris."))
    assert events == ["reset"] and s.decisions[-1].reason == "fragment"
    await s.process_frame(VADUserStartedSpeakingFrame())
    assert await s.process_frame(final("Stop.")) == ProcessFrameResult.STOP
    assert events == ["reset", "start"]


async def test_the_agent_no_longer_busy_lets_any_words_start_the_turn():
    s, state, events = await strategy()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(interim("Paris"))                # a fragment while busy: held
    state["run"] = "idle"
    await s.process_frame(BotStoppedSpeakingFrame())       # the reply is over
    await s.process_frame(interim("Paris"))
    assert events == ["start"] and s.decisions[-1].reason == "no_longer_busy"


async def test_no_words_after_the_vad_stop_resumes_or_interrupts_as_set():
    s, _, events = await strategy(final_wait_s=0.05)
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame())  # a cough
    await asyncio.sleep(0.15)
    assert events == [] and s._hold is None and s.decisions[-1].reason == "no_words"
    assert s.decisions[-1].action == "resume" and s.pause.calls[-2:] == ["resume", "over"]
    s, _, events = await strategy(final_wait_s=0.05, no_words="interrupt")
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame())
    await asyncio.sleep(0.15)
    assert events == ["start"] and s.decisions[-1].reason == "no_words"


async def test_speech_with_no_words_for_max_hold_interrupts_unless_echo_was_seen():
    s, _, events = await strategy(max_hold_s=0.1)
    await s.process_frame(VADUserStartedSpeakingFrame())
    await asyncio.sleep(0.2)
    assert events == ["start"] and s.decisions[-1].reason == "max_hold"
    await s.handle_user_turn_stopped()
    s, _, events = await strategy(max_hold_s=0.1)
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.on_echo(False, "")
    await asyncio.sleep(0.2)
    assert events == [] and s._hold is not None           # echo goes on: its final decides


def test_the_tail_keeps_the_agent_busy_after_the_bot_stops_speaking():
    """Live echo lags Pipecat's "Bot stopped speaking" (the server's audio has gone out) by the echo path: the agent
    stays busy for tail_s after it, idle once it is over, and busy only by the tail is not `playing`."""
    now = [100.0]
    a = AgentActivity(lambda: "idle", tail_s=1.0, clock=lambda: now[0])
    assert not a.busy()                                    # never spoke: no tail
    a.saw(BotStartedSpeakingFrame())
    now[0] = 103.0
    a.saw(BotStoppedSpeakingFrame())
    now[0] = 103.5
    a.saw(BotStoppedSpeakingFrame())                       # the filter's copy, later: the first one set the time
    now[0] = 103.9
    assert a.busy() and a.in_tail() and not a.playing() and a.why().startswith("just done speaking")
    now[0] = 104.0
    assert not a.busy() and a.why() == "idle"
    b = AgentActivity(lambda: "idle", clock=lambda: now[0])     # tail 0: idle at once
    b.saw(BotStartedSpeakingFrame())
    b.saw(BotStoppedSpeakingFrame())
    assert not b.busy()
    assert load_config().echo["tail_s"] == 1.0
    with pytest.raises(ConfigError, match="tail_s"):
        load_config(overrides={"echo": {"tail_s": 9}})


async def test_in_the_tail_a_vad_start_holds_echo_starts_no_turn_and_a_final_answer_starts_one():
    s, _, events = await strategy(run="idle", tail_s=1.0)      # the run is over, the reply's audio still going out
    await s.process_frame(BotStoppedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())       # 137 ms later (the bench): its echo, or the person
    assert events == [] and (s.decisions[-1].action, s.decisions[-1].reason) == ("hold", "busy")
    await s.on_echo(True, "")                                  # the filter dropped it as echo: no turn
    assert events == [] and s._hold is None and s.decisions[-1].reason == "echo"
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(interim("Paris"))                    # an interim fragment still waits (a longer echo's start)
    await s.process_frame(VADUserStoppedSpeakingFrame())
    assert events == []
    assert await s.process_frame(final("Paris.")) == ProcessFrameResult.STOP   # nothing plays: it is an answer
    assert events == ["start"] and (s.decisions[-1].reason, s.decisions[-1].final) == ("tail", True)
    await s.handle_user_turn_stopped()
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(VADUserStoppedSpeakingFrame())
    assert await s.process_frame(final("Yes.")) == ProcessFrameResult.STOP     # "Want me to go on?" "Yes."
    assert events == ["start", "start"] and s.decisions[-1].reason == "tail"


async def test_after_the_tail_a_vad_start_starts_at_once():
    s, _, events = await strategy(run="idle", tail_s=0.15)
    await s.process_frame(BotStoppedSpeakingFrame())
    await asyncio.sleep(0.2)
    assert await s.process_frame(VADUserStartedSpeakingFrame()) == ProcessFrameResult.STOP
    assert events == ["start"] and s.decisions[-1].reason == "idle"


async def test_the_tts_tells_the_guard_each_generation_it_says():
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TTSSpeakFrame
    from test_speech_path import sentence, speak

    guard = EchoGuard()
    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Hey!"), sentence("What can I help you with?"),
                          LLMFullResponseEndFrame(), TTSSpeakFrame("Let me check.")], guard=guard)
    assert [s for _, s in guard.recent()] == eng.spoken == ["Hey! What can I help you with?", "Let me check."]


# ---------------------------------------------------------------------------------------------- whole turns

LONG = " ".join(f"Sentence number {i} is here." for i in range(1, 13))
ECHO_ON = {"echo": {"guard": True, "hold_for_words": "all"}}


async def _while_playing(c, played_s: float = 1.5):
    """Stream silence until the reply has played `played_s`; returns the reply."""
    quiet = asyncio.create_task(c.silence(30.0))
    await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=10)
    while True:
        rep = c.current
        if rep is not None and rep.first_audio_at is not None and c.now() - rep.first_audio_at > played_s:
            break
        await asyncio.sleep(0.02)
    quiet.cancel()
    return rep


def _session(r):
    return r.orch.clients["test-client"].session


@pytest.mark.needs_pi
async def test_echo_while_the_bot_speaks_resumes_the_reply_and_starts_no_turn(tmp_path):
    from harness import rig
    from local_voice.client import tone_pcm

    async with rig(tmp_path, stt="fake_stream", stt_script=["tell me a story", "Sentence number 1 is here sentence number 2"],
                   tts_rtf=0.5, overrides=ECHO_ON) as r:
        r.stub.script([{"text": LONG, "chunk_words": 4, "delay_ms": 20}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        rep = await _while_playing(c)
        echo_at = c.now()
        await c.speak(tone_pcm(1.2))                       # the reply's own words, heard back
        tail = asyncio.create_task(c.silence(30.0))
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") == rep.id, timeout=30)
        tail.cancel()
        s = _session(r)
        assert not c.of_type("interrupt", since=echo_at) and rep.interrupted_at is None and end
        assert len(r.stub.requests()) == 1                 # no turn
        d = [(x.action, x.reason) for x in s.turn_start.decisions]
        assert d[0] == ("start", "idle") and d[1] == ("hold", "busy") and d[-1] == ("resume", "echo")
        assert ("start", "words") not in d
        assert [e.kind for e in s.reply_pause.events] == ["paused", "resumed"]
        assert s.echo_guard.dropped and not s.echo_guard.passed
        assert not [m for m in c.of_type("transcript", since=echo_at) if m.get("final")]   # no caption for echo
        await c.close()


@pytest.mark.needs_pi
async def test_the_persons_words_while_the_bot_speaks_interrupt(tmp_path):
    from harness import rig
    from local_voice.client import tone_pcm

    async with rig(tmp_path, stt="fake_stream", stt_script=["tell me a story", "what time is it now"], tts_rtf=0.5,
                   overrides=ECHO_ON) as r:
        r.stub.script([{"text": LONG, "chunk_words": 4, "delay_ms": 20}, {"text": "It is noon."}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        rep = await _while_playing(c)
        at = c.now()
        barge = asyncio.create_task(c.speak(tone_pcm(1.0), tail_s=2.0))
        intr = await c.wait_for(lambda m: m.get("t") == "interrupt", timeout=5, since=at)
        intr_at = next(t for t, m in c.messages if m is intr)
        assert intr["reply_id"] == rep.id
        # the first interim decides (the fake recogniser has two words as soon as the VAD opens it, 0.2 s in)
        assert intr_at - at < 0.6, f"interrupt {1000 * (intr_at - at):.0f} ms after speech"
        d = _session(r).turn_start.decisions
        assert (d[-1].action, d[-1].reason, d[-1].final) == ("start", "words", False)
        await barge
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") != rep.id, timeout=15,
                         since=intr_at)
        assert r.user_texts()[-1].endswith("what time is it now")
        await c.close()


@pytest.mark.needs_pi
async def test_a_backchannel_resumes_and_is_not_in_the_next_turns_text(tmp_path):
    from harness import rig
    from local_voice.client import tone_pcm

    async with rig(tmp_path, stt="fake_stream", stt_script=["tell me a story", "mm-hmm", "what time is it"],
                   tts_rtf=0.5, overrides=ECHO_ON) as r:
        reply = " ".join(f"Sentence number {i} is here." for i in range(1, 6))
        r.stub.script([{"text": reply, "chunk_words": 4, "delay_ms": 20}, {"text": "It is noon."}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        rep = await _while_playing(c, 1.0)
        at = c.now()
        await c.speak(tone_pcm(0.5))
        tail = asyncio.create_task(c.silence(30.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") == rep.id, timeout=30)
        tail.cancel()
        assert not c.of_type("interrupt", since=at) and rep.interrupted_at is None
        s = _session(r)
        assert ("resume", "backchannel") in [(d.action, d.reason) for d in s.turn_start.decisions]
        assert [e.kind for e in s.reply_pause.events] == ["paused", "resumed"]
        await asyncio.sleep(0.6)                           # the bot has stopped speaking: idle
        quiet_from = c.now()
        await c.speak(tone_pcm(0.8))
        tail = asyncio.create_task(c.silence(30.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=quiet_from)
        tail.cancel()
        assert r.user_texts() == ["tell me a story", "what time is it"]
        await c.close()


@pytest.mark.needs_pi
async def test_a_fragment_while_the_bot_speaks_is_echo_cut_short(tmp_path):
    from harness import rig
    from local_voice.client import tone_pcm

    async with rig(tmp_path, stt="fake_stream", stt_script=["tell me a story", "is here"], tts_rtf=0.5,
                   overrides=ECHO_ON) as r:
        r.stub.script([{"text": LONG, "chunk_words": 4, "delay_ms": 20}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        rep = await _while_playing(c)
        at = c.now()
        await c.speak(tone_pcm(0.4))
        tail = asyncio.create_task(c.silence(30.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") == rep.id, timeout=30)
        tail.cancel()
        assert not c.of_type("interrupt", since=at) and len(r.stub.requests()) == 1
        d = _session(r).turn_start.decisions
        assert ("resume", "fragment", True) in [(x.action, x.reason, x.final) for x in d]
        await c.close()


@pytest.mark.needs_pi
async def test_echo_while_idle_is_a_turn_with_its_words_marked_for_the_model(tmp_path):
    from harness import rig
    from local_voice.client import tone_pcm

    said = "I plan from the garden folder. Tomatoes, peppers and a compost heap layer on top."
    async with rig(tmp_path, stt="fake_stream", stt_script=["where do you plan from", "and a compost heap layer on top"],
                   overrides=ECHO_ON) as r:
        r.stub.script([{"text": said, "delay_ms": 5}, {"text": "That was my own line.", "delay_ms": 5}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        tail = asyncio.create_task(c.silence(30.0))
        first = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15)
        tail.cancel()
        await c.silence(1.0)                               # the reply is over and the run settled: idle
        at = c.now()
        await c.speak(tone_pcm(1.4))
        tail = asyncio.create_task(c.silence(30.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") != first["reply_id"],
                         timeout=15, since=at)
        tail.cancel()
        s = _session(r)
        assert r.user_texts()[-1] == "and a compost heap layer on top"
        assert s.turn_start.decisions[-1].reason == "idle"
        assert len(s.echo_guard.passed) == 1 and not s.echo_guard.dropped
        note, record = s.echo_guard.turn_note("and a compost heap layer on top")
        assert note.startswith("(These words match what you said ") and note.endswith("your own voice.)")
        assert record["matched_words"] == 7 and record["heard_words"] == 7 and record["said_ago_s"] >= 0
        assert s.echo_guard.turn_note("and a compost heap layer on top") == (None, None)   # noted once
        await c.close()


@pytest.mark.needs_pi
async def test_a_pending_yes_no_starts_the_turn_at_the_vad_start(tmp_path):
    from harness import rig
    from local_voice.client import tone_pcm

    write = {"tool_calls": [{"name": "write", "arguments": {"path": "Hello.txt", "content": "hi\n"}}], "delay_ms": 5}
    ov = {**ECHO_ON, "agent": {"approvals": {"card_wait_s": 20.0, "reask_after_s": 0.0, "voice_wait_s": 20.0}}}
    async with rig(tmp_path, stt="fake_stream", stt_script=["yes"], overrides=ov,
                   spaces={"tools": ["read", "ls"], "act_tools": ["write"]}) as r:
        r.stub.script([write, {"text": "Done.", "delay_ms": 5}])
        c = await r.client(client="mac")
        t0 = c.now()
        await c.send({"t": "text", "text": "Act mode."})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=t0)
        t0 = c.now()
        await c.send({"t": "text", "text": "Make a file called Hello.txt"})
        await c.wait_for(lambda m: m.get("t") == "confirm_request", timeout=15, since=t0)
        await c.speak(tone_pcm(0.6))
        tail = asyncio.create_task(c.silence(30.0))
        cancel = await c.wait_for(lambda m: m.get("t") == "confirm_cancel", timeout=15, since=t0)
        await c.wait_for(lambda m: m.get("t") == "reply_text" and "Done" in m.get("delta", ""), timeout=15, since=t0)
        tail.cancel()
        assert cancel["why"] == "answered"                 # by voice: the spoken yes
        d = _session(r).turn_start.decisions
        starts = [x for x in d if x.action == "start"]
        assert starts and (starts[-1].reason, starts[-1].text) == ("confirm", "")   # at the VAD start, before words
        assert not [x for x in d if x.reason == "backchannel"]                     # "yes" was never one
        assert (r.workdir / "Hello.txt").read_text() == "hi\n"
        await c.close()


@pytest.mark.needs_pi
async def test_off_is_pipecats_own_turn_start(tmp_path):
    from harness import rig
    from local_voice.client import tone_pcm
    from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy

    async with rig(tmp_path, stt="fake_stream", stt_script=["tell me a story", "Sentence number 1 is here sentence"],
                   tts_rtf=0.5, overrides={"echo": {"guard": False, "hold_for_words": "off"}}) as r:
        r.stub.script([{"text": LONG, "chunk_words": 4, "delay_ms": 20}, {"text": "Okay."}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        rep = await _while_playing(c)
        s = _session(r)
        assert s.turn_start is None and s.echo_guard is None
        assert not [p for p in s.pipeline.processors if "EchoFilter" in str(p)]
        agg = next(p for p in s.pipeline.processors if type(p).__name__ == "LLMUserAggregator")
        assert [type(x) for x in agg._user_turn_controller.user_turn_strategies.start] == [
            VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]
        at = c.now()
        barge = asyncio.create_task(c.speak(tone_pcm(0.8), tail_s=2.0))
        intr = await c.wait_for(lambda m: m.get("t") == "interrupt", timeout=5, since=at)
        intr_at = next(t for t, m in c.messages if m is intr)
        assert intr["reply_id"] == rep.id and intr_at - at < 0.3     # at the VAD start, echo or not
        await barge
        await c.close()


@pytest.mark.needs_pi
@pytest.mark.parametrize("tail_s", [1.0, 0.0])
async def test_the_echo_of_a_short_reply_just_after_it_ends(tmp_path, tail_s):
    """The planted-echo bench's finding, model-free (2026-10-05): the echo of a short reply begins after the
    server's "Bot stopped speaking" (the client still plays the last chunk, then the room and the microphone add their
    delay), when the agent was idle, so the guard let it through and it became a turn. Here the tone the client speaks
    right after `state: listening` (the Mouth's word for "Bot stopped speaking") stands for that echo, and the scripted
    transcript is the reply's own words. With the tail it is held, dropped as echo and starts no turn; with tail_s 0 it
    is a turn, as before."""
    from harness import rig
    from local_voice.client import tone_pcm

    said = "Okay then, here it is now."
    ov = {"echo": {"guard": True, "hold_for_words": "all", "tail_s": tail_s}}
    async with rig(tmp_path, stt="fake_stream", stt_script=["tell me something", "okay then here it is now"],
                   overrides=ov) as r:
        r.stub.script([{"text": said, "delay_ms": 5}, {"text": "That was my own line.", "delay_ms": 5}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        quiet = asyncio.create_task(c.silence(30.0))
        await c.wait_for(lambda m: m.get("t") == "audio_end", timeout=15)
        stopped = await c.wait_for(lambda m: m.get("t") == "state" and m.get("v") == "listening", timeout=5,
                                   since=next(t for t, m in c.messages if m.get("t") == "audio_start"))
        quiet.cancel()
        at = c.now()
        await c.speak(tone_pcm(0.4))                       # the echo: its VAD start lands ~0.2 s into the tail
        quiet = asyncio.create_task(c.silence(30.0))
        for _ in range(60):                                # a turn, if one comes, is asked within 3 s
            if len(r.stub.requests()) > 1:
                break
            await asyncio.sleep(0.05)
        quiet.cancel()
        s = _session(r)
        d = [(x.action, x.reason) for x in s.turn_start.decisions]
        assert stopped and at - next(t for t, m in c.messages if m is stopped) < 0.2
        if tail_s:
            assert len(r.stub.requests()) == 1, r.user_texts()
            # held at the VAD start, the first interim ("okay then") a fragment that waits, the final dropped as echo
            assert d[1] == ("hold", "busy") and d[-1] == ("resume", "echo") and d.count(("start", "idle")) == 1, d
            assert [h for h, _ in s.echo_guard.dropped] == ["okay then here it is now"] and not s.echo_guard.passed
        else:
            assert r.user_texts()[-1] == "okay then here it is now", d     # the finding, reproduced
            assert d[-1] == ("start", "idle") and len(s.echo_guard.passed) == 1
        await c.close()
