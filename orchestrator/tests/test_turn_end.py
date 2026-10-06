"""The final transcript as a second opinion on Smart Turn (local_voice/turn_end.py): whole spoken turns over protocol v1
with the energy VAD, a Smart Turn whose verdicts are scripted, and a scripted streaming recogniser."""
from __future__ import annotations

import asyncio

import pytest
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams

from harness import rig
from local_voice.client import tone_pcm
from local_voice.pipeline import TurnSetup
from local_voice.testing import CueEnergyVADAnalyzer
from local_voice.turn_end import WordsTurnStopStrategy, ends_turn, goes_on, holds


class ScriptedSmartTurn(BaseSmartTurn):
    """Smart Turn with its verdicts scripted, one per VAD stop: (prediction, probability)."""

    def __init__(self, verdicts: list[tuple[int, float]], **kw):
        super().__init__(**kw)
        self.verdicts = list(verdicts)
        self.calls = 0

    def _predict_endpoint(self, audio_array):
        v = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return {"prediction": v[0], "probability": v[1]}


def turns(verdicts, *, fallback=3.0, punct=None, veto=None, command=None, hold=None):
    def factory(mic: str) -> TurnSetup:
        st = WordsTurnStopStrategy(turn_analyzer=ScriptedSmartTurn(verdicts, params=SmartTurnParams(stop_secs=fallback)),
                                   punctuation_stop_secs=punct, veto_wait_secs=veto, command_stop_secs=command,
                                   hold_secs=hold)
        return TurnSetup(vad=CueEnergyVADAnalyzer(), stop=[st], stop_timeout_s=8.0)
    return factory


def test_the_words_rules():
    assert ends_turn("What does a heat pump do?") and ends_turn("Set a timer for ten minutes.") and ends_turn("Yes.")
    assert not ends_turn("What is the capital of.") and not ends_turn("I was thinking,") and not ends_turn("Can you tell me")
    assert goes_on("I was thinking,") and goes_on("What's the difference between") and goes_on("My question is about the")
    assert goes_on("Well...") and not goes_on("Why is the sky blue?") and not goes_on("Can you tell me")
    assert goes_on("Can you tell me", unpunctuated=True) and not goes_on("", unpunctuated=True)


async def _turn_end_after(r, segments: list[tuple[float, float]]) -> tuple[float, list[str]]:
    """Speak tones (seconds, then seconds of silence); return how long after the last tone the agent's turn began,
    and every prompt the stub got."""
    c = await r.client()
    eos = 0.0
    for speech, quiet in segments:
        eos = await c.speak(tone_pcm(speech), tail_s=0.0)
        if quiet:
            await c.silence(quiet)
    tail = asyncio.create_task(c.silence(6.0))
    await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=eos)
    tail.cancel()
    think = next(at for at, m in c.messages if m.get("t") == "state" and m.get("v") == "thinking" and at > eos)
    await c.close()
    return think - eos, r.user_texts()


@pytest.mark.parametrize("punct", [None, 0.4])
async def test_an_unsure_smart_turn_ends_the_turn_on_a_finished_sentence(tmp_path, punct):
    async with rig(tmp_path, stt="fake_stream", stt_script=["What does a heat pump do?"],
                   turn_factory=turns([(0, 0.02)], punct=punct)) as r:
        r.stub.script([{"text": "It moves heat.", "delay_ms": 5}])
        waited, prompts = await _turn_end_after(r, [(0.8, 0.0)])
    assert prompts == ["What does a heat pump do?"]
    if punct is None:
        assert waited >= 3.0, "without the rule the turn waits for Smart Turn's 3 s fallback"
    else:
        assert waited < 2.0, f"the finished sentence ended the turn {waited:.2f} s after the speech"


async def test_a_sentence_that_goes_on_keeps_the_turn_open_across_its_pause(tmp_path):
    async with rig(tmp_path, stt="fake_stream", stt_script=["What is the capital of", "Australia?"],
                   turn_factory=turns([(0, 0.1), (0, 0.1)], punct=0.4)) as r:
        r.stub.script([{"text": "Canberra.", "delay_ms": 5}])
        waited, prompts = await _turn_end_after(r, [(0.6, 1.0), (0.5, 0.0)])
    assert prompts == ["What is the capital of Australia?"]
    assert waited < 2.0


@pytest.mark.parametrize("veto", [None, 1.5])
async def test_a_complete_verdict_on_words_that_go_on_waits_for_the_rest(tmp_path, veto):
    async with rig(tmp_path, stt="fake_stream", stt_script=["I was thinking about,", "the weather."],
                   turn_factory=turns([(1, 0.9), (1, 0.9)], veto=veto)) as r:
        r.stub.script([{"text": "Sunny.", "delay_ms": 5}, {"text": "Sunny.", "delay_ms": 5}])
        _waited, prompts = await _turn_end_after(r, [(0.6, 0.9), (0.5, 0.0)])
    if veto is None:
        assert prompts[0] == "I was thinking about,", "Smart Turn alone cuts the person off at the pause"
    else:
        assert prompts == ["I was thinking about, the weather."]


async def test_a_vetoed_turn_still_ends_when_nothing_more_is_said(tmp_path):
    async with rig(tmp_path, stt="fake_stream", stt_script=["My question is about the"],
                   turn_factory=turns([(1, 0.9)], veto=1.0)) as r:
        r.stub.script([{"text": "About what?", "delay_ms": 5}])
        waited, prompts = await _turn_end_after(r, [(0.6, 0.0)])
    assert prompts == ["My question is about the"]
    assert 1.0 <= waited < 2.6


@pytest.mark.parametrize("carry", [0.0, 2.0])
async def test_a_short_first_word_is_decoded_again_with_what_follows(tmp_path, carry):
    """Nemotron returns nothing for a short word on its own ("Wait," before a pause); with carry_s the next segment's
    session hears it again together with what follows."""
    ov = {"stt": {"adapters": {"fake_stream": {"min_heard_s": 0.6, "carry_s": carry}}}}
    async with rig(tmp_path, stt="fake_stream", stt_text="", stt_script=["(lost)", "Wait, does it work in winter?"],
                   overrides=ov) as r:
        r.stub.script([{"text": "Yes.", "delay_ms": 5}])
        c = await r.client()
        await c.speak(tone_pcm(0.4), tail_s=0.0)
        await c.silence(0.6)
        await c.speak(tone_pcm(1.0), tail_s=0.0)
        tail = asyncio.create_task(c.silence(4.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15)
        tail.cancel()
        await c.close()
        sessions = r.stt.opened
        assert r.user_texts() == ["Wait, does it work in winter?"]
    assert sessions[0].loud_s < 0.6, "the first word alone gave no words"
    if carry:
        assert sessions[1].loud_s >= 1.3, "the second session heard the first word again"
    else:
        assert sessions[1].loud_s < 1.1


async def test_the_tone_hook_is_off_by_default_and_its_hint_reaches_only_the_prompt(tmp_path):
    """tone/README.md: off by default (nothing imported or computed); when on, the bracketed hint goes to Pi before the
    words, and the turn log's user_text stays the words alone."""
    async with rig(tmp_path, stt_text="what a day") as r:
        assert r.orch.tone is None and r.orch.cfg.tone["mode"] == "off"

    class FakeTone:
        def __init__(self):
            self.calls = []

        def analyze(self, pcm, transcript, *, session, turn, channel):
            self.calls.append((len(pcm), transcript, channel))
            return type("R", (), {"hint": "[delivery vs usual, automatic and uncertain: pace faster (+3.0 sd)]",
                                  "due": "[delivery vs usual, automatic and uncertain: pace faster (+3.0 sd)]"})()

    tone = FakeTone()
    async with rig(tmp_path / "on", stt="fake_stream", stt_text="what a day") as r:
        r.orch.tone = tone
        r.stub.script([{"text": "You sound rushed.", "delay_ms": 5}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        tail = asyncio.create_task(c.silence(4.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15)
        tail.cancel()
        await c.close()
        await asyncio.sleep(1.0)
        prompt = r.user_texts()[-1]
    assert tone.calls and tone.calls[0][1] == "what a day" and tone.calls[0][0] > 16000 and tone.calls[0][2] == "test"
    assert prompt == "[delivery vs usual, automatic and uncertain: pace faster (+3.0 sd)]\nwhat a day"
    import json as _json
    rows = [_json.loads(x) for p in (tmp_path / "on" / "turns").glob("*.jsonl") for x in p.read_text().splitlines()]
    assert rows[-1]["user_text"] == "what a day" and rows[-1]["tone"]["shown"] is True


@pytest.mark.parametrize("said, command", [("Act mode.", None), ("Act mode.", 0.3), ("What does a heat pump do?", 0.3)])
async def test_an_unsure_smart_turn_ends_the_turn_at_once_on_a_command(tmp_path, said, command):
    """"Back home.", "Just talk." and "Yes." waited for Smart Turn's fallback in the M3 e2e (2026-10-05 16:22): a
    command the orchestrator answers itself is complete as said. A question is not a command: it still waits."""
    async with rig(tmp_path, stt="fake_stream", stt_script=[said],
                   turn_factory=turns([(0, 0.02)], command=command)) as r:
        r.stub.script([{"text": "It moves heat.", "delay_ms": 5}])
        waited, _ = await _turn_end_after(r, [(0.8, 0.0)])
        mode = r.orch.hub.mode
    if command is not None and said == "Act mode.":
        assert waited < 1.5 and mode == "act", f"the command ended its turn {waited:.2f} s after the speech"
    else:
        assert waited >= 3.0, f"the turn ended {waited:.2f} s after the speech, before Smart Turn's fallback"


def test_the_hold_words():
    """A final that stops on a filler or a joining word, whatever punctuation the recogniser put after it (as in a
    rambling request: "And uh.", "Just put a Um.", "So.", "a short note that just.")."""
    for t in ("And uh.", "Just put a Um.", "So.", "I think like a short note that just.", "Add milk and",
              "It looks good but", "Okay so.", "Read me the"):
        assert holds(t), t
    for t in ("Something like that.", "What is the capital of France?", "Hello World from the voice agent.", "Yes.", ""):
        assert not holds(t), t
    assert holds("What does a heat pump look like?")                              # the brief's list alone holds it,
    assert not holds("What does a heat pump look like?", except_questions=True)   # but it is a finished question
    assert holds("Can you make a, like,", except_questions=True) and holds("And um?", except_questions=True)


@pytest.mark.parametrize("hold", [None, 2.0])
async def test_a_complete_verdict_on_a_filler_keeps_the_turn_open_across_the_pause(tmp_path, hold):
    async with rig(tmp_path, stt="fake_stream", stt_script=["Please make a note. And um.", "Just say hello."],
                   turn_factory=turns([(1, 0.9), (1, 0.9)], fallback=2.0, hold=hold)) as r:
        r.stub.script([{"text": "Done.", "delay_ms": 5}, {"text": "Done.", "delay_ms": 5}])
        _waited, prompts = await _turn_end_after(r, [(0.6, 1.2), (0.5, 0.0)])
    if hold is None:
        assert prompts[0] == "Please make a note. And um.", "Smart Turn alone ends the turn at the filler"
    else:
        assert prompts == ["Please make a note. And um. Just say hello."]


async def test_a_turn_held_on_a_filler_still_ends_at_the_fallback(tmp_path):
    async with rig(tmp_path, stt="fake_stream", stt_script=["So, um."],
                   turn_factory=turns([(1, 0.9)], fallback=2.0, hold=2.0)) as r:
        r.stub.script([{"text": "Take your time.", "delay_ms": 5}])
        waited, prompts = await _turn_end_after(r, [(0.6, 0.0)])
    assert prompts == ["So, um."] and 2.0 <= waited < 3.0, waited


async def test_a_finished_sentence_still_ends_at_once_with_the_hold_on(tmp_path):
    async with rig(tmp_path, stt="fake_stream", stt_script=["What is the capital of France?"],
                   turn_factory=turns([(1, 0.9)], fallback=2.0, hold=2.0)) as r:
        r.stub.script([{"text": "Paris.", "delay_ms": 5}])
        waited, _ = await _turn_end_after(r, [(0.8, 0.0)])
    assert waited < 1.0, waited
