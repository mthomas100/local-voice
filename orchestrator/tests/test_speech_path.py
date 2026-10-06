"""The speech path's text rules: how a reply's sentences are grouped into generations, and where a sentence ends;
and the grouping as the TTS service does it while the sentences stream in (services/tts.py, tts.group)."""
from __future__ import annotations

import time

import pytest

from local_voice.engines.fakes import FakeSynthesizer
from local_voice.speech_text import group_sentences, word_count


def test_word_count():
    assert word_count("I couldn't check — each one.") == 5
    assert word_count("") == 0


def test_grouping_off_is_one_generation_per_sentence():
    assert group_sentences(["Hey!", "What can I help you with?"]) == ["Hey!", "What can I help you with?"]


def test_a_short_sentence_is_said_with_the_next():
    # an early live test's first reply (2026-10-05): "Hey!" was generated alone
    assert group_sentences(["Hey!", "What can I help you with?"], min_words=4) == ["Hey! What can I help you with?"]
    assert group_sentences(["Sure.", "Okay.", "The capital is Rome."], min_words=4) == ["Sure. Okay. The capital is Rome."]


def test_a_short_last_sentence_joins_the_group_before_and_a_lone_one_stays():
    assert group_sentences(["The file is saved in your notes.", "Done."], min_words=4) == [
        "The file is saved in your notes. Done."]
    assert group_sentences(["Six."], min_words=4) == ["Six."]


def test_up_to_max_sentences_per_generation():
    s = ["One two three four.", "Five six seven.", "Eight nine ten eleven.", "Twelve."]
    assert group_sentences(s, max_sentences=2) == ["One two three four. Five six seven.", "Eight nine ten eleven. Twelve."]
    assert group_sentences(s, min_words=4, max_sentences=3) == [
        "One two three four. Five six seven. Eight nine ten eleven. Twelve."]


# ------------------------------------------------------------------- grouping in the TTS service (services/tts.py)

class TimedSynth(FakeSynthesizer):
    """The fake voice, noting when each generation started."""

    def __init__(self, settings: dict):
        super().__init__(settings)
        self.started: list[tuple[float, str]] = []

    def stream(self, text: str):
        self.started.append((time.monotonic(), text))
        return super().stream(text)


def sentence(text: str):
    from pipecat.frames.frames import AggregatedTextFrame
    return AggregatedTextFrame(text, "sentence")


async def speak(frames: list, *, min_words: int = 4, hold_s: float = 0.3, guard=None) -> tuple[TimedSynth, list]:
    """The frames through MLXTTSService as runtime.make_tts builds it (code skipped), with the fake voice; `guard`: the
    EchoGuard the pipeline gives it when echo.guard is on."""
    from pipecat.tests.utils import run_test

    from local_voice.mlx_worker import MLXWorker
    from local_voice.services.tts import MLXTTSService

    eng = TimedSynth({"seconds_per_char": 0.002, "lead_s": 0.0})
    svc = MLXTTSService(engine=eng, worker=MLXWorker(use_mlx=False), sample_rate=24000, skip_aggregator_types=["code"],
                        group_min_words=min_words, group_hold_s=hold_s)
    svc.echo_guard = guard
    down, _ = await run_test(svc, frames_to_send=frames)
    return eng, down


async def test_a_short_sentence_is_one_generation_with_the_next():
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TTSTextFrame

    eng, down = await speak([LLMFullResponseStartFrame(), sentence("Hey!"), sentence(" What can I help you with?"),
                             LLMFullResponseEndFrame()])
    assert eng.spoken == ["Hey! What can I help you with?"]
    assert [f.text for f in down if isinstance(f, TTSTextFrame)] == ["Hey! What can I help you with?"]   # one caption
    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Sure."), sentence("Okay."),
                          sentence("The capital is Rome."), LLMFullResponseEndFrame()])
    assert eng.spoken == ["Sure. Okay. The capital is Rome."]                             # a run of short ones too
    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Hey!"), sentence("What can I help you with?"),
                          LLMFullResponseEndFrame()], min_words=0)
    assert eng.spoken == ["Hey!", "What can I help you with?"]                            # off: as before


async def test_a_short_last_sentence_is_said_at_the_response_end():
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TTSSpeakFrame
    from pipecat.tests.utils import SleepFrame

    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Six."), LLMFullResponseEndFrame(), SleepFrame(0.6),
                          TTSSpeakFrame("Marker.")], hold_s=3.0)
    assert eng.spoken == ["Six.", "Marker."]
    (t_six, _), (t_marker, _) = eng.started
    assert t_marker - t_six >= 0.5                       # said at the end of the response, not held for hold_s


async def test_a_short_sentence_before_a_tool_call_is_said_after_hold_s():
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame
    from pipecat.tests.utils import SleepFrame

    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Okay."), SleepFrame(0.8),   # the tool runs
                          sentence("It says scratch README."), LLMFullResponseEndFrame()], hold_s=0.3)
    assert eng.spoken == ["Okay.", "It says scratch README."]
    (t_ok, _), (t_answer, _) = eng.started
    assert t_answer - t_ok >= 0.35                       # alone after 0.3 s, not with the answer 0.8 s in


async def test_what_is_held_goes_before_anything_else_and_never_into_code():
    from pipecat.frames.frames import (AggregatedTextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
                                       TTSSpeakFrame)

    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Okay."), TTSSpeakFrame("Let me check."),
                          LLMFullResponseEndFrame()], hold_s=3.0)
    assert eng.spoken == ["Okay.", "Let me check."]
    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Here:"), AggregatedTextFrame("```x = 1```", "code"),
                          sentence("That is the whole of it."), LLMFullResponseEndFrame()], hold_s=3.0)
    assert eng.spoken == ["Here:", "That is the whole of it."]


async def test_an_interruption_drops_what_is_held():
    from pipecat.frames.frames import InterruptionFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame
    from pipecat.tests.utils import SleepFrame

    eng, _ = await speak([LLMFullResponseStartFrame(), sentence("Hey!"), SleepFrame(0.1), InterruptionFrame(),
                          SleepFrame(0.5), LLMFullResponseStartFrame(), sentence("A fresh reply starts here."),
                          LLMFullResponseEndFrame()], hold_s=0.3)
    assert eng.spoken == ["A fresh reply starts here."]


@pytest.mark.needs_pi
async def test_grouping_from_config_through_the_whole_orchestrator(tmp_path):
    import asyncio

    from harness import rig
    from local_voice.client import tone_pcm

    async with rig(tmp_path, stt_text="hello", overrides={"tts": {"group": {"min_words": 4, "hold_s": 0.5}}}) as r:
        r.stub.script([{"text": "Hey! What can I help you with?", "delay_ms": 5}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        tail = asyncio.create_task(c.silence(6.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15)
        tail.cancel()
        assert r.tts.spoken[1:] == ["Hey! What can I help you with?"]     # [0] is the busy notice, rendered at start
        await c.close()
