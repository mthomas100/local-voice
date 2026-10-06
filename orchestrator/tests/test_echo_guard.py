"""The echo guard's matching (local_voice/echo_guard.py) on synthetic echoes of the kind an early live test had, and on
what must be kept."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from local_voice.echo_guard import EchoGuard, align, best_match, words

ROOT = Path(__file__).resolve().parents[1]

# A synthetic reply, and its last words heard back the way a recogniser writes replayed audio (slips, its own breaks)
REPLY = ("Let me dig into the garden plan I keep. I plan from ~/garden/beds: a seed list picks the beds and the order, "
         "then tomatoes, peppers, beans, herbs, and a compost heap layer on top. I couldn't check inside each "
         "one — want the full breakdown?")
ECHOES = ["And a compost heap layer on top.", "Peppers beans herbs and a compost heap layer on top.",
          "I couldn't check inside each one. Want a full breakdown.",
          "Those peppers beans herbs and a compost heap layer on top.", "I couldn't check inside each. One."]


def test_words_drop_punctuation_and_keep_contractions():
    assert words("I couldn't check — each. One!") == ["i", "couldn't", "check", "each", "one"]
    assert words("~/.pi/agent") == ["pi", "agent"]
    assert words("") == [] and words(None) == []


def test_align_finds_the_stretch_despite_recogniser_slips():
    m, b, e, _, _ = align(words("want a full breakdown"), words(REPLY))
    assert m == 3 and words(REPLY)[b:e][-3:] == ["the", "full", "breakdown"]


def test_every_echo_of_the_reply_is_echo():
    for heard in ECHOES:
        m = best_match(heard, [(REPLY, 0.0)])
        assert m is not None and m.is_echo(), (heard, m)


def test_the_persons_own_turns_are_not_echo():
    own = ["Okay, so how about you uh walk me through the plan? For the weekend.",
           "What time does the bakery open?", "Yeah.", "What's three plus three?",
           "Yes, go ahead and add it.", "And let me know when you're done."]
    for heard in own:
        m = best_match(heard, [(REPLY, 0.0)])
        assert m is not None and not m.is_echo(), (heard, m)


def test_quoting_the_agent_inside_a_question_is_kept():
    m = best_match("Hang on, what exactly do you mean when you say there is a compost heap layer, and where is it?",
                   [(REPLY, 0.0)])
    assert m.matched >= 3 and not m.is_echo()


def test_short_words_are_never_echo_by_text_alone():
    m = best_match("Six.", [("Six.", 0.0)])
    assert m.matched == 1 and not m.is_echo()


def test_guard_window_and_sentences_joined_into_a_reply():
    now = [1000.0]
    g = EchoGuard(window_s=300, clock=lambda: now[0])
    for i, s in enumerate(["I plan from the garden folder.", "Tomatoes, peppers, beans, herbs,",
                           "and a compost heap layer on top."]):
        g.said(s, at=1000.0 + i)
    now[0] = 1100.0
    assert g.judge("herbs and a compost heap layer") is not None     # runs across a sentence break
    assert g.judge("what is in my compost heap") is None
    now[0] = 1400.0
    assert g.judge("and a compost heap layer on top") is None          # out of the window


def test_detector_counts_the_echo_turns(tmp_path):
    turns = tmp_path / "t.jsonl"
    rows = [("10:00:40.076", "10:00:58.522", "Why don't you describe the plan?", REPLY)]
    rows += [(f"10:0{i + 1}:00.000", f"10:0{i + 1}:05.000", heard, "Something else entirely.")
             for i, heard in enumerate(ECHOES)]
    rows += [("10:09:00.000", "10:09:02.000", "What's three plus three?", "Six.")]
    turns.write_text("".join(json.dumps({
        "v": 1, "type": "turn", "session": "s", "turn": n + 1, "t_start": f"2026-10-05T{a}-07:00",
        "t_end": f"2026-10-05T{b}-07:00", "input": "voice", "user_text": u, "reply_text": r}) + "\n"
        for n, (a, b, u, r) in enumerate(rows)))
    out = subprocess.run([sys.executable, str(ROOT / "tools/echo_turns.py"), str(turns), "--window-s", "600"],
                         capture_output=True, text=True, check=True).stdout
    assert "5 echo turns of 7" in out, out


async def _run_filter(busy: bool, text: str, final: bool = True):
    from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    from local_voice.echo_guard import echo_filter

    g = EchoGuard(window_s=300)
    g.said(REPLY)
    seen, pushed = [], []
    f = echo_filter(g, busy=lambda: busy, on_echo=lambda final, rest: _note(seen, final, rest))

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)
    f.push_frame = push
    cls = TranscriptionFrame if final else InterimTranscriptionFrame
    frame = cls(text, "user", "2026-10-05T10:02:00Z")
    await f.process_frame(frame, FrameDirection.DOWNSTREAM)
    return g, seen, pushed


async def _note(seen, final, rest):
    seen.append((final, rest))


def test_filter_drops_echo_only_while_the_agent_is_busy():
    import asyncio

    g, seen, pushed = asyncio.run(_run_filter(True, "And a compost heap layer on top."))
    assert pushed == [] and seen == [(True, "")] and len(g.dropped) == 1
    # idle: the person may be reading the agent's line aloud to ask about it; the words go on, the match is kept
    g, seen, pushed = asyncio.run(_run_filter(False, "And a compost heap layer on top."))
    assert [p.text for p in pushed] == ["And a compost heap layer on top."] and seen == [] and len(g.passed) == 1


def test_filter_keeps_the_persons_own_words_after_echo_while_busy():
    import asyncio

    heard = "Peppers beans herbs and a compost heap layer on top. What does that mean?"
    g, seen, pushed = asyncio.run(_run_filter(True, heard))
    assert [p.text for p in pushed] == ["What does that mean?"] and seen == [(True, "What does that mean?")]
    # 7 of 12 words matched (0.58): not echo, kept whole
    g, seen, pushed = asyncio.run(_run_filter(True, "And a compost heap layer on top. What does that part mean?"))
    assert len(pushed) == 1 and seen == []
    g, seen, pushed = asyncio.run(_run_filter(True, "What can I help you with", final=False))
    assert [p.text for p in pushed] == ["What can I help you with"] and seen == []


def test_the_match_says_when_the_matched_sentence_was_said_not_the_reply():
    g = EchoGuard(window_s=300, clock=lambda: 1000.0)
    for i, s in enumerate(["Let me dig into the plan.", "I plan from the garden folder.",
                           "And a compost heap layer on top."]):
        g.said(s, at=900.0 + 10 * i)                     # one reply: sentences under 30 s apart
    m = g.match("and a compost heap layer on top")
    assert m.is_echo() and m.said_at == 920.0           # the last sentence's time; the reply began at 900


def test_a_fragment_is_echo_only_when_all_its_words_were_in_the_last_sentences():
    g = EchoGuard(window_s=300, clock=lambda: 100.0)    # fragment_sentences 3
    for i, s in enumerate(["The weather is mild.", "I checked your calendar.", "You have two meetings.",
                           "The first is at ten."]):
        g.said(s, at=90.0 + i)
    assert g.fragment_is_echo("At ten.") and g.fragment_is_echo("Meetings.")
    assert not g.fragment_is_echo("Mild.")              # four sentences back
    assert not g.fragment_is_echo("Stop.") and not g.fragment_is_echo("")
    assert not g.fragment_is_echo("you have two")       # three words: judge() decides those


def test_turn_note_marks_echo_heard_while_idle_once():
    g = EchoGuard(window_s=300, clock=lambda: 1000.0)
    g.said("I plan from the garden folder.", at=900.0)
    g.said("Tomatoes, peppers and a compost heap layer on top.", at=905.0)
    heard = "and a compost heap layer on top"
    m = g.match(heard)
    m.heard_at = 950.0                                  # the filter passed it then (the idle case)
    g.passed.append((heard, m))
    assert g.turn_note("What time is it?") == (None, None)
    note, record = g.turn_note("And a compost heap layer on top. What does that mean?")
    assert note == ('(Some of these words ("and a compost heap layer on top") match what you said 45 s ago; '
                    'the person may be quoting you, or the microphone heard your own voice.)')
    assert record == {"matched_words": 7, "heard_words": 7, "said_ago_s": 45.0,
                      "said": "and a compost heap layer on top"}
    assert g.turn_note("And a compost heap layer on top.") == (None, None)   # each passed transcript once


def test_filter_marks_when_idle_echo_passed_and_lets_the_bot_frames_be_watched():
    import asyncio

    from pipecat.frames.frames import BotStartedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    from local_voice.echo_guard import echo_filter

    g, _, _ = asyncio.run(_run_filter(False, "And a compost heap layer on top."))
    assert g.passed[0][1].heard_at is not None
    watched, pushed = [], []
    f = echo_filter(EchoGuard(), busy=lambda: False, watch=watched.append)

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append((frame, direction))
    f.push_frame = push
    bot = BotStartedSpeakingFrame()
    asyncio.run(f.process_frame(bot, FrameDirection.UPSTREAM))
    assert watched == [bot] and pushed == [(bot, FrameDirection.UPSTREAM)]


# ------------------------------------------------------------------- the verdict before the STT's push (the caption leak)

ECHO = "and a compost heap layer on top"


class FirstPushes:
    """What Pipecat's RTVI observer captions from (rtvi/observer.py 1.12: a transcript at its first push): every
    transcript pushed, by whom."""

    def __new__(cls):
        from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
        from pipecat.observers.base_observer import BaseObserver

        class _Obs(BaseObserver):
            def __init__(self):
                super().__init__(observe_every_push=False)
                self.seen: list[tuple[str, bool, str]] = []

            async def on_push_frame(self, data):
                f = data.frame
                if isinstance(f, (TranscriptionFrame, InterimTranscriptionFrame)):
                    self.seen.append((type(data.source).__name__, isinstance(f, TranscriptionFrame), f.text))
        return _Obs()


async def _stt_through_filter(stream: bool, script: list[str], busy: bool):
    """The STT service as runtime.make_stt builds it (a fake engine) with the echo filter after it, wired as
    pipeline.build_session wires them; one VAD segment of loud audio per scripted transcript."""
    from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame, run_test

    from local_voice.echo_guard import echo_filter
    from local_voice.engines.fakes import FakeStreamingTranscriber, FakeTranscriber
    from local_voice.mlx_worker import MLXWorker
    from local_voice.services.stt import SegmentedMLXSTTService, StreamingMLXSTTService

    g = EchoGuard(window_s=300)
    g.said(REPLY)
    worker = MLXWorker(use_mlx=False)
    if stream:
        stt = StreamingMLXSTTService(engine=FakeStreamingTranscriber({"script": script, "words_per_s": 8}),
                                     worker=worker, preroll_s=0.0, tail_pad_s=0.0)
    else:
        stt = SegmentedMLXSTTService(engine=FakeTranscriber({"script": script}), worker=worker,
                                     trailing_silence_secs=0.0)
    told = []

    async def on_echo(final, rest):
        told.append((final, rest))
    filt = echo_filter(g, busy=lambda: busy, on_echo=on_echo)
    stt.transcript_gate = filt.gate
    obs = FirstPushes()
    loud = (0.3 * 32767 * np.sin(2 * np.pi * 300 * np.arange(320) / 16000)).astype("<i2").tobytes()
    frames = []
    for _ in script:          # 1.5 s of speech at a microphone's pace (in 0.1 s steps), so interims come as it grows
        frames.append(VADUserStartedSpeakingFrame())
        for _ in range(15):
            frames += [InputAudioRawFrame(loud, 16000, 1) for _ in range(5)] + [SleepFrame(0.1)]
        frames += [VADUserStoppedSpeakingFrame(), SleepFrame(0.4)]
    down, _ = await run_test(Pipeline([stt, filt]), frames_to_send=frames, observers=[obs])
    return g, filt, obs.seen, told, [f for f in down if type(f).__name__.endswith("TranscriptionFrame")]


async def test_dropped_echo_never_leaves_the_live_stt_so_no_caption_sees_it():
    """Before, the STT pushed the echo and the filter after it dropped it: Pipecat's RTVI observer had already sent it to
    the browser page as the person's caption. Now the STT asks the guard first: no final of the echo, and no interim of
    3 or more of its words, is pushed by anyone; the guard, the strategy (on_echo) and the log hear of it as before."""
    g, filt, seen, told, down = await _stt_through_filter(True, [ECHO], busy=True)
    assert not [s for s in seen if s[1]] and not [f for f in down if type(f).__name__ == "TranscriptionFrame"], \
        seen                                                                     # no final anywhere
    assert seen and all(len(words(t)) < 3 for _, _, t in seen), seen             # only the 1-2 word prefixes
    assert (False, "") in told                                                   # longer interims: echo, dropped
    assert filt.dropped_finals == 1 and [h for h, _ in g.dropped] == [ECHO] and told[-1] == (True, "")


async def test_the_persons_words_pass_judged_once_and_echo_cut_from_them():
    from local_voice.echo_guard import JUDGED

    heard = "Peppers beans herbs and a compost heap layer on top. What does that mean?"
    g, filt, seen, told, down = await _stt_through_filter(True, [heard, "what time is it now"], busy=True)
    finals = [t for src, final, t in seen if final]
    assert finals == ["What does that mean?", "what time is it now"], seen      # echo cut out before the push
    assert {src for src, _, _ in seen} == {"StreamingMLXSTTService"}             # nothing else pushed a transcript
    assert all(f.metadata.get(JUDGED) for f in down) and told[-1] == (True, "What does that mean?")
    # idle: the words go on, and the match is kept once (the filter does not judge the judged frame again)
    g, filt, seen, told, down = await _stt_through_filter(True, [ECHO], busy=False)
    assert [t for _, final, t in seen if final] == [ECHO] and len(g.passed) == 1 and not told


async def test_the_segmented_stt_asks_the_guard_too():
    g, filt, seen, told, down = await _stt_through_filter(False, [ECHO, "what time is it now"], busy=True)
    assert seen == [("SegmentedMLXSTTService", True, "what time is it now")] and filt.dropped_finals == 1


async def test_echo_during_a_reply_never_shows_on_the_browser_page(tmp_path):
    """The leak as the person would see it: the browser page (static/index.html, in a separate headless Chromium,
    tests/test_browser_page.py's set-up) gets RTVI's user-transcription for every transcript the STT pushes and shows the
    finals as the person's captions. A tone spoken 6 s into the microphone file, while a 12-sentence reply plays,
    stands for its echo; the scripted transcript is the reply's own words. The guard drops it before the STT pushes it,
    so the page never hears of it."""
    import asyncio

    import pytest

    from harness import free_port, rig
    from test_browser_page import CHROMIUM, one_utterance_wav, open_page

    if not CHROMIUM.exists():
        pytest.skip("needs the cached Playwright Chromium 1243")
    if not __import__("shutil").which("pi"):
        pytest.skip("pi is not on PATH")
    from playwright.async_api import async_playwright

    bport = free_port()
    story = " ".join(f"Sentence number {i} is here." for i in range(1, 13))
    echo = "sentence number 3 is here sentence number 4"
    async with rig(tmp_path, stt_script=["tell me a story", echo], stt_text="",
                   overrides={"server": {"browser": {"enabled": True, "port": bport}}}) as r:
        r.stub.script([{"text": story, "chunk_words": 4, "delay_ms": 20}])
        async with async_playwright() as p:
            browser, page = await open_page(p, bport, one_utterance_wav(tmp_path / "mic.wav", at=(0.5, 6.0)))
            await page.wait_for_function("document.body.innerText.includes('Sentence number 12 is here.')",
                                         timeout=30000)
            await asyncio.sleep(1.0)
            session = next(iter(r.orch.browser_sessions.values()))
            dropped = [h for h, _ in session.echo_guard.dropped]
            users = await page.evaluate("Array.from(document.querySelectorAll('.msg.user')).map(e => e.textContent)")
            rtvi = await page.evaluate("window.__rtvi")
            await browser.close()
    finals = [m["data"]["text"] for m in rtvi if m.get("type") == "user-transcription" and m["data"].get("final")]
    assert dropped == [echo], dropped                     # the echo was heard, and dropped while the reply played
    assert finals == ["tell me a story"] and users == ["tell me a story"], (finals, users)
    assert len(r.stub.requests()) == 1                     # and it started no turn
