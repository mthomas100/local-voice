"""The orchestrator end to end without any model: protocol v1 client -> pipeline -> Pi (against the stub LLM) -> fake
TTS -> client. These are the M1 behaviours that do not depend on a real model's output."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from harness import rig
from local_voice.client import V1Client, tone_pcm

pytestmark = pytest.mark.needs_pi

REPLY = "Hi there. The hold gate pauses model calls while a render runs."


async def test_a_spoken_turn_gets_a_spoken_reply(tmp_path):
    async with rig(tmp_path, stt_text="hello there") as r:
        r.stub.script([{"text": REPLY, "delay_ms": 5}])
        c = await r.client()
        assert c.welcome["space"] == "home" and c.welcome["state"] == "listening"
        eos = await c.speak(tone_pcm(1.0), tail_s=0.0)
        tail = asyncio.create_task(c.silence(4.0))
        tr = await c.wait_for(lambda m: m.get("t") == "transcript" and m.get("final"), timeout=10)
        assert tr["text"] == "hello there"
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15)
        tail.cancel()
        rep = c.replies[end["reply_id"]]
        assert rep.bytes > 0 and rep.first_audio_at is not None and rep.first_audio_at > eos
        assert "".join(rep.text).replace(" ", "") == REPLY.replace(" ", "")
        assert r.user_texts()[-1] == "hello there"
        assert r.tts.spoken[0] == r.orch.cfg.busy_notice          # rendered once at startup
        assert r.tts.spoken[1:] == ["Hi there.", "The hold gate pauses model calls while a render runs."]
        kinds = [m["t"] for _, m in c.messages]
        assert kinds.index("audio_start") < kinds.index("audio_end") < kinds.index("end_of_turn")
        assert "interrupt" not in kinds            # turn-start interruptions are not forwarded while nothing plays
        states = [m["v"] for _, m in c.messages if m["t"] == "state"]
        assert {"thinking", "speaking", "listening"} <= set(states)
        assert all(a != b for a, b in zip(states, states[1:])), states   # every state came twice (a real-server run)
        async with httpx.AsyncClient() as h:
            st = (await h.get(r.status_url)).json()
        assert st["space"] == "home" and st["hold"]["phase"] == "open" and st["clients"][0]["device"] == "test-client"
        line = r.orch.clients["test-client"].session.latency.lines[-1]
        assert line["hold_open"] is True and line["hold_phase"] == "open" and line["load1"] >= 0
        assert line["eos_to_first_audio_ms"] > 0 and "stages" in line
        assert line["stages"]["llm_idle_s"] is None        # the process's first LLM run: no idle time to report
        await c.close()


async def test_a_tool_turn_is_acknowledged_then_answered(tmp_path):
    async with rig(tmp_path, stt_text="read me the readme") as r:
        r.stub.script([{"tool_calls": [{"name": "read", "arguments": {"path": "README.md"}, "split": 4}], "delay_ms": 30},
                       {"text": "It says scratch README.", "delay_ms": 5}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        tail = asyncio.create_task(c.silence(6.0))
        start = await c.wait_for(lambda m: m.get("t") == "tool" and m.get("phase") == "start", timeout=10)
        assert start["name"] == "read" and start["label"] == "reading a file"
        end_tool = await c.wait_for(lambda m: m.get("t") == "tool" and m.get("phase") == "end", timeout=10)
        assert end_tool["ok"] is True
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15)
        tail.cancel()
        # the canned acknowledgement is spoken by the orchestrator, before the answer, and kept out of Pi's context
        assert r.tts.spoken[1:] == ["Let me check.", "It says scratch README."]
        reqs = r.stub.requests()
        assert len(reqs) == 2 and "Let me check" not in json.dumps(reqs[1]["body"]["messages"])
        rep = c.replies[end["reply_id"]]
        assert "".join(rep.text).startswith("Let me check.")
        await c.close()


async def test_the_models_words_before_a_tool_call_are_said_first_and_stand_for_the_acknowledgement(tmp_path):
    """Speech in the order written: the sentence the model wrote before its tool call is said before anything the
    orchestrator says (Pipecat speaks a TTSSpeakFrame at once, and the code-fence splitter held the sentence for
    lookahead: "Let me check.Let me check current prices rather than guessing.", 2026-10-05 16:04), and it stands for
    the acknowledgement, so no canned "Let me check." follows it. Without any words first, the canned one is said."""
    read = {"name": "read", "arguments": {"path": "README.md"}}
    async with rig(tmp_path, stt_script=["read me the readme", "and again"]) as r:
        # the answer after the tool takes 2 s to start: the words before the call must not wait for it
        r.stub.script([{"text": "I'll read the readme for you.", "tool_calls": [read], "delay_ms": 5},
                       {"text": "It says scratch README.", "delay_ms": 5, "pre_delay_ms": 2000},
                       {"tool_calls": [read], "delay_ms": 5}, {"text": "Still scratch README.", "delay_ms": 5}])
        c = await r.client()
        hows = []
        for _ in range(2):
            at = c.now()
            await c.speak(tone_pcm(0.8))
            tail = asyncio.create_task(c.silence(6.0))
            await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=at)
            tail.cancel()
            hows.append(r.orch.clients["test-client"].session.agent.last_reply["ack"])
        await c.close()
    assert r.tts.spoken[1:] == ["I'll read the readme for you.", "It says scratch README.", "Let me check.",
                                "Still scratch README."]
    assert hows == ["model", "canned"]
    said = {m["delta"].strip(): at for at, m in c.messages if m.get("t") == "reply_text"}
    assert said["It says scratch README."] - said["I'll read the readme for you."] > 1.5


async def test_barge_in_interrupts_fast_and_the_next_prompt_knows_what_was_heard(tmp_path):
    long = " ".join(f"Sentence number {i} is here." for i in range(1, 30))
    async with rig(tmp_path, stt_script=["tell me a long story", "stop, what time is it"], tts_rtf=0.5) as r:
        r.stub.script([{"text": long, "chunk_words": 4, "delay_ms": 20}, {"text": "It is noon."}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        quiet = asyncio.create_task(c.silence(3.0))
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=10)
        while True:   # let roughly a second of the reply play
            rep = c.current
            if rep is not None and rep.first_audio_at is not None and c.now() - rep.first_audio_at > 1.0:
                break
            await asyncio.sleep(0.02)
        quiet.cancel()
        speech_start = c.now()
        barge = asyncio.create_task(c.speak(tone_pcm(0.8), tail_s=1.5))
        intr = await c.wait_for(lambda m: m.get("t") == "interrupt", timeout=5, since=speech_start)
        intr_at = next(at for at, m in c.messages if m is intr)
        assert intr["reply_id"] == rep.id
        assert (intr_at - speech_start) < 0.30, f"interrupt came {1000 * (intr_at - speech_start):.0f} ms after speech"
        # the reply's audio stopped at the first speech frame (bargein.py), before the VAD confirmed the interruption
        stop = rep.silent_from(speech_start)
        assert stop is not None and stop - speech_start < 0.15 and stop < intr_at, (stop - speech_start, intr_at - speech_start)
        pause = r.orch.clients["test-client"].session.reply_pause
        assert [e.kind for e in pause.events][-2:] == ["paused", "confirmed"]
        await barge
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") != rep.id, timeout=15,
                         since=intr_at)
        last = r.user_texts()[-1]
        assert last.startswith("(You were interrupted. The user heard only this much of your last reply: \"Sentence number 1")
        assert last.endswith("stop, what time is it")
        heard = last.split('"')[1]
        # played_ms decides the words: about a second of a 0.03 s/char fake voice, nowhere near the whole reply
        assert 3 <= len(heard.split()) < len(long.split()) / 2
        assert r.tts.closed_early >= 1                # the sentence being synthesized was closed, not finished
        idle = r.orch.clients["test-client"].session.latency.lines[-1]["stages"]["llm_idle_s"]
        assert idle is not None and idle >= 0         # the second run says how long the LLM sat idle before it
        await c.close()


def _assistant_texts(body: dict) -> list[str]:
    out = []
    for m in body["messages"]:
        if m.get("role") == "assistant":
            c = m.get("content")
            out.append(c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict)))
    return out


async def _barge_in_mid_generation(r, after_s: float = 0.0):
    """Turn 1 streams its reply word by word; the person speaks over it `after_s` into its audio, while the rest is
    still being written; turn 2 follows. Returns the client and the reply that was cut off."""
    c = await r.client()
    await c.speak(tone_pcm(0.8))
    quiet = asyncio.create_task(c.silence(5.0))
    await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=10)
    rep = c.current
    while rep.first_audio_at is None or c.now() - rep.first_audio_at < after_s:
        await asyncio.sleep(0.01)
    quiet.cancel()
    at = c.now()
    barge = asyncio.create_task(c.speak(tone_pcm(0.8), tail_s=2.5))
    await c.wait_for(lambda m: m.get("t") == "interrupt", timeout=5, since=at)
    await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") != rep.id, timeout=20, since=at)
    await barge
    return c, rep


async def test_a_barge_in_mid_generation_lets_the_run_finish_silently_and_keeps_it(tmp_path):
    """brief step 4: the run is not aborted, so the next request carries the whole first reply and extends what the
    LLM server already holds (ds4 replays its recurrent state from a checkpoint otherwise: 9-28 s on 2026-10-05)."""
    first = "Sure thing, here we go. " + " ".join(f"word{i}" for i in range(1, 25)) + "."
    async with rig(tmp_path, stt_script=["tell me something", "stop, what time is it"], tts_rtf=0.5) as r:
        r.stub.script([{"text": first, "delay_ms": 60}, {"text": "It is noon."}])
        c, rep = await _barge_in_mid_generation(r, after_s=0.5)
        assert r.orch.clients["test-client"].session.agent.silent_runs == {"settled": 1, "aborted": 0}
        reqs = r.stub.requests()
        assert len(reqs) == 2
        assert _assistant_texts(reqs[1]["body"])[-1].replace(" ", "") == first.replace(" ", "")
        last = r.user_texts()[-1]
        assert last.startswith("(You were interrupted. The user heard only this much of your last reply: \"Sure")
        assert last.endswith("stop, what time is it")
        assert not any("word24" in t for t in r.tts.spoken)      # the silent remainder was never spoken
        assert "word24" not in "".join(rep.text)
        await c.close()


async def test_a_silent_run_still_waiting_on_the_llm_is_never_aborted(tmp_path):
    """The person carries on talking while the run waits on the LLM (a replay after an earlier abort, here a 3.5 s
    stand-in). It is left to finish even past wait_s: an abort during a prefill makes ds4 delete the checkpoint the
    prefill started from, and the next request prefilled 15,738 tokens from zero (e2e 2026-10-05 14:15)."""
    async with rig(tmp_path, stt_script=["stop", "what is the capital of italy"], tts_rtf=0.5) as r:
        r.stub.script([{"text": "Okay.", "pre_delay_ms": 3500}, {"text": "Rome."}])
        c = await r.client()
        await c.speak(tone_pcm(0.6))
        await c.silence(0.7)                 # the turn ends and its run starts waiting on the LLM
        at = c.now()
        await c.speak(tone_pcm(0.8))         # the person carries on
        tail = asyncio.create_task(c.silence(15.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=20, since=at)
        tail.cancel()
        reqs = r.stub.requests()
        assert len(reqs) == 2
        assert _assistant_texts(reqs[1]["body"])[-1] == "Okay."     # the first run finished, it was not aborted
        assert r.orch.clients["test-client"].session.agent.silent_runs == {"settled": 1, "aborted": 0}
        assert r.user_texts()[-1].endswith("what is the capital of italy")
        await c.close()


async def test_a_barge_in_mid_generation_still_aborts_a_long_reply(tmp_path):
    first = "Sure. " + " ".join(f"word{i}" for i in range(1, 25)) + "."
    async with rig(tmp_path, stt_script=["tell me something", "stop, what time is it"], tts_rtf=0.5,
                   overrides={"agent": {"barge_in_finish": {"max_chars": 5}}}) as r:
        r.stub.script([{"text": first, "delay_ms": 60}, {"text": "It is noon."}])
        c, _rep = await _barge_in_mid_generation(r, after_s=0.3)
        reqs = r.stub.requests()
        assert len(reqs) == 2
        assert not any("word24" in t for t in _assistant_texts(reqs[1]["body"]))   # Pi dropped the aborted reply
        assert r.user_texts()[-1].endswith("stop, what time is it")
        await c.close()


async def test_a_barge_in_while_a_tool_call_is_pending_aborts_the_run(tmp_path):
    async with rig(tmp_path, stt_script=["read me the readme", "stop, never mind"], tts_rtf=0.5) as r:
        r.stub.script([{"tool_calls": [{"name": "read", "arguments": {"path": "README.md"}, "split": 8}],
                        "delay_ms": 250},
                       {"text": "Okay."}])
        c, _rep = await _barge_in_mid_generation(r, after_s=0.1)
        reqs = r.stub.requests()
        assert len(reqs) == 2, [x["body"]["messages"][-1] for x in reqs]   # the tool never ran, no second model call
        assert "tool" not in [m.get("role") for m in reqs[1]["body"]["messages"]]
        assert r.user_texts()[-1].endswith("stop, never mind")
        await c.close()


async def test_a_short_noise_over_a_reply_pauses_it_and_the_reply_carries_on(tmp_path):
    """A cough-length sound over a reply (shorter than the VAD's start_secs) stops the reply's audio at once and the
    reply then carries on from where it stopped: no interrupt, nothing lost, the next prompt has no heard-text note."""
    reply = " ".join(f"Sentence number {i} is here." for i in range(1, 8))
    async with rig(tmp_path, stt_script=["tell me a story", "and then"], tts_rtf=0.5) as r:
        r.stub.script([{"text": reply, "chunk_words": 4, "delay_ms": 20}, {"text": "The end."}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        quiet = asyncio.create_task(c.silence(20.0))
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=10)
        while True:
            rep = c.current
            if rep is not None and rep.first_audio_at is not None and c.now() - rep.first_audio_at > 1.0:
                break
            await asyncio.sleep(0.02)
        quiet.cancel()
        noise_at = c.now()
        await c.stream(tone_pcm(0.1))                 # 100 ms: 3 VAD frames, under the 6 that start_secs 0.2 needs
        tail = asyncio.create_task(c.silence(20.0))
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") == rep.id, timeout=20,
                               since=noise_at)
        tail.cancel()
        assert not c.of_type("interrupt", since=noise_at)
        pause = r.orch.clients["test-client"].session.reply_pause
        kinds = [e.kind for e in pause.events]
        assert kinds == ["paused", "resumed"], kinds
        held = pause.events[-1].held_ms
        assert 350 <= held <= 700, held
        stop = rep.silent_from(noise_at)
        assert stop - noise_at < 0.15, f"the reply kept playing {1000 * (stop - noise_at):.0f} ms into the noise"
        gaps = [(b[0] - a[1]) for a, b in zip(rep.segments, rep.segments[1:])]
        assert max(gaps) >= 0.3, rep.segments       # the person heard the reply stop, then carry on
        # nothing was lost: every sentence's text and audio reached the client
        assert "".join(rep.text).replace(" ", "") == reply.replace(" ", "")
        assert abs(rep.audio_s - sum(e - s for s, e in rep.segments)) < 1e-6
        assert end["reply_id"] == rep.id
        await c.close()


async def test_typed_text_runs_a_turn(tmp_path):
    async with rig(tmp_path) as r:
        r.stub.script([{"text": "Typed reply."}])
        c = await r.client()
        await c.send({"t": "text", "text": "hello by keyboard"})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        assert r.user_texts()[-1] == "hello by keyboard" and r.tts.spoken[-1] == "Typed reply."
        await c.close()


async def test_a_closed_connections_turn_is_aborted_and_never_holds_up_the_next(tmp_path):
    """A connection closed mid-turn ends its turn like a barge-in. Pipecat leaves a processor's own tasks running at
    cleanup, so in the e2e of 2026-10-05 12:05 the closed connection's turn ran on, holding the space's lock, and the
    next connection's turn waited for it and was then aborted by its late cleanup."""
    async with rig(tmp_path) as r:
        r.stub.script([{"text": "Slowly. " * 40, "delay_ms": 100}, {"text": "Typed reply."}])
        a = await r.client(device="first")
        await a.send({"t": "text", "text": "tell me something long"})
        await a.wait_for(lambda m: m.get("t") == "state" and m.get("v") == "thinking", timeout=10)
        await asyncio.sleep(0.5)                     # the first reply is streaming (4 s in all)
        await a.close()
        b = await r.client(device="second")
        t0 = time.monotonic()
        await b.send({"t": "text", "text": "hello by keyboard"})
        await b.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        assert time.monotonic() - t0 < 3.0, "the second turn waited for the closed connection's turn"
        assert r.user_texts()[-1].endswith("hello by keyboard") and r.tts.spoken[-1] == "Typed reply."
        await b.close()


async def test_push_to_talk(tmp_path):
    async with rig(tmp_path, stt_text="push to talk works") as r:
        r.stub.script([{"text": "Yes it does."}])
        c = await r.client(mic="ptt")
        await c.send({"t": "start"})
        await c.speak(tone_pcm(0.6))
        await c.silence(0.1)                 # PROTOCOL.md: ~100 ms of silence before stop
        await c.send({"t": "stop"})
        tr = await c.wait_for(lambda m: m.get("t") == "transcript" and m.get("final"), timeout=10)
        assert tr["text"] == "push to talk works"
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        assert r.tts.spoken[-1] == "Yes it does."
        await c.close()


async def test_a_client_interrupt_mid_generation_lets_the_run_finish_silently(tmp_path):
    """A push-to-talk barge-in is the client's own `interrupt` (PROTOCOL.md), not the server's VAD, and it takes the
    same silent finish. A test run looked as if it did not: there the story had 171 tokens written when
    the person pressed talk, over finish_max_chars, so the run was aborted by design and ds4 replayed 2,879 tokens."""
    first = "Sure thing, here we go. " + " ".join(f"word{i}" for i in range(1, 25)) + "."
    async with rig(tmp_path, stt_script=["tell me something", "what time is it"], tts_rtf=0.5) as r:
        r.stub.script([{"text": first, "delay_ms": 60}, {"text": "It is noon."}])
        c = await r.client(mic="ptt")
        await c.send({"t": "start"})
        await c.speak(tone_pcm(0.6), tail_s=0.1)
        await c.send({"t": "stop"})
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=10)
        rep = c.current
        while rep.first_audio_at is None or c.now() - rep.first_audio_at < 0.5:
            await asyncio.sleep(0.01)
        at = c.now()
        rep.interrupted_at = at                  # the client flushes its own player first, then tells the server
        await c.send({"t": "interrupt", "reply_id": rep.id})
        await c.send({"t": "played_ms", "reply_id": rep.id, "ms": round(c.played_ms(rep, at))})
        await c.send({"t": "start"})
        await c.speak(tone_pcm(0.6), tail_s=0.1)
        await c.send({"t": "stop"})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") != rep.id, timeout=20, since=at)
        assert r.orch.clients["test-client"].session.agent.silent_runs == {"settled": 1, "aborted": 0}
        reqs = r.stub.requests()
        assert len(reqs) == 2
        assert _assistant_texts(reqs[1]["body"])[-1].replace(" ", "") == first.replace(" ", "")
        last = r.user_texts()[-1]
        # the played_ms right behind the client's interrupt decides what was heard (it used to be dropped: "before
        # the user heard any of your last reply")
        assert last.startswith("(You were interrupted. The user heard only this much of your last reply: \"Sure")
        assert last.endswith("what time is it")
        assert not c.of_type("interrupt", since=at)          # the client asked for it: nothing to echo
        await c.close()


async def test_markdown_in_a_reply_is_not_spoken_and_the_captions_keep_it(tmp_path):
    """Asked for a long story through the apps, qwen38 wrote a rule and a bold title despite the persona (a test run,
    2026-10-05); the voice spent 730 ms on the rule. What is synthesized has no Markdown; reply_text keeps it."""
    md = "Here you go.\n\n---\n\n**The Weight of Light**\n\nOnce upon a time, a *keeper* lived alone."
    async with rig(tmp_path) as r:
        r.stub.script([{"text": md, "delay_ms": 5}])
        c = await r.client()
        await c.send({"t": "text", "text": "tell me a story"})
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        spoken = r.tts.spoken[1:]
        assert spoken and not any(ch in s for s in spoken for ch in "*#`-"), spoken
        assert "The Weight of Light" in " ".join(spoken) and "a keeper lived" in " ".join(spoken), spoken
        assert "**The Weight of Light**" in "".join(c.replies[end["reply_id"]].text)
        await c.close()


async def test_no_pipecat_settings_errors_at_a_connection(tmp_path):
    """Pipecat logs an ERROR for each settings field a service leaves NOT_GIVEN, twice at every connection (a test
    run's serve.log, 2026-10-05): noise that could hide a real error."""
    from loguru import logger

    errors: list[str] = []
    sink = logger.add(lambda m: errors.append(str(m)), level="ERROR")
    try:
        async with rig(tmp_path) as r:
            r.stub.script([{"text": "Hi.", "delay_ms": 5}])
            c = await r.client()
            await c.send({"t": "text", "text": "hello"})
            await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
            await c.close()
    finally:
        logger.remove(sink)
    assert not [e for e in errors if "NOT_GIVEN" in e], errors


async def test_streaming_stt_partials_then_final(tmp_path):
    async with rig(tmp_path, stt="fake_stream", stt_text="what is on my calendar today") as r:
        r.stub.script([{"text": "Nothing today."}])
        c = await r.client()
        await c.speak(tone_pcm(1.6))
        tail = asyncio.create_task(c.silence(4.0))
        final = await c.wait_for(lambda m: m.get("t") == "transcript" and m.get("final"), timeout=10)
        tail.cancel()
        partials = [m["text"] for m in c.of_type("transcript") if not m["final"]]
        assert final["text"] == "what is on my calendar today"
        assert partials and all(final["text"].startswith(p) for p in partials)
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        assert r.user_texts()[-1] == "what is on my calendar today"
        # the session got the room's own quiet up to the VAD stop, then the tail pad, before it closed
        assert r.orch.runtime.stt_engine.last_session.quiet_tail_at_close_s >= 0.45   # 0.2 s without the pad
        await c.close()


async def test_a_held_gate_plays_the_notice_and_calls_no_model_until_it_opens(tmp_path):
    async with rig(tmp_path, stt_text="are you there") as r:
        r.stub.script([{"text": "I am here now."}])
        r.gate.set("held", "render the-film")
        await asyncio.sleep(0.3)
        c = await r.client()
        assert c.welcome["state"] == "held" and c.welcome["hold"] == "held"
        mlx_calls = r.orch.runtime.worker.calls
        await c.speak(tone_pcm(0.8))
        quiet = asyncio.create_task(c.silence(10.0))
        hold_msg = await c.wait_for(lambda m: m.get("t") == "hold" and m.get("phase") == "held", timeout=10)
        assert "render the-film" in hold_msg["why"]
        await c.wait_for(lambda m: m.get("t") == "state" and m.get("v") == "held", timeout=10)
        notice_end = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        notice = c.replies[notice_end["reply_id"]]
        assert notice.bytes > 0                                   # the pre-rendered notice played
        await asyncio.sleep(1.0)
        assert r.stub.requests() == []                            # no model call while held
        assert r.orch.runtime.worker.calls == mlx_calls           # and no speech inference
        assert r.orch.runtime.worker.refused >= 1                 # the STT was refused at the MLX thread's door
        assert r.stt.calls == []
        opened = c.now()
        r.gate.set("open")
        tr = await c.wait_for(lambda m: m.get("t") == "transcript" and m.get("final"), timeout=10, since=opened)
        assert tr["text"] == "are you there"                      # the kept audio, transcribed after the hold
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10, since=opened)
        quiet.cancel()
        assert r.user_texts() == ["are you there"] and r.tts.spoken[-1] == "I am here now."
        await c.close()


async def test_a_bad_hello_closes_with_1002_and_a_second_socket_takes_over_with_4409(tmp_path):
    import websockets

    async with rig(tmp_path) as r:
        async with websockets.connect(r.url) as ws:
            await ws.send(json.dumps({"t": "hello", "v": 2}))
            with pytest.raises(websockets.ConnectionClosed) as e:
                await ws.recv()
            assert e.value.rcvd.code == 1002
        a = await r.client(device="phone")
        b = await r.client(device="phone")
        await asyncio.sleep(0.5)
        assert a.close_code == 4409 and b.close_code is None
        assert b.welcome["session"] == a.welcome["session"]      # the same device resumes the same session
        await b.close()


async def test_a_peer_outside_the_allowed_logins_is_closed_with_4403(tmp_path, monkeypatch):
    async with rig(tmp_path) as r:
        async def deny(host, port):
            return False, "someone@else"
        monkeypatch.setattr(r.orch, "peer_allowed", deny)
        c = V1Client(r.url)
        with pytest.raises(ConnectionError):
            await c.connect()
        assert c.close_code == 4403


async def test_a_hold_that_starts_mid_reply_cuts_it_off_and_resumes_after(tmp_path):
    long = " ".join(f"Part {i} of the answer goes here." for i in range(1, 25))
    async with rig(tmp_path, stt_text="explain it all", tts_rtf=0.5) as r:
        r.stub.script([{"text": long, "chunk_words": 4, "delay_ms": 20}, {"text": "Picking up where I stopped."}])
        c = await r.client()
        await c.speak(tone_pcm(0.8))
        quiet = asyncio.create_task(c.silence(20.0))
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=10)
        await asyncio.sleep(1.0)
        held_at = c.now()
        r.gate.set("held", "render scene-3")
        await c.wait_for(lambda m: m.get("t") == "interrupt", timeout=5, since=held_at)
        await c.wait_for(lambda m: m.get("t") == "state" and m.get("v") == "held", timeout=5, since=held_at)
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10, since=held_at)   # the notice
        n_req = len(r.stub.requests())
        await asyncio.sleep(1.0)
        assert len(r.stub.requests()) == n_req == 1        # nothing sent to the model while held
        opened = c.now()
        r.gate.set("open")
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=opened + 0.01)
        quiet.cancel()
        last = r.user_texts()[-1]
        assert last.startswith("(You were interrupted. The user heard only this much of your last reply: \"Part 1")
        assert "GPU was taken for a render" in last
        assert r.tts.spoken[-1] == "Picking up where I stopped."
        await c.close()
