"""The orchestrator's side of the brain's interface (brain/TURN_LOG.md v1): the turn log it writes (§1) and the spoken
digests it reads (§2). Records are checked with the brain's own parser, imported by path from brain/, so the two sides
cannot drift; everything is written under the test's tmp dir."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from harness import rig
from local_voice.client import tone_pcm

pytestmark = pytest.mark.needs_pi
BRAIN = Path(__file__).resolve().parents[2] / "brain" / "local_voice_brain"


def brain_module(name: str):
    key = f"brain_{name}_under_test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, BRAIN / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def read_turns(tmp_path: Path):
    tl = brain_module("turnlog")
    day = datetime.now().astimezone().strftime("%Y-%m-%d")
    turns, problems = tl.read_day(tmp_path / "turns", day)
    raw = [json.loads(line) for line in (tmp_path / "turns" / f"{day}.jsonl").read_text().splitlines()]
    return turns, problems, raw


async def test_spoken_interrupted_and_typed_turns_are_logged_as_the_brain_reads_them(tmp_path):
    long = " ".join(f"Sentence number {i} is here." for i in range(1, 25))
    async with rig(tmp_path, stt_script=["tell me a long story", "stop, what time is it"], tts_rtf=0.5) as r:
        r.stub.script([{"text": long, "chunk_words": 4, "delay_ms": 20},
                       {"tool_calls": [{"name": "read", "arguments": {"path": "README.md"}}]}, {"text": "It is noon."},
                       {"text": "Typed answer."}])
        c = await r.client(device="phone", client="iphone")
        session = c.welcome["session"]
        # 1: a spoken turn that is interrupted after about a second of its reply
        await c.speak(tone_pcm(0.8))
        quiet = asyncio.create_task(c.silence(3.0))
        await c.wait_for(lambda m: m.get("t") == "audio_start", timeout=10)
        while True:
            rep = c.current
            if rep is not None and rep.first_audio_at is not None and c.now() - rep.first_audio_at > 1.0:
                break
            await asyncio.sleep(0.02)
        quiet.cancel()
        t_barge = c.now()
        # 2: the barge-in itself is the second spoken turn; its reply calls a tool and completes
        await c.speak(tone_pcm(0.8), tail_s=0.2)
        quiet = asyncio.create_task(c.silence(4.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and m.get("reply_id") != rep.id, timeout=15,
                         since=t_barge)
        quiet.cancel()
        # 3: a typed turn
        t_typed = c.now()
        await c.send({"t": "text", "text": "and one typed by hand"})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10, since=t_typed)
        await asyncio.sleep(1.2)       # the interrupted turn's record waits up to heard_wait_s for played_ms
        await c.close()
    turns, problems, raw = read_turns(tmp_path)
    assert problems == [] and len(turns) == 3 and len(raw) == 3
    a, b, typed = turns
    assert [t.turn for t in turns] == [1, 2, 3] and {t.session for t in turns} == {session}
    assert all(t.client == "iphone" and t.space == "home" and t.mode == "conversation" for t in turns)
    assert all(t.t_start.utcoffset() is not None and t.t_end and t.t_end >= t.t_start for t in turns)
    # the interrupted turn: the words heard come from played_ms, a proper prefix of the reply
    assert a.input == "voice" and a.user_text == "tell me a long story" and a.interrupted
    assert a.reply_text.startswith("Sentence number 1 is here.") and a.heard_text
    assert a.reply_text.startswith(a.heard_text) and len(a.heard_text) < len(a.reply_text)
    assert a.spoken_text == a.heard_text
    # the next spoken turn: the person's words only, never the heard-text note the prompt carried to Pi
    assert b.user_text == "stop, what time is it" and "interrupted" not in b.user_text and not b.interrupted
    assert b.tools == (("read", True),) and b.reply_text == "It is noon." and b.heard_text is None
    assert r.user_texts()[1].startswith("(You were interrupted.")      # ... which Pi did get
    # the typed turn
    assert typed.input == "text" and typed.user_text == "and one typed by hand" and typed.reply_text == "Typed answer."
    assert all(rec["atlas"] is None and rec["tone"] is None and rec["v"] == 1 for rec in raw)


async def test_a_held_turn_is_logged_with_its_own_start_and_a_resume_note_is_not(tmp_path):
    async with rig(tmp_path, stt_text="are you there") as r:
        r.stub.script([{"text": "Here now."}])
        r.gate.set("held")
        await asyncio.sleep(0.3)
        c = await r.client()
        t0 = datetime.now().astimezone()
        await c.speak(tone_pcm(0.8))
        quiet = asyncio.create_task(c.silence(8.0))
        await c.wait_for(lambda m: m.get("t") == "state" and m.get("v") == "held", timeout=10)
        await asyncio.sleep(1.5)
        r.gate.set("open")
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and c.replies.get(m["reply_id"])
                         and "".join(c.replies[m["reply_id"]].text).startswith("Here"), timeout=15)
        quiet.cancel()
        await c.close()
    turns, problems, _ = read_turns(tmp_path)
    assert problems == [] and len(turns) == 1
    t = turns[0]
    assert t.user_text == "are you there" and t.input == "voice" and t.reply_text == "Here now."
    assert abs((t.t_start - t0).total_seconds()) < 1.5          # when the person spoke, not when the hold ended


def queue_digest(tmp_path: Path, day: str, text: str, *, expires_in_days: float = 2.0) -> Path:
    ob = brain_module("outbox")
    now = datetime.now().astimezone()
    return ob.queue(tmp_path / "brain", day=day, text=text, now=now - timedelta(hours=1), expiry_days=expires_in_days,
                    cites=[{"session": "s-x", "turn": 1}])


async def test_pending_digests_are_spoken_once_after_the_first_completed_turn_of_a_new_session(tmp_path):
    queue_digest(tmp_path, "2026-10-03", "From Saturday: keep replies short.")
    queue_digest(tmp_path, "2026-10-04", "From Sunday: Kokoro stays the fallback voice.")
    queue_digest(tmp_path, "2026-10-01", "This one expired.", expires_in_days=0.01)
    async with rig(tmp_path, stt_text="good morning") as r:
        r.stub.script([{"text": "Morning."}], default={"text": "Again."})
        import httpx
        async with httpx.AsyncClient() as h:
            assert (await h.get(r.status_url)).json()["digest_pending"] is True
        c = await r.client(device="mac")
        spoken_before = list(r.tts.spoken)
        assert not any("From Saturday" in s for s in spoken_before)       # never before the first request
        await c.speak(tone_pcm(0.6))
        quiet = asyncio.create_task(c.silence(12.0))
        await c.wait_for(lambda m: m.get("t") == "end_of_turn" and c.replies.get(m["reply_id"])
                         and "".join(c.replies[m["reply_id"]].text).startswith("Morning"), timeout=10)
        for _ in range(100):
            if len(list((tmp_path / "brain" / "outbox" / "delivered").glob("*.json"))) == 2:
                break
            await asyncio.sleep(0.1)
        quiet.cancel()
        said = [s for s in r.tts.spoken if s.startswith("From ")]
        assert said == ["From Saturday: keep replies short.", "From Sunday: Kokoro stays the fallback voice."]
        delivered = sorted((tmp_path / "brain" / "outbox" / "delivered").glob("*.json"))
        assert [p.stem for p in delivered] == ["2026-10-03", "2026-10-04"]
        assert all(json.loads(p.read_text())["delivered"] for p in delivered)
        assert [p.stem for p in (tmp_path / "brain" / "outbox").glob("*.json")] == ["2026-10-01"]   # expired: left
        assert all("From Saturday" not in json.dumps(q["body"]) for q in r.stub.requests())          # not in context
        # a second session (another device) has nothing left to say
        c2 = await r.client(device="phone")
        await c2.speak(tone_pcm(0.6))
        quiet2 = asyncio.create_task(c2.silence(4.0))
        await c2.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=10)
        await asyncio.sleep(1.0)
        quiet2.cancel()
        assert [s for s in r.tts.spoken if s.startswith("From ")] == said
        await c.close()
        await c2.close()
