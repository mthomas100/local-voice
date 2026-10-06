"""How often does a barge-in land while the model is still generating its reply? (brief step 4, 2026-10-05)

When the Pi run has settled before the person barges in, only the speech is cut and the next request extends ds4's
state as usual. When it has not, the run is aborted, the next request no longer extends ds4's state, and ds4 replays
the conversation from its nearest checkpoint (9-28 s on 2026-10-05). This measures the window: typed conversational
questions through the real server and models; per reply, when its audio started at the client, how long it played,
and when the run settled on the server. A barge-in t seconds into a reply lands mid-generation if t < settled - first
audio. Reported per reply, and as the share of all playback time inside that window (the chance for a barge-in at a
random moment) and the share of replies for barge-ins 0.5, 1 and 2 s in.

Run:  LV_MEASURE=1 .venv/bin/python -m pytest -m e2e tests/e2e/test_reply_timing.py -s
"""
from __future__ import annotations

import os
import statistics

import pytest

from e2e_fixtures import RUN_DIR, gpu_still_ours, kb_clone, orch, run_dir  # noqa: F401 - fixtures
from local_voice.client import V1Client
from test_turn import record

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio(loop_scope="module"),
              pytest.mark.skipif(not os.environ.get("LV_MEASURE"), reason="a measurement: set LV_MEASURE=1")]

QUESTIONS = ["What's a quick dinner I can make with eggs and spinach?", "How can I sleep better?",
             "What does a heat pump do?", "What's the difference between weather and climate?",
             "Tell me a fun fact about octopuses.", "How long should I boil an egg for a runny yolk?",
             "What's a good way to start learning Spanish?", "Why is the sky blue?",
             "Suggest a name for a grey cat.", "What should I pack for a day hike?",
             "How does compound interest work?", "What's the tallest mountain in Europe?"]


async def test_reply_timing(orch):
    c = V1Client(orch.e2e_url, device="e2e-timing")
    await c.connect()
    rows = []
    try:
        for q in QUESTIONS:
            why = gpu_still_ours()
            if why:
                pytest.skip(f"stopping: {why}")
            at = c.now()
            await c.send({"t": "text", "text": q})
            end = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=90, since=at)
            r = c.replies.get(end.get("reply_id"))
            agent = orch.clients["e2e-timing"].session.agent
            if r is None or r.first_audio_at is None or agent.last_settled_at is None:
                continue
            first_audio = c.t0 + r.first_audio_at
            window = agent.last_settled_at - first_audio
            rows.append({"q": q, "reply": "".join(r.text), "audio_s": round(r.audio_s, 2),
                         "ttft_ms": round(agent.last_ttft_ms or 0), "gen_after_audio_s": round(window, 3),
                         "words": agent.last_reply})
    finally:
        await c.close()
    played = sum(x["audio_s"] for x in rows)
    inside = sum(min(max(0.0, x["gen_after_audio_s"]), x["audio_s"]) for x in rows)
    audio = sorted(x["audio_s"] for x in rows)
    summary = {"replies": len(rows), "median_audio_s": statistics.median(audio),
               "p90_audio_s": audio[min(len(audio) - 1, round(0.9 * (len(audio) - 1)))], "max_audio_s": audio[-1],
               "over_15_s": sum(a > 15 for a in audio),
               "median_words_said": statistics.median((x["words"] or {}).get("said", 0) for x in rows),
               "median_words_written": statistics.median((x["words"] or {}).get("written", 0) for x in rows),
               "cut": sum(bool((x["words"] or {}).get("cut")) for x in rows),
               "median_gen_after_audio_s": statistics.median(x["gen_after_audio_s"] for x in rows),
               "max_gen_after_audio_s": max(x["gen_after_audio_s"] for x in rows),
               "share_of_playback_mid_generation": round(inside / played, 3),
               **{f"replies_mid_generation_at_{t}s": sum(x["gen_after_audio_s"] > t for x in rows)
                  for t in (0.5, 1.0, 2.0)}}
    record("reply-timing", rows=rows, summary=summary)
    print(summary)
    assert rows
