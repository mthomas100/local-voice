"""The spoken cap (local_voice/spoken_cap.py): replies said up to the last sentence that ends within the cap, every
sentence held until it ends, an over-long first sentence ended at a clause break."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness import rig
from local_voice.client import tone_pcm
from local_voice.spoken_cap import SpokenCap, clause_cut, sentence_end, wants_more, words

# qwen38's answers to "What does a heat pump do?" (test runs, 2026-10-05): off-1 was this one
# sentence, 49 words and 16.4 s of speech; HEAT_PUMP is three sentences written like it
ONE_SENTENCE = ("A heat pump moves heat instead of making it — it works like an air conditioner, pulling warmth out of one "
                "place and dumping it somewhere else, so it can heat your home in winter and cool it in summer, usually "
                "more efficiently than a furnace or electric heater.")
HEAT_PUMP = ("A heat pump moves heat instead of making it — it works like an air conditioner, pulling warmth out of one "
             "place and dumping it somewhere else, so it can heat or cool. In winter it pulls heat from the outside air, "
             "even cold air, and pumps it indoors. That's why it uses far less electricity than a heater.")


def stream(cap: SpokenCap, text: str, step: int = 3) -> list[str]:
    """Feed the text a few characters at a time, as the model's deltas come; then the end of the run."""
    out = [cap.feed(text[i:i + step]) for i in range(0, len(text), step)]
    return [*out, cap.boundary()]


def test_sentence_ends():
    assert sentence_end("Hello there. How") == len("Hello there.")
    assert sentence_end("Hello there.") is None                         # a full stop at the end waits for what follows
    assert sentence_end("Hello there!") == len("Hello there!")         # ! ? … are trusted at the end of the text so far
    assert sentence_end("I run from ~/.") is None                       # an early live test: "~/.pi/agent"
    assert sentence_end("I run from ~/.pi/agent: a file. Then") == len("I run from ~/.pi/agent: a file.")
    assert sentence_end("It costs 3.") is None                          # may go on "5 dollars"
    assert sentence_end("It costs 3.5 dollars. Then") == len("It costs 3.5 dollars.")
    assert sentence_end("Climb Mt. Elbrus first. Then") == len("Climb Mt. Elbrus first.")
    assert sentence_end("Bring snacks, e.g. nuts. Then") == len("Bring snacks, e.g. nuts.")
    assert sentence_end("You need:\n1. Eggs\n2. Milk.\nThen") == len("You need:\n1. Eggs\n2. Milk.")
    assert sentence_end('He said "stop." Then') == len('He said "stop."')
    assert sentence_end("Really?! Yes") == len("Really?!")
    assert sentence_end("No. Never") == len("No.")
    assert sentence_end("Hello wor") is None


def test_a_short_reply_is_said_whole():
    cap = SpokenCap(40)
    out = stream(cap, "Sure. Paris is the capital of France.")
    assert "".join(out) == "Sure. Paris is the capital of France."
    assert out[1] == "Sure."                                            # a sentence goes out as soon as it ends
    assert not cap.cut and cap.said == 7


def test_a_long_reply_is_cut_at_the_last_sentence_within_the_cap():
    cap = SpokenCap(40)
    said = "".join(stream(cap, HEAT_PUMP))
    first = HEAT_PUMP.split(". ")[0] + "."
    second = " In winter it pulls heat from the outside air, even cold air, and pumps it indoors."
    assert words(first) == 32 and words(second) == 16                  # the dash is not a word
    assert said == first                                                # 32 + 16 > 40: cut before the second
    assert cap.cut and cap.spoken == said
    wide = SpokenCap(50)
    assert "".join(stream(wide, HEAT_PUMP)) == first + second and wide.cut


def test_a_first_sentence_over_the_cap_ends_at_a_clause_break_within_it():
    cap = SpokenCap(40)
    assert words(ONE_SENTENCE) == 48
    assert "".join(stream(cap, ONE_SENTENCE)) == (
        "A heat pump moves heat instead of making it — it works like an air conditioner, pulling warmth out of one "
        "place and dumping it somewhere else, so it can heat your home in winter and cool it in summer.")
    assert cap.cut and cap.said == 39
    assert clause_cut("Well, " + "word " * 30 + "end.", 20) is None   # "Well." alone is no answer: said whole
    no_break = SpokenCap(5)
    out = "".join(stream(no_break, "This first sentence is a good deal longer than five words. And this is not said."))
    assert out == "This first sentence is a good deal longer than five words." and no_break.cut


def test_nothing_after_the_cut_and_no_cap_says_everything():
    cap = SpokenCap(8)
    stream(cap, "One two three four. Five six seven eight nine ten. Eleven.")
    assert cap.spoken == "One two three four." and cap.cut
    assert cap.feed(" More text.") == "" and cap.boundary() == ""
    off = SpokenCap(0)
    assert "".join(stream(off, HEAT_PUMP)) == HEAT_PUMP and not off.cut


def test_a_tool_call_ends_a_block_and_the_answer_after_it_says_something():
    cap = SpokenCap(12)
    assert "".join(stream(cap, "Let me check the list.")) == "Let me check the list."   # then a tool call: boundary()
    out = [cap.feed(x) for x in ("You have ", "eggs, oat milk and coffee beans", " on it, and a note about the ",
                                 "party on Friday at eight. ", "Want the rest?")]
    # 5 said, 7 left: the answer's first sentence (16 words) is cut at its clause break after 9 words, not dropped
    assert "".join(out) == "You have eggs, oat milk and coffee beans on it." and cap.cut


def test_the_last_sentence_after_a_number_is_judged_at_the_end():
    cap = SpokenCap(20)
    out = [cap.feed("It is 3."), cap.feed("5 degrees. "), cap.feed("Bring a coat")]
    assert out == ["", "It is 3.5 degrees.", ""]
    assert cap.boundary() == " Bring a coat" and not cap.cut


def test_markdown_is_not_counted_as_words_and_numbers_count_as_said():
    assert words("**The Weight of Light**\n\n---\n\n- one\n- two") == 6
    assert words("Elbrus, at 5,642 meters") == 2 + 6 + 1
    assert words("$10,000 at 7% a year") == 3 + 1 + 3 + 1 + 1     # "ten thousand dollars at seven percent a year"


def test_asking_for_more():
    for s in ("Tell me a long story about a lighthouse keeper.", "Go on.", "Tell me more about that.",
              "Explain it in detail.", "Read me the whole thing.", "Can you walk me through it?", "Keep going"):
        assert wants_more(s), s
    for s in ("How long should I boil an egg?", "What does a heat pump do?", "What's on my shopping list?",
              "Hi there, how is your day going?", "What does my knowledge base say about the whole gate?"):
        assert not wants_more(s), s


async def test_a_long_reply_is_cut_with_an_offer_and_a_yes_goes_on_from_there(tmp_path):
    """Whole spoken turns against the stub LLM: the cut, the offer, the note on the next prompt, the long cap after a
    yes, and the turn log's reply_text as said."""
    ov = {"agent": {"spoken_cap": {"words": 12, "long_words": 100, "offer": "Want me to go on?"}}}
    rest = "In winter it pulls heat from the outside air, even cold air. That is why it is cheap to run."
    async with rig(tmp_path, stt_script=["what does a heat pump do", "yes"], overrides=ov) as r:
        r.stub.script([{"text": HEAT_PUMP, "delay_ms": 5}, {"text": rest, "delay_ms": 5}])
        c = await r.client()
        words_seen = []
        for _ in range(2):
            at = c.now()
            await c.speak(tone_pcm(0.8))
            tail = asyncio.create_task(c.silence(10.0))
            await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=25, since=at)
            tail.cancel()
            words_seen.append(dict(r.orch.clients["test-client"].session.agent.last_reply or {}))
        await c.close()
        prompts = r.user_texts()
        turn_dir = Path(r.orch.cfg.turn_log_dir)
        rows = [json.loads(x) for p in sorted(turn_dir.glob("*.jsonl")) for x in p.read_text().splitlines()]
    first = "A heat pump moves heat instead of making it."      # the first sentence, cut at its dash (12-word cap)
    assert r.tts.spoken[1:3] == [first, "Want me to go on?"]
    assert words_seen[0] == {"written": 58, "said": 9, "cap": 12, "cut": True, "ack": None}
    assert prompts[1] == (f'(Your last reply was too long to say aloud. The user heard it only up to: "{first}", and was '
                          'then asked "Want me to go on?" If they want more, go on from there without repeating what '
                          'they heard.)\n\nyes')
    # "yes" after a cut reply gets the long cap: the whole continuation is said
    assert " ".join(r.tts.spoken[3:]) == rest
    assert words_seen[1] == {"written": 20, "said": 20, "cap": 100, "cut": False, "ack": None}
    assert [x["reply_text"] for x in rows] == [first, rest]
