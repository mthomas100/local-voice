"""The validator decides what the model may write: citations of shown turns only, quotes only as exact source spans,
no verdicts, diagnoses or advice in the model's own words about the person."""
from __future__ import annotations

from datetime import datetime

import pytest

from local_voice_brain import validate as v
from local_voice_brain.turnlog import Turn


def mk(session="s1", n=1, user="hello there", reply="hi", heard=None, interrupted=False):
    return Turn(session=session, turn=n, t_start=datetime.fromisoformat("2026-10-04T09:00:00-07:00"), t_end=None,
                space="home", mode="conversation", client="test", input="voice", user_text=user, reply_text=reply,
                heard_text=heard, interrupted=interrupted, tools=(), atlas=None, raw={})


@pytest.mark.parametrize("source,quote,expected", [
    ("I've even thought about them", "I’ve even thought", "I've even thought"),        # curly vs straight
    ("Keep KOKORO as   the fallback", "keep kokoro as the fallback", "Keep KOKORO as   the fallback"),
    ("it was “fine” — really", 'was "fine" - really', "was “fine” — really"),
    ("wait… what", "wait... what", "wait… what"),
    ("the river walk", "  'the river walk'  ", "the river walk"),
    ("the river walk", "a lake walk", None),
    ("the river walk", "ri", None),                                                     # too short to be a quote
])
def test_find_span_returns_the_exact_source_text(source, quote, expected):
    assert v.find_span(source, quote) == expected


def test_parse_json_reply_tolerates_fences_and_chatter():
    assert v.parse_json_reply('```json\n{"facts": [], "spoken": "NO_REPLY"}\n```') == {"facts": [], "spoken": "NO_REPLY"}
    assert v.parse_json_reply('Sure! {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        v.parse_json_reply("no json here")
    with pytest.raises(ValueError):
        v.parse_json_reply('{"a": 1,,}')


@pytest.mark.parametrize("raw,expected", [
    ("NO_REPLY", None), (" no_reply. ", None), ("", None), (None, None),
    ("**Kokoro** is the `fallback`.", "Kokoro is the fallback ."),
])
def test_clean_spoken(raw, expected):
    assert v.clean_spoken(raw, 45) == expected


def test_clean_spoken_caps_words_at_a_sentence_end():
    s = "One two three four five six. Seven eight nine ten eleven twelve thirteen fourteen."
    assert v.clean_spoken(s, 10) == "One two three four five six."
    assert v.clean_spoken("a b c d e f g h i j k l", 5) == "a b c d e."


def test_validate_tech_keeps_cited_facts_and_drops_invented_evidence():
    handles = {"S1T1": mk(n=1, user="let's keep kokoro as the fallback voice"),
               "S1T2": mk(n=2, user="how long did it take", reply="It ran for sixty-four seconds at sixty ms per frame",
                          heard="It ran for sixty-four seconds", interrupted=True)}
    obj = {"facts": [
        {"text": "Kokoro is the fallback voice.", "kind": "DECISION", "cites": ["s1t1"], "quote": "Keep Kokoro"},
        {"text": "It ran for 64 s.", "kind": "weird", "cites": "S1T2", "quote": "sixty-four seconds"},
        {"text": "The bench ran at sixty ms per frame.", "cites": ["S1T2"], "quote": "sixty ms per frame"},  # not heard
        {"text": "Something from nowhere.", "cites": ["S7T1"]},
        {"text": "x", "cites": ["S1T1"]},
        "not an object",
    ], "spoken": "Kokoro is the fallback."}
    p = v.validate_tech(obj, handles, max_facts=10, max_words=45)
    assert [(f.kind, f.cites, f.quote) for f in p.facts] == [
        ("decision", ["S1T1"], "keep kokoro"), ("finding", ["S1T2"], "sixty-four seconds")]
    assert p.spoken == "Kokoro is the fallback."
    assert any("quote not found" in d for d in p.dropped)        # the words the person never heard
    assert any("cites no turn" in d for d in p.dropped)
    assert len(p.dropped) == 4


def test_validate_tech_respects_the_fact_limit():
    handles = {"S1T1": mk()}
    obj = {"facts": [{"text": f"Fact number {i} here.", "cites": ["S1T1"]} for i in range(5)], "spoken": "NO_REPLY"}
    p = v.validate_tech(obj, handles, max_facts=2, max_words=45)
    assert len(p.facts) == 2 and p.spoken is None and "3 more facts" in p.dropped[-1]


WORDS = {"S1T1": "been thinking about how stretched thin I feel with work lately, I keep saying yes to things",
         "S1T2": "honestly I've felt anxious all week about the move"}


def life(reflections, words=WORDS, tags=frozenset({"area/work", "journal"})):
    handles = {h: mk(n=int(h[-1])) for h in words}
    return v.validate_life({"reflections": reflections}, handles, words, known_tags=set(tags), max_reflections=3)


def test_validate_life_quotes_are_exact_spans_and_tags_are_reused():
    p = life([{"title": "Yes to things", "quotes": [{"turn": "S1T1", "text": "I keep Saying yes to things"}],
               "notice": "You mentioned work once today.", "questions": ["What does a yes cost?", ""],
               "tags": ["area/work", "#journal", "Bad Tag!", "invented", "area/new-area"]}])
    (r,) = p.reflections
    assert r.quotes[0].text == "I keep saying yes to things"     # the person's casing, not the model's
    assert r.questions == ["What does a yes cost?"]
    assert r.tags == ["area/work", "journal", "area/new-area"]


@pytest.mark.parametrize("notice,reason", [
    ("You seem stressed about work.", "verdict"),
    ("This sounds like burnout.", "clinical"),
    ("You should say no more often.", "advice"),
    ("You are clearly overwhelmed by it all.", "verdict"),
])
def test_validate_life_drops_verdicts_diagnoses_and_advice(notice, reason):
    p = life([{"title": "Work", "quotes": [{"turn": "S1T1", "text": "stretched thin"}], "notice": notice}])
    assert not p.reflections and reason.split()[0] in " ".join(p.dropped).replace("a clinical word", "clinical")


def test_validate_life_allows_a_feeling_word_they_used_themselves():
    p = life([{"title": "The move", "quotes": [{"turn": "S1T2", "text": "I've felt anxious all week"}],
               "notice": "You used the word anxious about the move.", "questions": ["What would make the move easier to "
                                                                                    "think about?"]}])
    assert len(p.reflections) == 1


def test_validate_life_drops_quotes_that_are_not_their_words_and_unknown_handles():
    p = life([{"title": "Invented", "quotes": [{"turn": "S1T1", "text": "I hate my job"},
                                               {"turn": "S4T4", "text": "stretched thin"}],
               "notice": "n", "questions": ["q?"]}])
    assert not p.reflections
    assert any("not in S1T1" in d for d in p.dropped) and any("S4T4" in d for d in p.dropped)


def test_validate_life_needs_something_noticed_or_asked():
    p = life([{"title": "Bare", "quotes": [{"turn": "S1T1", "text": "stretched thin"}], "notice": "", "questions": []}])
    assert not p.reflections and "nothing noticed" in p.dropped[0]
