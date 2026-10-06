"""Approvals in the orchestrator (PROTOCOL.md "Approvals", 2026-10-05), model-free.

Units: the spoken answer (only a plain yes approves; "yes, but ..." and any other reply go to the model as the person's
words; an echo of the question answers nothing), the spoken lines, file names said with their dot, the request parsed
from the gate's select, session grants. Whole turns over protocol v1 against the stub LLM and a real Pi child with the
gate: the card (confirm_request with summary, action and choices) answered by a button, by voice, by "yes, for this
session" (and the next call in the same scope runs without a question, and /v1/status lists the grant), by silence (asked
once more, then refused aloud, nothing run, no model call after), and by a reply of the person's own."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from harness import rig
from local_voice import approvals
from local_voice.approvals import Approval, ApprovalSettings, Grants, heard_answer, speakable_names
from local_voice.pi_rpc import ConfirmRequest, SelectRequest

ONCE = ["allow_once", "deny"]
SESSION = ["allow_once", "allow_session", "deny"]
ASKED = "I'd like to create a new file Hello dot T X T in the scratch folder. Shall I? Shall I create Hello dot T X T? Yes or no."


@pytest.mark.parametrize("text,offered,want", [
    ("Yes.", SESSION, "allow_once"),
    ("Yeah, go ahead and create it.", SESSION, "allow_once"),
    ("Yes, go ahead and add it.", SESSION, "allow_once"),
    ("Sure, why not.", SESSION, "allow_once"),
    ("Do it.", SESSION, "allow_once"),
    ("Yes, for this session.", SESSION, "allow_session"),
    ("Yes, for the rest of the session please.", SESSION, "allow_session"),
    ("Don't ask me again.", SESSION, "allow_session"),
    ("Always.", SESSION, "allow_session"),
    ("Yes, for this session.", ONCE, "allow_once"),          # no session scope offered: a yes is still a yes, once
    ("No.", SESSION, "deny"),
    ("No thanks.", SESSION, "deny"),
    ("Please don't.", SESSION, "deny"),
    ("Nope, don't do it.", SESSION, "deny"),
    ("Yes, but call it notes.txt.", SESSION, "deny_said"),   # never approves the call as it stands
    ("Yes, in my inbox folder.", SESSION, "deny_said"),
    ("Okay no.", SESSION, "deny_said"),
    ("No, put it in the inbox instead.", SESSION, "deny_said"),
    ("What will it write?", SESSION, "deny_said"),
    ("Wait.", SESSION, None),                                # not an answer yet
    ("Um.", SESSION, None),
    ("Shall I create Hello dot T X T?", SESSION, None),     # the question's own echo
    ("Yes or no.", SESSION, None),
    ("", SESSION, None),
])
def test_only_a_plain_yes_approves(text, offered, want):
    assert heard_answer(text, offered, ASKED) == want


def test_a_clean_answer_ends_the_turn_early_and_a_longer_one_waits():
    assert approvals.is_clean_answer("Yes.", SESSION) and approvals.is_clean_answer("No thanks.", SESSION)
    assert approvals.is_clean_answer("Yes, for this session.", SESSION)
    assert not approvals.is_clean_answer("Yes, but", SESSION) and not approvals.is_clean_answer("What?", SESSION)


def test_file_names_are_said_with_their_dot():
    assert speakable_names("Create a new file Hello.txt in the scratch folder.") == \
        "Create a new file Hello dot T X T in the scratch folder."
    assert speakable_names("change 2026-10-05.md and atlas.py and data.json") == \
        "change 2026-10-05 dot M D and atlas dot P Y and data dot json"
    assert speakable_names("It is done. Next.") == "It is done. Next."


def _approval(**kw) -> Approval:
    d = dict(id="u1", kind="select", title="May I create Hello.txt?", message="/w/Hello.txt",
             summary="Create a new file Hello.txt in the scratch folder.", short="create Hello.txt",
             action={"tool": "write"}, scope={"key": "write /w", "label": "writing files in the scratch folder"},
             options=SESSION)
    d.update(kw)
    return Approval(**d)


def test_the_spoken_lines_say_what_will_happen_and_what_did_not():
    s, a = ApprovalSettings(), _approval()
    assert approvals.spoken_question(a, s) == "I'd like to create a new file Hello dot T X T in the scratch folder. Shall I?"
    assert approvals.spoken_again(a, s) == "Shall I create Hello dot T X T? Yes or no."
    assert approvals.spoken_outcome(a, s, "deny") == "Okay, I didn't create Hello dot T X T."
    assert approvals.spoken_outcome(a, s, "deny_said") == "Okay, I didn't create Hello dot T X T."
    assert approvals.spoken_outcome(a, s, "timeout") == "I didn't hear an answer, so I didn't create Hello dot T X T."
    assert approvals.spoken_outcome(a, s, "allow_session") == \
        "Okay. For the rest of this session I won't ask again about writing files in the scratch folder."
    assert approvals.spoken_outcome(a, s, "allow_once") is None
    assert a.choices(s) == [{"id": "allow_once", "label": "Do it"},
                            {"id": "allow_session", "label": "Allow writing files in the scratch folder for the rest of this session"},
                            {"id": "deny", "label": "Don't"}]


def test_the_request_is_read_from_the_gates_select_or_a_plain_confirm():
    payload = {"lv": "approval", "v": 1, "title": "May I run a command?", "message": "ls", "summary": "Run ls.",
               "short": "run that command", "action": {"tool": "bash", "command": "ls"}, "scope": None}
    a = approvals.from_request(SelectRequest({}, id="s1", method="select", title=json.dumps(payload),
                                             options=["allow_once", "allow_session", "deny"], timeout_ms=1000))
    assert a.kind == "select" and a.options == ["allow_once", "deny"] and a.tool == "bash"   # no scope: no session choice
    assert approvals.from_request(SelectRequest({}, id="s2", method="select", title="Pick one", options=["a"],
                                                timeout_ms=None)) is None
    c = approvals.from_request(ConfirmRequest({}, id="c1", method="confirm", title="May I delete the cache?",
                                              message="rm -r cache", timeout_ms=None))
    assert c.kind == "confirm" and c.summary == "Delete the cache: rm -r cache." and c.options == ONCE


def test_grants_cover_one_scope_in_one_space_and_session():
    g = Grants()
    g.add("s1", "home", {"key": "kb new", "label": "creating knowledge base pages"})
    assert g.match("s1", "home", {"key": "kb new"}) is not None
    assert g.match("s1", "atlas", {"key": "kb new"}) is None
    assert g.match("s2", "home", {"key": "kb new"}) is None
    assert g.match("s1", "home", {"key": "kb ingest"}) is None and g.match("s1", "home", None) is None
    assert [x["scope"] for x in g.as_json()] == ["kb new"]
    g.drop_session("s1")
    assert g.as_json() == []


# ------------------------------------------------------------------------------------------- whole turns

def _spaces(tmp_path) -> dict:
    """home on the rig's scratch folder, with write and bash in act mode."""
    return {"tools": ["read", "ls"], "act_tools": ["write", "bash"]}


def _ov(**approvals_cfg) -> dict:
    return {"agent": {"approvals": {"card_wait_s": 4.0, "reask_after_s": 1.5, "voice_wait_s": 2.0, **approvals_cfg},
                      "progress": {"after_s": 0.5, "text": "Still looking."}}}


def write_call(name: str = "Hello.txt", text: str = "Hello world from the voice agent\n") -> dict:
    return {"tool_calls": [{"name": "write", "arguments": {"path": name, "content": text}}], "delay_ms": 5}


async def status(r) -> dict:
    async with httpx.AsyncClient() as h:     # never a blocking client: the server runs on this test's event loop
        return (await h.get(r.status_url)).json()


def arrived(c, t: str, since: float) -> list[float]:
    """When each message of type t arrived (the test client keeps (time, message) pairs)."""
    return [at for at, m in c.messages if at >= since and m.get("t") == t]


async def _turn(c, text: str, timeout: float = 20) -> dict:
    t0 = c.now()
    await c.send({"t": "text", "text": text})
    return await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=timeout, since=t0)


async def _ask(c, text: str) -> tuple[dict, float]:
    t0 = c.now()
    await c.send({"t": "text", "text": text})
    q = await c.wait_for(lambda m: m.get("t") == "confirm_request", timeout=15, since=t0)
    return q, t0


async def test_a_card_is_answered_by_its_button(tmp_path):
    async with rig(tmp_path, spaces=_spaces(tmp_path), overrides=_ov()) as r:
        r.stub.script([write_call(), {"text": "Done, it's there.", "delay_ms": 5}])
        c = await r.client(client="mac")
        await _turn(c, "Act mode.")
        q, t0 = await _ask(c, "Make a file called Hello.txt")
        assert q["summary"] == "Create a new file Hello.txt in the scratch folder."
        assert q["action"]["tool"] == "write" and q["action"]["effect"] == "create"
        assert q["action"]["path"] == str(r.workdir / "Hello.txt")
        assert q["action"]["preview"] == "Hello world from the voice agent\n"
        assert q["action"]["space"] == "home" and q["action"]["mode"] == "act"
        assert [x["id"] for x in q["choices"]] == ["allow_once", "allow_session", "deny"]
        assert q["timeout_ms"] == 4000 and q["title"] and q["message"]
        st = await status(r)
        assert st["approval"]["summary"] == q["summary"] and st["approval"]["wait_s"] == 4.0
        await asyncio.sleep(0.3)
        await c.send({"t": "confirm_response", "id": q["id"], "confirmed": True, "choice": "allow_once"})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=t0)
        assert (r.workdir / "Hello.txt").read_text() == "Hello world from the voice agent\n"
        assert not c.of_type("confirm_cancel", since=t0)          # the client closed its own card
        spoken = r.tts.spoken
        assert "I'd like to create a new file Hello dot T X T in the scratch folder. Shall I?" in spoken
        assert spoken[-1] == "Done, it's there." and "One moment." not in spoken   # the question was the acknowledgement
        assert (await status(r))["approval"] is None
        await c.close()


async def test_yes_for_this_session_allows_the_same_scope_without_asking_again(tmp_path):
    async with rig(tmp_path, spaces=_spaces(tmp_path), overrides=_ov()) as r:
        r.stub.script([write_call("a.txt", "a\n"), {"text": "Made a.", "delay_ms": 5},
                       write_call("b.txt", "b\n"), {"text": "Made b.", "delay_ms": 5},
                       write_call("sub/c.txt", "c\n"), {"text": "Made c.", "delay_ms": 5}])
        c = await r.client(client="mac")
        await _turn(c, "Act mode.")
        q, t0 = await _ask(c, "Make a.txt")
        await c.send({"t": "text", "text": "Yes, for this session."})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=t0)
        cancel = c.of_type("confirm_cancel", since=t0)
        assert [m["why"] for m in cancel] == ["answered"] and cancel[0]["id"] == q["id"]
        assert "Okay. For the rest of this session I won't ask again about writing files in the scratch folder." in r.tts.spoken
        grants = (await status(r))["grants"]
        assert [(g["space"], g["scope"]) for g in grants] == [("home", f"write {r.workdir}")]
        t1 = c.now()
        await _turn(c, "Make b.txt")
        assert not c.of_type("confirm_request", since=t1)          # the same folder: no question
        assert (r.workdir / "b.txt").read_text() == "b\n"
        (r.workdir / "sub").mkdir(exist_ok=True)
        q3, t2 = await _ask(c, "Make sub/c.txt")                   # another folder is another scope: asked
        assert q3["action"]["path"] == str(r.workdir / "sub/c.txt")
        await c.send({"t": "confirm_response", "id": q3["id"], "choice": "deny", "confirmed": False})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=t2)
        assert not (r.workdir / "sub/c.txt").exists() and r.tts.spoken[-1] == "Okay, I didn't create c dot T X T."
        await c.close()


async def test_silence_with_a_card_is_asked_again_then_refused_aloud(tmp_path):
    async with rig(tmp_path, spaces=_spaces(tmp_path), overrides=_ov()) as r:
        r.stub.script([write_call(), {"text": "UNEXPECTED", "delay_ms": 5}])
        c = await r.client(client="mac")
        await _turn(c, "Act mode.")
        n0 = len(r.stub.requests())
        q, t0 = await _ask(c, "Make Hello.txt")
        end = await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=20, since=t0)
        cancel = c.of_type("confirm_cancel", since=t0)
        assert [m["why"] for m in cancel] == ["timeout"]
        waited = arrived(c, "confirm_cancel", t0)[0] - arrived(c, "confirm_request", t0)[0]
        assert 3.8 <= waited <= 6.0, waited                      # card_wait_s 4.0
        spoken = r.tts.spoken
        assert "Shall I create Hello dot T X T? Yes or no." in spoken   # asked once more (reask_after_s 1.5)
        assert spoken[-1] == "I didn't hear an answer, so I didn't create Hello dot T X T."
        assert not (r.workdir / "Hello.txt").exists()
        assert len(r.stub.requests()) - n0 == 1                  # no model call after the refusal
        assert "UNEXPECTED" not in spoken and end
        await c.close()


async def test_silence_without_a_card_waits_the_short_time_and_is_not_asked_again(tmp_path):
    async with rig(tmp_path, spaces=_spaces(tmp_path), overrides=_ov()) as r:
        r.stub.script([write_call(), {"text": "UNEXPECTED", "delay_ms": 5}])
        c = await r.client()                                      # the test client shows no card
        await _turn(c, "Act mode.")
        q, t0 = await _ask(c, "Make Hello.txt")
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=20, since=t0)
        waited = arrived(c, "confirm_cancel", t0)[0] - arrived(c, "confirm_request", t0)[0]
        assert 1.8 <= waited <= 3.5, waited                      # voice_wait_s 2.0
        assert q["timeout_ms"] == 2000
        assert "Shall I create Hello dot T X T? Yes or no." not in r.tts.spoken
        await c.close()


async def test_a_reply_of_their_own_is_not_approval_and_reaches_the_model(tmp_path):
    async with rig(tmp_path, spaces=_spaces(tmp_path), overrides=_ov()) as r:
        r.stub.script([write_call(), {"text": "Okay, which name then?", "delay_ms": 5}])
        c = await r.client(client="mac")
        await _turn(c, "Act mode.")
        q, t0 = await _ask(c, "Make Hello.txt")
        await c.send({"t": "text", "text": "Yes, but call it notes.txt."})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=t0)
        assert not (r.workdir / "Hello.txt").exists() and not (r.workdir / "notes.txt").exists()
        assert [m["why"] for m in c.of_type("confirm_cancel", since=t0)] == ["answered"]
        spoken = r.tts.spoken
        assert spoken[-2:] == ["Okay, I didn't create Hello dot T X T.", "Okay, which name then?"]
        msgs = r.stub.requests()[-1]["body"]["messages"]
        assert "Yes, but call it notes.txt." in json.dumps(msgs[-1]["content"])
        await c.close()


async def test_an_echo_of_the_question_answers_nothing(tmp_path):
    async with rig(tmp_path, spaces=_spaces(tmp_path), overrides=_ov()) as r:
        r.stub.script([write_call(), {"text": "Done.", "delay_ms": 5}])
        c = await r.client(client="mac")
        await _turn(c, "Act mode.")
        q, t0 = await _ask(c, "Make Hello.txt")
        await c.send({"t": "text", "text": "Shall I create Hello dot TXT?"})   # the speaker heard by the microphone
        await asyncio.sleep(0.5)
        assert not c.of_type("confirm_cancel", since=t0) and r.orch.hub.approval is not None
        await c.send({"t": "text", "text": "Yes."})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=t0)
        assert (r.workdir / "Hello.txt").exists()
        await c.close()


async def test_a_kb_write_is_asked_about_without_a_lookup_acknowledgement_first(tmp_path):
    """An early version said "Let me look that up." and then "May I change your knowledge base?": a kb verb that writes is
    asked about with nothing said before the question; a kb search still gets its acknowledgement."""
    from conftest import HERE
    kb = tmp_path / "kbhome"
    kb.mkdir()
    fake_kb = str(HERE / "tests/fixtures/fake_kb_tool.ts")      # never the real kb.ts in a test
    async with rig(tmp_path, spaces={"tools": ["read", "kb"]},
                   overrides={**_ov(), "pi": {"extensions": {"kb": fake_kb}, "env": {"KB_HOME": str(kb)}}}) as r:
        r.stub.script([{"tool_calls": [{"name": "kb", "arguments": {"args": ["search", "hold gate"]}}], "delay_ms": 5},
                       {"text": "It pauses the models.", "delay_ms": 5},
                       {"tool_calls": [{"name": "kb", "arguments": {"args": ["new", "Analysis", "voice-agent-write",
                                                                           "--title", "Voice agent write access"]}}],
                        "delay_ms": 5}])
        c = await r.client(client="mac")
        await _turn(c, "What does my kb say about the hold gate?")
        assert r.tts.spoken[-2:] == ["Let me look that up.", "It pauses the models."]
        n = len(r.tts.spoken)
        q, t0 = await _ask(c, "Add an issue to my wiki.")
        assert q["summary"] == 'Create a new analysis page in your knowledge base titled "Voice agent write access".'
        assert q["action"]["command"] == "kb new Analysis voice-agent-write --title 'Voice agent write access'"
        await c.send({"t": "text", "text": "No."})
        await c.wait_for(lambda m: m.get("t") == "end_of_turn", timeout=15, since=t0)
        assert r.tts.spoken[n:] == ["I'd like to create a new analysis page in your knowledge base titled "
                                    "\"Voice agent write access\". Shall I?", "Okay, I didn't create that page."]
        await c.close()
