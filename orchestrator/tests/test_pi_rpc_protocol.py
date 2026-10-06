"""Pi's RPC protocol through the bridge, against the stub LLM: the behaviour the voice layer relies on (05c)."""
import asyncio
import subprocess
import time

import pytest

from local_voice.pi_rpc import (AgentEnd, MessageEnd, MessageStart, PiCommandError, QueueUpdate, Retry, Settled, TextDelta,
                           ToolCallStarted, ToolEnd, ToolStart, ToolUpdate, heredoc_command)

pytestmark = pytest.mark.needs_pi
LONG = " ".join(f"w{i}" for i in range(60))


async def until(child, pred, timeout=20.0):
    got = []
    while True:
        ev = await asyncio.wait_for(child.events.get(), timeout)
        got.append(ev)
        if pred(ev):
            return got


async def test_turn_streams_text_then_settles(make_child, stub):
    child = await make_child()
    stub.script([{"text": "Hello there, this is the stub.", "delay_ms": 5}])
    r = await child.run_turn("Hi")
    assert r.text == "Hello there, this is the stub." and r.stop_reason == "stop"
    kinds = [type(e).__name__ for e in r.events]
    assert kinds.index("TextDelta") < kinds.index("MessageEnd", kinds.index("TextDelta")) < kinds.index("AgentEnd") < kinds.index("Settled")
    # the first turn also carries the transcript's system message as a message_start/end pair
    assert any(isinstance(e, MessageStart) and e.role == "system" for e in r.events)
    assert r.first_text_s is not None and r.total_s >= r.first_text_s


async def test_thinking_off_sends_disabled(make_child, stub):
    child = await make_child()
    stub.script([{"text": "ok"}])
    await child.run_turn("Hi")
    body = stub.requests()[-1]["body"]
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body


async def test_read_and_ls_round_trip(make_child, stub):
    child = await make_child()
    stub.script([{"tool_calls": [{"name": "read", "arguments": {"path": "README.md"}}, {"name": "ls", "arguments": {"path": "sub"}}]},
                 {"text": "Done."}])
    r = await child.run_turn("read and list")
    assert [n for n, _ in r.tools] == ["read", "ls"]
    res = {n: (ok, t) for n, ok, t in r.tool_results}
    assert res["read"][0] and "Line two." in res["read"][1]
    assert res["ls"] == (True, "a.txt\nb.txt")
    msgs = stub.requests()[-1]["body"]["messages"]
    assert [m["role"] for m in msgs[-3:]] == ["assistant", "tool", "tool"]


async def test_toolcall_started_precedes_execution(make_child, stub):
    child = await make_child(tools=["bash"])
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "echo hi"}, "split": 6}], "delay_ms": 20}, {"text": "ok"}])
    r = await child.run_turn("run it")
    started = next(e.at for e in r.events if isinstance(e, ToolCallStarted))
    executing = next(e.at for e in r.events if isinstance(e, ToolStart))
    assert executing - started > 0.08   # the arguments streamed in between: acknowledge at ToolCallStarted


async def test_bash_updates_are_cumulative(make_child, stub):
    child = await make_child(tools=["bash"])
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "for i in 1 2 3; do echo t$i; sleep 0.2; done"}}]}, {"text": "ok"}])
    r = await child.run_turn("count")
    ups = [e.text for e in r.events if isinstance(e, ToolUpdate) and e.text]
    assert ups[-1] == "t1\nt2\nt3\n" and all(ups[-1].startswith(u) for u in ups)


async def test_abort_mid_stream_drops_the_partial_reply_from_context(make_child, stub):
    child = await make_child()
    stub.script([{"text": LONG, "delay_ms": 60}])
    await child.prompt("long one")
    await until(child, lambda e: isinstance(e, TextDelta) and e.text.strip() == "w3")
    await child.abort()
    evs = await until(child, lambda e: isinstance(e, Settled))
    me = next(e for e in evs if isinstance(e, MessageEnd) and e.role == "assistant")
    assert me.stop_reason == "aborted" and me.text.startswith("w0 w1 w2 w3")
    stub.script([{"text": "ok"}])
    await child.run_turn("go on")
    roles = [m["role"] for m in stub.requests()[-1]["body"]["messages"]]
    assert roles[-2:] == ["user", "user"]   # Pi sends the model nothing of what it had said


async def test_interrupt_tells_the_model_what_was_heard(make_child, stub):
    child = await make_child()
    stub.script([{"text": LONG, "delay_ms": 60}])
    await child.prompt("long one")
    await until(child, lambda e: isinstance(e, TextDelta) and e.text.strip() == "w2")
    await child.interrupt(heard="w0 w1")        # waits for the settle and drains the aborted run's events
    stub.script([{"text": "sorry"}])
    await child.run_turn("stop, what time is it")
    last = stub.requests()[-1]["body"]["messages"][-1]
    text = last["content"] if isinstance(last["content"], str) else last["content"][0]["text"]
    assert text.startswith('(You were interrupted. The user heard only this much of your last reply: "w0 w1")')


async def test_abort_mid_tool_kills_the_command(make_child, stub):
    child = await make_child(tools=["bash"])
    marker = f"bridge-test-{time.time_ns()}"
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": f"echo {marker} >/dev/null; for i in $(seq 1 40); do echo t$i; sleep 0.25; done"}}]},
                 {"text": "SHOULD NOT BE REQUESTED"}])
    n0 = len(stub.requests())
    await child.prompt("slow")
    await until(child, lambda e: isinstance(e, ToolUpdate) and "t2" in e.text)
    await child.abort()
    evs = await until(child, lambda e: isinstance(e, Settled))
    end = next(e for e in evs if isinstance(e, ToolEnd))
    assert not end.ok and "Command aborted" in end.text
    await asyncio.sleep(0.5)
    assert subprocess.run(["pgrep", "-f", marker], capture_output=True).returncode == 1
    assert len(stub.requests()) - n0 == 1


async def test_steer_lands_after_the_running_tool(make_child, stub):
    child = await make_child(tools=["bash"])
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "sleep 0.6; echo done"}}]}, {"text": "OK."}])
    await child.prompt("run")
    await until(child, lambda e: isinstance(e, ToolStart))
    assert await child.steer("never mind, just say OK") == "queued"
    evs = await until(child, lambda e: isinstance(e, Settled))
    assert any(isinstance(e, QueueUpdate) and e.steering == [] for e in evs)
    msgs = stub.requests()[-1]["body"]["messages"]
    assert [m["role"] for m in msgs[-2:]] == ["tool", "user"]


async def test_follow_up_runs_before_settling(make_child, stub):
    child = await make_child()
    stub.script([{"text": "first answer here", "delay_ms": 60}, {"text": "Goodbye."}])
    await child.prompt("first")
    await until(child, lambda e: isinstance(e, TextDelta))
    assert await child.follow_up("also say goodbye") == "queued"
    evs = await until(child, lambda e: isinstance(e, Settled))
    finals = [e.text for e in evs if isinstance(e, MessageEnd) and e.role == "assistant"]
    assert finals == ["first answer here", "Goodbye."]
    assert sum(isinstance(e, AgentEnd) for e in evs) == 1


async def test_clear_queue_returns_texts_and_they_never_reach_the_model(make_child, stub):
    child = await make_child()
    stub.script([{"text": "streaming a while longer now", "delay_ms": 60}], default={"text": "UNEXPECTED"})
    n0 = len(stub.requests())
    await child.prompt("go")
    await until(child, lambda e: isinstance(e, TextDelta))
    await child.steer("S1")
    await child.follow_up("F1")
    assert await child.clear_queue() == (["S1"], ["F1"])
    await until(child, lambda e: isinstance(e, Settled))
    assert len(stub.requests()) - n0 == 1


async def test_prompt_while_busy_needs_a_streaming_behavior(make_child, stub):
    child = await make_child()
    stub.script([{"text": "busy busy busy", "delay_ms": 60}, {"text": "second"}])
    await child.prompt("go")
    await until(child, lambda e: isinstance(e, TextDelta))
    with pytest.raises(PiCommandError, match="already processing"):
        await child.command("prompt", message="no behaviour")
    assert await child.prompt("queued", streaming_behavior="followUp") == "queued"
    evs = await until(child, lambda e: isinstance(e, Settled))
    assert [e.text for e in evs if isinstance(e, MessageEnd) and e.role == "assistant"][-1] == "second"


async def test_retry_then_success_settles_once(make_child, stub):
    child = await make_child()
    stub.script([{"status": 503, "error_message": "loading"}, {"text": "recovered"}])
    r = await child.run_turn("hi")
    assert r.text == "recovered"
    assert [e.raw["type"] for e in r.events if isinstance(e, Retry)] == ["auto_retry_start", "auto_retry_end"]
    assert any(isinstance(e, AgentEnd) and e.will_retry for e in r.events)
    assert sum(isinstance(e, Settled) for e in r.events) == 1


async def test_error_without_retry_reaches_the_bridge(make_child, stub):
    child = await make_child()
    await child.command("set_auto_retry", enabled=False)
    stub.script([{"status": 500, "error_message": "boom"}])
    r = await child.run_turn("hi")
    assert r.stop_reason == "error" and "boom" in (r.error or "")


async def test_records_over_64k_and_unicode_separators(make_child, stub, workdir):
    (workdir / "big.txt").write_text(("x" * 1000 + "\n") * 120)
    child = await make_child()
    stub.script([{"tool_calls": [{"name": "read", "arguments": {"path": "big.txt"}}]},
                 {"text": "a b c", "chunk_words": 5}])
    r = await child.run_turn("read the big file")
    assert r.tool_results[0][1] and max(len(str(e.raw)) for e in r.events) > 65536
    assert r.text == "a b c"


async def test_rpc_bash_heredoc_is_verbatim(make_child, workdir):
    child = await make_child()
    text = "it's $HOME & `x` — \"q\" back\\slash\nATLAS_END\nsecond line"
    res = await child.bash(heredoc_command("cat > out.txt", text), exclude_from_context=True)
    assert res["exitCode"] == 0
    assert (workdir / "out.txt").read_text() == text + "\n"
