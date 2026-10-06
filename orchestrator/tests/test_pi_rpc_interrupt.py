"""Barge-in hygiene (05c E10, after the 04c review): an aborted run's leftovers never leak into the next turn,
the abort is safe on a cancellation path, and an open confirm dialog does not hold the abort."""
import asyncio
import time

import pytest

from conftest import EXT, HERE
from local_voice.pi_rpc import ConfirmRequest, MessageEnd, PiBusyError, SelectRequest, Settled, TextDelta

pytestmark = pytest.mark.needs_pi
LONG = " ".join(f"w{i}" for i in range(80))
FRESH = "Fresh answer to the new question."


async def until(child, pred, timeout=20.0):
    got = []
    while True:
        ev = await asyncio.wait_for(child.events.get(), timeout)
        got.append(ev)
        if pred(ev):
            return got


def clean(r):
    """The turn holds exactly its own reply: complete, one settle, nothing from the aborted run."""
    assert r.text == FRESH and r.stop_reason == "stop"
    assert sum(isinstance(e, Settled) for e in r.events) == 1
    assert not any(isinstance(e, MessageEnd) and e.stop_reason == "aborted" for e in r.events)
    assert not any(isinstance(e, TextDelta) and e.text.strip().startswith("w") for e in r.events)


async def test_awaited_interrupt_then_an_immediate_turn_is_clean(make_child, stub):
    child = await make_child()
    stub.script([{"text": LONG, "delay_ms": 50}, {"text": FRESH}])
    await child.prompt("long story")
    await until(child, lambda e: isinstance(e, TextDelta) and e.text.strip() == "w2")
    await child.interrupt(heard="w0 w1")
    assert not child.busy and child.events.empty()
    r = await child.run_turn("new question")
    clean(r)
    assert child.events.empty()
    last = stub.requests()[-1]["body"]["messages"][-1]["content"]
    text = last if isinstance(last, str) else last[0]["text"]
    assert text.startswith('(You were interrupted. The user heard only this much of your last reply: "w0 w1")')


async def test_stopping_at_settled_never_aborts_the_next_run(make_child, stub):
    """The orchestrator breaks out of turn() at agent_settled, so the generator's cleanup runs later, maybe after the
    next prompt went out (a second connection waiting on the space's lock: e2e 2026-10-05 12:05:21, clear_queue and
    abort 1 ms after the new prompt). The run that generator was for is over: nothing may be aborted."""
    child = await make_child()
    stub.script([{"text": "First."}, {"text": FRESH, "pre_delay_ms": 300, "delay_ms": 30}])
    first = child.turn("one")
    async for ev in first:
        if isinstance(ev, Settled):
            break
    second = asyncio.create_task(child.run_turn("two"))
    deadline = time.monotonic() + 10
    while len(stub.requests()) < 2 and time.monotonic() < deadline:   # the new run is open
        await asyncio.sleep(0.01)
    await first.aclose()                 # what the event loop's finalizer does with the first generator, late
    clean(await second)


async def test_an_interrupt_meant_for_an_earlier_turn_never_aborts_a_newer_run(make_child, stub):
    child = await make_child()
    stub.script([{"text": "First."}, {"text": FRESH, "pre_delay_ms": 300, "delay_ms": 30}])
    await child.run_turn("one")
    old = child._turn
    second = asyncio.create_task(child.run_turn("two"))
    deadline = time.monotonic() + 10
    while len(stub.requests()) < 2 and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    await child.request_interrupt(turn=old)
    clean(await second)


async def test_cancelled_consumer_aborts_without_blocking(make_child, stub):
    """What Pipecat does on barge-in: cancel the task reading turn() and wait at most 1 s for it."""
    child = await make_child()
    stub.script([{"text": LONG, "delay_ms": 50}, {"text": FRESH}])
    seen = []

    async def consume():
        async for ev in child.turn("long story"):
            seen.append(ev)

    task = asyncio.create_task(consume())
    while not any(isinstance(e, TextDelta) and e.text.strip() == "w2" for e in seen):
        await asyncio.sleep(0.01)
    t = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1.0)
    assert time.monotonic() - t < 0.2          # the generator's cleanup only schedules the abort
    child.request_interrupt(heard="w0 w1")      # the service also records what was heard; same task
    r = await child.run_turn("new question")    # starts at once: quiesce waits for the abort and the settle
    clean(r)
    # the aborted stream was cut, not finished: the stub notices at its next write, so poll briefly
    for _ in range(100):
        if any(" DISCONNECT 1 " in line for line in stub.log_lines()):
            break
        await asyncio.sleep(0.01)
    assert any(" DISCONNECT 1 " in line for line in stub.log_lines())
    assert not any(" DONE 1 " in line for line in stub.log_lines())


async def test_a_stale_settled_can_never_end_a_later_turn(make_child, stub):
    child = await make_child()
    stub.script([{"text": FRESH, "pre_delay_ms": 300}])
    task = asyncio.create_task(child.run_turn("new question"))
    while len(stub.requests()) < 1:
        await asyncio.sleep(0.01)
    child.events.put_nowait(Settled({"type": "agent_settled"}, turn=child._turn - 1))   # a leftover from before
    r = await task
    clean(r)
    assert any(isinstance(e, Settled) and e.turn == child._turn - 1 for e in child.stray)


async def test_abort_dismisses_the_gates_open_confirm(make_child, stub):
    child = await make_child(tools=["bash"], extensions=[EXT / "voice_gate.ts"],
                             env={"VOICE_TIER": "ask", "VOICE_CONFIRM_TIMEOUT_MS": "20000"})
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "echo needs-a-yes"}}]}, {"text": FRESH}],
                default={"text": "UNEXPECTED"})
    n0 = len(stub.requests())
    await child.prompt("run it")
    await until(child, lambda e: isinstance(e, SelectRequest))    # the gate asks with an approval (a select)
    t = time.monotonic()
    await child.interrupt()
    assert time.monotonic() - t < 2.0           # not the dialog's 20 s
    assert len(stub.requests()) - n0 == 1       # no model call after the refusal
    stub.script([{"text": FRESH}])
    clean(await child.run_turn("new question"))


async def test_interrupt_cancels_dialogs_of_extensions_without_the_signal(make_child, stub):
    child = await make_child(tools=["bash"], extensions=[HERE / "tests/fixtures/confirm_without_signal.ts"])
    stub.script([{"tool_calls": [{"name": "bash", "arguments": {"command": "echo needs-a-yes"}}]}, {"text": "UNEXPECTED"}],
                default={"text": "UNEXPECTED"})
    n0 = len(stub.requests())
    await child.prompt("run it")
    await until(child, lambda e: isinstance(e, ConfirmRequest))
    t = time.monotonic()
    await child.interrupt()
    assert time.monotonic() - t < 2.0
    assert len(stub.requests()) - n0 == 1
    stub.script([{"text": FRESH}])
    clean(await child.run_turn("new question"))


async def test_interrupts_share_one_task_and_idle_interrupt_is_harmless(make_child, stub):
    child = await make_child()
    a = child.request_interrupt()
    assert child.request_interrupt() is a
    await child.interrupt()
    assert not child.busy
    stub.script([{"text": FRESH}])
    clean(await child.run_turn("hello"))


async def test_turn_refuses_while_a_run_will_not_settle(make_child, stub):
    child = await make_child()
    stub.script([{"text": LONG, "delay_ms": 50}])
    await child.prompt("long story")             # a run started outside turn()
    await until(child, lambda e: isinstance(e, TextDelta))
    with pytest.raises(PiBusyError):
        async for _ in child.turn("too early", settle_timeout=0.3):
            pass
    await child.interrupt()
    stub.script([{"text": FRESH}])
    clean(await child.run_turn("now"))
