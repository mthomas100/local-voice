"""Config, the MLX thread's GPU guard, the hold monitor and the fake engines: no model, no network but localhost."""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from fake_gate import FakeGate
from local_voice.config import ConfigError, load_config
from local_voice.engines.fakes import FakeStreamingTranscriber, FakeSynthesizer, FakeTranscriber
from local_voice.hold import HoldMonitor
from local_voice.mlx_worker import GpuHeldError, MLXWorker


def test_config_loads_and_names_models():
    cfg = load_config()
    assert cfg.stt.name == "nemotron" and cfg.stt.kind == "streaming"
    assert cfg.tts.name == "qwen3" and cfg.tts.settings["streaming_interval"] == 0.32
    assert cfg.tts.settings["repetition_penalty"] >= 1.05
    assert cfg.adapter("tts", "kokoro").impl.endswith(":KokoroEngine")
    assert set(cfg.hosts) <= {"127.0.0.1", "tailnet"} and cfg.port == 8770
    assert not any("film-rig" in str(p) for p in cfg.pi_extensions.values())


@pytest.mark.parametrize("override,msg", [
    ({"stt": {"adapter": "whisper"}}, "stt.adapters has only"),
    ({"server": {"hosts": ["0.0.0.0"]}}, "server.hosts may hold only"),
    ({"pi": {"extensions": {"film": "~/x/film-rig.ts"}}}, "never include film-rig.ts"),
    ({"server": {"port": "8770"}}, "server.port must be int"),
    ({"version": 2}, "version must be 1"),
])
def test_config_errors_name_the_key(override, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(overrides=override)


def test_adapters_build_from_impl():
    cfg = load_config(overrides={"stt": {"adapter": "fake"}, "tts": {"adapter": "fake"}})
    assert isinstance(cfg.stt.build(), FakeTranscriber) and isinstance(cfg.tts.build(), FakeSynthesizer)


async def test_mlx_worker_runs_on_one_thread_and_refuses_when_guarded():
    reason = {"v": None}
    w = MLXWorker(guard=lambda: reason["v"], use_mlx=False)
    ids = {await w.run(threading.get_ident) for _ in range(5)}
    assert len(ids) == 1 and threading.get_ident() not in ids
    reason["v"] = "hold gate is held (render)"
    with pytest.raises(GpuHeldError, match="held"):
        await w.run(lambda: 1)
    assert w.submit(lambda: 2).result() == 2          # cleanup is never refused
    assert w.refused == 1
    w.shutdown()


async def test_hold_monitor_phases_guard_and_wait_open():
    gate = FakeGate().start()
    try:
        mon = HoldMonitor(gate.url, poll_s=0.05)
        changes = []
        mon.subscribe(lambda old, new: changes.append((old.phase, new.phase)))
        assert (await mon.start()).phase == "open" and mon.gpu_refusal() is None
        gate.set("held", "render the-film")
        await asyncio.sleep(0.2)
        assert mon.phase == "held" and "render the-film" in mon.gpu_refusal()
        waiter = asyncio.create_task(mon.wait_open())
        await asyncio.sleep(0.2)
        assert not waiter.done()
        gate.set("open")
        st = await asyncio.wait_for(waiter, 5)
        assert st.phase == "open" and ("held", "open") in changes
        await mon.stop()
    finally:
        gate.stop()


async def test_hold_monitor_treats_a_missing_gate_as_absent_and_allowed():
    mon = HoldMonitor("http://127.0.0.1:9", poll_s=0.05, timeout_s=0.3)
    st = await mon.check()
    assert st.phase == "absent" and st.allows_gpu and mon.gpu_refusal() is None
    await mon.stop()


def test_fake_synthesizer_enforces_one_stream_and_records_early_close():
    s = FakeSynthesizer({"seconds_per_char": 0.05})
    a = s.stream("hello there")
    next(a)
    with pytest.raises(RuntimeError, match="second stream"):
        next(s.stream("again"))
    a.close()
    assert s.closed_early == 1 and s.spoken == ["hello there"]
    assert len(list(s.stream("short"))) >= 4


def test_fake_streaming_transcriber_reveals_words_and_finishes_on_close():
    eng = FakeStreamingTranscriber({"text": "check the weather please", "words_per_s": 4})
    s = eng.open()
    s.feed(np.full(8000, 0.2, np.float32))       # 0.5 s of loud audio: 2 words
    assert "".join(s.step()) == "check the"
    s.close()
    assert "".join(s.step()) == " weather please" and s.done


async def test_a_whole_buffer_gets_silence_after_it_before_the_streaming_session_closes():
    """Nemotron dropped a last word with only the VAD's 0.2 s after it ("...says moo?" -> "...says", e2e 2026-10-05)."""
    from local_voice.services.stt import TAIL_PAD_S, transcribe_buffer

    eng = FakeStreamingTranscriber({"text": "which animal says moo", "words_per_s": 4})
    pcm = (np.full(16000, 0.2, np.float32) * 32767).astype("<i2").tobytes()     # 1 s of speech, nothing after it
    text = await transcribe_buffer(MLXWorker(use_mlx=False), eng, pcm)
    assert text == "which animal says moo"
    assert eng.last_session.quiet_tail_at_close_s == pytest.approx(TAIL_PAD_S)


async def test_live_stt_drain_returns_when_the_vad_stops_mid_drain():
    """Regression (e2e 2026-10-05): a VAD stop landing during a drain kept it stepping an unclosed session 10,000 times
    (2.4 s), so a short first segment's final transcript came after the turn had already ended."""
    from local_voice.services.stt import _Utterance

    class Session:
        def __init__(self):
            self.closed = False
            self.steps = 0

        @property
        def done(self):
            return self.closed and self.steps > 2

        def step(self):
            self.steps += 1
            utt.closed = True                 # the VAD stop arrives while we are stepping
            return []

        def close(self):
            self.closed = True

    class Svc:
        _worker = MLXWorker(use_mlx=False)

    utt = _Utterance.__new__(_Utterance)
    utt.svc, utt.session, utt.closed, utt.session_closed, utt.text = Svc(), Session(), False, False, ""
    await utt._drain()
    assert utt.session.steps <= 2               # back to the loop to close the session, not 10,000 steps
    utt.session.close()
    utt.session_closed = True
    await utt._drain()
    assert utt.session.done
    Svc._worker.shutdown()
