"""Import every skeleton and construct both pipelines without loading any model.

    uv run --python <venv with pipecat-ai[webrtc,runner,silero]==1.12.0> python check_construct.py
    python check_construct.py --with-onnx   # also builds Silero + Smart Turn (2.3 MB + 8.7 MB ONNX on CPU)

Guards: onnxruntime.InferenceSession is replaced by a function that raises, unless --with-onnx, so a
constructor that loads an ONNX model fails the check; afterwards no mlx, mlx_audio, torch or transformers
module may have been imported. Nothing is served, no socket is bound, no frame flows through a pipeline.
The serializer is exercised directly on synthetic bytes and frames (pure functions).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FORBIDDEN_MODULES = ("mlx", "mlx_audio", "mlx_lm", "torch", "transformers", "parakeet_mlx", "mlx_whisper")


def guard_onnx(allow: bool) -> None:
    import onnxruntime

    if allow:
        return

    def refuse(*args, **kwargs):
        raise RuntimeError(f"model load attempted: onnxruntime.InferenceSession{args[:1]}")

    onnxruntime.InferenceSession = refuse  # type: ignore[assignment]


class StubWebSocket:
    """Enough of starlette's WebSocket for FastAPIWebsocketTransport.__init__ (fastapi.py:629-672)."""

    headers: dict = {}
    client = ("127.0.0.1", 50000)


async def main(with_onnx: bool) -> int:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from pipecat.frames.frames import (
        InputAudioRawFrame,
        InterruptionFrame,
        InterruptionWorkerFrame,
        LLMMessagesAppendFrame,
        OutputAudioRawFrame,
        OutputTransportMessageFrame,
        OutputTransportMessageUrgentFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

    import browser_entry
    import latency_jsonl
    import mlx_services
    import pi_bridge_service
    import pipeline_factory
    import protocol_v1
    import thinking_sound
    import ws_server

    results: list[tuple[str, str]] = []

    def ok(what: str, detail: str = "") -> None:
        results.append((what, detail))

    ok("pipecat-ai version", version("pipecat-ai"))

    def services():
        stt = mlx_services.MLXSegmentedSTTService(engine=mlx_services.FakeTranscribeEngine())
        tts = mlx_services.MLXStreamingTTSService(
            engine=mlx_services.FakeSpeechEngine(), sample_rate=24000, skip_aggregator_types=["code"]
        )
        llm = pi_bridge_service.PiBridgeLLMService(
            backend=pi_bridge_service.FakeAgentBackend(),
            ack_phrases={"kb": "Let me look that up."},
            tool_labels={"kb": "searching your knowledge base"},
        )
        return stt, llm, tts

    stt, llm, tts = services()
    ok("MLXSegmentedSTTService", f"wants_wav_segments={stt.wants_wav_segments}")
    ok("MLXStreamingTTSService", "push_start_frame/push_stop_frames on")
    ok("PiBridgeLLMService", f"adapter={type(llm.get_llm_adapter()).__name__}")

    # ---- serializer, both directions (pure)
    ser = protocol_v1.ProtocolV1Serializer()
    pcm_up = bytes(640)  # 20 ms at 16 kHz
    f = await ser.deserialize(pcm_up)
    assert isinstance(f, InputAudioRawFrame) and f.sample_rate == 16000 and f.num_frames == 320
    assert isinstance(await ser.deserialize('{"t":"interrupt","reply_id":"r1"}'), InterruptionWorkerFrame)
    assert isinstance(await ser.deserialize('{"t":"start"}'), VADUserStartedSpeakingFrame)
    assert isinstance(await ser.deserialize('{"t":"stop"}'), VADUserStoppedSpeakingFrame)
    assert isinstance(await ser.deserialize('{"t":"text","text":"hi"}'), LLMMessagesAppendFrame)
    cm = await ser.deserialize('{"t":"played_ms","reply_id":"r1","ms":2140}')
    assert isinstance(cm, protocol_v1.ClientMessageFrame) and cm.message["ms"] == 2140
    ser.client_interrupted = False
    assert await ser.serialize(InterruptionFrame()) is None  # nothing playing: no interrupt sent
    start = await ser.serialize(OutputTransportMessageFrame(message={"t": "audio_start", "reply_id": "r2", "rate": 24000}))
    assert json.loads(start)["reply_id"] == "r2"
    down = await ser.serialize(OutputAudioRawFrame(audio=bytes(1920), sample_rate=24000, num_channels=1))
    assert isinstance(down, bytes) and len(down) == 1920
    intr = json.loads(await ser.serialize(InterruptionFrame()))
    assert intr == {"t": "interrupt", "reply_id": "r2"}
    state = await ser.serialize(OutputTransportMessageUrgentFrame(message={"t": "state", "v": "speaking"}))
    assert json.loads(state)["v"] == "speaking"
    rtvi = OutputTransportMessageUrgentFrame(message={"label": "rtvi-ai", "type": "bot-ready"})
    assert await ser.serialize(rtvi) is None
    ok("ProtocolV1Serializer", "audio up/down, interrupt, start/stop, text, played_ms, rtvi filtered")

    # ---- protocol v1 pipeline on the FastAPI transport (no socket: a stub stands in for the WebSocket)
    tmp = Path(tempfile.mkdtemp(prefix="pipecat_skeletons_"))
    log = latency_jsonl.TurnLatencyLog(tmp / "latency.jsonl", session="check")
    ws_transport = FastAPIWebsocketTransport(
        websocket=StubWebSocket(),  # type: ignore[arg-type]
        params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=16000,
                                      audio_out_sample_rate=24000, audio_out_10ms_chunks=4,
                                      audio_out_end_silence_secs=0, serializer=protocol_v1.ProtocolV1Serializer()),
    )
    for mic in ("vad", "ptt"):
        params = pipeline_factory.user_params(mic=mic, load_models=with_onnx)
        stt, llm, tts = services()
        pipeline, pair = pipeline_factory.build_pipeline(transport=ws_transport, stt=stt, llm=llm, tts=tts,
                                                         params=params, protocol_v1=True)
        worker = pipeline_factory.build_worker(pipeline, observers=log.observers, rtvi=False, name=f"check-{mic}")
        names = [type(p).__name__ for p in pipeline.processors]
        ok(f"protocol v1 pipeline ({mic})", " -> ".join(names))
        ok(f"PipelineWorker ({mic})", f"rtvi={worker._rtvi is not None}, idle_timeout={worker._idle_timeout_secs}")

    # ---- browser pipeline on SmallWebRTC with the working-sound mixer (connection object only; no SDP)
    wav = thinking_sound.write_working_loop(tmp / "working.wav")
    conn = SmallWebRTCConnection()
    rtc_transport = SmallWebRTCTransport(
        webrtc_connection=conn,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=16000,
                               audio_out_sample_rate=24000,
                               audio_out_mixer=thinking_sound.make_working_mixer(wav)),
    )
    stt, llm, tts = services()
    llm = pi_bridge_service.PiBridgeLLMService(backend=pi_bridge_service.FakeAgentBackend(),
                                               ui_event=browser_entry.rtvi_message, use_mixer_for_tools=True)
    pipeline, _ = pipeline_factory.build_pipeline(transport=rtc_transport, stt=stt, llm=llm, tts=tts,
                                                  params=pipeline_factory.user_params(load_models=with_onnx),
                                                  protocol_v1=False)
    worker = pipeline_factory.build_worker(pipeline, observers=log.observers, rtvi=True, name="check-browser")
    ok("browser pipeline", " -> ".join(type(p).__name__ for p in pipeline.processors))
    ok("PipelineWorker (browser)", f"rtvi={worker._rtvi is not None}")
    await conn.disconnect()

    # ---- the two FastAPI apps (constructed, not served)
    app = ws_server.create_app(allowed_logins={"owner@example"}, make_services=services, status=lambda: {"v": 1})
    ok("protocol v1 app routes", ", ".join(sorted(r.path for r in app.routes if r.path.startswith("/v1"))))
    bapp = browser_entry.create_browser_app(make_services=services)
    ok("browser app routes", ", ".join(sorted({r.path for r in bapp.routes})))

    loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    assert not loaded, f"model libraries imported: {loaded}"
    ok("no model library imported", ", ".join(FORBIDDEN_MODULES))
    ok("ONNX loads", "Silero + Smart Turn constructed" if with_onnx else "none (InferenceSession guarded)")

    width = max(len(w) for w, _ in results)
    for what, detail in results:
        print(f"OK  {what.ljust(width)}  {detail}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-onnx", action="store_true", help="also build Silero and Smart Turn (CPU ONNX)")
    a = ap.parse_args()
    guard_onnx(a.with_onnx)
    sys.exit(asyncio.run(main(a.with_onnx)))
