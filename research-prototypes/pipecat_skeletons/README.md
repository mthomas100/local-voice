# Pipecat 1.12 skeletons for the orchestrator

Starting points for the `orchestrator` agent, written 2026-10-05 for due-diligence task 04c. Every claim in the
code comments cites the installed pipecat-ai 1.12.0 source as `file:line`; the full reasoning, with the source
excerpts, is research note 04c, kept with the design notes in a private wiki.

These files construct and import without loading any model. They have not run a turn: no audio has gone
through them, no model was called, and no socket was bound. Treat every behaviour described here as read from
the source, not observed, until the orchestrator's tests run it.

## Files

| File | What it shows |
|---|---|
| `mlx_thread.py` | the single MLX executor thread, and why Pipecat's own `asyncio.to_thread` pattern is not safe for MLX |
| `mlx_services.py` | `MLXSegmentedSTTService` (Parakeet per utterance, raw PCM in, finalized transcripts out) and `MLXStreamingTTSService` (one chunk per MLX call, stop flag on barge-in, leading-silence trim with Pipecat's own onset detector); engine protocols plus fakes |
| `pi_bridge_service.py` | `PiBridgeLLMService(LLMService)`: sends only new user messages to the agent, streams `text_delta` as `LLMTextFrame`, speaks a canned acknowledgement on the first tool call, emits `tool` UI events, aborts the agent on barge-in without blocking, tells the agent what the user heard; a fake backend |
| `protocol_v1.py` | `ProtocolV1Serializer` (PROTOCOL.md on the wire), `ProtocolV1Ears` (transcript, listening/thinking, pong), `ProtocolV1Mouth` (audio_start, reply_text, audio_end, end_of_turn, speaking) |
| `pipeline_factory.py` | the one pipeline shape for both entry points, turn configuration for open mic, push-to-talk and model-free tests, the code-fence splitter, the worker settings (no idle timeout, metrics on, RTVI only for the browser) |
| `ws_server.py` | FastAPI app for protocol v1 on :8770: `tailscale whois` check, close 4403, hello/welcome, one pipeline per connection, `/v1/status`, binding to exactly 127.0.0.1 and the tailnet address |
| `browser_entry.py` | the dev runner's SmallWebRTC routes rebuilt without the public STUN server, the prebuilt client at `/client`, an optional tailnet-only ICE filter |
| `latency_jsonl.py` | one JSON line per turn from `UserBotLatencyObserver`'s breakdown, plus TTFA (leading silence split out), Smart Turn and usage metrics |
| `thinking_sound.py` | the working loop on `SoundfileMixer`, with the reasons to use it only on the browser path |
| `check_construct.py` | imports everything and builds both pipelines with fakes; refuses any ONNX model load and fails if an MLX, torch or transformers module was imported |

## Run the check

```bash
cd ~/repos/local-voice/research-prototypes/pipecat_skeletons
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python "pipecat-ai[webrtc,runner,silero]==1.12.0"
.venv/bin/python check_construct.py              # no model loads; about 1 s of CPU
.venv/bin/python check_construct.py --with-onnx  # also builds Silero and Smart Turn (CPU ONNX, ~11 MB)
```

Result on 2026-10-05 (scratch venv, Python 3.12.14, pipecat-ai 1.12.0): every constructor passed, the
serializer round-trips passed, no model library was imported, no ONNX session was opened. `--with-onnx` was
not run (no model was loaded during that check).

## What the orchestrator still has to build and test

- Real engines behind `TranscribeEngine` and `SpeechEngine` (mlx-audio 0.5.7 calls: research note 09c), warmed
  at startup on the MLX thread, behind the hold gate.
- The Pi child per space from `research-prototypes/pi_rpc_bridge/` as the bridge's backend. Integration
  point to verify: after an abort, the aborted run's tail (`message_end`, `agent_end`, `agent_settled`) lands
  in `PiChild.events`; the next `turn()` would read it and stop at the stale `agent_settled`. The bridge waits
  for `busy` to clear and calls `drain()` before every new prompt; test that with the stub LLM.
- Model-free frame-flow tests with `pipecat.tests.utils.run_test` (tests/utils.py:123-165): a context frame
  produces start, text, end; an `InterruptionFrame` mid-turn calls `interrupt()` once and the next turn waits
  for it; a second context frame with no new user message sends nothing; `run_llm=False` on tool frames.
- Push-to-talk: the client's `stop` can overtake the last audio frames (see `protocol_v1.py`); either the
  client sends ~100 ms of silence before `stop`, or a small processor delays the VAD stop frame.
- The spoken yes/no for `confirm_request` (M3), the busy notice while the hold gate is held, keepalive.
