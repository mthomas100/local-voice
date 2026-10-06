# orchestrator

The voice agent's server: you talk, it answers aloud, and it acts through your Pi agent. Python 3.12, Pipecat 1.12.0,
mlx-audio 0.5.7, Pi over RPC. The wire contract is in `PROTOCOL.md` and spaces in `SPACES.md` (both in the repo's
docs); the speech measurements behind the defaults are in `../measure/`.

## Run it

```bash
cd orchestrator
./run.sh                      # refuses (exit 75) while the hold gate is draining or held
```

It loads and warms the speech models named in `config.yaml`, renders the busy notice, and serves:

| What | Where | Who may connect |
|---|---|---|
| protocol v1 (the iPhone and Mac apps) | `ws://127.0.0.1:8770/v1/voice` and `ws://<tailnet ip>:8770/v1/voice` | loopback, and tailnet peers whose Tailscale login is in `server.allowed_logins` (default: your own) |
| status (the dashboard JSON) | `http://127.0.0.1:8770/v1/status` (and the tailnet address) | same |
| browser page | `http://127.0.0.1:7860/` | this Mac only |

`./run.sh --check` loads, warms and renders, then exits. Try a spoken turn without an app:

```bash
.venv/bin/python -m local_voice.client --say "what does my knowledge base say about the hold gate?"
```

## Choose models and behaviour

Everything is in `config.yaml`; local overrides go in `config.local.yaml` beside it (gitignored), merged key by key.

- Ears: `stt.adapter` is `nemotron` (live session, partials, final 53-54 ms after you stop) or `parakeet` (per
  utterance).
- Voice: `tts.adapter` is `qwen3` (Qwen3-TTS 1.7B CustomVoice, speaker Ryan) or `kokoro`, `pocket`, `marvis`. Any
  mlx-audio model works through an adapter entry: `impl: module:Class` plus its settings.
- Turn-taking: `turn.vad` (Silero) and `turn.smart_turn` (Smart Turn v3.2, with a silence fallback).
- What the agent can reach: `spaces.yaml` (SPACES.md). The persona is `persona.md`, sent as Pi's `--system-prompt`.
- The spoken acknowledgements and the labels clients show while a tool runs: `agent.acks`, `agent.labels`.

## Layout

| Path | What |
|---|---|
| `local_voice/server.py` | the process: startup, protocol v1 endpoint (access check, close codes, keepalive, resume), `/v1/status`, serving |
| `local_voice/pipeline.py` | one Pipecat pipeline per connection: STT, user aggregator (VAD, Smart Turn), agent, TTS, notices |
| `local_voice/bargein.py` | barge-in: the reply's audio stops at the first speech frame, `interrupt` follows when the VAD confirms, an unconfirmed pause carries on (protocol v1 output) |
| `local_voice/agent.py` | `AgentHub` (one Pi child per space, process-wide) and `PiAgentService` (Pi as Pipecat's "LLM") |
| `local_voice/pi_rpc.py` | the Pi RPC bridge, carried over from the tested 05c prototype |
| `local_voice/services/` | the STT services (live and per utterance) and the TTS service over the MLX adapters |
| `local_voice/engines/` | the model adapters (Nemotron, Parakeet, Qwen3-TTS, Kokoro, any streaming mlx-audio TTS) and fakes |
| `local_voice/mlx_worker.py` | the single MLX thread, with the hold gate's refusal enforced at its door |
| `local_voice/hold.py`, `session.py` | the hold gate as the orchestrator sees it; the busy notice, held turns and their resumption |
| `local_voice/protocol_v1.py` | the serializer and the two processors that speak PROTOCOL.md; the `played_ms` timeline |
| `local_voice/browser.py`, `static/index.html` | the browser entry: our own WebRTC routes and page (nothing fetched off the Mac) |
| `local_voice/client.py` | the protocol v1 test client (also a command) |
| `local_voice/spaces.py`, `agent_dir.py` | `spaces.yaml` loading and checks; the Pi agent dir derived from your models.json |
| `pi/` | this project's Pi extensions: `voice_gate.ts` (the risk gate), `voice_mode.ts` (conversation and act modes) |
| `tests/` | model-free tests (fakes, the stub LLM, a real Pi child, a test hold gate); `tests/e2e/` needs the GPU |
| `tools/ttft_probe.py` | the LLM time-to-first-token probe: our exact Pi child through a logging proxy, turns at chosen gaps, ds4's own prefill times and keepalive period (makes LLM calls: gpu_clear.sh first) |
| `tools/bargein_bench.py`, `tools/fetch_esc50.sh` | the barge-in bench (CPU): Silero settings and pause designs against `say` interruptions and planted ESC-50 controls (coughs, knocks, typing...) |
| `tools/smart_turn_bench.py` | Smart Turn v3.2 on `say` speech (CPU): finished requests judged unfinished |

## Tests

```bash
.venv/bin/python -m pytest -q                      # model-free: fakes, stub LLM, real Pi, test gate (~90 s)
.venv/bin/python -m pytest -m e2e tests/e2e -s     # real models and qwen38; only when gpu_clear.sh says CLEAR
```

The model-free suite never reaches `:8090`, `~/.pi/agent` (its `models.json` is only read), the knowledge base or the journal. The
e2e suite runs on a clone of the knowledge base (`KB_HOME`; Pi's `kb.ts` writes a session digest) and writes its results to
`../state/orchestrator-e2e/<time>/`.
