# pi_rpc_bridge: the voice layer's handle on Pi

A tested prototype (due diligence 05c, 2026-10-05) of how the voice orchestrator drives Pi: one `pi --mode rpc`
child per space, prompts in, typed events out, interruption, steering, and spoken confirmation. Everything here
was exercised against a stub LLM; nothing has met a real model yet. The evidence is research note 05c,
kept with the design notes in a private wiki.

| File | What it is |
|---|---|
| `pi_rpc_bridge.py` | the bridge: `SpaceConfig`, `PiChild`, `SpacePool`, typed events, helpers. Standard library only |
| `stub_llm.py` | a scriptable OpenAI-compatible SSE server in llama.cpp's `--jinja` shapes; logs every request verbatim |
| `extensions/voice_gate.ts` | the risk gate from SPACES.md: tiers, a bash allowlist that cannot be stretched, `ctx.ui.confirm` with a timeout |
| `extensions/voice_mode.ts` | `/voice-mode act` and `/voice-mode conversation`: switch the active tool set |
| `tests/` | 42 pytest tests against the stub (about 33 s); `run_tests.sh` runs them |
| `qwen38_voice_tooluse_test.py` | 13 voice requests on the real rig per configuration; waits for the GPU. Run 2026-10-05: `results/` |
| `results/05d-qwen-voice-tooluse-results.md` | what that run found: qwen38 as the live brain, persona via `--system-prompt`, qwen27 needs the chat-template compat, Atlas capture in the orchestrator |

## Run the tests

```
./run_tests.sh                                  # 41 pass, the Atlas test skips without ATLAS_REPO
ATLAS_REPO=/path/to/a/journal/repo ./run_tests.sh   # an Atlas-style journal repo with atlas.py
```

The hold-gate test starts the real `hold` binary (`~/repos/local-rig/hold/hold`, from the companion `local-rig` repo) as a test gate on a free port
with `--no-implicit` and a temporary `--state`; it skips if the binary or `hold.ts` is missing. No test reaches
127.0.0.1:8090, `~/.pi/agent`, the real knowledge base or a real journal repo.

## Using the bridge

```python
from pi_rpc_bridge import SpaceConfig, SpacePool, TextDelta, ToolCallStarted, ToolEnd, ConfirmRequest, Settled

spaces = {"home": SpaceConfig(name="home", root=Path.home(), tools=["read", "grep", "find", "ls", "kb"],
                              extensions=[".../hold.ts", ".../kb.ts", ".../voice_gate.ts"],
                              session_dir=state / "sessions/home", session_name="voice-home",
                              append_system_prompt=[persona])}
pool = SpacePool(spaces, on_ui=speak_and_listen)      # on_ui answers confirm/select by voice; None = cancel
child = await pool.switch("home")
async for ev in child.turn(transcript):
    if isinstance(ev, ToolCallStarted): say_ack(ev.name)        # earliest moment a tool is coming
    elif isinstance(ev, TextDelta): tts.feed(ev.text)
    elif isinstance(ev, Settled): break
# barge-in on a cancellation path (Pipecat cancels the turn task and waits at most 1 s): never await there
child.request_interrupt(heard=words_actually_played)             # returns at once; the abort runs in the background
# elsewhere: await child.interrupt(heard=...) returns once the aborted run has settled and its leftovers are drained
```

Barge-in hygiene (05c E10): every event carries the turn it was read in. `turn()` first waits for a pending
interrupt and for the aborted run's `agent_settled`, drains what that run left behind into `child.stray`, and
then yields only its own events, so a stale `agent_settled` can never end a later turn. If a run will not
settle within `settle_timeout` (10 s), `turn()` raises `PiBusyError` instead of prompting into it. A consumer that
stops reading a turn before `agent_settled` (cancellation, `break`, `aclose`) triggers `request_interrupt()`.

`SpaceConfig.argv()` shows the exact command line. Defaults: `--thinking off`, `--no-extensions` (only the listed
`-e` files load), `--no-prompt-templates`, `--approve`, `--continue` when the session dir already holds a session,
and `PI_OFFLINE=1`, `PI_SKIP_VERSION_CHECK=1` in the child's environment, which also drops `KB_SESSION`,
`KB_ACTOR`, `CLAUDE_ENV_FILE` and herdr's variables inherited from the parent.

## What the bridge relies on (measured in 05c)

- A turn ends at `agent_settled`, not `agent_end`: retries (`agent_end` with `willRetry: true`) and queued
  follow-ups run inside the same settle. An extension's `agent_end` handler delays it (`kb.ts`: about 170 ms).
- `prompt` is answered only after every `before_agent_start` handler returns. `film-rig.ts` waits there for a
  whole render, with no `agent_start` and an `abort` that answers but cancels nothing. Keep it out of voice
  children; `hold.ts` waits in the `context` hook instead, which an abort does cancel.
- After `abort`, Pi drops the partial assistant reply from what it sends the model next. `interrupt(heard=...)`
  prefixes the next prompt with what the user actually heard.
- Pi writes an aborted run's tail (`message_end` "aborted", `turn_end`, `agent_end`, `agent_settled`) before the
  abort's own response, so those records are queued when an abort returns: hence the turn tags and the drain.
- An abort does not dismiss an extension's open confirm dialog unless the extension passes `signal: ctx.signal`
  (it waited the dialog's full 20 s otherwise). `voice_gate.ts` passes it; for other extensions the bridge sends
  the abort first and then cancels every open dialog, so the run settles at once with no extra model call.
- `tool_execution_start` precedes the gate's confirm request: do not say "running it" until the tool's
  updates or end arrive, or a `ConfirmRequest` has been answered yes.
- A late `extension_ui_response` (after the dialog's timeout) is ignored silently.
- `--tools` decides what is registered: `voice_mode.ts` can only activate tools the child was started with.
- Records exceed asyncio's 64 KiB line limit (a 50 KB file read made a 100 KB record), hence `STREAM_LIMIT`.

## The voice gate

Env, set per child: `VOICE_TIER` (`readonly`, `ask`, `trusted`), `VOICE_CONFIRM_TIMEOUT_MS` (default 20000),
`VOICE_BASH_ALLOW` and `VOICE_WRITE_ALLOW` (JSON arrays of globs, write globs relative to the root),
`VOICE_READ_TOOLS`, `VOICE_DENY_TERMINATES=1`. A bash command matches the allowlist only if its first line has no
shell control characters; the one multi-line form allowed is a quoted heredoc whose marker line appears once, at
the end, which is exactly what `atlas_capture_command()` builds. One leading `cd` into the space root is stripped first
(a plain, quoted or escaped path inside the root, or the literal no-ops `"$(pwd)"`, `$PWD`, `${PWD}`, `.`): qwen38
wraps every Atlas capture in one (05d). Chained commands, other `$(...)`, unquoted heredocs, smuggled markers and a
`cd` that leaves the root fall through to asking.

## The GPU test

`qwen38_voice_tooluse_test.py` waits for its turn: it exits 75 and does nothing unless
`measure/bench/gpu_clear.sh` exits 0, and stops if a render or a hold starts between requests. Options added for the
2026-10-05 run: `--legs model:append|replace[:compat]`, `--compat-on-leak`, `--repeat N`, `--spaces home|atlas`,
`--orchestrator-capture` (RPC `bash` capture, byte check, then the reply turn), `--leak-abort-chars`.
By default it runs the 13 requests on `local/qwen38`, then `local/qwen27-262k`,
then one request on `local/qwen38` so the rig is left on its default, and writes `results/voice-tooluse-<time>.json`
and `.md`. `--dry-run` prints the plan; `--stub` runs the whole harness against the stub (a self-test; it passed on
2026-10-05); `--qwen27-compat qwen-chat-template` and `--prompt-mode replace` are the two follow-ups the results
may call for. Its header explains the model switches and the isolation.
