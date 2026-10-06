# Wire protocol v1: native clients ↔ voice orchestrator

Frozen for implementation on 2026-10-05. The orchestrator and every native client (the SwiftUI app on iPhone and
Mac, and the Python test client) implement exactly this. A change that breaks a v1 client gets a new path
(`/v2/voice`), never an edit here; additive message types are allowed in v1 because receivers ignore unknown ones.

The browser client does not use this protocol. It uses Pipecat's SmallWebRTC transport, because browsers only
apply echo cancellation to audio inside a WebRTC call, not to a raw WebSocket stream (Pipecat client docs). Both entry points feed the same pipeline code.

Why a WebSocket with raw PCM, not WebRTC, for native clients: on iOS and macOS the echo canceller is Apple's
voice-processing unit either way; a WebSocket keeps the playback buffer under our control, which is what makes
barge-in instant; the measured Tailscale path is direct WireGuard at single-digit to tens of milliseconds and raw
PCM is about 0.3-0.4 Mbit/s each way (measured 2026-10-04).

## Endpoint and access

- `ws://<host>:8770/v1/voice`. The orchestrator listens on 127.0.0.1 and on this Mac's tailnet address only, never
  on a LAN or public interface, and never through `tailscale funnel`.
- A connection from a non-loopback address is accepted only if `tailscale whois <peer ip:port>` names a login in the
  orchestrator's `allowed_logins` config (the owner's own devices). Loopback is allowed explicitly (`whois` fails
  for 127.0.0.1); the check takes about 30 ms. The server accepts the socket and then closes it with 4403, because a
  close before accept surfaces to the client as a bare HTTP 403.
- The WireGuard tunnel already encrypts the traffic. When HTTPS certificates are enabled on the tailnet, the same
  path is served as `wss://` through `tailscale serve`; until then the iOS app carries an App Transport Security
  exception for the tailnet domain.
- `GET http://<host>:8770/v1/status` returns the dashboard JSON (section "Status endpoint"); same access rules.

## Audio

| Direction | Format | Framing |
|---|---|---|
| client → server | PCM signed 16-bit little-endian, mono, 16 000 Hz | one binary WebSocket message per 20-40 ms (640-1280 bytes) |
| server → client | PCM signed 16-bit little-endian, mono, 24 000 Hz | one binary WebSocket message per 20-40 ms (960-1920 bytes) |

- Keep every binary message under 3 000 bytes (a reported `URLSessionWebSocketTask` failure above that size).
- The client converts from the hardware rate (usually 48 kHz) with `AVAudioConverter`; never ask the hardware for
  16 kHz. The client sends echo-cancelled audio (voice-processing input on Apple platforms).
- Server audio for one reply is bracketed by `audio_start` and `audio_end` control messages carrying the reply id.

## Control messages

JSON text messages. Every message has `"t"` (type). Times are milliseconds. Unknown types are ignored by both sides.

### Client → server

| t | Fields | Meaning |
|---|---|---|
| `hello` | `v` (1), `client` (`iphone`, `mac`, `test`), `device` (name), `mic` (`vad` or `ptt`), `space` (optional) | first message; the server answers `welcome` |
| `start` | | push-to-talk pressed: start of a user turn (`mic: ptt`) |
| `stop` | | push-to-talk released: end of the user turn, no turn detection needed |
| `interrupt` | `reply_id` | the client stopped playback itself (user tapped stop, or client-side barge-in) |
| `played_ms` | `reply_id`, `ms` | how much of a reply's audio was actually played; sent after any interrupt and at the end of every reply |
| `text` | `text` | typed input, treated as a user turn |
| `space` | `name` | ask to switch space (same as saying "go to …") |
| `mode` | `name` (`conversation` or `act`) | switch mode |
| `confirm_response` | `id`, `confirmed` (bool) (+ `choice`: see "Approvals") | answer to a `confirm_request` (a button, or the client's own speech UI) |
| `ping` | `n` | keepalive, every 15 s |

### Server → client

| t | Fields | Meaning |
|---|---|---|
| `welcome` | `v`, `session`, `space`, `mode`, `tier`, `state`, `hold` | handshake answer |
| `state` | `v` (`idle`, `listening`, `thinking`, `speaking`, `held`, `error`) | drive the UI and the Live Activity |
| `transcript` | `final` (bool), `text` | what the user said; partials may be replaced |
| `reply_text` | `reply_id`, `delta` | assistant text as it streams (captions); code and tool output never appear here |
| `audio_start` | `reply_id`, `rate` (24000) | binary audio for this reply follows |
| `audio_end` | `reply_id` | no more audio for this reply |
| `interrupt` | `reply_id` | flush local playback now; the user barged in or the reply was cancelled |
| `end_of_turn` | `reply_id` | the agent is done; the client may arm its mic gate |
| `tool` | `phase` (`start`, `update`, `end`), `name`, `label`, `ok` | what the agent is doing ("reading your journal") |
| `confirm_request` | `id`, `title`, `message`, `timeout_ms` (+ `summary`, `action`, `choices`: see "Approvals") | the agent wants permission; it is also asked aloud |
| `confirm_cancel` | `id`, `why` | the question is withdrawn (answered by voice, timed out, or overtaken); close its card |
| `space` | `name`, `mode`, `tier`, `description` | current space and mode, after any switch |
| `hold` | `phase` (`open`, `draining`, `held`), `why` | the Mac's GPU lock; while held the agent cannot think or speak with its models |
| `error` | `code`, `message` | non-fatal problem; fatal ones close the socket |
| `pong` | `n` | keepalive answer |

### Answers to `space` and `mode` (added 2026-10-05, additive)

The protocol gives `space` and `mode` no request id, so a switch is answered by state: the server sends a `space`
message with the new space and mode when the switch happened (also after a spoken "go to …"), and an `error` with
`code` `space_unknown`, `space_unavailable` (its root is missing, or its child cannot start) or `mode_unknown` when it
did not. A client treats an `error` that arrives while its switch is pending as that switch's refusal, and no answer
within 5 s as "no answer" (the M1 server ignores both messages; M3 implements them).

### Approvals (added 2026-10-05, additive)

In early live use the plain v1 question ("May I change your knowledge base? It starts with kb new. Yes or no?") proved
too vague: an approval should give what coding agents give, exactly what will happen and a choice the person can see
and pick. So a `confirm_request` also carries:

| Field | Meaning |
|---|---|
| `summary` | one plain sentence saying exactly what will happen and to what ("Create a new page in your knowledge base titled 'Weekly plan'.") |
| `action` | `{tool, effect, command, path, cwd, space, mode, preview}`: the tool (`write`, `edit`, `bash`, `kb`, …); the effect (`create`, `modify`, `delete`, `run`, `network`); the exact command line for a shell or kb call; the absolute path of the file written or edited; the working directory; the space and mode; and a preview: the text to be written, or a unified diff for an edit, cut at 4,000 characters with a visible marker |
| `choices` | `[{id, label}]` in order: at least `allow_once` ("Do it") and `deny` ("Don't"); the server may add `allow_session` ("Allow this for the rest of the session", scoped to the same tool and the same command prefix or folder, never broader) |

and `confirm_response` also carries `choice` (one of the offered ids); `confirmed` stays, true for any allow and false for
`deny`, for clients that send only it. A new server message, `confirm_cancel` (`id`, `why`), withdraws a question
(answered by voice, timed out, or overtaken by a barge-in), so a client closes its card.

Behaviour: the voice asks with the summary ("I'd like to create a new page in your knowledge base titled …. Shall I?")
and takes "yes" as `allow_once`, "no" as `deny`, and "yes, for this session" as `allow_session`. A client that can show
it renders a card with the summary as its headline, the exact command or path, the preview, and one button per choice,
and keeps it until it is answered or cancelled. While such a client is connected the server waits at least 120 s and
asks once more aloud before giving up; silence is still no, and the agent then says what it did not do.

Details both sides rely on (settled 2026-10-05 from what the server and the apps built):

- **The cut marker** is the preview's last line, `… [cut here: N more characters not shown]`, after the first 4,000
  characters; clients pin a last line that starts with "…" and speaks of a cut.
- **`confirm_cancel.why`** is one of `answered` (by voice, or by another client), `timeout`, `overtaken` (a new turn,
  a barge-in or an abort ended the run that asked). These are codes, not text to show: a client says each in its own
  words (the browser page: "Answered by voice.", "No answer, so it was not done."; "Overtaken by what you said next.")
  and shows an unknown value as written.
- **`timeout_ms` is the whole wait.** The second spoken ask (at the halfway point, with a card client connected) is
  speech only: the server does not send the `confirm_request` again. A `confirm_request` whose id is already waiting
  replaces it and carries the time that remains.
- **Reconnects.** A dropped connection closes the client's card. When a client resumes its session (same `device`
  within 5 minutes) while a question is still open, the server sends that `confirm_request` again with the time that
  remains.
- **Answers** carry `confirmed`, plus `choice` when choices were offered; a choice that was not offered is never sent;
  a client always shows a way to refuse, even if `deny` is missing.
- **`/v1/status`** gains `approval` (the question waiting, as sent, or null) and `grants` (the session allowances, each
  `{session, space, scope, label, since, uses}`).

### Typed turns

For a `text` turn the server sends no `transcript`; the client shows what it sent (as the M1 server and the apps do).

### Proposed, not implemented

- `ptt_token` (client → server), `token` (hex): the iPhone's Push to Talk ephemeral push token, so the server could wake
  the phone for a reply through Apple's push service. Proposed 2026-10-05; Push to Talk is
  behind a build flag in the app and needs your own APNs key, and the server side is not built. Additive, so it
  needs no new protocol version when it lands.

## Close codes

| Code | Meaning |
|---|---|
| 1000 | normal close |
| 1002 | protocol error: a missing or malformed `hello`, an unknown `v`, a binary frame before `hello` |
| 4403 | not allowed: the peer is not one of the owner's tailnet logins |
| 4409 | another session with the same `device` took over |

## Behaviour both sides rely on

- **When `interrupt` is sent.** Only while a reply is active, between `audio_start` and the client's `played_ms`.
  The pipeline also signals an interruption at the start of every user turn; the server does not forward those.
- **Push-to-talk tail.** Before `stop`, a push-to-talk client sends about 100 ms of silence so the last word is
  flushed through the VAD before the turn closes.
- **Barge-in.** On a server `interrupt` the client stops its player, drops queued buffers, and replies with
  `played_ms`. The flush happens locally before anything else; a client holding 800 ms of audio otherwise keeps
  talking for most of a second. The server truncates the assistant turn in its own context to the words that were
  actually played (Pipecat records spoken text) and tells the agent backend it was interrupted.
- **Mic gate.** After `audio_end`, the client ignores its microphone for 500-800 ms once its playback drains, unless
  new audio arrives, so the residual echo tail of Apple's voice processing is not heard as the user (a reported field issue). Push-to-talk clients skip the gate.
- **Pre-connect buffering.** A client starts capturing the moment the user presses talk and sends the buffered frames
  once the socket is up, so the first words are not lost.
- **Held GPU.** While `hold.phase` is `held`, the server answers a turn with a short pre-rendered spoken notice and
  state `held`, keeps the transcript if it can, and runs the turn when the hold ends. It never loads a model during a
  hold (a rule since 2026-10-04: a deliberate GPU hold always wins).
- **Keepalive.** Either side closes after 60 s without any message.
- **Reconnect.** A client that reconnects with the same `device` within 5 minutes resumes the same space and agent
  session; audio in flight is lost.

## Status endpoint

`GET /v1/status` returns JSON for the dashboard and the clients:

```json
{"v": 1, "state": "idle", "space": "home", "mode": "conversation", "tier": "ask",
 "model": "local/qwen38", "hold": {"phase": "open", "why": ""},
 "tool": null, "last_turn": {"eos_to_first_audio_ms": 0, "stt_ms": 0, "llm_ttft_ms": 0, "tts_first_audio_ms": 0},
 "clients": [{"device": "iphone", "connected_s": 0}]}
```

Fields the orchestrator added in M1 (2026-10-05; additive, so clients ignore what they do not use): `spaces` (each
space from `spaces.yaml` by name: `name`, `description`, `root`, `tier`, `model`, `tools`, `act_tools`, `skills`; the
apps list spaces from it), `turns`, `children` (the Pi children running), `digest_pending` (a background-brain
digest waits to be spoken), `speech` (the speech models loaded), `uptime_s`, and `client` on each entry of `clients`.

## Example session

```
C: {"t":"hello","v":1,"client":"iphone","device":"iphone","mic":"vad"}
S: {"t":"welcome","v":1,"session":"s1","space":"home","mode":"conversation","tier":"ask","state":"listening","hold":"open"}
C: <binary PCM 16k> ... (continuous)
S: {"t":"transcript","final":true,"text":"what did I write in my journal yesterday"}
S: {"t":"state","v":"thinking"}
S: {"t":"tool","phase":"start","name":"read","label":"reading your journal"}
S: {"t":"audio_start","reply_id":"r1","rate":24000}
S: <binary PCM 24k: "Let me look."> 
S: {"t":"tool","phase":"end","name":"read","ok":true}
S: {"t":"reply_text","reply_id":"r1","delta":"Yesterday you wrote about"}
S: {"t":"state","v":"speaking"}
S: <binary PCM 24k> ...
C: <user starts talking over it>
S: {"t":"interrupt","reply_id":"r1"}
C: {"t":"played_ms","reply_id":"r1","ms":2140}
S: {"t":"state","v":"listening"}
```
