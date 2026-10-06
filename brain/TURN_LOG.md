# The turn log and the spoken-digest outbox: what the orchestrator writes and reads for the brain

Interface v1, 2026-10-05. The background brain (M4) reads nothing else from a conversation: not Pi's session files,
not audio. The orchestrator owns writing the turn log and reading the outbox; the brain owns reading the turn log and
writing the outbox. Both live under the repo's gitignored `state/`, because they hold the person's words.

## 1. Turn log: `state/turns/YYYY-MM-DD.jsonl`

One file per local calendar day (the Mac's timezone), named by the day the turn **started**. UTF-8, one JSON object
per line, `\n`-terminated, append-only (open, append one line, close; never rewrite a file). A record is written once
per user turn, when the turn is over: the agent settled, or the reply was interrupted or cancelled.

```json
{"v": 1, "type": "turn",
 "session": "s-20261004-0914-k3", "turn": 3,
 "t_start": "2026-10-04T09:16:02.120-07:00", "t_end": "2026-10-04T09:16:09.870-07:00",
 "client": "iphone", "space": "home", "mode": "conversation", "input": "voice",
 "user_text": "let's keep kokoro as the fallback voice",
 "reply_text": "Done. Kokoro stays the fallback; Ryan is still the main voice.",
 "heard_text": null, "interrupted": false,
 "tools": [{"name": "read", "ok": true}],
 "atlas": null,
 "tone": null}
```

| Field | Required | Meaning |
|---|---|---|
| `v` | yes | `1` |
| `type` | yes | `"turn"`. Other types (`session_start`, `space_switch`, …) may be written and are ignored |
| `session` | yes | the voice session id, the same string as `welcome.session` in PROTOCOL.md |
| `turn` | yes | 1-based number of the user turn within the session |
| `t_start` | yes | ISO 8601 **with offset**: speech start (VAD), push-to-talk `start`, or a typed `text` arriving |
| `t_end` | no | ISO 8601 with offset: `agent_settled`, or the interruption |
| `client` | no | from `hello` (`iphone`, `mac`, `test`) or `browser` |
| `space` | yes | the space the turn ran in, after any router switch (SPACES.md names) |
| `mode` | no | `conversation` or `act` |
| `input` | no | `voice` or `text` |
| `user_text` | yes | the final transcript as the person's words: exactly what the speech recogniser produced (or what they typed). **Never** the delivery hint, the heard-text note, the "already saved" note or anything else the orchestrator adds to Pi's prompt |
| `reply_text` | yes | the assistant's whole reply text as generated for speech (what went to TTS and `reply_text` captions); `""` if none |
| `heard_text` | no | when the reply was interrupted: the words actually played (from `played_ms`); `null` otherwise |
| `interrupted` | no | `true` when the user barged in or the reply was cancelled |
| `tools` | no | the tools the agent called in this turn: `[{"name": "kb", "ok": true}]` |
| `atlas` | no | set only when the person's words were saved into Atlas during this turn (below); `null` otherwise |
| `tone` | no | the Tier-1 delivery hint when one was computed (`tone/README.md`): `{"hint": "[delivery …]", "shown": true}` |

`atlas`, when the orchestrator captured the words with `python3 atlas.py capture --via voice` (the default path,
SPACES.md), or saw the model capture them (the fallback path):

```json
{"root": "~/atlas", "path": "journal/2026-10-04.md", "by": "orchestrator", "text": null}
```

- `root`: the Atlas root the capture ran in (absolute). The brain writes inbox pages only into the root its own config
  names and skips turns whose `root` differs (tests point both at a clone).
- `path`: the note the words went to, relative to `root` (`journal/<day>.md` by default, or the `--to` path).
- `by`: `orchestrator` or `model`.
- `text`: the exact bytes sent to `capture` on stdin when they differ from `user_text` (for example a leading "note
  this:" left off); `null` means `user_text` was captured as is. The brain quotes only words it finds verbatim in the
  note, so a wrong `text` costs a quote, never a misquote.

Rules the brain relies on:

- **Words are verbatim.** `user_text` and `atlas.text` are never cleaned, summarised or re-cased.
- **Turns of the `atlas` space are life content.** The brain never sends them to the kb, and it reflects only on the
  ones whose words are in Atlas (`atlas` set and found in the note). Anything the person says in other spaces is
  treated as technical and may reach the kb's operational lane (`.sessions/digests/`), as kb.ts already does for
  every Pi session in those spaces.
- **Activity.** The newest `t_end` (or `t_start`) is how the brain knows a conversation is recent; a turn in progress
  is not logged yet, so the brain also asks `GET /v1/status` (busy while `thinking` or `speaking`).
- **Malformed lines** are skipped and counted in the brain's report; one bad line never blocks a day.

## 2. Outbox: `state/brain/outbox/YYYY-MM-DD.json`

When a reflection ends with something worth saying, the brain queues one spoken digest per digested day:

```json
{"v": 1, "id": "digest-2026-10-04", "day": "2026-10-04",
 "created": "2026-10-05T03:12:40-07:00", "expires": "2026-10-07T03:12:40-07:00",
 "text": "From Sunday's voice conversations: Kokoro stays the fallback voice, and you want replies kept to two sentences.",
 "cites": [{"session": "s-20261004-0914-k3", "turn": 3}],
 "kb_digest": ".sessions/digests/2026/10/pi-voice-brain-2026-10-04-tech-7f3a.md",
 "atlas_page": "inbox/voice-reflections-2026-10-04.md"}
```

A `NO_REPLY` day queues nothing. What the orchestrator does with it (the brain cannot speak; it never loads a TTS
model):

1. At the first idle moment after the **first completed turn of a new session** (never before the person's first
   request is answered), while the hold gate is open, speak the pending digests oldest first through the normal TTS
   path, as a reply of their own (`append_to_context` off). Skip any whose `expires` has passed.
2. After it was played (or interrupted), move the file to `state/brain/outbox/delivered/` and add `"delivered"`
   (ISO 8601 with offset). Never speak one twice.
3. Optional: show it in `/v1/status` as `"digest_pending": true`, and give the active Pi child the text as context
   so "what did you note yesterday?" works.

`brain/local_voice_brain/outbox.py` implements `pending(state_dir, now)` and `mark_delivered(path, now)` in the
standard library only, so the orchestrator can import that one file by path or copy it.
