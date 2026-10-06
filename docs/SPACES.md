# Spaces: one agent with computer-level reach, scoped per domain

Frozen for implementation on 2026-10-05. The schema below is the one source the orchestrator derives everything
from: the Pi child it spawns for a space, the trigger phrases the router listens for, the permission gate, and what
the dashboard shows. The research behind it is kept with the design notes in a private wiki.

## Why it is layered

In every harness the working directory is where skills are discovered, not a sandbox (Pi `docs/security.md`). So a
space limits three different things in three different places:

1. **Capability reach**, fixed when the space's Pi child is spawned: the tool allowlist, MCP servers, model.
2. **Knowledge reach**, from the space's root directory: `AGENTS.md`, `.agents/skills`, `.pi/skills`, `.pi/mcp.json`.
   Pi discovers these itself; a space with `skills: auto` gets exactly what the repo declares.
3. **Risk gating**, at run time: a `tool_call` extension (`voice-gate.ts`) that allows, asks or refuses each call by
   the space's tier and command allowlist. "Asks" means the agent says what it wants to do and waits for a spoken
   yes; silence or "no" cancels.

## `spaces.yaml`

```yaml
version: 1
defaults:
  model: local/qwen38            # fast MoE: the live conversation
  deep_model: local/qwen27-262k  # slow dense model: long do-it-for-me tasks and reflection
  thinking: off                  # reasoning off on the live path (field reports: it eats the reply budget)
  mode: conversation             # conversation | act
  tier: ask                      # readonly | ask | trusted
  voice: ryan                    # key into the TTS config
spaces:
  home:
    root: ~
    description: "your Mac"      # spoken on entry: "Back on your Mac."
    triggers: ["home", "back home", "my mac", "the computer"]
    skills: [wiki-query, web-research, chrome-read, reddit, x]   # an explicit list: --no-skills plus --skill <dir> each
    tools: [read, grep, find, ls, kb]          # conversation mode
    act_tools: [bash, write, edit]             # added in act mode; tier ask makes every one of them need a spoken yes
    tier: ask
  atlas:
    root: "~/atlas"              # an Atlas-style markdown journal repo (atlas.py CLI); point it at yours
    description: "your atlas journal"
    triggers: ["atlas", "journal", "my journal", "life wiki", "my notes"]
    skills: auto                 # the repo's own .agents/skills: atlas-journal, atlas-process, atlas-ask, atlas-review, atlas-file
    tools: [read, grep, find, ls, bash]        # bash is needed in conversation mode too: capture runs through atlas.py
    act_tools: [write, edit]                   # agent pages go to inbox/; wiki/ only on "file that"
    tier: trusted
    bash_allow:                  # argv prefixes, not shell globs; see "bash_allow" below
      - ["python3", "atlas.py", "capture"]
      - ["python3", "atlas.py", "queue"]
      - ["python3", "atlas.py", "context"]
      - ["python3", "atlas.py", "show"]
      - ["python3", "atlas.py", "meta"]
      - ["python3", "atlas.py", "stamp"]
      - ["python3", "atlas.py", "answer"]
      - ["python3", "atlas.py", "start"]
      - ["python3", "atlas.py", "finish"]
      - ["git", "status"]
      - ["git", "log"]
    rules:                       # appended to the system prompt in this space
      - "When the person muses or says note this, save their exact words first with python3 atlas.py capture --via voice, then reply."
      - "Never edit, tidy or rewrite their words. Your own pages go to inbox/; wiki/ only when they say file that."
  kb:
    root: ~/kb                   # an optional knowledge base: a `kb` CLI over a markdown wiki
    description: "the knowledge base"
    triggers: ["knowledge base", "the kb", "the wiki"]
    skills: [wiki-query]
    tools: [read, grep, find, ls, kb]
    tier: readonly
  voice:
    root: ~/repos/local-voice
    description: "this voice project"
    triggers: ["voice project", "local voice"]
    skills: auto
    tools: [read, grep, find, ls]
    tier: readonly
```

### Fields

| Field | Required | Meaning |
|---|---|---|
| `root` | yes | working directory of the space's Pi child; `~` expands; must exist |
| `description` | yes | short noun phrase the agent speaks on entry |
| `triggers` | no | phrases the router matches before the LLM (lowercase; whole-phrase match after "go to", "switch to", "open", or alone) |
| `skills` | no | `auto` (Pi discovers from the root and the user's global dirs) or a list of skill names (spawned with `--no-skills` and one `--skill <dir>` each); default `auto` |
| `extensions` | no | extra Pi extension paths for this space; `voice-gate.ts` and `hold.ts` are always added |
| `tools` | yes | Pi `--tools` allowlist in `conversation` mode (applies to built-in and extension tools) |
| `act_tools` | no | tools added when the mode is `act` (switched in-process with `pi.setActiveTools`); the tier still gates each call |
| `tier` | no | `readonly`: no writes, no shell. `ask`: writes and shell need a spoken yes. `trusted`: writes inside `root` and commands matching `bash_allow` run without asking; everything else asks |
| `bash_allow` | no | argv prefixes a `trusted` space runs without asking. A command qualifies only if it parses as one simple command (no `;`, `&&`, `\|\|`, `\|`, `$(…)`, backticks or redirections) whose argv starts with a listed prefix; a heredoc body is allowed only behind a quoted marker that appears once, at the end. Shell globs are not used because `python3 atlas.py *` also matches `python3 atlas.py x; rm -rf journal` (shown in note 05c) |
| `model`, `thinking`, `voice` | no | override the defaults |
| `rules` | no | sentences appended to the system prompt for this space |
| `gpu_exclusive` | no | the space runs GPU jobs (renders) that unload the LLM; the agent says so and waits |

> **Security warning: the journal space is `trusted`.** In act mode, the agent can `write` and `edit` **any file
> under the journal's root without asking**. `voice_gate.ts` allows every path inside the root, because
> `VOICE_WRITE_ALLOW` defaults to `["*"]`. "Agent pages go to `inbox/`" is only an instruction in the persona and
> space rules, which a model can ignore, and nothing enforces it. The whitelisted `atlas.py` commands also run
> without asking, and `atlas.py start`/`finish` commit in the journal repo. Only files outside the root still ask.
> To restrict it, pick one:
> - Set `tier: ask` on the space. Every write, edit and non-read command then needs a spoken or tapped yes. The
>   orchestrator's own verbatim capture of "note this: …" goes through Pi's RPC `bash`, not a tool call, so it is
>   not affected.
> - Remove `write` and `edit` from its `act_tools`.
> - Export `VOICE_WRITE_ALLOW='["inbox/*"]'` before `./run.sh`. The Pi children inherit it, so trusted writes are
>   limited to `inbox/` and everything else asks. Globs are relative to the root, and `*` also matches `/`. The
>   variable applies to every `trusted` space.
>
> Keep the journal in git (Atlas-style repos are), so any unwanted change can be seen and reverted.

## Behaviour

- **One Pi child per space**, started lazily on first entry with `pi --mode rpc --session-dir <state>/sessions/<space>
  --name voice-<space> --model … --thinking off --tools <tools + act_tools> --approve --no-extensions -e …` and
  `cwd = root`. Pi RPC cannot change directory, hence one child per space (note 05b). Verified against a stub LLM on
  2026-10-05 (note 05c): a child starts in about 140 ms and 108 MB with no extensions; `--continue` with the
  session dir resumes; `switch_session` takes 10 ms.
  - **Extensions are an explicit list**: `voice_gate.ts` and `voice_mode.ts` (this project), `hold.ts` (waits during
    a GPU hold; an abort cancels the wait in about 200 ms), `today.ts` (appends the date at the very end of the
    system message, so the prefix cache survives), `kb.ts` only where the kb tool is listed (it adds about 210 ms to
    startup and 156-188 ms before `agent_settled` on every turn, and writes a session digest), and `-e builtin:mcp`
    for a space with MCP servers (`--no-extensions` drops it). **Never `film-rig.ts`**: it waits in
    `before_agent_start` while any render runs, Pi then answers nothing, and `abort` reports success but cancels
    nothing.
  - **Every tool is registered at spawn** (`--tools` decides registration); `voice_mode.ts` narrows to `tools` at
    session start and widens to `tools + act_tools` in act mode. A mode switch rewrites the tool list in the system
    message, so the next turn re-processes the whole prompt; that cost is paid on explicit switches only.
  - **The persona** goes in `--system-prompt`, which replaces Pi's coding preamble, tool list, rules and docs and keeps
    `AGENTS.md`, skills and the working directory; `--append-system-prompt` would land mid-prompt under the coding
    preamble. The GPU-phase test measures both before this is final.
  - **`--thinking off` is required**: it sends `"thinking": {"type": "disabled"}`; the Flash Next model runs on
    ds4-server, which honours that field and otherwise thinks at HIGH. Whether llama-server honours it for qwen27 is
    being measured; the fallback is a per-model compat `thinkingFormat: "qwen-chat-template"`.
- **Switching.** The router matches trigger phrases on the final transcript before any model call. On a match the
  orchestrator speaks "Switching to <description>.", makes that child active, and forwards the rest of the sentence as
  the first prompt. No match falls through to the active child, which has a `space_switch` tool for requests like "go
  where my journal is". Ambiguity is asked about, never guessed.
- **Modes.** `conversation`: the fast model, short spoken replies, `tools` only; the persona says that anything more
  needs act mode. `act`: `tools` plus `act_tools`, and long tasks run as a separate Pi child whose completion comes back as a `follow_up` ("The render
  finished."). The mode is switched by voice ("act mode", "just talk") or the `mode` control message.
- **Spoken confirmation.** `voice-gate.ts` turns a call that needs permission into `ctx.ui.confirm(title, message,
  {timeout: 20000})`; the orchestrator speaks the message, listens for yes or no, and answers
  `extension_ui_response`. No answer within the timeout means no.
- **Atlas** (the journal space: an Atlas-style markdown journal repo driven by its own `atlas.py` CLI). The person's
  words are never rewritten (the journal's own rule). A voice transcript is their words: it is
  captured verbatim with `atlas.py capture --via voice` before the agent replies, transcription quirks included.
  When the orchestrator knows an utterance is a musing (the person said "note this" or "journal this", or journal
  mode is on: "just take notes"), it captures it itself through Pi's RPC `bash` command with `excludeFromContext`,
  then sends the prompt with a note that the words are already saved: deterministic, byte-exact in 4 of 4 real-rig tries by
  the 05d report's reading (its automatic check passed 3; the fourth repeated a sentence saved just before), 38-48 ms, always `--via voice` into today's note, first spoken word 4.6 s instead of 8.5-9.8 s (note 05d). This is the
  default. Otherwise the model decides through the atlas-journal skill (24 of 24 captures first, the
  words right in 23 of 24, once `voice_gate.ts` allowed one leading `cd` into the space root; earlier refusals of `cd <root> &&` were the
  gate, not the model); against the stub the capture finished at 251 ms and the first spoken word came at 255 ms. Both paths kept the bytes exact
  (ellipsis, dashes, quotes, `$`, `&`, backticks) and passed `atlas.py finish`'s untouched check (note 05c).
  Delivery cues from the tone tier never enter their words; at most they go into frontmatter through `atlas.py meta`,
  and only once the person turns that on. A journal kept in iCloud Drive can hold evicted placeholder files;
  read errors are reported, never "fixed" by writing.
- **Dashboard.** `/v1/status` (PROTOCOL.md) shows space, mode, tier, model, the tool running, the GPU hold, and the
  last turn's latency breakdown.

## Test rule

Tests never touch a real space. They run against `git clone`s of Atlas and copies of anything else under a temporary
directory, with `KB_HOME` pointed at a copy if the kb could be written. The real Atlas holds uncommitted edits of
the person's at any time.
