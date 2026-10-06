# brain: the background brain (milestone M4)

While nobody is using the Mac, the brain reads a finished day of voice conversations and leaves three things behind:
the technical facts worth keeping, as notes in the session lane of an optional knowledge base (a `kb` CLI over a
markdown wiki); a reflection on what the user said into their journal, as a draft page in the `inbox/` of an
Atlas-style markdown journal repo (one with an `atlas.py` CLI); and one or two sentences the agent says at the start of the next
conversation, or nothing at all (`NO_REPLY`). It never speaks, never loads a speech model, and never asks the model
anything while the GPU is held or busy.

Built 2026-10-05 against a stub LLM, real Pi, a git clone of a kb repo and a git clone of an Atlas journal. It has
not yet met a real model or a real day of conversation.

## How a heartbeat goes

launchd runs `brain/run.sh` every 30 minutes (modelled on OpenClaw's heartbeat and Hermes' deferred review):

1. **Anything due?** The oldest day before today that has a turn log (`state/turns/<day>.jsonl`, the interface in
   `TURN_LOG.md`) and is not yet done. Nothing due: exit at once, no gate checked, no model touched. Days older than a
   week are marked skipped (a week-old spoken digest helps nobody).
2. **Is the Mac idle?** No voice turn in the last 20 minutes (turn log), nobody at the keyboard or mouse for 15
   minutes (macOS HID idle time), the orchestrator not mid-turn (`GET :8770/v1/status`), and
   `measure/bench/gpu_clear.sh` CLEAR (hold gate open, no render or other GPU job, the LLM idle for two minutes).
   Any no: exit 75 and try again next time. `gpu_clear.sh` is run again right before **every** model call, and no
   flag skips it (`--force` skips only the first three checks). A missing script counts as BUSY.
3. **The technical pass.** The day's turns outside the Atlas space go to the model through `pi --mode json` with no
   tools at all. It answers with JSON: facts (decision, preference, finding, todo, question), each citing the turns it
   came from, and a spoken line or `NO_REPLY`. The run loads the kb's own Pi adapter (`kb.ts`), so Pi registers and
   closes it as a `pi:` session in the kb exactly like any other Pi session.
4. **The life pass.** Only the user's own words from the Atlas space, and only those found verbatim in the Atlas note
   they were captured to, go to a second Pi run (no tools, no kb adapter). It drafts at most three reflections: exact
   quotes, one observation, a question or two.
5. **Validation decides, the model proposes.** Every citation must name a turn the pass was shown. Every quote must be
   found in the cited turn's words, and what is written is the exact span from the source (a curly quote or a changed
   capital in the model's copy can never change the user's words). A fact whose quote is not there is dropped whole:
   it claimed evidence it does not have. A reflection whose own wording names a feeling the user did not name, uses a
   clinical word, or gives advice is dropped (Atlas: observations with evidence, never verdicts). Everything dropped
   is listed in the day's report.
6. **The writes**, each retried on its own if it fails, without asking the model again:
   - kb: one `kb session note` per fact on the technical pass's kb session (kb verbs only; nothing hand-written).
   - Atlas: one new page `inbox/voice-reflections-<day>.md`, committed alone.
   - outbox: `state/brain/outbox/<day>.json`, the spoken digest the orchestrator plays after the first finished turn
     of the next conversation (`TURN_LOG.md` §2). Life content is never spoken; the digest only says a reflection is
     waiting.
   - the day's report: `state/brain/days/<day>/report.md`, citing every session.

If the model chosen in `config.toml` is a qwen27 row whose Pi `models.json` entry lacks
`compat.thinkingFormat: "qwen-chat-template"`, the brain uses `fallback_model` (qwen38) and says why: llama-server
ignores Pi's thinking switch for qwen27 otherwise (found 2026-10-05). After a pass on any model
other than qwen38 it makes one tiny call to load qwen38 again, so the morning's first conversation pays no reload.

## Why it writes where it writes

- **The kb's session lane, not wiki pages.** The kb treats session material as operational memory: it
  becomes knowledge only when a human promotes a distilled note, and raw transcripts never become topic evidence. So
  facts are notes on a session digest (`.sessions/digests/…/pi-voice-brain-…md`),
  each one line with its kind and the voice turns it came from, ready to promote by copying. The brain never writes a
  wiki page and never edits anything under `.sessions/` by hand.
- **Atlas `inbox/`, through atlas.py where atlas.py has the logic.** atlas.py has no verb for inbox pages; every Atlas
  skill writes them directly in documented shapes, and so does the brain (atlas-process's "In your words / What the AI
  notices / Questions"). The frontmatter is rendered by the repo's own `atlas.py` (`render`), the page is checked to
  parse as an agent's page (`split`, `parse`, `human`), tags come from `atlas.py context`, and `atlas.py queue` must
  then list it as waiting on the person.
- **No `atlas.py start`/`finish`.** Both run `git add -A` over the whole repo (which may be cloud-synced), which from a
  background job would commit the user's half-finished edits, and any sync placeholder state, under the brain's name. The
  brain commits only its new page (`git commit -- inbox/<page>`, atlas-review's precedent), then proves the commit
  holds that one file and that the rest of the working tree is exactly as it was. It never pushes.
- **Two passes.** The technical pass never sees a turn of the Atlas space, so life content cannot reach the kb even if
  the model misbehaves; the life pass never sees the assistant's replies, so it cannot quote the assistant as the
  user.

## Files

| Path | What |
|---|---|
| `TURN_LOG.md` | the interface: what the orchestrator logs per turn, and the outbox it reads |
| `config.toml` | every path, gate, model and limit (`config.local.toml` beside it overrides, untracked) |
| `run.sh` | the entry point launchd runs; sets PATH because launchd has none |
| `launchd/com.example.local-voice-brain.plist` | the 30-minute heartbeat, as a template (`__HOME__`, `__REPO__`); not loaded |
| `local_voice_brain/` | standard library only: `job.py` (the heartbeat), `gates.py`, `turnlog.py`, `prompts.py`, `llm.py` (Pi), `validate.py`, `kbwriter.py`, `atlaswriter.py`, `outbox.py`, `state.py`, `report.py` |
| `tests/` | 60 tests; `demo_day.py` runs one whole heartbeat on a synthetic day and prints every artifact |

State, all under the repo's gitignored `state/brain/`: `ledger.json` (each day's status and writes),
`heartbeat.log` (one line per heartbeat), `days/<day>/` (the exact prompts, the model's replies, the validated
proposal, `report.md`), `pi-sessions/` (the reflection runs' Pi transcripts, which the kb digest's
`transcript_path` points at), `outbox/`.

## Commands

```bash
brain/run.sh status                              # ledger, pending outbox, the model it would use, the gates right now
brain/run.sh                                     # one heartbeat (what launchd runs)
brain/run.sh run --day 2026-10-04 --dry-run      # ask the model, write nothing (no kb session either)
brain/run.sh run --day 2026-10-04 --force        # skip the idle checks, never gpu_clear
brain/run.sh retry 2026-10-04                    # after three failures a day is left alone until this
uv run --project brain pytest brain -q           # the tests (stub LLM, clones; about 45 s; no GPU, no model)
uv run --project brain python brain/tests/demo_day.py   # one heartbeat on a synthetic Sunday, every artifact printed
```

Exit codes: 0 done or nothing to do, 75 deferred, 1 failed (recorded; retried next heartbeat, up to three times),
78 bad configuration.

## Installing it (a manual step; nothing here loads it)

Only worth doing once the orchestrator writes `state/turns/` (until then every heartbeat ends at step 1). Run from
the repo root; the plist is a template because launchd expands neither `$HOME` nor `~`.

```bash
mkdir -p state/brain                              # launchd will not create the log's directory
sed -e "s|__HOME__|$HOME|g" -e "s|__REPO__|$PWD|g" brain/launchd/com.example.local-voice-brain.plist \
  > ~/Library/LaunchAgents/com.example.local-voice-brain.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.example.local-voice-brain.plist
# check: launchctl print gui/$(id -u)/com.example.local-voice-brain | head; tail state/brain/heartbeat.log
# remove: launchctl bootout gui/$(id -u)/com.example.local-voice-brain && rm ~/Library/LaunchAgents/com.example.local-voice-brain.plist
```

If your journal's own rules say nothing runs in the background, note that with this installed one thing does: it
only adds pages to `inbox/`, which you accept or delete as with any agent's draft.
