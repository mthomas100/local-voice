# 05d: qwen38 against qwen27-262k as the voice brain, through Pi RPC on the real rig

Date: 2026-10-05, 09:00 to 10:00 PDT. Pi 0.99.1. `gpu_clear.sh` read CLEAR before each run (gpu_jobs=0, hold open), and its gpu_jobs and hold fields were
re-checked before every request; nothing took the GPU during either run. Two runs:

- **Main run** (`20261005-090038`, 09:00:38 to 09:10): 13 voice-style requests per configuration, 10 in `home` (cwd
  `~`, tools read, grep, find, ls, kb) and 3 in a fresh `git clone` of Atlas (tools read, grep, find, ls, bash), in this
  order: qwen38 with the persona appended (`--append-system-prompt`), qwen38 with the persona as the system prompt
  (`--system-prompt`, "replace"), qwen27-262k appended, then qwen27-262k with `thinkingFormat: "qwen-chat-template"`
  in both placements, then one qwen38 request to restore the default.
- **Atlas rerun** (`20261005-095230`, 09:52 to 09:58): the 3 Atlas requests 3 times per configuration on a fresh clone
  with the corrected `voice_gate.ts`, then the orchestrator-side capture path on qwen38. qwen38 is loaded at the end.

Isolation held: a throwaway agent dir with the real `local` provider, `--no-extensions` plus hold.ts, today.ts,
voice_gate.ts and (home) kb.ts, `KB_HOME` on a fresh clone of the knowledge base searched with ripgrep, the journal only as clones with
push disabled, every confirm answered "no". Per-request tables: `voice-tooluse-20261005-090038.md` and
`voice-tooluse-20261005-095230.md` in this folder (replies withheld in the public copy: they describe the author's own
files, notes and journal).

## Headline

All 13 requests per configuration (main run). Timing leaves out each model's first request, which includes the
model load. "First text" is the first word that would be spoken.

| Leg | Turns | Expected tool | First text median / p90 s | Total median / p90 s | Words median / max | Model thinking leaks | Markdown | Errors |
|---|---|---|---|---|---|---|---|---|
| qwen38:append | 13 | 13/13 | 1.9 / 12.5 | 2.8 / 14.0 | 18 / 119 | 0 | 2 | 0 |
| qwen38:replace | 13 | 13/13 | 1.4 / 10.9 | 2.4 / 13.7 | 19 / 99 | 0 | 0 | 0 |
| qwen27-262k:append | 2 | 2/2 | 60.0 / 60.0 | 65.8 / 65.8 | 42 / 72 | 1 | 0 | 0 |
| qwen27-262k:append:qwen-chat-template | 13 | 12/13 | 3.9 / 18.3 | 5.9 / 23.1 | 22 / 138 | 0 | 0 | 0 |
| qwen27-262k:replace:qwen-chat-template | 13 | 13/13 | 3.4 / 16.2 | 6.2 / 19.8 | 25 / 105 | 0 | 0 | 0 |

The same, home requests only (10 per configuration; no Atlas captures involved):

| Leg | Turns | Expected tool | First text median / p90 s | Total median / p90 s | Words median / max | Model thinking leaks | Markdown | Errors |
|---|---|---|---|---|---|---|---|---|
| qwen38:append | 10 | 10/10 | 1.2 / 7.9 | 3.1 / 16.1 | 15 / 119 | 0 | 2 | 0 |
| qwen38:replace | 10 | 10/10 | 1.4 / 11.9 | 2.9 / 14.8 | 20 / 99 | 0 | 0 | 0 |
| qwen27-262k:append | 2 | 2/2 | 60.0 / 60.0 | 65.8 / 65.8 | 42 / 72 | 1 | 0 | 0 |
| qwen27-262k:append:qwen-chat-template | 10 | 9/10 | 3.6 / 18.1 | 6.4 / 28.5 | 20 / 138 | 0 | 0 | 0 |
| qwen27-262k:replace:qwen-chat-template | 10 | 10/10 | 3.4 / 11.7 | 6.5 / 18.8 | 26 / 105 | 0 | 0 | 0 |

First text by kind of request (medians, seconds):

| Leg | no tool (#1, #7, #10): first text median s | one file tool (#3-#6, #9): first text median s | knowledge base (#2, #8): first text median s |
|---|---|---|---|
| qwen38:append | 0.2 | 1.2 | 12.9 |
| qwen38:replace | 0.2 | 1.3 | 14.5 |
| qwen27-262k:append | - | - | 60.0 |
| qwen27-262k:append:qwen-chat-template | 0.6 | 3.0 | 37.6 |
| qwen27-262k:replace:qwen-chat-template | 0.6 | 3.4 | 26.1 |

Model loads, from llama-swap's own loading banner: qwen38 19.8 s (first word 25.7 s after the prompt), qwen27-262k
15.4 s and 14.3 s (first word 21.2 s and 30.1 s), qwen38 again at the end 18.3 s.

## Verdicts

**Which brain: qwen38.** On every kind of request it is two to three times faster than qwen27-262k, it picked the
expected tool 26 of 26 times (qwen27 with compat 25 of 26: it answered "is there a SPACES.md" without looking), and it
never reasoned out loud with thinking off. No-tool replies start in 0.2 s against 0.6 s, single-file lookups in 1.2 to
1.3 s against 3.0 to 3.4 s.

**Persona placement: `--system-prompt` (replace).** It removed the markdown leaks (0 against 2: the two
knowledge-base answers in append mode used backticks), shortened the worst reply (99 against 119 words) and the p90
(10.9 against 12.5 s to first text over all 13 requests), and drops about 2.2k characters of Pi's coding preamble,
tool list, rules and docs from the head of the prompt (05c E7b), with nothing lost on tool choice. Median latency differences are within noise. Neither placement made
knowledge-base answers short (90 to 138 words, about 40 to 55 s of speech), so length needs more than a persona line
(below).

**qwen27-262k thinking: the compat setting is required.** With the rig's provider compat (`thinkingFormat:
"deepseek"`), `--thinking off` sends `"thinking": {"type": "disabled"}`. ds4-server honours it for qwen38;
llama-server b10869 does not for qwen27, which reasoned 2,159 characters on the second request ("The user is asking
about \"hold gate\" in the knowledge base. I should use the wiki-query skill...") and took 60.0 s to its first word.
With `"compat": {"thinkingFormat": "qwen-chat-template"}` on the qwen27-262k entry of models.json, Pi sends
`chat_template_kwargs: {"enable_thinking": false, "preserve_thinking": true}` and qwen27 produced no reasoning in any
of 44 turns (26 in the main run, 18 in the Atlas rerun). That is a models.json change on the model-server side.

**Not thinking: llama-swap's loading banner.** During a model load llama-swap (`sendLoadingState: true`) streams a
banner as `reasoning_content` ("llama-swap loading model: qwen38 ... Done! (19.83s)"), which Pi turns into thinking
deltas. The voice layer must never speak thinking deltas, and can use this one as a "waking up, about 15-20 s" cue.

| Leg | # | Loading banner chars (load s) | Model thinking chars | First text s | Excerpt of the model's own reasoning |
|---|---|---|---|---|---|
| qwen38:append | 1 | 248 (19.8) | 0 | 25.7 | "" |
| qwen27-262k:append | 1 | 285 (15.4) | 0 | 21.2 | "" |
| qwen27-262k:append | 2 | 0 (-) | 2159 | 60.0 | "The user is asking about \"hold gate\" in the knowledge base. I should use the wiki-query skill (read-only query). Let me first read that skill file, and then sea" |

## Long tool chains: knowledge-base questions take 11 to 25 s on qwen38

| Leg | # | Tools in order | First sign s | First text s | Total s | Words | Input tokens |
|---|---|---|---|---|---|---|---|
| qwen38:append | 2 | kb, read | 3.6 | 3.6 | 13.6 | 119 | 0 |
| qwen38:append | 8 | kb, read | 0.8 | 22.2 | 25.8 | 112 | 0 |
| qwen38:replace | 2 | read, read, kb, read | 0.7 | 11.2 | 14.1 | 90 | 0 |
| qwen38:replace | 8 | kb, read | 0.8 | 17.7 | 21.0 | 99 | 0 |
| qwen27-262k:append | 2 | read, kb, read, read | 6.5 | 60.0 | 65.8 | 72 | 6061 |
| qwen27-262k:append:qwen-chat-template | 2 | kb, read | 0.9 | 13.2 | 23.5 | 130 | 3220 |
| qwen27-262k:append:qwen-chat-template | 8 | kb, read | 0.9 | 61.9 | 73.0 | 138 | 5223 |
| qwen27-262k:replace:qwen-chat-template | 2 | kb, kb | 0.8 | 7.4 | 15.2 | 105 | 50 |
| qwen27-262k:replace:qwen-chat-template | 8 | kb, kb | 1.0 | 44.7 | 52.0 | 83 | 50 |

Both models answered "what does my knowledge base say about X" by running `kb search` and then reading a whole wiki
page (sometimes two), then speaking 83 to 138 words. The page read is what costs: thousands of tokens to prefill
before the answer starts. qwen38 needed 11 to 22 s to its first word on these (3.6 s once, when it said "I'll search
the knowledge base for that." before the tools); qwen27 with compat 7 to 62 s. The first sign of life, the model
starting its tool call, came at 0.7 to 1.0 s in 7 of the 8 answers with thinking off (3.6 s in the other, where the
model spoke a preface first), so an acknowledgement spoken at `toolcall_start` (05c) covers the wait. To shorten the wait itself: tell the voice persona to answer from the search snippets and open a page
only when asked; give the kb results a one-line description per hit; cap knowledge answers at two sentences with an
offer of more; or hand deep questions to a background run that reports back.

Counting from tool output was unreliable: asked how often "barge-in" appears in SPEC.md (9 times when the test ran,
one per line), the four configurations said 11, 11, 7 and 10. Two grepped the whole project and quoted a SPEC.md
count they never measured (11 and 10); qwen38 in append mode grepped SPEC.md itself, saw 9 lines and still said eleven;
qwen27 in append mode grepped case-sensitively and reported the 7 lines it saw, missing two capitalised ones. Other facts
checked out (the README's first line, the extensions folder, the arithmetic).

## Atlas capture: what went wrong, request by request

The main run's capture column (qwen38 0/2 in both placements, qwen27 with compat 1/2) had two different causes.

**qwen38: the gate refused a harmless `cd`.** qwen38 did the right thing in substance every time: it read the
atlas-journal skill and called `python3 atlas.py capture` with a quoted heredoc before replying. But it wrapped the
command in `cd <atlas root> && ...` (append) or `cd "$(pwd)" && ...` (replace). voice_gate.ts treated `&&` as chaining,
asked for permission, the test answered no, and the model then told the person it had not saved their words. In real
use the person would have heard "May I run a command?" on every musing.

**qwen27 with compat: one fidelity slip.** Both placements captured the musing exactly. On "Note this: call the
plumber about the kitchen tap on Tuesday." both dropped the instruction prefix, which the atlas-journal skill allows,
and also the final period, which it does not.

| Leg | Request | Capture command | Gate | Words saved | Spoken before the capture call |
|---|---|---|---|---|---|
| qwen27-262k-append-qwen-chat-template | #11 | `atlas.py capture --via pi` | allowed | exact | - |
| qwen27-262k-append-qwen-chat-template | #13 | `atlas.py capture --via pi` | allowed | prefix and final period dropped | - |
| qwen27-262k-replace-qwen-chat-template | #11 | `atlas.py capture --via claude-code` | allowed | exact | - |
| qwen27-262k-replace-qwen-chat-template | #13 | `atlas.py capture --via claude-code` `--to "asks/Plumber.md"` | allowed | prefix and final period dropped | - |
| qwen38-append | #11 | `cd ~/repos/ &&` + `atlas.py capture --via pi` | asked, refused | exact | - |
| qwen38-append | #13 | `cd ~/repos/ &&` + `atlas.py capture --via pi` | asked, refused | prefix and final period dropped | - |
| qwen38-replace | #11 | `cd "$(pwd)" &&` + `atlas.py capture --via voice` | asked, refused | exact | "That's a lovely image \u2014 let me save it first." |
| qwen38-replace | #13 | `cd "$(pwd)" &&` + `atlas.py capture --via voice` | asked, refused | exact minus the allowed prefix | - |

**Fix to the gate** (in `research-prototypes/pi_rpc_bridge/extensions/voice_gate.ts`): one leading `cd` is stripped
before the allowlist check when its target is the space root or inside it, written as a plain, quoted or escaped
path, or as the literal no-op forms `"$(pwd)"`, `$PWD`, `${PWD}` and `.`. Anything with another expansion
(`cd "$(pwd)/.."`, `cd "$(rm ...)"`, `cd "$HOME"`), a target outside the root, or a second chained command still
asks. A new test covers 15 forms; the suite is 42 tests, all passing.

**Rerun with the fixed gate, 3 runs per configuration:**

| Leg (3 runs each) | Captures run / asked | cd prefix | Musing (#11) words | Note (#13) words | --via used | #13 filed to | #11 first spoken word median s | #13 first spoken word median s |
|---|---|---|---|---|---|---|---|---|
| qwen27-262k:append:qwen-chat-template | 6/6 run, 0 asked | 2/6 | exact 3/3 | exact, prefix left off 2/3, exact 1/3 | pi | journal/ideas.md 1, today's note 2 | 9.8 | 3.6 |
| qwen27-262k:replace:qwen-chat-template | 6/6 run, 0 asked | 2/6 | exact 3/3 | exact, prefix left off 3/3 | claude-code, voice | journal/ideas.md 3 | 8.5 | 3.8 |
| qwen38:append | 6/6 run, 0 asked | 6/6 | exact 3/3 | prefix and final period dropped 1/3, exact, prefix left off 2/3 | pi | today's note 3 | 9.1 | 1.7 |
| qwen38:replace | 6/6 run, 0 asked | 6/6 | exact 3/3 | exact, prefix left off 3/3 | omp | journal/ideas.md 2, today's note 1 | 8.6 | 1.8 |

All 24 captures ran, none asked, and none was preceded by a spoken reply. The words are right: the musing exact 12 of
12, the note exact or with the allowed prefix left off 11 of 12 (one qwen38 run dropped the final period again). What
the models do not agree on is bookkeeping: `--via` came out as `pi`, `omp`, `claude-code` or `voice` (SPACES.md wants
`voice`), and the plumber reminder was filed in `journal/ideas.md` in 6 of 12 runs, today's note in the other 6. A
musing takes 8.5 to 9.8 s to the first spoken word, since it is the first turn of a fresh session and includes the
skill read and the capture round trip.

**Orchestrator-side capture on the real rig** (the bridge saves the utterance with Pi's RPC `bash`,
`excludeFromContext: true`, then prompts the model with the words plus "already saved; do not save them again"):

| Placement | Request | Capture (RPC bash) ms | Saved bytes equal the utterance | Reply first text s | Reply total s | Tools in the reply turn | Saved again | Reply words |
|---|---|---|---|---|---|---|---|---|
| append | #11 | 48 | yes (see note) | 17.51 | 18.14 | - | no | 26 |
| append | #13 | 44 | yes | 4.56 | 4.88 | - | no | 15 |
| replace | #11 | 40 | yes | 4.58 | 5.31 | - | no | 32 |
| replace | #13 | 38 | yes | 0.24 | 3.24 | bash | no | 65 |

Byte-exact 4 of 4, 38 to 48 ms per capture, with `--via voice` and today's note every time, and the model never saved
again. The first word came 4.6 s after a fresh session's first prompt, against 8.5 to 9.8 s when the model captures.
The first row includes the reload of qwen38 after qwen27. The note on it: the test's own check reported false only
because the preceding model capture had saved the identical sentence; the journal shows the orchestrator's block
exact. One warm turn waited 3.2 s for the server before streaming; its twin in the other placement started in 0.24 s.
One reply ran an unrequested `tail` of the journal plus `atlas.py queue` to check the save; the persona should say not
to.

## Recommendation

1. **Live brain: qwen38 on ds4-server, persona via `--system-prompt`, thinking off.** Keep qwen27-262k for long or
   background work only, and only after its models.json entry gets `"compat": {"thinkingFormat":
   "qwen-chat-template"}`.
2. **Atlas: capture in the orchestrator, not the model.** When the router knows an utterance in the atlas space is a
   musing or a "note this", save it with RPC `bash` (`atlas_capture_command`, `--via voice`, excludeFromContext),
   then prompt with the "already saved" note. Exact, deterministic, about 4 s sooner to the reply, and the filing and
   `--via` no longer depend on the model. Keep the model-driven path as the fallback; with the fixed gate it works.
3. **Ship the gate's cd rule**, and treat any remaining "May I run a command?" during a capture as a bug.
4. **Knowledge questions:** acknowledge at `toolcall_start`, answer from search snippets first, cap spoken answers at
   two sentences with an offer of more, and consider a voice-only model alias with a low `maxTokens`.
5. **Never speak thinking deltas;** use llama-swap's loading banner as a "loading" cue.
6. **Do not let the agent speak counts it did not read off a tool's output.**

## Caveats

One run per configuration for the home requests (10 to 13 samples per configuration), so medians are firmer than
p90s. Each Atlas request in the rerun opened a fresh Pi session, so musing timings include the system prompt's
prefill. ds4-server reports no input token counts. The GPU was otherwise idle.

## Raw files

The raw rows, run logs and Pi RPC transcripts are not published: they contain the author's own files, notes and
journal text. The per-request tables in this folder keep every timing, tool and count.
