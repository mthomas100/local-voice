#!/usr/bin/env python3
"""qwen38_voice_tooluse_test.py: which local brain suits voice? Prepared 2026-10-05 (05c); run the same day (results/).

It runs 13 voice-style requests through Pi's RPC mode on the real rig, for each model in turn: 10 in the `home`
space (cwd ~, tools read,grep,find,ls,kb) and 3 in a fresh `git clone` of Atlas (tools read,grep,find,ls,bash). Per
request it records time to first text delta, time to the first sign of life (a tool call starting or text),
total time to agent_settled, the tools called and whether the expected one was among them, the reply's length in
words, and leaks into the spoken reply: thinking (reasoning deltas, <think> tags) and markdown. For the Atlas
musings it also checks that the capture ran before the first spoken word and that the bytes saved equal the
transcript.

GPU RULE. It does nothing unless ~/repos/local-voice/measure/bench/gpu_clear.sh exits 0 right before it starts
(exit code 75 otherwise). It checks once, at the start: once running it is itself the LLM traffic gpu_clear.sh
watches for. It never runs inside a `hold` (a hold closes the gate to LLM calls); hold.ts is loaded, so if a
render takes the GPU mid-run the turn waits, and the request is marked `held` and its timings invalid.

MODEL SWITCHES. Default order: local/qwen38, then local/qwen27-262k (the question
being: which brain suits voice). Under llama-swap every model runs ALONE (routing set `big_alone`), so switching evicts the first
model and costs a reload (13-17 s for a big model, measured 2026-09-11); the first request after each switch is
flagged `first_after_switch` and its time includes that load. At the end it requests local/qwen38 once more
(`restore`), so the rig is left on its usual default. qwen38 is served by ds4-server, qwen27-262k by llama-server
b10869 (--jinja): Pi sends both `thinking: {type: "disabled"}` for --thinking off (provider compat
thinkingFormat "deepseek", measured against a stub in 05c E7). ds4-server honours that field (ds4_server.c reads
"thinking"); whether llama-server does is not verified, and the thinking_chars column answers it. If qwen27 leaks
thinking, rerun with --qwen27-compat qwen-chat-template, which sends chat_template_kwargs.enable_thinking=false.

ISOLATION. A throwaway PI_CODING_AGENT_DIR (models.json copied from ~/.pi/agent/models.json, so the `local`
provider is the live gate on :8090; no packages), --no-extensions plus explicit hold.ts, kb.ts, today.ts and
the prototype voice_gate.ts. KB_HOME is a fresh `git clone` of ~/kb under the state dir, searched with ripgrep
(KB_SEARCH_BACKEND=rg) unless --kb-backend qmd. Atlas is `git clone`d with its push URL disabled; atlas.py runs
only in the clone. Any confirm request is answered "no" and recorded. Nothing is written to ~/kb, the real Atlas
or ~/.pi/agent. Results: research-prototypes/pi_rpc_bridge/results/voice-tooluse-<time>.{json,md}; transcripts
and clones: ~/repos/local-voice/state/voice-tooluse/<time>/ (gitignored). Replies and tool arguments are stored.

LEGS (added 2026-10-05 08:58 for the GPU phase). `--legs model:placement[:compat] ...` runs several configurations
in one guarded run, e.g. `local/qwen38:append local/qwen38:replace local/qwen27-262k:append`. `--compat-on-leak`:
when a qwen27 leg leaks thinking with --thinking off, that leg is rerun with thinkingFormat qwen-chat-template and
later qwen27 legs use it from the start. Between requests the run re-checks gpu_clear.sh's gpu_jobs and hold
fields (not its LLM-idle field, which this run's own traffic trips) and stops if a render or a hold has started;
the restore request is then skipped. A turn whose reasoning passes --leak-abort-chars is interrupted, and a leg
stops after --leak-skip-after leaked turns, so a thinking leak cannot burn the GPU for minutes per request.
Published results (results/ and --publish-dir) withhold the Atlas replies, which can quote the person's journal;
the full rows stay in the state dir, which git ignores.

Usage (Python 3.11+, standard library only):
  python3.12 qwen38_voice_tooluse_test.py                       # the real run, after the GPU check
  python3.12 qwen38_voice_tooluse_test.py --models local/qwen38 # one model
  python3.12 qwen38_voice_tooluse_test.py --dry-run             # print the plan; touches nothing
  python3.12 qwen38_voice_tooluse_test.py --stub                # self-test of this harness against stub_llm.py
                                                                # (scripted replies; never contacts the rig)
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pi_rpc_bridge import (ConfirmRequest, Notify, PiChild, PiError, Retry, SpaceConfig, Status, TextDelta,  # noqa: E402
                           ThinkingDelta, ToolCallStarted, ToolEnd, ToolStart, Settled, atlas_capture_command)

HOME = Path.home()
GPU_CLEAR = Path(os.environ.get("VOICE_TEST_GPU_CLEAR", HOME / "repos/local-voice/measure/bench/gpu_clear.sh"))
REAL_MODELS = HOME / ".pi/agent/models.json"
REAL_EXT = HOME / ".pi/agent/extensions"
ATLAS_REAL = Path(os.environ.get("ATLAS_REPO", HOME / "atlas"))   # the journal repo; cloned, never written
KB_REAL = Path(os.environ.get("KB_HOME", HOME / "kb"))
STATE = HOME / "repos/local-voice/state/voice-tooluse"
RESULTS = HERE / "results"
EX_TEMPFAIL = 75
TURN_TIMEOUT_S = 420.0

VOICE_PERSONA = (
    "You are the user's voice assistant on their Mac, and you are speaking, not writing. Answer in one or two short "
    "sentences of plain spoken English: no markdown, no lists, no headings, no code, and no file paths or URLs read out "
    "unless asked. When a question needs a file, a folder or the knowledge base, use a tool first, then say what you "
    "found in a sentence. If you cannot find it, say so briefly.")
ATLAS_RULES = (
    "When the person muses or says note this, save their exact words first with python3 atlas.py capture --via voice, "
    "then reply.\nNever edit, tidy or rewrite their words. Your own pages go to inbox/; wiki/ only when they say file that.")
ATLAS_BASH_ALLOW = ["python3 atlas.py *", "git status*", "git log*"]


@dataclass
class Req:
    space: str
    prompt: str
    expected: list[str]           # any of these counts as the expected tool; [] = no tool expected
    capture: bool = False         # an Atlas musing: the words must be saved before the reply
    prefix_ok: str = ""           # an instruction prefix atlas-journal allows the agent to leave off


REQUESTS = [
    Req("home", "Hi there, can you hear me okay?", []),
    Req("home", "What does my knowledge base say about the hold gate?", ["kb"]),
    Req("home", "Which folders are in my repos directory?", ["ls", "find"]),
    Req("home", "Read me the first line of the README in my local voice repo.", ["read"]),
    Req("home", "Is there a file called SPACES.md anywhere in my local voice project?", ["find", "ls"]),
    Req("home", "How many times does the word barge-in appear in the local voice spec?", ["grep"]),
    Req("home", "Quick one: what's twelve times eight?", []),
    Req("home", "Look up in my knowledge base what kb search uses under the hood.", ["kb"]),
    Req("home", "What extensions does my pi agent have installed? Just list the folder.", ["ls", "find"]),
    Req("home", "Thanks, that's all for now.", []),
    Req("atlas", "You know, I keep thinking about how quiet the garden was this morning, like the whole street was holding its breath.",
        ["bash"], capture=True),
    Req("atlas", "What's in my ideas file?", ["read", "bash", "grep"]),
    Req("atlas", "Note this: call the plumber about the kitchen tap on Tuesday.", ["bash"], capture=True, prefix_ok="Note this: "),
]

MARKDOWN = re.compile(r"(\*\*|__|^#{1,6} |^\s*[-*] |^\s*\d+\. |`|^\|.*\|$|\[[^\]]+\]\([^)]+\))", re.M)
THINK_TAG = re.compile(r"</?think>", re.I)


@dataclass
class Row:
    model: str
    space: str
    index: int
    prompt: str
    expected: list[str]
    tools: list[str] = field(default_factory=list)
    tool_args: list[str] = field(default_factory=list)
    expected_called: bool | None = None
    first_sign_s: float | None = None
    first_text_s: float | None = None
    total_s: float | None = None
    reply: str = ""
    reply_words: int = 0
    thinking_chars: int = 0
    think_tag_in_reply: bool = False
    markdown_in_reply: bool = False
    capture_before_reply: bool | None = None
    capture_verbatim: bool | None = None
    confirms_denied: int = 0
    held: bool = False
    retries: int = 0
    first_after_switch: bool = False
    stop_reason: str | None = None
    error: str | None = None
    usage: dict | None = None
    leg: str = ""
    prompt_mode: str = ""
    compat: str | None = None
    rep: int = 1


def guard() -> None:
    try:
        r = subprocess.run([str(GPU_CLEAR)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"gpu_clear.sh could not run ({e}); doing nothing.")
        sys.exit(EX_TEMPFAIL)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        print("The GPU is not clear; doing nothing. Run again when gpu_clear.sh exits 0.")
        sys.exit(EX_TEMPFAIL)


def make_agent_dir(d: Path, base_url: str | None, qwen27_compat: str | None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    m = json.loads(REAL_MODELS.read_text())
    p = m["providers"]["local"]
    prov = {k: v for k, v in p.items() if k != "models"}
    if base_url:
        prov["baseUrl"] = base_url
    models = [x for x in p["models"] if x["id"] in ("qwen38", "qwen27-262k")]
    if qwen27_compat:
        for x in models:
            if x["id"] == "qwen27-262k":
                x["compat"] = {**x.get("compat", {}), "thinkingFormat": qwen27_compat}
    (d / "models.json").write_text(json.dumps({"providers": {"local": {**prov, "models": models}}}, indent=1))
    (d / "settings.json").write_text(json.dumps({"defaultProvider": "local", "defaultModel": "qwen38", "packages": []}))
    return d


def clone(src: Path, dst: Path) -> Path:
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(src), str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "remote", "set-url", "--push", "origin", "DISABLED-voice-test-clone"], check=True)
    top = subprocess.run(["git", "-C", str(dst), "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    assert Path(top).resolve() == dst.resolve() != src.resolve(), f"{dst} is not a separate clone"
    return dst


def last_capture(note: Path) -> str | None:
    if not note.exists():
        return None
    s = note.read_text(encoding="utf-8")
    i = s.rfind(" · said to ")
    if i < 0:
        return None
    body = s[s.index("\n\n", i) + 2:]
    return body[:-1] if body.endswith("\n") else body


def space_configs(model: str, thinking: str, agent: Path, kb: Path, atlas: Path, run_dir: Path, kb_backend: str,
                  prompt_mode: str, stub: bool = False) -> dict[str, SpaceConfig]:
    proto_ext = HERE / "extensions"
    common_env = {"PI_CODING_AGENT_DIR": str(agent), "KB_HOME": str(kb), "UV_OFFLINE": "1"}
    if stub:
        common_env["HOLD_GATE"] = "http://127.0.0.1:9"   # never gated against the stub; belt and braces
    if kb_backend == "rg":
        common_env["KB_SEARCH_BACKEND"] = "rg"
    ext = [REAL_EXT / "hold.ts", REAL_EXT / "today.ts", proto_ext / "voice_gate.ts"]
    tag = model.split("/")[-1]

    def prompt_args(extra: str | None):
        text = VOICE_PERSONA + (("\n\n" + extra) if extra else "")
        if prompt_mode == "replace":
            return {"extra_args": ["--system-prompt", text]}
        return {"append_system_prompt": [text]}

    return {
        "home": SpaceConfig(name="home", root=HOME, model=model, thinking=thinking, tools=["read", "grep", "find", "ls", "kb"],
                            extensions=ext + [REAL_EXT / "kb.ts"], session_dir=run_dir / "sessions" / f"home-{tag}",
                            session_name=f"voice-test-home-{tag}", resume=False,
                            env={**common_env, "VOICE_TIER": "ask"}, **prompt_args(None)),
        "atlas": SpaceConfig(name="atlas", root=atlas, model=model, thinking=thinking, tools=["read", "grep", "find", "ls", "bash"],
                             extensions=ext, session_dir=run_dir / "sessions" / f"atlas-{tag}", session_name=f"voice-test-atlas-{tag}",
                             resume=False, env={**common_env, "VOICE_TIER": "trusted", "VOICE_BASH_ALLOW": json.dumps(ATLAS_BASH_ALLOW)},
                             **prompt_args(ATLAS_RULES)),
    }


async def run_request(child: PiChild, model: str, i: int, req: Req, atlas: Path, first_after_switch: bool,
                      leak_abort_chars: int = 0) -> Row:
    row = Row(model=model, space=req.space, index=i, prompt=req.prompt, expected=req.expected, first_after_switch=first_after_switch)
    note = atlas / "journal" / f"{date.today().isoformat()}.md"
    before = last_capture(note) if req.capture else None
    t0 = time.monotonic()
    tool_end_at = None
    try:
        async with asyncio.timeout(TURN_TIMEOUT_S), contextlib.aclosing(child.turn(req.prompt, timeout=TURN_TIMEOUT_S)) as events:
            async for ev in events:
                if isinstance(ev, (ToolCallStarted, TextDelta)) and row.first_sign_s is None:
                    row.first_sign_s = ev.at - t0
                if isinstance(ev, TextDelta):
                    if row.first_text_s is None:
                        row.first_text_s = ev.at - t0
                    row.reply += ev.text
                elif isinstance(ev, ThinkingDelta):
                    row.thinking_chars += len(ev.text)
                    if leak_abort_chars and row.thinking_chars > leak_abort_chars:
                        row.error = f"interrupted: reasoning passed {leak_abort_chars} characters with thinking off"
                        break
                elif isinstance(ev, ToolStart):
                    row.tools.append(ev.name)
                    row.tool_args.append(json.dumps(ev.args, ensure_ascii=False)[:300])
                elif isinstance(ev, ToolEnd) and ev.name == "bash" and "Saved word for word" in ev.text and tool_end_at is None:
                    tool_end_at = ev.at
                elif isinstance(ev, ConfirmRequest):
                    row.confirms_denied += 1
                elif isinstance(ev, (Notify, Status)) and "held" in (getattr(ev, "message", "") or getattr(ev, "text", "") or ""):
                    row.held = True
                elif isinstance(ev, Retry) and ev.kind == "auto_retry_start":
                    row.retries += 1
                elif ev.__class__.__name__ == "MessageEnd" and ev.role == "assistant":
                    row.stop_reason, row.error, row.usage = ev.stop_reason, ev.error, ev.usage
                elif isinstance(ev, Settled):
                    row.total_s = ev.at - t0
    except TimeoutError:
        row.error = f"turn timed out after {TURN_TIMEOUT_S:.0f} s"
        try:
            await child.interrupt()
        except PiError as e:
            row.error += f"; the run did not settle after the abort ({e})"
    if row.error and row.error.startswith("interrupted"):
        try:
            await child.interrupt()
        except PiError as e:
            row.error += f"; the run did not settle after the abort ({e})"
    row.reply_words = len(row.reply.split())
    row.think_tag_in_reply = bool(THINK_TAG.search(row.reply))
    row.markdown_in_reply = bool(MARKDOWN.search(row.reply))
    row.expected_called = (not row.tools) if not req.expected else any(t in req.expected for t in row.tools)
    if req.capture:
        after = last_capture(note)
        first_text_at = (t0 + row.first_text_s) if row.first_text_s is not None else None
        row.capture_before_reply = tool_end_at is not None and (first_text_at is None or tool_end_at < first_text_at)
        row.capture_verbatim = after is not None and after != before and after in (req.prompt, req.prompt[len(req.prefix_ok):] if req.prefix_ok else req.prompt)
    return row


async def answer_no(child, ev):
    return False


def stub_scripts(leak: bool = False, spaces=("home", "atlas")) -> list[dict]:
    """--stub only: a plausible tool call for each request, so the harness itself is exercised end to end.
    leak=True adds reasoning_content to every reply, as a model that ignores thinking-off would."""
    q = []
    for r in REQUESTS:
        if r.space not in spaces:
            continue
        if r.space == "home" and r.expected:
            args = {"kb": {"args": ["search", "hold gate"]}, "ls": {"path": "repos"}, "find": {"pattern": "SPACES.md", "path": "repos/local-voice"},
                    "read": {"path": "repos/local-voice/README.md"}, "grep": {"pattern": "barge-in", "path": "repos/local-voice/SPEC.md"}}[r.expected[0]]
            q.append({"tool_calls": [{"name": r.expected[0], "arguments": args}], "pre_delay_ms": 30})
        if r.capture:
            words = r.prompt[len(r.prefix_ok):] if r.prefix_ok else r.prompt
            q.append({"tool_calls": [{"name": "bash", "arguments": {"command": atlas_capture_command(words)}, "split": 6}], "delay_ms": 10, "pre_delay_ms": 30})
        elif r.space == "atlas":
            q.append({"tool_calls": [{"name": "read", "arguments": {"path": "journal/ideas.md"}}], "pre_delay_ms": 30})
        reply = {"text": "Stub reply in one short spoken sentence.", "delay_ms": 10, "pre_delay_ms": 30}
        if leak:
            reply["reasoning"] = " ".join(["hmm, let me think about this carefully"] * 40)
            reply["chunk_words"] = 20
        q.append(reply)
    return q


@dataclass
class Leg:
    model: str
    prompt_mode: str = "append"
    compat: str | None = None

    @property
    def label(self) -> str:
        return f"{self.model.split('/')[-1]}:{self.prompt_mode}" + (f":{self.compat}" if self.compat else "")


def parse_legs(a) -> list[Leg]:
    if a.legs:
        out = []
        for spec in a.legs:
            parts = spec.split(":")
            model = parts[0] if "/" in parts[0] else f"local/{parts[0]}"
            out.append(Leg(model, parts[1] if len(parts) > 1 else a.prompt_mode, parts[2] if len(parts) > 2 else None))
        return out
    return [Leg(m, a.prompt_mode, a.qwen27_compat if "qwen27" in m else None) for m in a.models]


GPU_STATUS = re.compile(r"gpu_jobs=(\d+) hold=(\S+)")


def gpu_taken() -> str | None:
    """Between requests: has a render or a hold started? gpu_clear.sh's LLM-idle field is ignored here, since this
    run's own requests trip it. Returns the status line if the GPU was taken, else None."""
    try:
        r = subprocess.run([str(GPU_CLEAR)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"gpu_clear.sh could not run: {e}"
    line = (r.stdout.strip() or r.stderr.strip()).splitlines()[-1] if (r.stdout.strip() or r.stderr.strip()) else ""
    m = GPU_STATUS.search(line)
    if not m:
        return f"unreadable gpu_clear.sh status: {line!r}"
    jobs, hold = int(m.group(1)), m.group(2)
    return None if jobs == 0 and hold in ("open", "absent") else line


def redact(row: dict) -> dict:
    """Published copy: Atlas replies and tool results can quote the person's journal, so their text is withheld."""
    if row.get("space") != "atlas":
        return row
    r = dict(row)
    r["reply"] = f"[withheld: Atlas reply, {row.get('reply_words', 0)} words; full text in the state dir]"
    r["tool_args"] = [a if "atlas.py capture" in a else "[withheld]" for a in row.get("tool_args", [])]
    return r


async def run_leg(leg: Leg, a, agent: Path, kb: Path, atlas: Path, run_dir: Path, stub, first_of_model: bool,
                  rows: list[Row], rep: int = 1) -> str | None:
    """One configuration, 13 requests. Returns why the whole run must stop (GPU taken), or None."""
    cfgs = space_configs(leg.model, a.thinking, agent, kb, atlas, run_dir / leg.label.replace(":", "-"), a.kb_backend,
                         leg.prompt_mode, bool(stub))
    if stub:
        stub.script(stub_scripts(leak=("qwen27" in leg.model and leg.compat is None and a.stub_leak), spaces=a.spaces),
                    default={"text": "Stub default reply."})
    first, leaks, n = first_of_model, 0, 0
    for space in a.spaces:
        suffix = f"-r{rep}" if a.repeat > 1 else ""
        child = PiChild(cfgs[space], on_ui=answer_no, transcript=run_dir / f"{space}-{leg.label.replace(':', '-')}{suffix}.log")
        await child.start()
        child.drain()
        try:
            for i, req in enumerate(REQUESTS, 1):
                if req.space != space:
                    continue
                if n and not stub:
                    taken = gpu_taken()
                    if taken:
                        return f"stopped before {leg.label} #{i}: {taken}"
                if a.leak_skip_after and leaks >= a.leak_skip_after:
                    print(f"{leg.label} #{i:2d} skipped: {leaks} turns were interrupted for runaway reasoning already")
                    continue
                row = await run_request(child, leg.model, i, req, atlas, first, a.leak_abort_chars)
                row.leg, row.prompt_mode, row.compat, row.rep = leg.label, leg.prompt_mode, leg.compat, rep
                first, n = False, n + 1
                # only heavy leaks (turns interrupted at --leak-abort-chars) end a leg early; a short reasoning
                # prefix (qwen38's first turn after a load streamed 248 characters, 2026-10-05) is recorded, not skipped
                leaks += bool(row.error and row.error.startswith("interrupted: reasoning"))
                rows.append(row)
                print(f"{leg.label}{f' r{rep}' if a.repeat > 1 else ''} #{i:2d} {space:5s} tools={row.tools} expected={row.expected_called} first_text={row.first_text_s} "
                      f"total={row.total_s} words={row.reply_words} think={row.thinking_chars} md={row.markdown_in_reply}"
                      + (f" capture_first={row.capture_before_reply} verbatim={row.capture_verbatim}" if req.capture else "")
                      + (f" ERROR={row.error}" if row.error else ""), flush=True)
        finally:
            await child.close()
    return None


async def orchestrator_capture_phase(a, agent: Path, kb: Path, atlas: Path, run_dir: Path, stub) -> list[dict]:
    """The capture path SPACES.md can take instead of trusting the model: the bridge saves the utterance with Pi's RPC
    `bash` (excludeFromContext, so the transcript is not repeated to the model), checks the bytes, then prompts the
    model with the words and a note that they are saved. Runs on local/qwen38, both persona placements."""
    out = []
    note = atlas / "journal" / f"{date.today().isoformat()}.md"
    for mode in ("append", "replace"):
        if stub:
            stub.script([{"text": "Got it, that's saved.", "delay_ms": 10}] * 4, default={"text": "Stub default reply."})
        cfg = space_configs("local/qwen38", a.thinking, agent, kb, atlas, run_dir / f"orchestrator-capture-{mode}", a.kb_backend,
                            mode, bool(stub))["atlas"]
        child = PiChild(cfg, on_ui=answer_no, transcript=run_dir / f"atlas-orchestrator-capture-{mode}.log")
        await child.start()
        child.drain()
        try:
            for i, req in enumerate(REQUESTS, 1):
                if not req.capture:
                    continue
                if not stub:
                    taken = gpu_taken()
                    if taken:
                        print(f"orchestrator capture stopped: {taken}", flush=True)
                        return out
                before = last_capture(note)
                t0 = time.monotonic()
                res = await child.bash(atlas_capture_command(req.prompt), exclude_from_context=True)
                cap_s = time.monotonic() - t0
                after = last_capture(note)
                prompt = (f"{req.prompt}\n\n(The voice system has already saved these words verbatim in journal/{note.name}. "
                          "Do not save them again; just reply.)")
                t1 = time.monotonic()
                r = await child.run_turn(prompt, timeout=TURN_TIMEOUT_S)
                again = any(n == "bash" and "atlas.py capture" in json.dumps(args) for n, args in r.tools)
                row = {"mode": mode, "index": i, "capture_s": round(cap_s, 3), "exit_code": res.get("exitCode"),
                       "output": (res.get("output") or "").strip(), "byte_exact": after == req.prompt and after != before,
                       "first_text_s": r.first_text_s, "total_s": r.total_s, "tools": [n for n, _ in r.tools],
                       "captured_again": again, "reply_words": len(r.text.split()), "reply": r.text,
                       "markdown_in_reply": bool(MARKDOWN.search(r.text)), "thinking_chars": len(r.thinking),
                       "wall_after_capture_s": round(time.monotonic() - t1, 3)}
                out.append(row)
                print(f"orchestrator-capture {mode} #{i}: capture {cap_s*1000:.0f} ms exit={row['exit_code']} byte_exact={row['byte_exact']} "
                      f"reply first_text={r.first_text_s} total={r.total_s} tools={row['tools']} again={again} words={row['reply_words']}", flush=True)
        finally:
            await child.close()
    return out


async def main_async(a) -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (Path(a.state_dir) if a.state_dir else STATE) / stamp
    legs = parse_legs(a)
    restore = None if a.no_restore or legs[-1].model == "local/qwen38" and not a.compat_on_leak else "local/qwen38"
    if a.dry_run:
        print("Plan (nothing run):")
        print("  guard: " + str(GPU_CLEAR) + " must exit 0 first; between requests its gpu_jobs and hold fields must stay clear")
        for leg in legs:
            print(f"  leg {leg.label}: {len(REQUESTS)} requests ({sum(r.space == 'home' for r in REQUESTS)} home, "
                  f"{sum(r.space == 'atlas' for r in REQUESTS)} atlas)")
        if a.compat_on_leak:
            print("  then, for any qwen27 leg that leaks thinking: the same leg again with thinkingFormat qwen-chat-template")
        if restore:
            print(f"  restore: one short request on {restore}")
        for i, r in enumerate(REQUESTS, 1):
            print(f"  {i:2d} [{r.space}] expect {r.expected or 'no tool'}{' + verbatim capture before reply' if r.capture else ''}: {r.prompt}")
        print(f"  results -> {RESULTS}/voice-tooluse-<time>.json/.md; state -> {run_dir}")
        return 0
    stub = None
    if a.stub:
        from stub_llm import StubLLM
        os.environ["HOLD_GATE"] = "http://127.0.0.1:9"   # every child, the restore one too: never the live gate
        run_dir.mkdir(parents=True, exist_ok=True)
        stub = StubLLM(0, run_dir / "stub").start()
        base_url = stub.base_url
    else:
        guard()
        base_url = None
    run_dir.mkdir(parents=True, exist_ok=True)
    agents = {None: make_agent_dir(run_dir / "pi-agent", base_url, None)}
    kb = clone(Path(a.kb_repo), run_dir / "kb")
    atlas = clone(Path(a.atlas_repo), run_dir / "atlas")
    print(f"state: {run_dir}", flush=True)
    rows: list[Row] = []
    stopped = None
    leaking: set[str] = set()   # models that leaked thinking with thinking off
    queue = list(legs)
    seen_models: set[str] = set()
    t_run = time.monotonic()
    while queue and not stopped:
        leg = queue.pop(0)
        if leg.model in leaking and leg.compat is None and a.compat_on_leak:
            leg = Leg(leg.model, leg.prompt_mode, "qwen-chat-template")
        if leg.compat not in agents:
            agents[leg.compat] = make_agent_dir(run_dir / f"pi-agent-{leg.compat}", base_url, leg.compat)
        n_before = len(rows)
        for rep in range(1, a.repeat + 1):
            stopped = await run_leg(leg, a, agents[leg.compat], kb, atlas, run_dir, stub, leg.model not in seen_models, rows, rep)
            seen_models.add(leg.model)
            if stopped:
                break
        leg_rows = rows[n_before:]
        leaked = sum(r.thinking_chars > 0 or r.think_tag_in_reply for r in leg_rows)
        if leaked and leg.compat is None and a.compat_on_leak and "qwen27" in leg.model:
            leaking.add(leg.model)
            queue.insert(0, Leg(leg.model, leg.prompt_mode, "qwen-chat-template"))
            print(f"{leg.label}: thinking leaked in {leaked} turns; rerunning it with thinkingFormat qwen-chat-template", flush=True)
    if stopped:
        print(stopped, flush=True)
    oc_rows: list[dict] = []
    if a.orchestrator_capture and not stopped:
        oc_rows = await orchestrator_capture_phase(a, agents[None], kb, atlas, run_dir, stub)
        restore = None   # that phase ran on qwen38, which is now loaded
    restore_row = None
    if restore and not stopped:
        cfg = SpaceConfig(name="restore", root=HOME, model=restore, thinking=a.thinking, tools=[], extensions=[REAL_EXT / "hold.ts"],
                          env={"PI_CODING_AGENT_DIR": str(agents[None])})
        child = PiChild(cfg, transcript=run_dir / "restore.log")
        await child.start()
        r = await child.run_turn("Say OK.", timeout=TURN_TIMEOUT_S)
        restore_row = {"model": restore, "first_text_s": r.first_text_s, "total_s": r.total_s, "reply": r.text}
        await child.close()
        print(f"restore {restore}: first_text={r.first_text_s} total={r.total_s}", flush=True)
    if stub:
        stub.stop()
    meta = {"stamp": stamp, "stub": bool(a.stub), "legs": [l.label for l in legs], "ran": sorted({r.leg for r in rows}, key=[r.leg for r in rows].index),
            "thinking": a.thinking, "kb_backend": a.kb_backend, "state_dir": str(run_dir), "persona": VOICE_PERSONA,
            "atlas_rules": ATLAS_RULES, "stopped": stopped, "wall_s": round(time.monotonic() - t_run, 1),
            "leak_abort_chars": a.leak_abort_chars, "leak_skip_after": a.leak_skip_after,
            "pi_version": subprocess.run(["pi", "--version"], capture_output=True, text=True).stdout.strip()}
    meta["orchestrator_capture"] = bool(oc_rows)
    full = {"meta": meta, "rows": [asdict(r) for r in rows], "restore": restore_row, "orchestrator_capture": oc_rows}
    (run_dir / f"voice-tooluse-{stamp}-full.json").write_text(json.dumps(full, indent=1))
    pub = {"meta": meta, "rows": [redact(asdict(r)) for r in rows], "restore": restore_row,
           "orchestrator_capture": [{k: v for k, v in r.items() if k != "reply"} | {"reply": f"[withheld: {r['reply_words']} words]"}
                                    for r in oc_rows]}
    out_dirs = [run_dir / "results"] if a.stub else [RESULTS] + ([Path(a.publish_dir)] if a.publish_dir else [])
    name = f"voice-tooluse-{stamp}{'-stub' if a.stub else ''}"
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(pub, indent=1))
        (d / f"{name}.md").write_text(summary_md(meta, [Row(**{k: v for k, v in x.items()}) for x in pub["rows"]], restore_row))
        with open(d / f"{name}-rows.jsonl", "w") as f:
            for x in pub["rows"]:
                f.write(json.dumps(x) + "\n")
    print(f"results: {out_dirs[0] / name}.json, .md, -rows.jsonl (full rows: {run_dir / f'voice-tooluse-{stamp}-full.json'})", flush=True)
    return 0 if not stopped else 3


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary_md(meta, rows, restore_row) -> str:
    f = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    lines = [f"# Voice tool-use test {meta['stamp']}{' (STUB self-test, not a measurement)' if meta['stub'] else ''}", "",
             f"Pi {meta['pi_version']}; thinking {meta['thinking']}; kb search {meta['kb_backend']}; legs run: {', '.join(meta['ran'])}; "
             f"wall time {meta['wall_s']} s." + (f" **Stopped:** {meta['stopped']}." if meta['stopped'] else ""), "",
             "Timing columns leave out each model's first request (it includes the model load) and any request made while the GPU was "
             "held. Seconds from sending the prompt: first sign = a tool call starting or text; first text = the first word "
             "that would be spoken; total = agent_settled.", "",
             "| leg | turns | expected tool | first text median / p90 s | first sign median s | total median / p90 s | words median / max | thinking leaks | markdown leaks | captures first + verbatim | errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for leg in meta["ran"]:
        rs = [r for r in rows if r.leg == leg]
        ok = [r for r in rs if not r.first_after_switch and not r.held]
        ft = [r.first_text_s for r in ok if r.first_text_s is not None]
        fs = [r.first_sign_s for r in ok if r.first_sign_s is not None]
        tt = [r.total_s for r in ok if r.total_s is not None]
        words = [r.reply_words for r in rs]
        caps = [r for r in rs if r.capture_before_reply is not None]
        lines.append(f"| {leg} | {len(rs)} | {sum(bool(r.expected_called) for r in rs)}/{len(rs)} | {f(pct(ft, .5))} / {f(pct(ft, .9))} | "
                     f"{f(pct(fs, .5))} | {f(pct(tt, .5))} / {f(pct(tt, .9))} | {f(pct(words, .5))} / {max(words) if words else '-'} | "
                     f"{sum(r.thinking_chars > 0 or r.think_tag_in_reply for r in rs)} | {sum(r.markdown_in_reply for r in rs)} | "
                     f"{sum(bool(r.capture_before_reply and r.capture_verbatim) for r in caps)}/{len(caps)} | {sum(bool(r.error) for r in rs)} |")
    lines += ["", "| leg | # | space | tools | expected | first sign s | first text s | total s | words | think chars | md | capture first / verbatim | in / out / cached tokens | reply |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cap = "-" if r.capture_before_reply is None else f"{r.capture_before_reply} / {r.capture_verbatim}"
        u = r.usage or {}
        tok = f"{u.get('input', '-')} / {u.get('output', '-')} / {u.get('cacheRead', '-')}"
        reply = (r.reply or "").replace("|", "/").replace(chr(10), " ")[:160] + (f" [error: {r.error}]" if r.error else "")
        lines.append(f"| {r.leg} | {r.index}{' (load)' if r.first_after_switch else ''}{' (held)' if r.held else ''} | {r.space} | "
                     f"{','.join(r.tools) or '-'} | {r.expected_called} | {f(r.first_sign_s)} | {f(r.first_text_s)} | {f(r.total_s)} | "
                     f"{r.reply_words} | {r.thinking_chars} | {r.markdown_in_reply} | {cap} | {tok} | {reply} |")
    if restore_row:
        lines += ["", f"Restore: {restore_row['model']} answered in {f(restore_row['total_s'])} s (first text {f(restore_row['first_text_s'])} s, includes the reload)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["local/qwen38", "local/qwen27-262k"])
    ap.add_argument("--legs", nargs="+", default=None, help="model:append|replace[:compat] ...; overrides --models and --prompt-mode")
    ap.add_argument("--compat-on-leak", action="store_true", help="rerun a qwen27 leg that leaks thinking with thinkingFormat qwen-chat-template")
    ap.add_argument("--leak-abort-chars", type=int, default=4000, help="interrupt a turn once its reasoning passes this many characters (0 = never)")
    ap.add_argument("--leak-skip-after", type=int, default=2, help="skip the rest of a leg after this many turns interrupted for runaway reasoning (0 = never)")
    ap.add_argument("--thinking", default="off")
    ap.add_argument("--prompt-mode", choices=["append", "replace"], default="append",
                    help="append = --append-system-prompt (SPACES.md); replace = --system-prompt (drops Pi's coding preamble, tools, rules, docs)")
    ap.add_argument("--kb-backend", choices=["rg", "qmd"], default="rg")
    ap.add_argument("--qwen27-compat", choices=["qwen-chat-template", "qwen", "chat-template"], default=None)
    ap.add_argument("--kb-repo", default=str(KB_REAL), help="cloned, never written")
    ap.add_argument("--atlas-repo", default=str(ATLAS_REAL), help="cloned, never written")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--publish-dir", default=None, help="also write the published (redacted) results here")
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--spaces", nargs="+", choices=["home", "atlas"], default=["home", "atlas"], help="which spaces' requests to run")
    ap.add_argument("--repeat", type=int, default=1, help="run each leg this many times (fresh Pi child and session each time)")
    ap.add_argument("--orchestrator-capture", action="store_true",
                    help="afterwards, on qwen38: save each musing with RPC bash (excludeFromContext), check the bytes, then prompt")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-leak", action="store_true", help="--stub only: qwen27 legs without compat emit reasoning, to exercise --compat-on-leak")
    a = ap.parse_args()
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
