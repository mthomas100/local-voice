"""The model call, through Pi in JSON mode (`pi --mode json -p`), one Pi run per reflection pass.

The model only proposes: it gets no tools at all (`--no-tools`), and deterministic code validates what it says
and does every write. Pi still earns its place: it resolves `local/<model>` through the same models.json every
other client uses (compat settings included), and in the technical pass it loads the kb's own Pi adapter (kb.ts),
which registers the run as a `pi:` session in the kb's operational lane exactly like any other Pi session, so the
brain's kb notes have a session to belong to and `kb trace` can find the transcript.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Variables a parent session leaks that would mis-attribute kb writes or confuse Pi (the bridge drops the same set).
_DROP_PREFIXES = ("KB_", "PI_", "CLAUDE", "HERDR")

# Found 2026-10-05: llama-server ignores Pi's `thinking: disabled` for the qwen27 rows and the model reasons for
# about a minute, leaking its thinking, unless the row's compat carries this thinkingFormat.
NEEDS_CHAT_TEMPLATE = ("qwen27",)
CHAT_TEMPLATE_FORMAT = "qwen-chat-template"


class LLMError(RuntimeError):
    """A Pi run failed: exit code, timeout, provider error, or no assistant text."""


@dataclass
class PiResult:
    text: str
    session_id: str
    session_file: Path | None
    model: str
    provider: str
    stop_reason: str
    seconds: float
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def actor(self) -> str:
        return f"pi/{self.model}"


def child_env(agent_dir: Path | None, extra: dict[str, str] | None = None, pi_bin: Path | None = None
              ) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_DROP_PREFIXES)}
    env.update(PI_OFFLINE="1", PI_SKIP_VERSION_CHECK="1")
    if pi_bin is not None:
        # pi is a `#!/usr/bin/env node` script under nvm; launchd's PATH has no node, the one beside pi is the right one.
        env["PATH"] = f"{pi_bin.parent}:{env.get('PATH', '/usr/bin:/bin')}"
    if agent_dir:
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env.update(extra or {})
    return env


def models_json(agent_dir: Path | None) -> dict[str, Any] | None:
    p = (agent_dir or Path.home() / ".pi" / "agent") / "models.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def effective_compat(models: dict[str, Any] | None, model: str) -> dict[str, Any] | None:
    """Provider compat merged with the model row's compat, or None when the row is not found."""
    provider, _, mid = model.partition("/")
    if not mid:
        provider, mid = "", model
    for pname, prov in ((models or {}).get("providers") or {}).items():
        if provider and pname != provider:
            continue
        for row in prov.get("models") or []:
            if row.get("id") == mid:
                return {**(prov.get("compat") or {}), **(row.get("compat") or {})}
    return None


def choose_model(model: str, fallback: str, agent_dir: Path | None) -> tuple[str, str]:
    """The model to run and why. A qwen27 row without the chat-template compat falls back (see the comment above)."""
    mid = model.partition("/")[2] or model
    if not mid.startswith(NEEDS_CHAT_TEMPLATE):
        return model, ""
    compat = effective_compat(models_json(agent_dir), model)
    if compat is None:
        return model, f"{model} not found in Pi's models.json; Pi will report it"
    fmt = compat.get("thinkingFormat")
    if fmt == CHAT_TEMPLATE_FORMAT:
        return model, ""
    return fallback, (f"{model} has compat.thinkingFormat {fmt!r}, not {CHAT_TEMPLATE_FORMAT!r}, so its thinking "
                      f"cannot be switched off (give its models.json row the chat-template compat); using {fallback}")


def find_session_file(session_dir: Path, session_id: str) -> Path | None:
    hits = sorted(session_dir.glob(f"*_{session_id}.jsonl"))
    return hits[-1] if hits else None


def parse_events(stdout: str) -> dict[str, Any]:
    """Session header id, the last assistant message, and any provider error, from Pi's JSONL stream."""
    out: dict[str, Any] = {"session_id": None, "message": None, "errors": []}
    for line in stdout.split("\n"):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = ev.get("type")
        if typ == "session":
            out["session_id"] = ev.get("id")
        elif typ == "message_end":
            msg = ev.get("message") or {}
            if msg.get("role") == "assistant":
                out["message"] = msg
        elif typ == "auto_retry_end" and not ev.get("success", True):
            out["errors"].append(str(ev.get("finalError") or "retry failed"))
    return out


def message_text(msg: dict[str, Any]) -> str:
    parts = msg.get("content") or []
    if isinstance(parts, str):
        return parts
    # Text blocks only: a thinking block is never part of the answer (never use reasoning output).
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")


class PiRunner:
    """Runs one `pi --mode json` pass and returns the final assistant text."""

    def __init__(self, pi_bin: Path, *, agent_dir: Path | None, session_dir: Path, timeout_s: float):
        self.pi_bin = pi_bin
        self.agent_dir = agent_dir
        self.session_dir = session_dir
        self.timeout_s = timeout_s

    def check(self) -> None:
        if not self.pi_bin.is_file():
            raise LLMError(f"pi not found at {self.pi_bin}; set llm.pi_bin in brain/config.toml (`which pi`)")

    def argv(self, *, model: str, session_id: str | None, system_prompt_file: Path | None, task: str,
             extensions: list[Path]) -> list[str]:
        a = [str(self.pi_bin), "--mode", "json", "--no-tools", "--no-skills", "--no-extensions"]
        for e in extensions:
            a += ["-e", str(e)]
        a += ["--no-context-files", "--no-prompt-templates", "--no-approve", "--thinking", "off", "--model", model]
        if session_id:
            a += ["--session-dir", str(self.session_dir), "--session-id", session_id]
        else:
            a += ["--no-session"]
        if system_prompt_file is not None:
            if not system_prompt_file.is_file():
                # Pi reads an argument that is not an existing file as the prompt text itself.
                raise LLMError(f"system prompt file missing: {system_prompt_file}")
            a += ["--system-prompt", str(system_prompt_file)]
        # Pi prepends piped stdin to the first prompt with no separator (measured 2026-10-05), hence the blank line.
        a += ["--", "\n\n" + task]
        return a

    def run(self, *, model: str, task: str, stdin_text: str = "", session_id: str | None = None,
            system_prompt_file: Path | None = None, cwd: Path, extensions: list[Path] | None = None,
            env_extra: dict[str, str] | None = None) -> PiResult:
        self.check()
        if session_id:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        argv = self.argv(model=model, session_id=session_id, system_prompt_file=system_prompt_file, task=task,
                         extensions=list(extensions or []))
        t0 = time.monotonic()
        try:
            p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True, cwd=str(cwd),
                               env=child_env(self.agent_dir, env_extra, self.pi_bin), timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"pi timed out after {self.timeout_s:g} s") from e
        except OSError as e:
            raise LLMError(f"pi did not start: {e}") from e
        secs = time.monotonic() - t0
        ev = parse_events(p.stdout)
        msg = ev["message"]
        if p.returncode != 0 or msg is None:
            tail = (p.stderr.strip().splitlines() or ["no stderr"])[-1]
            raise LLMError(f"pi exited {p.returncode} without an answer: {tail}")
        stop = str(msg.get("stopReason") or "")
        if stop in ("error", "aborted") or ev["errors"]:
            raise LLMError(f"the model call failed ({stop}): {msg.get('errorMessage') or '; '.join(ev['errors'])}")
        text = message_text(msg)
        if not text.strip():
            raise LLMError("the model returned no text")
        sid = ev["session_id"] or session_id or ""
        return PiResult(text=text, session_id=sid,
                        session_file=find_session_file(self.session_dir, sid) if session_id else None,
                        model=str(msg.get("model") or model.partition("/")[2] or model),
                        provider=str(msg.get("provider") or model.partition("/")[0]), stop_reason=stop,
                        seconds=secs, usage=dict(msg.get("usage") or {}))
