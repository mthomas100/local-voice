#!/usr/bin/env python3
"""A protocol v1 voice server with no models, for testing the Apple clients (PROTOCOL.md, frozen 2026-10-05).

It speaks the wire protocol the orchestrator serves on :8770, listens on 127.0.0.1 only (a scratch port, never
8770), and answers every user turn with a planted tone (or a `say` voice) so the client side can be checked against
machine ground truth: the mock reports the pitch, length and level of what the client sent, and the client reports
what it played. Every message in and out goes to a JSONL log for the e2e tests.

Behaviour copied from the real server's design (research-prototypes/pipecat_skeletons):
- accept, then close 4403 for a refused peer (a close before accept surfaces as a bare HTTP 403);
- 1002 for a missing or malformed hello, an unknown version, or a binary frame before hello;
- a second connection with the same device takes over: the old one is closed with 4409;
- reply audio is 24 kHz PCM16 in 40 ms messages (1,920 bytes), paced at real time;
- `interrupt` is sent only while a reply is active, never for a turn start with nothing playing, and not when the
  client interrupted first;
- push-to-talk turns are bracketed by start/stop; open-mic turns are found by an energy VAD (the real server uses
  Silero and Smart Turn; this is enough to exercise barge-in and the client's mic gate);
- `/v1/status` follows the session (state, space, mode, tier, model, tool, hold, the last turn's latency, clients)
  with the orchestrator's additions (`spaces`, `turns`, `uptime_s`); `space` and `mode` switch among SPACES.md's
  spaces and answer with a `space` message, or an `error` for a name it does not know (an assumption: the protocol
  does not say how a refusal looks). `--no-switch` ignores both, as the M1 orchestrator does.
- approvals (PROTOCOL.md "Approvals", 2026-10-05): `--approval SCENARIO[,SCENARIO...]` asks a question before each
  answer, one scenario per turn (the last repeats): `confirm_request` with summary, action and choices (an old-style
  one for `legacy`), then the question aloud; a button (`confirm_response`), a spoken turn (taken as a yes, withdrawn
  with `confirm_cancel` why "answered"), a new typed turn ("overtaken") or silence ("timeout") answers it. The why
  codes are the orchestrator's (2026-10-05).

Usage: .venv/bin/python mock_server.py --port 0 --log runs/x/server.jsonl   (port 0 picks a free port and prints it)
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

# The spaces of SPACES.md, as the orchestrator's /v1/status lists them (descriptions and tiers from spaces.yaml there;
# the model names say mock, so no screenshot of the mock passes for the real thing).
SPACES = {
    "home": {"name": "home", "description": "your Mac", "tier": "ask", "model": "mock/qwen38"},
    "atlas": {"name": "atlas", "description": "your atlas journal", "tier": "trusted", "model": "mock/qwen38"},
    "kb": {"name": "kb", "description": "the knowledge base", "tier": "readonly", "model": "mock/qwen38"},
    "voice": {"name": "voice", "description": "this voice project", "tier": "readonly", "model": "mock/qwen38"},
}
MODES = ("conversation", "act")

# Approvals: the question each scenario asks before its answer. The paths name a made-up home; the mock writes nothing.
HOME = "/Users/owner"
ALLOW_ONCE = {"id": "allow_once", "label": "Do it"}
ALLOW_SESSION = {"id": "allow_session", "label": "Allow this for the rest of the session"}
DENY = {"id": "deny", "label": "Don't"}
LONG_TEXT = "\n".join(f"Line {i}: the agent's notes from the session, kept word for word." for i in range(1, 141))
APPROVALS = {
    "write": {
        "summary": "Create a new file plan.txt in your notes folder with the text below.",
        "action": {"tool": "write", "effect": "create", "path": f"{HOME}/notes/plan.txt", "cwd": f"{HOME}/notes",
                   "space": "home", "mode": "act",
                   "preview": "Plan for Saturday\n\n- walk before breakfast\n- call Mum at 11\n"},
        "title": "May I write a file?", "message": f"write {HOME}/notes/plan.txt", "label": "writing plan.txt",
        "done": "Done. plan.txt is in your notes folder.", "not_done": "OK, I didn't create plan.txt.",
        "silence": "I didn't create plan.txt, because there was no answer.",
    },
    "edit": {
        "summary": "Change one line in todo.md in your notes folder: 'buy milk' becomes 'buy oat milk'.",
        "action": {"tool": "edit", "effect": "modify", "path": f"{HOME}/notes/todo.md", "cwd": f"{HOME}/notes",
                   "space": "home", "mode": "act",
                   "preview": "--- a/todo.md\n+++ b/todo.md\n@@ -1,4 +1,4 @@\n # This week\n-- buy milk\n+- buy oat milk\n"
                              " - post the parcel\n - book the dentist"},
        "title": "May I edit a file?", "message": f"edit {HOME}/notes/todo.md", "label": "editing todo.md",
        "done": "Done. It says buy oat milk now.", "not_done": "OK, I left todo.md as it was.",
        "silence": "I didn't change todo.md, because there was no answer.",
    },
    "bash": {
        "summary": "Run a command that lists the 20 newest files in your Downloads folder.",
        "action": {"tool": "bash", "effect": "run", "command": "ls -lt ~/Downloads | head -20", "cwd": HOME,
                   "space": "home", "mode": "act"},
        "title": "May I run a command?", "message": "ls -lt ~/Downloads | head -20", "label": "listing Downloads",
        "done": "The newest is a PDF from this morning.", "not_done": "OK, I didn't run it.",
        "silence": "I didn't run the command, because there was no answer.",
    },
    "session": {
        "summary": "Create a new page in your knowledge base titled 'Tea brewing notes'.",
        "action": {"tool": "kb", "effect": "create",
                   "command": "kb new Analysis tea-brewing-notes --title 'Tea brewing notes'",
                   "cwd": f"{HOME}/kb", "space": "home", "mode": "act",
                   "preview": "# Tea brewing notes\n\nGreen tea: 80 °C for two minutes. Black tea: just off the "
                              "boil, four minutes.\n"},
        "choices": [ALLOW_ONCE, ALLOW_SESSION, DENY],
        "title": "May I change your knowledge base?", "message": "kb new Analysis tea-brewing-notes",
        "label": "adding a page to your knowledge base", "done": "Done. The page is in your knowledge base.",
        "not_done": "OK, I didn't add the page.", "silence": "I didn't add the page, because there was no answer.",
    },
    "long": {
        "summary": "Create a new file session-notes.md in your notes folder with the text below.",
        "action": {"tool": "write", "effect": "create", "path": f"{HOME}/notes/session-notes.md",
                   "cwd": f"{HOME}/notes", "space": "home", "mode": "act",
                   # As the server cuts a preview: 4,000 characters and a marker (PROTOCOL.md).
                   "preview": LONG_TEXT[:4000] + f"\n… [cut: {len(LONG_TEXT) - 4000:,} more characters]"},
        "title": "May I write a file?", "message": f"write {HOME}/notes/session-notes.md",
        "label": "writing session-notes.md", "done": "Done. The notes are saved.",
        "not_done": "OK, I didn't save the notes.", "silence": "I didn't save the notes, because there was no answer.",
    },
    # An old-style question, as early servers asked it: title and message only, yes or no, and no confirm_cancel
    # when its time ran out.
    "legacy": {
        "title": "May I change your knowledge base?",
        "message": "kb new Issue tea-brewing-notes --title Tea brewing notes",
        "label": "adding a page to your knowledge base", "done": "Done. The page is in your knowledge base.",
        "not_done": "OK, I didn't add the page.", "silence": "I didn't add the page, because there was no answer.",
    },
}
APPROVALS["cancel"] = APPROVALS["bash"]  # answered aloud while the card is up
APPROVALS["timeout"] = APPROVALS["write"]  # no answer: withdrawn when its time runs out

IN_RATE = 16_000
OUT_RATE = 24_000
OUT_CHUNK = 1_920  # 40 ms at 24 kHz, as the orchestrator's audio_out_10ms_chunks=4
SPEECH_DBFS = -35.0  # open-mic VAD threshold
VAD_START_FRAMES = 3  # 60 ms of speech starts a turn
VAD_END_MS = 500  # this much silence ends it (--vad-end-ms)


def samples_of(data: bytes) -> array.array:
    a = array.array("h")
    a.frombytes(data[: len(data) // 2 * 2])
    if sys.byteorder != "little":
        a.byteswap()
    return a


def rms_dbfs(s) -> float:
    if not s:
        return -120.0
    acc = sum((x / 32768.0) ** 2 for x in s) / len(s)
    return max(-120.0, 10 * math.log10(acc)) if acc > 0 else -120.0


def zero_cross_hz(s, rate: int) -> float:
    crossings = [i for i in range(1, len(s)) if s[i - 1] < 0 <= s[i]]
    if len(crossings) < 3:
        return 0.0
    return (len(crossings) - 1) / ((crossings[-1] - crossings[0]) / rate)


def tone_pcm(freq: float, seconds: float, rate: int = OUT_RATE, amp: float = 0.3) -> bytes:
    n = int(seconds * rate)
    fade = int(0.01 * rate)
    out = array.array("h")
    for i in range(n):
        edge = min(1.0, min(i, n - 1 - i) / fade)
        out.append(int(32767 * amp * edge * math.sin(2 * math.pi * freq * i / rate)))
    return out.tobytes()


_say_cache: dict[str, bytes] = {}


def say_pcm(text: str, voice: str = "Samantha") -> bytes:
    """`say` rendered to 24 kHz PCM16 mono (CPU only; macOS's own synthesizer)."""
    if text in _say_cache:
        return _say_cache[text]
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = os.path.join(d, "s.aiff"), os.path.join(d, "s.wav")
        subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{OUT_RATE}", "-c", "1", aiff, wav], check=True)
        raw = open(wav, "rb").read()
    # Walk the RIFF chunks to the PCM data.
    pos = 12
    while pos + 8 <= len(raw):
        cid, size = raw[pos : pos + 4], int.from_bytes(raw[pos + 4 : pos + 8], "little")
        if cid == b"data":
            _say_cache[text] = raw[pos + 8 : pos + 8 + size]
            return _say_cache[text]
        pos += 8 + size + (size & 1)
    raise RuntimeError("no data chunk in afconvert output")


class Log:
    def __init__(self, path: str | None):
        self.t0 = time.monotonic()
        self.f = open(path, "a", buffering=1) if path else None

    def __call__(self, event: str, **fields):
        rec = {"t": round(time.monotonic() - self.t0, 4), "event": event, **fields}
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
        if self.f:
            self.f.write(line + "\n")
        else:
            print(line, flush=True)


@dataclass
class Turn:
    mic: str
    samples: array.array = field(default_factory=lambda: array.array("h"))
    frames: int = 0
    bad_sizes: list = field(default_factory=list)
    frame_dbfs: list = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    def add(self, data: bytes):
        self.frames += 1
        if len(data) % 2 or not (640 <= len(data) <= 1280):
            self.bad_sizes.append(len(data))
        s = samples_of(data)
        self.samples.extend(s)
        self.frame_dbfs.append(round(rms_dbfs(s), 1))

    def summary(self) -> dict:
        s = self.samples
        loud = [i for i, db in enumerate(self.frame_dbfs) if db > SPEECH_DBFS]
        zero_tail = 0
        for db in reversed(self.frame_dbfs):
            if db > -100:
                break
            zero_tail += 1
        speech = s[loud[0] * 320 : (loud[-1] + 1) * 320] if loud else array.array("h")
        return {
            "mic": self.mic,
            "frames": self.frames,
            "seconds": round(len(s) / IN_RATE, 3),
            "bad_sizes": self.bad_sizes[:10],
            "speech_seconds": round(len(loud) * 0.02, 3),
            "speech_hz": round(zero_cross_hz(speech, IN_RATE), 1),
            "speech_dbfs": round(rms_dbfs(speech), 1),
            "first_speech_frame": loud[0] if loud else None,
            "zero_tail_ms": zero_tail * 20,
            "frame_dbfs": self.frame_dbfs,
        }


class MockServer:
    def __init__(self, args):
        self.args = args
        self.log = Log(args.log)
        self.devices: dict[str, Session] = {}
        self.connections = 0
        self.replies = 0
        self.hold = "held" if args.hold else "open"
        self.started = time.monotonic()
        # What /v1/status reports: kept as the sessions go.
        self.space, self.mode, self.state = "home", "conversation", "idle"
        self.tool: dict | None = None
        self.last_turn = {"eos_to_first_audio_ms": 0, "stt_ms": 0, "llm_ttft_ms": 0, "tts_first_audio_ms": 0}
        self.turns = 0
        self.questions = 0
        self.approval_plan = args.approval.split(",") if args.approval else []
        # --turns: what the n-th user turn "said" and the reply to speak, for a dry run of the real-server tests
        # (tests/test_real_server.py). Replies are rendered now, so a reply never waits for `say`.
        self.scripted: list[dict] = json.load(open(args.turns)) if args.turns else []
        for turn in self.scripted:
            say_pcm(turn["reply"])

    def next_scripted(self) -> dict | None:
        return self.scripted.pop(0) if self.scripted else None

    def next_approval(self) -> str | None:
        """--approval: this turn's scenario (`none` asks nothing); the last one repeats."""
        if not self.approval_plan:
            return None
        scenario = self.approval_plan.pop(0) if len(self.approval_plan) > 1 else self.approval_plan[0]
        return None if scenario == "none" else scenario

    def status(self) -> dict:
        now = time.monotonic()
        sp = SPACES[self.space]
        return {
            "v": 1, "state": self.state, "space": self.space, "mode": self.mode, "tier": sp["tier"],
            "model": sp["model"], "hold": {"phase": self.hold, "why": "mock render" if self.hold != "open" else ""},
            "tool": self.tool, "last_turn": self.last_turn, "turns": self.turns,
            "spaces": {n: {**sp_, "tools": [], "act_tools": []} for n, sp_ in SPACES.items()},
            "clients": [{"device": d, "client": s.client, "connected_s": round(now - s.opened, 1)}
                        for d, s in self.devices.items()],
            "uptime_s": round(now - self.started),
        }

    def space_message(self) -> dict:
        sp = SPACES[self.space]
        return {"t": "space", "name": self.space, "mode": self.mode, "tier": sp["tier"],
                "description": sp["description"]}

    def process_request(self, connection: ServerConnection, request):
        if request.path == "/v1/status":
            self.log("status-request", served=not self.args.no_status)
            if self.args.no_status:
                return connection.respond(404, "not found\n")
            response = connection.respond(200, json.dumps(self.status()) + "\n")
            del response.headers["Content-Type"]  # Headers is a multidict: assigning adds a second one
            response.headers["Content-Type"] = "application/json"
            return response
        if request.path != "/v1/voice":
            return connection.respond(404, "not found\n")
        return None

    async def handler(self, ws: ServerConnection):
        self.connections += 1
        conn = self.connections
        self.log("connect", conn=conn, peer=str(ws.remote_address))
        session = Session(self, ws, conn)
        try:
            await session.run()
        except ConnectionClosed:
            pass
        finally:
            session.cancel_reply()
            if self.devices.get(session.device) is session:
                del self.devices[session.device]
            self.log("disconnect", conn=conn, code=ws.close_code, reason=ws.close_reason)


class Session:
    def __init__(self, server: MockServer, ws: ServerConnection, conn: int):
        self.server, self.ws, self.conn, self.log = server, ws, conn, server.log
        self.args = server.args
        self.opened = time.monotonic()
        self.device = ""
        self.client = ""
        self.mic = "vad"
        self.turn_ended_at: float | None = None
        self.turn_tools: list[str] = []
        self.turn: Turn | None = None
        self.reply_task: asyncio.Task | None = None
        self.active_reply: str | None = None  # between audio_start and the client's played_ms, as the serializer tracks
        self.cancel = False
        self.client_interrupted = False
        self.reply_sent_ms: dict[str, int] = {}
        self.reply_full_ms: dict[str, int] = {}
        # The approval question waiting for an answer: {"id", "future", "scenario"}.
        self.question: dict | None = None
        self.answer_turn = False  # this user turn answers the question aloud, it asks nothing
        self.session_allowed: set[str] = set()  # tools allowed for the rest of the session (allow_session)
        self.vad_speech = 0
        self.vad_silence = 0
        self.vad_preroll: list[bytes] = []

    async def send(self, msg: dict):
        if msg.get("t") == "state":
            self.server.state = msg["v"]
        self.log("out", conn=self.conn, msg=msg)
        await self.ws.send(json.dumps(msg, separators=(",", ":")))

    async def close(self, code: int, reason: str):
        self.log("close", conn=self.conn, code=code, reason=reason)
        await self.ws.close(code, reason)

    async def run(self):
        a = self.args
        if a.reject:
            await self.close(4403, "not one of the owner's tailnet logins")
            return
        try:
            first = await asyncio.wait_for(self.ws.recv(), 10)
        except TimeoutError:
            await self.close(1002, "no hello")
            return
        if isinstance(first, bytes):
            self.log("binary-before-hello", conn=self.conn, bytes=len(first))
            await self.close(1002, "binary frame before hello")
            return
        try:
            hello = json.loads(first)
        except ValueError:
            await self.close(1002, "hello is not JSON")
            return
        self.log("in", conn=self.conn, msg=hello)
        if hello.get("t") != "hello" or hello.get("v") != 1 or hello.get("mic") not in ("vad", "ptt"):
            await self.close(1002, "bad hello")
            return
        self.device, self.mic, self.client = str(hello.get("device", "")), hello["mic"], str(hello.get("client", ""))
        old = self.server.devices.get(self.device)
        if old is not None:
            await old.close(4409, "another session with this device took over")
        self.server.devices[self.device] = self
        if a.protocol_error_after_hello:
            await self.close(1002, "test: protocol error")
            return
        if a.slow_welcome_ms:
            await asyncio.sleep(a.slow_welcome_ms / 1000)
        hold = {"phase": self.server.hold, "why": "mock render"} if a.hold_object else self.server.hold
        srv = self.server
        await self.send({"t": "welcome", "v": 1, "session": f"s{uuid.uuid4().hex[:8]}", "space": srv.space,
                         "mode": srv.mode, "tier": SPACES[srv.space]["tier"], "state": "listening", "hold": hold})
        # The orchestrator follows its welcome with the space and its description (server.py serve_session).
        await self.send(srv.space_message())
        if a.drop_after_welcome_s and self.conn == 1:
            asyncio.get_running_loop().call_later(a.drop_after_welcome_s, self.drop)
        if a.hold:
            await self.send({"t": "hold", "phase": "held", "why": "mock render"})
        while True:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), 60)
            except TimeoutError:
                await self.close(1000, "nothing received for 60 s")
                return
            if isinstance(msg, bytes):
                await self.on_audio(msg)
            else:
                await self.on_text(msg)

    def drop(self):
        self.log("drop", conn=self.conn)
        self.ws.transport.abort()  # no close frame: the client sees 1006

    # Client audio

    async def on_audio(self, data: bytes):
        if self.mic == "ptt":
            if self.turn is None:
                self.log("audio-outside-turn", conn=self.conn, bytes=len(data))
                return
            self.turn.add(data)
            return
        # Open mic: energy VAD.
        speech = rms_dbfs(samples_of(data)) > SPEECH_DBFS
        if self.turn is None:
            self.vad_preroll = (self.vad_preroll + [data])[-10:]
            self.vad_speech = self.vad_speech + 1 if speech else 0
            if self.vad_speech >= VAD_START_FRAMES:
                await self.user_turn_started()
                self.turn = Turn(mic="vad")
                for d in self.vad_preroll:
                    self.turn.add(d)
                self.vad_silence = 0
            return
        self.turn.add(data)
        self.vad_silence = 0 if speech else self.vad_silence + 1
        if self.vad_silence * 20 >= self.args.vad_end_ms:
            await self.user_turn_ended()

    async def user_turn_started(self):
        q = self.question
        if q and not q["future"].done():
            # Speech while a question waits answers it, as the orchestrator takes "yes" or "no" (the mock hears tones,
            # not words, and takes it as a yes); the card is withdrawn.
            q["future"].set_result({"by": "voice"})
            self.answer_turn = True
            await self.send({"t": "confirm_cancel", "id": q["id"], "why": "answered"})
            await self.send({"t": "state", "v": "listening"})
            return
        # Pipecat broadcasts an interruption at every user turn start; only an active reply is told (Pipecat research notes).
        if self.active_reply and not self.client_interrupted:
            rid = self.active_reply
            self.cancel_reply()
            self.log("barge-in", conn=self.conn, reply_id=rid, sent_ms=self.reply_sent_ms.get(rid, 0))
            await self.send({"t": "interrupt", "reply_id": rid})
            await self.send({"t": "state", "v": "listening"})
        else:
            await self.send({"t": "state", "v": "listening"})

    async def user_turn_ended(self):
        turn, self.turn = self.turn, None
        if turn is None:
            return
        if self.answer_turn:
            self.answer_turn = False
            self.log("answer-turn", conn=self.conn, **{k: v for k, v in turn.summary().items() if k != "frame_dbfs"})
            return
        self.turn_ended_at, self.turn_tools = time.monotonic(), []
        summary = turn.summary()
        self.log("turn", conn=self.conn, **summary)
        scripted = self.server.next_scripted()
        if scripted:
            self.start_reply(scripted["reply"], transcript=scripted.get("transcript", ""), voice="say")
            return
        heard = f"[{summary['speech_seconds']:.2f} s at {summary['speech_hz']:.0f} Hz]"
        reply = f"Mock reply: you spoke for {summary['speech_seconds']:.2f} seconds at {summary['speech_hz']:.0f} hertz."
        self.start_reply(reply, transcript=heard)

    # Client control

    async def on_text(self, raw: str):
        try:
            msg = json.loads(raw)
        except ValueError:
            self.log("malformed", conn=self.conn, raw=raw[:200])
            return
        self.log("in", conn=self.conn, msg=msg)
        t = msg.get("t")
        if t == "start":
            await self.user_turn_started()
            self.turn = Turn(mic="ptt")
        elif t == "stop":
            await self.user_turn_ended()
        elif t == "interrupt":
            rid = msg.get("reply_id")
            self.client_interrupted = True
            self.log("client-interrupt", conn=self.conn, reply_id=rid, sent_ms=self.reply_sent_ms.get(rid, 0))
            if self.question and not self.question["future"].done():
                self.cancel = True  # the spoken question stops; the question itself still waits for its answer
            else:
                self.cancel_reply()
        elif t == "played_ms":
            rid = msg.get("reply_id")
            self.log("played", conn=self.conn, reply_id=rid, ms=msg.get("ms"),
                     sent_ms=self.reply_sent_ms.get(rid), full_ms=self.reply_full_ms.get(rid))
            if rid == self.active_reply:
                self.active_reply = None
                self.client_interrupted = False
        elif t == "text":
            self.turn_ended_at, self.turn_tools = time.monotonic(), []
            # Like the orchestrator: typed words become the user's message with no `transcript` back.
            scripted = self.server.next_scripted()
            if scripted:
                self.start_reply(scripted["reply"], transcript=None, voice="say")
            else:
                self.start_reply(f"you typed: {msg.get('text', '')}", transcript=None)
        elif t in ("space", "mode"):
            name = str(msg.get("name", ""))
            if self.args.no_switch:
                self.log("switch-ignored", conn=self.conn, kind=t, name=name)
            elif t == "space" and name not in SPACES:
                await self.send({"t": "error", "code": "unknown_space", "message": f"there is no space called {name}"})
            elif t == "mode" and name not in MODES:
                await self.send({"t": "error", "code": "unknown_mode", "message": f"there is no {name} mode"})
            else:
                if t == "space":
                    self.server.space = name
                else:
                    self.server.mode = name
                await self.send(self.server.space_message())
        elif t == "confirm_response":
            q = self.question
            if q and not q["future"].done() and msg.get("id") == q["id"]:
                q["future"].set_result({"by": "button", "msg": msg})
            else:
                self.log("approval-stray", conn=self.conn, msg=msg)  # no such question waits
        elif t == "ping":
            await self.send({"t": "pong", "n": msg.get("n", 0)})

    # Replies

    def cancel_reply(self):
        self.cancel = True
        if self.question and not self.question["future"].done():
            self.question["future"].cancel()  # overtaken: ask_approval withdraws the card

    def start_reply(self, text: str, transcript: str | None, voice: str | None = None):
        if self.reply_task and not self.reply_task.done():
            self.cancel_reply()
        self.cancel = False
        self.reply_task = asyncio.create_task(self.reply(text, transcript, voice or self.args.reply_voice))

    async def reply(self, text: str, transcript: str | None, voice: str):
        a = self.args
        try:
            if transcript is not None:
                await self.send({"t": "transcript", "final": True, "text": transcript})
            await self.send({"t": "state", "v": "thinking"})
            if self.server.hold == "held":
                await self.speak(f"n{self.server.replies + 1}", tone_pcm(300, 0.5), "The Mac is busy.")
                await self.send({"t": "state", "v": "held"})
                await asyncio.sleep(a.hold_seconds)
                self.server.hold = "open"
                await self.send({"t": "hold", "phase": "open", "why": ""})
                await self.send({"t": "state", "v": "thinking"})
            if a.tool:
                self.server.tool = {"name": "read", "label": "reading your journal", "since": time.time()}
                self.turn_tools.append("read")
                await self.send({"t": "tool", "phase": "start", "name": "read", "label": "reading your journal"})
            if a.ack:
                await self.speak(f"a{self.server.replies + 1}", tone_pcm(880, 0.3), "Let me check.")
            scenario = self.server.next_approval()
            if scenario:
                text = await self.ask_approval(scenario)
                if text is None:
                    return
            if a.tool:
                await asyncio.sleep(a.tool_seconds)
                self.server.tool = None
                await self.send({"t": "tool", "phase": "end", "name": "read", "ok": True})
            await asyncio.sleep(a.think_ms / 1000)
            if self.cancel:
                return
            pcm = say_pcm(text) if voice == "say" else tone_pcm(a.reply_hz, a.reply_seconds)
            await self.speak(f"r{self.server.replies + 1}", pcm, text)
        except ConnectionClosed:
            pass

    async def ask_approval(self, scenario: str) -> str | None:
        """Asks a scenario's question as the orchestrator does since Approvals: the card (`confirm_request`), the
        question aloud, then a button, speech, a new turn or silence answers it. Returns what to say next; None when
        a new turn overtook it."""
        a, sc = self.args, APPROVALS[scenario]
        legacy = scenario == "legacy"
        tool = sc.get("action", {}).get("tool", "kb")
        if tool in self.session_allowed:
            self.log("approval-skipped", conn=self.conn, scenario=scenario, tool=tool, why="allowed for this session")
            return await self.run_tool(sc)
        self.server.questions += 1
        cid = f"c{self.server.questions}"
        timeout_ms = a.approval_timeout_ms or {"timeout": 3000, "legacy": 20000}.get(scenario, 120000)
        request = {"t": "confirm_request", "id": cid, "title": sc["title"], "message": sc["message"],
                   "timeout_ms": timeout_ms}
        offered = [] if legacy else sc.get("choices", [ALLOW_ONCE, DENY])
        if not legacy:
            request |= {"summary": sc["summary"], "action": sc["action"], "choices": offered}
        future = asyncio.get_running_loop().create_future()
        self.question = {"id": cid, "future": future, "scenario": scenario}
        await self.send(request)
        if legacy:
            spoken = f"{sc['title']} It starts with {' '.join(sc['message'].split()[:2])}. Yes or no?"
        else:
            spoken = f"I'd like to {sc['summary'][0].lower()}{sc['summary'][1:].rstrip('.')}. Shall I?"
        await self.speak(f"q{self.server.questions}", tone_pcm(660, 0.4), spoken)
        self.cancel = False  # a client interrupt cut only the spoken question
        try:
            answer = await asyncio.wait_for(future, timeout_ms / 1000)
        except TimeoutError:
            self.question = None
            if not legacy:  # the orchestrator before Approvals withdrew nothing; its card closed at its own time
                await self.send({"t": "confirm_cancel", "id": cid, "why": "timeout"})
            self.log("approval-answer", conn=self.conn, id=cid, scenario=scenario, by="silence", outcome="denied")
            return sc["silence"]
        except asyncio.CancelledError:
            self.question = None
            if not legacy:
                await self.send({"t": "confirm_cancel", "id": cid, "why": "overtaken"})
            self.log("approval-answer", conn=self.conn, id=cid, scenario=scenario, by="overtaken", outcome="denied")
            return None
        self.question = None
        if answer["by"] == "voice":
            self.log("approval-answer", conn=self.conn, id=cid, scenario=scenario, by="voice", outcome="allowed")
            return await self.run_tool(sc)
        msg = answer["msg"]
        choice, confirmed = msg.get("choice"), msg.get("confirmed")
        ids = [c["id"] for c in offered]
        allowed = choice.startswith("allow") if (choice and ids) else confirmed is True
        self.log("approval-answer", conn=self.conn, id=cid, scenario=scenario, by="button", choice=choice,
                 confirmed=confirmed, matches=msg.get("id") == cid, offered=(choice in ids) if ids else None,
                 consistent=confirmed is allowed, outcome="allowed" if allowed else "denied")
        if choice == "allow_session":
            self.session_allowed.add(tool)
        return await self.run_tool(sc) if allowed else sc["not_done"]

    async def run_tool(self, sc: dict) -> str:
        tool = sc.get("action", {}).get("tool", "kb")
        self.turn_tools.append(tool)
        self.server.tool = {"name": tool, "label": sc["label"], "since": time.time()}
        await self.send({"t": "tool", "phase": "start", "name": tool, "label": sc["label"]})
        await asyncio.sleep(0.2)
        self.server.tool = None
        await self.send({"t": "tool", "phase": "end", "name": tool, "ok": True})
        return sc["done"]

    def note_first_audio(self):
        """The turn's first reply audio is going out: /v1/status's last_turn (the mock has no STT or TTS stages, so
        its LLM figure is the think time and the others are zero)."""
        srv = self.server
        srv.turns += 1
        srv.last_turn = {"eos_to_first_audio_ms": round((time.monotonic() - self.turn_ended_at) * 1000, 1),
                         "stt_ms": 0, "llm_ttft_ms": float(self.args.think_ms), "tts_first_audio_ms": 0,
                         "tools": list(self.turn_tools), "space": srv.space}
        self.turn_ended_at = None

    async def speak(self, rid: str, pcm: bytes, text: str):
        self.server.replies += 1
        self.active_reply, self.client_interrupted = rid, False
        self.reply_full_ms[rid] = len(pcm) // 2 * 1000 // OUT_RATE
        await self.send({"t": "audio_start", "reply_id": rid, "rate": OUT_RATE})
        await self.send({"t": "reply_text", "reply_id": rid, "delta": text})
        await self.send({"t": "state", "v": "speaking"})
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        sent = 0
        for i in range(0, len(pcm), OUT_CHUNK):
            if self.cancel:
                break
            if i == 0 and self.turn_ended_at is not None:
                self.note_first_audio()
            await self.ws.send(pcm[i : i + OUT_CHUNK])
            sent += min(OUT_CHUNK, len(pcm) - i)
            self.reply_sent_ms[rid] = sent // 2 * 1000 // OUT_RATE
            # Real-time pace, at most one chunk ahead, like Pipecat's output transport.
            delay = t0 + (i // OUT_CHUNK + 1) * 0.04 - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
        self.log("reply", conn=self.conn, reply_id=rid, sent_ms=self.reply_sent_ms.get(rid, 0),
                 full_ms=self.reply_full_ms[rid], cancelled=self.cancel)
        if self.cancel:
            return
        await self.send({"t": "audio_end", "reply_id": rid})
        await self.send({"t": "end_of_turn", "reply_id": rid})
        await self.send({"t": "state", "v": "listening"})


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=18770, help="0 picks a free port (printed as JSON)")
    p.add_argument("--log", help="JSONL log path (default: stdout)")
    p.add_argument("--reply-voice", choices=["tone", "say"], default="tone")
    p.add_argument("--reply-seconds", type=float, default=1.2)
    p.add_argument("--reply-hz", type=float, default=523.25)
    p.add_argument("--think-ms", type=int, default=200)
    p.add_argument("--tool", action="store_true", help="tool start/end around each reply")
    p.add_argument("--ack", action="store_true", help="a separate acknowledgement reply before each answer")
    p.add_argument("--approval", metavar="SCENARIO[,...]",
                   help="an approval question before each answer, one scenario per turn (the last repeats; none "
                        "asks nothing): " + ", ".join(APPROVALS))
    p.add_argument("--approval-timeout-ms", type=int, default=0,
                   help="the question's timeout_ms (default 3000 for timeout, 20000 for legacy, else 120000)")
    p.add_argument("--hold", action="store_true", help="start held: the first turn gets the busy notice")
    p.add_argument("--hold-seconds", type=float, default=1.5)
    p.add_argument("--hold-object", action="store_true", help="welcome.hold as an object, not a string")
    p.add_argument("--reject", action="store_true", help="close every connection with 4403")
    p.add_argument("--protocol-error-after-hello", action="store_true", help="close with 1002 after hello")
    p.add_argument("--slow-welcome-ms", type=int, default=0)
    p.add_argument("--drop-after-welcome-s", type=float, default=0.0, help="abort the first connection (1006)")
    p.add_argument("--vad-end-ms", type=int, default=VAD_END_MS, help="silence that ends an open-mic turn")
    p.add_argument("--tool-seconds", type=float, default=0.3, help="how long the --tool runs")
    p.add_argument("--no-switch", action="store_true", help="ignore space and mode messages (as the M1 orchestrator)")
    p.add_argument("--no-status", action="store_true", help="/v1/status answers 404")
    p.add_argument("--turns", help='JSON list of {"transcript", "reply"}, one per user turn (spoken with say)')
    args = p.parse_args()
    server = MockServer(args)
    async with serve(server.handler, "127.0.0.1", args.port, process_request=server.process_request,
                     ping_interval=None, max_size=2**20) as ws_server:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        print(json.dumps({"listening": port}), flush=True)
        server.log("listening", port=port, args=vars(args))
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
