#!/usr/bin/env python3
"""stub_llm.py: a scriptable stand-in for llama-swap / llama-server (05c, 2026-10-05).

Pi's `local` provider talks OpenAI chat completions to it exactly as it talks to the rig, and gets back
the SSE chunk shapes llama.cpp emits with --jinja: a first chunk {"role":"assistant","content":null};
then reasoning_content and content deltas; tool calls as one delta carrying index, id, type and the
function name (arguments ""), followed by argument fragments carrying only index and arguments; a
finish chunk; a separate usage chunk with "choices": [] when stream_options.include_usage is set; then
[DONE]. Nothing here contacts anything else, and no model is involved.

Each /v1/chat/completions request pops the next script from a FIFO queue (POST /_script), or uses the
default script. Every request is logged verbatim (all headers, full body) to requests.jsonl, and a
timeline to stub.log. A client that disconnects mid-stream (Pi's abort) is logged as DISCONNECT.

Script fields (all optional):
  text             reply text, streamed word by word (chunk_words words per delta)
  reasoning        reasoning_content streamed before the text (what a thinking model leaks)
  tool_calls       [{"name": "read", "arguments": {...} or "raw json", "id": "call_x", "split": 3}]
  chunk_words      words per content delta (default 1)
  delay_ms         sleep between deltas (default 0)
  pre_delay_ms     sleep before the first chunk: a stand-in for prompt processing (default 0)
  status           HTTP status other than 200: answer with an OpenAI-style error body instead of a stream
  error_message    the message for that error body
  finish           finish_reason when there are no tool calls (default "stop")

Run:   python3 stub_llm.py --port 18391 --log-dir DIR
Embed: stub = StubLLM(port, log_dir).start(); stub.script([...]); stub.requests(); stub.stop()
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODELS = ("qwen38", "qwen27-262k")


class _State:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.lock = threading.Lock()
        self.queue: list[dict] = []
        self.default: dict = {"text": "PONG (unscripted default reply)"}
        self.n = 0
        self.loaded: list[str] = []

    def log(self, line: str) -> None:
        with self.lock, open(self.log_dir / "stub.log", "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f} {line}\n")

    def log_request(self, rec: dict) -> None:
        with self.lock, open(self.log_dir / "requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _pieces(text: str, words: int) -> list[str]:
    if not text:
        return []
    w = text.split(" ")
    return [(" " if i else "") + " ".join(w[i:i + words]) for i in range(0, len(w), words)]


def _split_n(s: str, n: int) -> list[str]:
    if not s:
        return [""]
    n = max(1, min(n, len(s)))
    k = -(-len(s) // n)
    return [s[i:i + k] for i in range(0, len(s), k)]


def _handler(state: _State):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence http.server's own stderr log
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            state.log(f"GET {self.path}")
            p = self.path
            if p.startswith("/v1/models"):
                return self._json({"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
            if p.startswith("/running"):
                return self._json({"running": [{"model": m, "state": "ready"} for m in state.loaded]})
            if p.startswith("/unload"):
                state.loaded = []
                state.log("UNLOADED")
                return self._json({"ok": True})
            if p.startswith("/health"):
                return self._json({"status": "ok"})
            if p.startswith("/_script"):
                with state.lock:
                    return self._json({"queue": state.queue, "n": state.n})
            self._json({"error": "no such path"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {"_unparsed": raw.decode("utf-8", "replace")}
            if self.path.startswith("/_script"):
                with state.lock:
                    if body.get("replace", True):
                        state.queue = []
                    state.queue.extend(body.get("queue", []))
                    if "default" in body:
                        state.default = body["default"]
                    return self._json({"ok": True, "queued": len(state.queue)})
            if self.path.startswith("/_reset"):
                with state.lock:
                    state.queue, state.n = [], 0
                return self._json({"ok": True})
            if not self.path.startswith("/v1/chat/completions"):
                state.log(f"POST {self.path} (404)")
                return self._json({"error": "no such path"}, 404)
            with state.lock:
                state.n += 1
                req = state.n
                script = state.queue.pop(0) if state.queue else dict(state.default)
            model = body.get("model", "qwen38")
            if model not in state.loaded:
                state.loaded = [model]  # one model at a time, like llama-swap
            msgs = body.get("messages") or []
            last = msgs[-1] if msgs else {}
            state.log_request({"ts": round(time.time(), 3), "n": req, "path": self.path,
                               "headers": dict(self.headers.items()), "body": body, "script": script})
            state.log(f"REQ {req} model={model} messages={len(msgs)} last_role={last.get('role')} "
                      f"tools={len(body.get('tools') or [])} script={json.dumps(script, ensure_ascii=False)[:160]}")
            if int(script.get("status", 200)) != 200:
                code = int(script["status"])
                state.log(f"ERROR {req} status={code}")
                return self._json({"error": {"code": code, "type": "server_error",
                                             "message": script.get("error_message", "stub error")}}, code)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            sent = 0

            def chunk(obj):
                nonlocal sent
                data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)) + "\n\n").encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                sent += 1

            base = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model, "system_fingerprint": "stub-05c"}
            delay = float(script.get("delay_ms", 0)) / 1000.0
            words = int(script.get("chunk_words", 1))
            try:
                if script.get("pre_delay_ms"):
                    time.sleep(float(script["pre_delay_ms"]) / 1000.0)
                chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]})
                for piece in _pieces(script.get("reasoning", ""), words):
                    chunk({**base, "choices": [{"index": 0, "delta": {"reasoning_content": piece}, "finish_reason": None}]})
                    if delay:
                        time.sleep(delay)
                for piece in _pieces(script.get("text", ""), words):
                    chunk({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                    if delay:
                        time.sleep(delay)
                calls = script.get("tool_calls") or []
                for i, call in enumerate(calls):
                    args = call.get("arguments", {})
                    args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                    cid = call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    chunk({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": i, "id": cid, "type": "function", "function": {"name": call["name"], "arguments": ""}}]}, "finish_reason": None}]})
                    for part in _split_n(args, int(call.get("split", 3))):
                        if delay:
                            time.sleep(delay)
                        chunk({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": i, "function": {"arguments": part}}]}, "finish_reason": None}]})
                finish = "tool_calls" if calls else script.get("finish", "stop")
                chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    usage = {"prompt_tokens": 100, "completion_tokens": max(1, sent), "total_tokens": 100 + max(1, sent)}
                    chunk({**base, "choices": [], "usage": usage,
                           "timings": {"prompt_n": 100, "prompt_ms": 10.0, "predicted_n": sent, "predicted_ms": 10.0}})
                chunk("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                state.log(f"DONE {req} chunks={sent} finish={finish}")
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                state.log(f"DISCONNECT {req} after {sent} chunks ({type(e).__name__})")

    return H


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, state: _State):
        super().__init__(addr, handler)
        self._state = state

    def handle_error(self, request, client_address):  # a client reset (Pi's abort) is not a stub failure
        import sys
        self._state.log(f"CONNECTION {client_address[1]} closed by the client ({sys.exc_info()[0].__name__})")


class StubLLM:
    """The stub in a background thread, for tests: start(), script(), requests(), stop()."""

    def __init__(self, port: int, log_dir: str | Path):
        self.port = port
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state = _State(self.log_dir)
        self.server: _Server | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> "StubLLM":
        self.server = _Server(("127.0.0.1", self.port), _handler(self.state), self.state)
        self.port = self.server.server_address[1]  # port 0 picks a free one
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.state.log(f"stub listening on 127.0.0.1:{self.port}")
        return self

    def script(self, queue: list[dict], default: dict | None = None, replace: bool = True) -> None:
        with self.state.lock:
            if replace:
                self.state.queue = []
            self.state.queue.extend(queue)
            if default is not None:
                self.state.default = default

    def requests(self) -> list[dict]:
        p = self.log_dir / "requests.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def log_lines(self) -> list[str]:
        p = self.log_dir / "stub.log"
        return p.read_text(encoding="utf-8").splitlines() if p.exists() else []

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18391)
    ap.add_argument("--log-dir", required=True)
    a = ap.parse_args()
    stub = StubLLM(a.port, a.log_dir)
    stub.server = _Server(("127.0.0.1", a.port), _handler(stub.state), stub.state)
    stub.state.log(f"stub listening on 127.0.0.1:{a.port}")
    stub.server.serve_forever()


if __name__ == "__main__":
    main()
