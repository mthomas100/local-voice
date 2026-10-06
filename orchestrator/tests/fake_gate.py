"""A stand-in for the hold gate's /hold API (~/repos/local-rig/hold/gate.go), for unit tests: a settable phase,
/hold/status in the gate's JSON shape, and /hold/wait-open's long poll. The DoD test against a real test gate uses
the `hold` binary itself (test_hold_gate_e2e.py)."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class FakeGate:
    def __init__(self):
        self.phase = "open"
        self.holds: list[dict] = []
        self.requests: list[str] = []
        self._changed = threading.Condition()
        self.server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def set(self, phase: str, reason: str = "render test") -> None:
        with self._changed:
            self.phase = phase
            self.holds = [] if phase == "open" else [{"id": "h1", "kind": "render", "reason": reason,
                                                       "state": "granted" if phase == "held" else "draining"}]
            self._changed.notify_all()

    def start(self) -> "FakeGate":
        gate = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                u = urlparse(self.path)
                gate.requests.append(u.path)
                if u.path == "/hold/status":
                    return self._json({"phase": gate.phase, "holds": gate.holds, "detected": [], "in_flight": [],
                                       "running": [], "running_ok": True, "unloading": False, "waiting": {}})
                if u.path == "/hold/wait-open":
                    timeout = float(parse_qs(u.query).get("timeout", ["30"])[0])
                    deadline = time.monotonic() + timeout
                    with gate._changed:
                        while gate.phase != "open" and time.monotonic() < deadline:
                            gate._changed.wait(max(0.01, deadline - time.monotonic()))
                    return self._json({"phase": gate.phase, "why": ""})
                self._json({"error": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.server:
            with self._changed:
                self.phase = "open"
                self._changed.notify_all()
            self.server.shutdown()
            self.server.server_close()
