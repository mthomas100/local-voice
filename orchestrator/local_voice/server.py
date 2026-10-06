"""The orchestrator process: protocol v1 on :8770 (loopback and the tailnet address only), /v1/status, and the
browser entry on 127.0.0.1.

Access (PROTOCOL.md): a non-loopback peer is accepted only if `tailscale whois` names one of
allowed_logins; the socket is accepted and then closed with 4403, because a close before accept reaches the client as
a bare HTTP 403 (uvicorn, note 04c §2). Loopback is allowed explicitly (`whois` fails for 127.0.0.1). The listening
sockets are bound to exactly 127.0.0.1 and this Mac's tailnet IPv4, never 0.0.0.0, never `tailscale funnel`.

Start: `python -m local_voice.server` (run.sh does it with the right environment). It refuses while the hold gate is
draining or held, loads and warms the speech models, renders the busy notice, then serves.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.websockets import WebSocketDisconnect, WebSocketState

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

from .agent import AgentHub
from .agent_dir import agent_dir_for
from .config import Config, ConfigError, load_config
from .hold import HoldMonitor
from .bargein import PausableWebsocketTransport
from .pipeline import IN_RATE, OUT_RATE, Session, TurnSetup, build_session
from .protocol_v1 import ProtocolV1Serializer, dumps
from .runtime import HoldNotOpen, SpeechRuntime
from .digests import DigestOutbox
from .spaces import SpacesError, load_spaces
from .tone_hook import make_tone_hook
from .turnlog import TurnLog

LOOPBACK = {"127.0.0.1", "::1"}
CLOSE_PROTOCOL = 1002
CLOSE_FORBIDDEN = 4403
CLOSE_TAKEN_OVER = 4409


async def tailnet_login(host: str, port: int, timeout: float = 2.0) -> str | None:
    """The Tailscale login that owns the peer at host:port (about 30 ms), or None if it is not a tailnet peer."""
    try:
        proc = await asyncio.create_subprocess_exec("tailscale", "whois", "--json", f"{host}:{port}",
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(out).get("UserProfile", {}).get("LoginName") or None
    except ValueError:
        return None


def self_login() -> str | None:
    """The login that owns this Mac's tailnet node ("self" in allowed_logins)."""
    try:
        out = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=5).stdout
        d = json.loads(out)
        return d.get("User", {}).get(str(d["Self"]["UserID"]), {}).get("LoginName")
    except Exception:  # noqa: BLE001
        return None


def tailnet_ipv4() -> str | None:
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
        return out[0] if out else None
    except Exception:  # noqa: BLE001
        return None


def new_session_id() -> str:
    """`s-<local date>-<time>-<4 hex>`: sortable, and unique across restarts (the brain's turn log keys on it)."""
    return f"s-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def warm_turn_models(cfg: Config) -> dict[str, float]:
    """Load Silero and Smart Turn v3.2 once at startup and run each on silence, so a broken model fails here rather
    than on the first connection, and their files are in the page cache. Both are CPU ONNX bundled with Pipecat
    (note 04c §7); every pipeline still builds its own pair, because their state is per audio stream."""
    import numpy as np

    from .pipeline import turn_setup

    t0 = time.monotonic()
    ts = turn_setup(cfg, "vad")
    out = {"vad_load_s": round(time.monotonic() - t0, 3)}
    t = time.monotonic()
    ts.vad.set_sample_rate(IN_RATE)
    ts.vad.voice_confidence(bytes(ts.vad.num_frames_required() * 2))
    out["vad_warm_s"] = round(time.monotonic() - t, 3)
    for strat in ts.stop:
        analyzer = getattr(strat, "_turn_analyzer", None)
        if analyzer is not None:
            t = time.monotonic()
            analyzer._predict_endpoint(np.zeros(IN_RATE * 2, dtype=np.float32))   # pinned 1.12.0: base_smart_turn.py
            out["smart_turn_warm_s"] = round(time.monotonic() - t, 3)
    return out


def bound_sockets(hosts: list[str], port: int) -> list[socket.socket]:
    socks = []
    for host in hosts:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        s = socket.socket(family, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(128)
        s.set_inheritable(True)
        socks.append(s)
    return socks


@dataclass
class Client:
    device: str
    client: str
    session: Session
    ws: WebSocket
    since: float = field(default_factory=time.monotonic)
    runner: WorkerRunner | None = None


class Orchestrator:
    def __init__(self, cfg: Config, *, use_mlx: bool = True, turn_factory: Callable[[str], TurnSetup] | None = None,
                 extra_env: dict[str, str] | None = None, base_url: str | None = None,
                 allowed_logins: set[str] | None = None):
        self.cfg = cfg
        self.state_dir = cfg.state_dir
        self.hold = HoldMonitor(cfg.gate, poll_s=cfg.hold_poll_s)
        self.runtime = SpeechRuntime(cfg, self.hold, use_mlx=use_mlx)
        self.turn_factory = turn_factory
        self.extra_env = extra_env
        self.base_url = base_url
        self._allowed = allowed_logins
        self.hub: AgentHub | None = None
        self.turn_log = TurnLog(cfg.turn_log_dir, heard_wait_s=cfg.heard_wait_s)
        self.digests = DigestOutbox(cfg.brain_state_dir, cfg.brain_outbox_module)
        self.clients: dict[str, Client] = {}            # device -> live client
        self.recent: dict[str, tuple[str, float]] = {}  # device -> (session id, disconnected at), the resume window
        self.browser_sessions: dict[str, Session] = {}
        self.started_at = time.time()
        self._servers: list[Any] = []
        self.tone = None

    @property
    def allowed_logins(self) -> set[str]:
        if self._allowed is None:
            out = set()
            for x in self.cfg.allowed_logins:
                if x == "self":
                    me = self_login()
                    if me:
                        out.add(me)
                else:
                    out.add(x)
            self._allowed = out
        return self._allowed

    async def start(self) -> dict[str, Any]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        spaces = load_spaces(self.cfg)
        await self.hold.start()
        timings = await self.runtime.start(state_dir=self.state_dir)
        if self.turn_factory is None:
            timings.update(await asyncio.to_thread(warm_turn_models, self.cfg))   # CPU ONNX, not MLX: any thread
        agent_dir = agent_dir_for(self.cfg, spaces, self.state_dir, base_url=self.base_url)
        self.hub = AgentHub(self.cfg, spaces, state_dir=self.state_dir, agent_dir=agent_dir, hold=self.hold,
                            extra_env=self.extra_env)
        self.tone = make_tone_hook(self.cfg)
        return timings

    async def stop(self) -> None:
        for srv in self._servers:
            srv.should_exit = True
        for c in list(self.clients.values()):
            with contextlib.suppress(Exception):
                await c.ws.close(code=1001)
        if self.hub:
            await self.hub.close()
        await self.hold.stop()
        self.runtime.shutdown()

    # ------------------------------------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        hub = self.hub
        hs = self.hold.state
        now = time.monotonic()
        base: dict[str, Any] = {"v": 1, "state": "held" if not hs.allows_gpu else (hub.state if hub else "idle"),
                                "hold": hs.as_json()}
        if hub is not None:
            st = hub.status()
            base.update(space=st["space"], mode=st["mode"], tier=st["tier"], model=st["model"], tool=st["tool"],
                        last_turn=st["last_turn"], turns=st["turns"], children=st["children"], journal=hub.journal,
                        spaces={n: s.dashboard() for n, s in hub.spaces.spaces.items()},
                        approval=st["approval"], grants=st["grants"])   # PROTOCOL.md "Approvals"
        base["clients"] = ([{"device": c.device, "client": c.client, "connected_s": round(now - c.since)}
                            for c in self.clients.values()]
                           + [{"device": f"browser-{sid}", "client": "browser", "connected_s": round(now - s.started)}
                              for sid, s in self.browser_sessions.items()])
        base["digest_pending"] = bool(self.digests.pending())
        base["speech"] = self.runtime.describe()
        base["uptime_s"] = round(time.time() - self.started_at)
        return base

    async def peer_allowed(self, host: str, port: int) -> tuple[bool, str]:
        if host in LOOPBACK:
            return True, "loopback"
        login = await tailnet_login(host, port)
        return (login is not None and login in self.allowed_logins), (login or "not a tailnet peer")

    # ------------------------------------------------------------------------------------- protocol v1

    def create_app(self) -> FastAPI:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)   # no schema pages on a tailnet port

        @app.get("/v1/status")
        async def v1_status(request: Request):
            ok, who = await self.peer_allowed(request.client.host, request.client.port)
            if not ok:
                return JSONResponse({"error": "forbidden", "who": who}, status_code=403)
            return JSONResponse(self.status())

        @app.websocket("/v1/voice")
        async def v1_voice(ws: WebSocket):
            await self.handle_voice(ws)

        return app

    async def handle_voice(self, ws: WebSocket) -> None:
        ok, who = await self.peer_allowed(ws.client.host, ws.client.port)
        await ws.accept()
        if not ok:
            logger.warning(f"refusing {ws.client.host}: {who}")
            await ws.close(code=CLOSE_FORBIDDEN)
            return
        try:
            hello = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=10))
        except (TimeoutError, ValueError, KeyError, WebSocketDisconnect, RuntimeError):
            with contextlib.suppress(Exception):
                await ws.close(code=CLOSE_PROTOCOL)
            return
        if not isinstance(hello, dict) or hello.get("t") != "hello" or hello.get("v") != 1:
            await ws.close(code=CLOSE_PROTOCOL)
            return
        device = str(hello.get("device") or f"anon-{uuid.uuid4().hex[:6]}")
        client = str(hello.get("client") or "unknown")
        mic = "ptt" if hello.get("mic") == "ptt" else "vad"
        old = self.clients.pop(device, None)
        if old is not None:   # the same device again: the new socket takes over (4409 to the old one)
            with contextlib.suppress(Exception):
                await old.ws.close(code=CLOSE_TAKEN_OVER)
            if old.runner:
                with contextlib.suppress(Exception):
                    await old.runner.cancel()
        prev = self.recent.get(device)
        resumed = bool(prev and time.monotonic() - prev[1] < self.cfg.resume_window_s)
        sid = prev[0] if resumed else new_session_id()
        if old is not None:
            sid, resumed = old.session.id, True
        await self.serve_session(ws, device=device, client=client, mic=mic, session_id=sid,
                                 space_hint=hello.get("space"), new_session=not resumed)

    async def serve_session(self, ws: WebSocket, *, device: str, client: str, mic: str, session_id: str,
                            space_hint: Any = None, new_session: bool = True) -> None:
        hub = self.hub
        assert hub is not None
        last_rx = {"t": time.monotonic()}

        def touched(_msg: dict | None = None) -> None:
            last_rx["t"] = time.monotonic()

        serializer = ProtocolV1Serializer(on_client_message=touched)
        transport = PausableWebsocketTransport(websocket=ws, params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=IN_RATE,
            audio_out_sample_rate=OUT_RATE, audio_out_10ms_chunks=4, audio_out_end_silence_secs=0,
            add_wav_header=False, serializer=serializer, session_timeout=None))
        orig_deserialize = serializer.deserialize

        async def deserialize(data):
            touched()
            return await orig_deserialize(data)

        serializer.deserialize = deserialize   # every message, binary or text, counts as keepalive
        session = build_session(cfg=self.cfg, runtime=self.runtime, hub=hub, transport=transport, protocol_v1=True,
                                mic=mic, device=device, client=client, session_id=session_id,
                                turn=self.turn_factory(mic) if self.turn_factory else None,
                                latency_dir=self.state_dir / "latency", turn_log=self.turn_log,
                                digests=self.digests, new_session=new_session, tone=self.tone)
        session.started = time.monotonic()
        serializer.timeline = session.timeline
        if isinstance(space_hint, str) and space_hint in hub.spaces and space_hint != hub.active:
            try:   # PROTOCOL.md hello.space: start where the client asks, if that space can start
                await hub.enter(space_hint)
            except Exception as e:  # noqa: BLE001 - the session still starts, in the space it was in
                logger.warning(f"{device}: hello asked for space {space_hint}: {e}")
        hs = await self.hold.check()
        sp = hub.space
        await ws.send_text(dumps({"t": "welcome", "v": 1, "session": session_id, "space": hub.active, "mode": hub.mode,
                                  "tier": sp.tier, "state": "listening" if hs.allows_gpu else "held",
                                  "hold": hs.phase if hs.phase != "absent" else "open"}))
        await ws.send_text(dumps({"t": "space", "name": hub.active, "mode": hub.mode, "tier": sp.tier,
                                  "description": sp.description}))
        runner = WorkerRunner(handle_sigint=False)
        entry = Client(device=device, client=client, session=session, ws=ws, runner=runner)
        self.clients[device] = entry

        @transport.event_handler("on_client_disconnected")
        async def _gone(_t, _ws):
            await runner.cancel()

        async def watchdog():
            while True:
                await asyncio.sleep(1.0)
                if time.monotonic() - last_rx["t"] > self.cfg.keepalive_s:
                    logger.info(f"{device}: nothing for {self.cfg.keepalive_s:.0f} s, closing")
                    with contextlib.suppress(Exception):
                        await ws.close(code=1000)
                    await runner.cancel()
                    return

        dog = asyncio.create_task(watchdog(), name=f"keepalive-{device}")
        logger.info(f"protocol v1: {device} ({client}, mic {mic}) connected as {session_id}")
        try:
            await runner.add_workers(session.worker)
            await runner.run()
        finally:
            dog.cancel()
            session.coordinator.close()
            if self.clients.get(device) is entry:
                del self.clients[device]
                self.recent[device] = (session_id, time.monotonic())
            logger.info(f"protocol v1: {device} disconnected")
            if ws.application_state != WebSocketState.DISCONNECTED:
                with contextlib.suppress(Exception):
                    await ws.close(code=1000)

    # ----------------------------------------------------------------------------------------- serving

    def hosts(self) -> list[str]:
        out = []
        for h in self.cfg.hosts:
            if h == "tailnet":
                ip = tailnet_ipv4()
                if ip:
                    out.append(ip)
                else:
                    logger.warning("tailnet address unavailable; serving loopback only")
            else:
                out.append(h)
        return out

    async def serve(self) -> None:
        import uvicorn

        servers = self._servers
        app = self.create_app()
        hosts = self.hosts()
        servers.append(uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off", ws_ping_interval=None)))
        sockets = [bound_sockets(hosts, self.cfg.port)]
        if self.cfg.browser_enabled:
            from .browser import create_browser_app

            bapp = create_browser_app(self)
            servers.append(uvicorn.Server(uvicorn.Config(bapp, log_level="warning", lifespan="on")))
            sockets.append(bound_sockets(["127.0.0.1"], self.cfg.browser_port))
        logger.info(f"serving protocol v1 on {', '.join(f'{h}:{self.cfg.port}' for h in hosts)}"
                    + (f"; browser on http://127.0.0.1:{self.cfg.browser_port}/" if self.cfg.browser_enabled else ""))
        await asyncio.gather(*(s.serve(sockets=socks) for s, socks in zip(servers, sockets)))


def setup_logging(state_dir: Path, level: str = "INFO") -> Path:
    logger.remove()
    logger.add(sys.stderr, level=level)
    logs = state_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"orchestrator-{time.strftime('%Y-%m-%d')}.log"
    logger.add(path, level="DEBUG", rotation="50 MB", retention=10)
    return path


async def amain(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.port is not None:
        overrides.setdefault("server", {})["port"] = args.port
    if args.browser_port is not None:
        overrides.setdefault("server", {}).setdefault("browser", {})["port"] = args.browser_port
    if args.record:
        overrides.setdefault("record", {})["enabled"] = True
    try:
        if args.scratch:
            from .scratch import make_scratch

            args.config = make_scratch(args.scratch, config=args.config)
            print(f"scratch config: {args.config}", file=sys.stderr)
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as e:
        print(f"config: {e}", file=sys.stderr)
        return 2
    log = setup_logging(cfg.state_dir, args.log_level)
    logger.info(f"config {cfg.path}; state {cfg.state_dir}; log {log}")
    orch = Orchestrator(cfg)
    try:
        timings = await orch.start()
    except HoldNotOpen as e:
        logger.error(f"not starting: {e}")
        await orch.hold.stop()
        return 75
    except (ConfigError, SpacesError) as e:
        logger.error(f"not starting: {e}")
        await orch.hold.stop()
        return 2
    logger.info(f"speech ready: {json.dumps(timings)}")
    if args.check:
        await orch.stop()
        return 0
    try:
        await orch.serve()
    finally:
        await orch.stop()
        logger.info("stopped: Pi children closed, speech models released")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="local-voice orchestrator")
    ap.add_argument("--config", default=None, help="config.yaml (default: the one beside this package)")
    ap.add_argument("--check", action="store_true", help="load, warm and render, then exit (no serving)")
    ap.add_argument("--log-level", default=os.environ.get("LV_LOG_LEVEL", "INFO"))
    ap.add_argument("--scratch", metavar="DIR", default=None,
                    help="run on a scratch copy: state, turn log, brain state, KB_HOME (a kb clone) and the spaces' "
                         "roots (clones) under DIR, loopback only (local_voice/scratch.py)")
    ap.add_argument("--port", type=int, default=None, help="protocol v1 port (default: the config's, 8770)")
    ap.add_argument("--browser-port", type=int, default=None, help="browser page port (default: the config's, 7860)")
    ap.add_argument("--record", action="store_true",
                    help="record every connection (record: in config.yaml; local_voice/recorder.py): mic, playback and "
                         "events under record.dir, read by tools/session_report.py")
    try:
        sys.exit(asyncio.run(amain(ap.parse_args())))
    except KeyboardInterrupt:
        # Ctrl-C: uvicorn shuts down gracefully and re-raises the signal, and amain's finally has already stopped
        # everything (it logs "stopped"); only the traceback is left to suppress.
        sys.exit(0)


if __name__ == "__main__":
    main()
