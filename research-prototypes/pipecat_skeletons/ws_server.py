"""Protocol v1 server: FastAPI + FastAPIWebsocketTransport on :8770, loopback and tailnet address only.

Construct-only skeleton (research note 04c); create_app() builds the app, serve() is never called by the
checks.

Why FastAPIWebsocketTransport and not WebsocketServerTransport (pipecat-ai 1.12.0):
- The server transport (now SingleClientWebsocketServerTransport) owns its own websockets.serve() with no
  hook before the handshake (transports/websocket/server.py:234-241), serves one client and closes any second
  one with 1013 (server.py:243-256), and is one pipeline per process. The FastAPI transport wraps a WebSocket
  the app has already accepted (websocket/fastapi.py:629-672), so the endpoint can check the peer, run the
  hello/welcome handshake and choose the pipeline (open mic or push-to-talk) before Pipecat sees anything,
  and /v1/status lives on the same port.
- A socket closed before accept() gets an HTTP 403 handshake response (uvicorn 0.54.0,
  protocols/websockets/websockets_impl.py:296-304); PROTOCOL.md wants close code 4403, so accept, then close(4403).
- allowed_origins (default from PIPECAT_ALLOWED_ORIGINS, fastapi.py:68-87, 648-651) rejects a missing Origin
  header when set; native clients send none, so leave it empty here.
- The output transport paces audio at real time: it sleeps one chunk duration per write
  (fastapi.py:456, 598-607), so the client's jitter buffer, not the server, absorbs network jitter.
- audio_out_end_silence_secs defaults to 2: two seconds of silence after an EndFrame (base_transport.py:75,
  base_output.py:959-962). Set it to 0 for this protocol.
- WorkerRunner(auto_end=...) ends when its workers finish; a long-lived host that keeps one runner would pass
  auto_end=False (workers/runner.py:238-262). One runner per connection is simpler and matches the dev
  runner's bot() pattern.

`tailscale whois --json <ip:port>` returns {"UserProfile": {"LoginName": ...}, "Node": {...}} and takes about
30 ms here; for 127.0.0.1 it exits 1 with "peer not found" (raw/04c-tailscale-cli-facts.md), so loopback is
allowed explicitly.
"""

from __future__ import annotations

import asyncio
import json
import socket
import uuid
from collections.abc import Callable, Iterable

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.websockets import WebSocketDisconnect

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from pipeline_factory import IN_RATE, OUT_RATE, build_pipeline, build_worker, user_params
from protocol_v1 import ProtocolV1Serializer

PORT = 8770
LOOPBACK = {"127.0.0.1", "::1"}


async def tailnet_login(host: str, port: int, timeout: float = 2.0) -> str | None:
    """The Tailscale login that owns the peer at host:port, or None if it is not a tailnet peer."""
    proc = await asyncio.create_subprocess_exec(
        "tailscale", "whois", "--json", f"{host}:{port}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
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


async def peer_allowed(host: str, port: int, allowed_logins: set[str]) -> tuple[bool, str]:
    """Loopback, or a tailnet peer whose login is in allowed_logins."""
    if host in LOOPBACK:
        return True, "loopback"
    login = await tailnet_login(host, port)
    return (login is not None and login in allowed_logins), (login or "not a tailnet peer")


def create_app(
    *,
    allowed_logins: set[str],
    make_services: Callable[[], tuple],
    status: Callable[[], dict],
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        allowed_logins: Tailscale logins allowed to connect (the owner's devices).
        make_services: Returns fresh (stt, llm, tts) processors for one connection. Processors belong to one
            pipeline; the MLX models behind them are process-wide and shared.
        status: Returns the /v1/status JSON.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)  # no schema pages on a tailnet port

    @app.get("/v1/status")
    async def v1_status(request: Request):
        ok, who = await peer_allowed(request.client.host, request.client.port, allowed_logins)
        if not ok:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return JSONResponse(status())

    @app.websocket("/v1/voice")
    async def v1_voice(ws: WebSocket):
        ok, who = await peer_allowed(ws.client.host, ws.client.port, allowed_logins)
        await ws.accept()
        if not ok:
            logger.warning(f"refusing {ws.client.host}: {who}")
            await ws.close(code=4403)
            return
        try:
            hello = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=10))
        except (TimeoutError, ValueError, WebSocketDisconnect):
            await ws.close(code=1002)  # PROTOCOL.md defines only 4403; 1002 is the standard protocol error
            return
        if hello.get("t") != "hello" or hello.get("v") != 1:
            await ws.close(code=1002)
            return
        session = f"s{uuid.uuid4().hex[:8]}"
        mic = "ptt" if hello.get("mic") == "ptt" else "vad"
        await ws.send_text(json.dumps({"t": "welcome", "v": 1, "session": session, "space": "home",
                                       "mode": "conversation", "tier": "ask", "state": "listening",
                                       "hold": "open"}))

        transport = FastAPIWebsocketTransport(
            websocket=ws,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=IN_RATE,
                audio_out_sample_rate=OUT_RATE,
                audio_out_10ms_chunks=4,  # 40 ms = 1,920-byte binary messages at 24 kHz PCM16
                audio_out_end_silence_secs=0,
                add_wav_header=False,
                serializer=ProtocolV1Serializer(),
                session_timeout=None,
            ),
        )
        stt, llm, tts = make_services()
        pipeline, _ = build_pipeline(transport=transport, stt=stt, llm=llm, tts=tts,
                                     params=user_params(mic=mic), protocol_v1=True)
        worker = build_worker(pipeline, rtvi=False, name=session)
        runner = WorkerRunner(handle_sigint=False)

        @transport.event_handler("on_client_disconnected")
        async def _on_disconnected(_transport, _ws):
            await runner.cancel()

        await runner.add_workers(worker)
        await runner.run()

    return app


def bound_sockets(hosts: Iterable[str], port: int = PORT) -> list[socket.socket]:
    """Listening sockets on exactly these addresses (127.0.0.1 and the Mac's tailnet IPv4), nothing else."""
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


async def serve(app: FastAPI, hosts: Iterable[str], port: int = PORT) -> None:
    """Run uvicorn on pre-bound sockets (uvicorn.Server.serve(sockets=...), uvicorn/server.py:88)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, log_level="info", lifespan="on"))
    await server.serve(sockets=bound_sockets(hosts, port))


def tailnet_ipv4() -> str:
    """This Mac's tailnet IPv4 (tailscale ip -4)."""
    import subprocess

    return subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, check=True).stdout.split()[0]

