"""Browser entry: SmallWebRTC with the project's own page, served by our own FastAPI app on 127.0.0.1.

Why not the dev runner (note 04c §3): its `/start` hands the prebuilt client Google's STUN server
(stun.l.google.com) and it has no TLS. Our `/start` returns no iceConfig, so the browser gathers host candidates
only, which is all a page on this Mac (or a phone on the tailnet) needs; the bot side already passes iceServers=[].
None of these routes is authenticated, so this app binds 127.0.0.1 only. Browsers grant the microphone on
http://localhost, so the Mac's own browser works today; a phone browser needs HTTPS (Tailscale HTTPS certificates,
off by default).

Echo cancellation comes from the browser (it applies AEC only inside a WebRTC call, PROTOCOL.md). UI events go as
RTVI server messages; the "working" loop while a tool runs is the browser path's alone (note 04c §5).

The page at / is plain WebRTC (static/index.html). Pipecat's prebuilt client can be mounted at /client
(server.browser.prebuilt), but it is off by default: when it loads it fetches Daily's call-machine bundle from
c.daily.co, which a fully local agent should not do (seen in the headless Chromium test, 2026-10-05).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from loguru import logger

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from .pipeline import IN_RATE, OUT_RATE, build_session

if TYPE_CHECKING:
    from .server import Orchestrator


def new_session_id() -> str:
    from .server import new_session_id as make

    return make()


def rtvi_message(message: dict) -> RTVIServerMessageFrame:
    return RTVIServerMessageFrame(data=message)


def create_browser_app(orch: "Orchestrator") -> FastAPI:
    handler = SmallWebRTCRequestHandler()   # no ice_servers: host candidates only

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await handler.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    sessions: dict[str, dict] = {}
    running: set[asyncio.Task] = set()

    async def run_bot(connection: SmallWebRTCConnection):
        transport = SmallWebRTCTransport(webrtc_connection=connection, params=TransportParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=IN_RATE, audio_out_sample_rate=OUT_RATE))
        sid = new_session_id()
        session = build_session(cfg=orch.cfg, runtime=orch.runtime, hub=orch.hub, transport=transport,
                                protocol_v1=False, mic="vad", device=f"browser-{sid}", client="browser",
                                session_id=sid, ui_event=rtvi_message,
                                turn=orch.turn_factory("vad") if orch.turn_factory else None,
                                latency_dir=orch.state_dir / "latency", turn_log=orch.turn_log, tone=orch.tone,
                                digests=orch.digests, new_session=True)
        session.started = time.monotonic()
        orch.browser_sessions[sid] = session
        runner = WorkerRunner(handle_sigint=False)

        @transport.event_handler("on_client_disconnected")
        async def _gone(_t, _c):
            await runner.cancel()

        try:
            await runner.add_workers(session.worker)
            await runner.run()
        finally:
            session.coordinator.close()
            orch.browser_sessions.pop(sid, None)

    async def on_connection(connection: SmallWebRTCConnection):
        task = asyncio.create_task(run_bot(connection))
        running.add(task)
        task.add_done_callback(running.discard)

    @app.post("/start")
    async def start(request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        session_id = str(uuid.uuid4())
        sessions[session_id] = body.get("body", {}) if isinstance(body, dict) else {}
        return {"sessionId": session_id}   # deliberately no iceConfig (no Google STUN)

    @app.post("/api/offer")
    async def offer(req: SmallWebRTCRequest, background_tasks: BackgroundTasks):
        return await handler.handle_web_request(request=req, webrtc_connection_callback=on_connection)

    @app.patch("/api/offer")
    async def candidates(req: SmallWebRTCPatchRequest):
        await handler.handle_patch_request(req)
        return {"status": "success"}

    @app.api_route("/sessions/{session_id}/api/offer", methods=["POST", "PATCH"])
    async def session_offer(session_id: str, request: Request, background_tasks: BackgroundTasks):
        if session_id not in sessions:
            return Response("unknown session", status_code=404)
        data = await request.json()
        if request.method == "POST":
            req = SmallWebRTCRequest(sdp=data["sdp"], type=data["type"], pc_id=data.get("pc_id"),
                                     restart_pc=data.get("restart_pc"),
                                     request_data=data.get("request_data") or data.get("requestData") or sessions[session_id])
            return await offer(req, background_tasks)
        patch = SmallWebRTCPatchRequest(pc_id=data["pc_id"],
                                        candidates=[IceCandidate(**c) for c in data.get("candidates", [])])
        return await candidates(patch)

    @app.get("/v1/status")
    async def status():
        return orch.status()

    page = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/", include_in_schema=False)
    async def root():
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    if orch.cfg.browser_prebuilt:
        try:
            from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI

            app.mount("/client", PipecatPrebuiltUI)
        except Exception as e:  # noqa: BLE001 - the prebuilt package is optional
            logger.warning(f"prebuilt client not mounted: {e}")

    return app
