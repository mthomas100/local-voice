"""Browser entry: SmallWebRTC + the prebuilt client, served by our own FastAPI app instead of the dev runner.

Construct-only skeleton (research note 04c). What the dev runner does (pipecat-ai 1.12.0, runner/run.py):
- `uv run bot.py -t webrtc` serves on --host (default localhost) and --port (default 7860)
  (run.py:187-188, 1714-1715) with plain uvicorn.run(app, host, port): there is no TLS option (run.py:2001).
- Routes: GET / redirects to /client/ (the prebuilt UI, run.py:942-955); POST /start registers a session and
  returns {"sessionId", "iceConfig"?} (run.py:682-760); POST /api/offer and PATCH /api/offer do SDP and
  trickle ICE (run.py:1000-1036); /sessions/{session_id}/api/offer proxies the same, Pipecat-Cloud style
  (run.py:1038-1076). None of it is authenticated.
- The prebuilt UI (pipecat-ai-prebuilt 1.3.0) posts /start with {"createDailyRoom": false,
  "enableDefaultIceServers": true, "transport": "webrtc"} and waitForICEGathering: true
  (raw/04c-pipecat-prebuilt-ui-start-params.md). With no --ice-servers the runner then hands the browser
  stun:stun.l.google.com:19302 (run.py:752-759), so the browser contacts Google's STUN server, which is not
  local. The /start below never returns iceConfig.
- The bot side gathers host candidates only: SmallWebRTCConnection passes iceServers=[] (connection.py:
  260-261, 313-316), so aiortc never falls back to its Google default (aiortc/rtcicetransport.py:185-226).
  aioice offers every non-loopback IPv4 as a host candidate (aioice/ice.py:81-92, 531-548). On this Mac that
  is the tailnet interface (utun) and the LAN interface (en0): checked 2026-10-05. A phone on the tailnet
  pairs on the 100.x addresses; nothing needs STUN or TURN. restrict_ice_to_tailnet() drops the LAN
  candidate (no constructor option exists; aioice reads the module-level function at gather time).

HTTPS: browsers grant the microphone only to secure contexts; http://localhost counts, http://100.x.y.z does
not. So the Mac's own browser works on http://localhost:7860/client today. The phone's browser needs HTTPS:
either `tailscale serve --bg --https=443 http://127.0.0.1:7860` or uvicorn with a `tailscale cert` key pair
(uvicorn.Config(ssl_certfile=..., ssl_keyfile=...)); both need HTTPS certificates enabled for the tailnet,
which are off by default in a tailnet's admin console. Behind `tailscale serve` the app sees
127.0.0.1 as the peer, so tailnet identity has to come from serve's identity headers (Tailscale
documentation, not verified here) rather than `tailscale whois`. Until then this app binds 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import ipaddress
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import RedirectResponse, Response
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

from pipeline_factory import IN_RATE, OUT_RATE, build_pipeline, build_worker, user_params

TAILNET = ipaddress.ip_network("100.64.0.0/10")


def restrict_ice_to_tailnet() -> None:
    """Offer only tailnet host candidates (drops en0 and any other LAN or bridge address). Not yet tested."""
    import aioice.ice as ice

    original = ice.get_host_addresses

    def tailnet_only(use_ipv4: bool, use_ipv6: bool) -> list[str]:
        return [a for a in original(use_ipv4, use_ipv6) if ipaddress.ip_address(a) in TAILNET]

    ice.get_host_addresses = tailnet_only


def rtvi_message(message: dict) -> RTVIServerMessageFrame:
    """UI events for the browser: RTVI server-message, which the RTVI observer forwards to the client."""
    return RTVIServerMessageFrame(data=message)


def create_browser_app(*, make_services: Callable[[], tuple], mixer_factory: Callable[[], object] | None = None) -> FastAPI:
    """The dev runner's WebRTC routes, minus the public STUN server, plus our pipeline."""
    handler = SmallWebRTCRequestHandler()  # no ice_servers: host candidates only

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await handler.close()  # what the dev runner's smallwebrtc_lifespan does (run.py:1078-1085)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    sessions: dict[str, dict] = {}
    running: set[asyncio.Task] = set()

    async def run_bot(connection: SmallWebRTCConnection):
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=IN_RATE,
                audio_out_sample_rate=OUT_RATE,
                audio_out_mixer=mixer_factory() if mixer_factory else None,
            ),
        )
        stt, llm, tts = make_services()
        pipeline, _ = build_pipeline(transport=transport, stt=stt, llm=llm, tts=tts,
                                     params=user_params(mic="vad"), protocol_v1=False)
        worker = build_worker(pipeline, rtvi=True, name=f"browser-{connection.pc_id}")
        runner = WorkerRunner(handle_sigint=False)

        @transport.event_handler("on_client_disconnected")
        async def _on_disconnected(_transport, _client):
            await runner.cancel()

        await runner.add_workers(worker)
        await runner.run()

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
        return {"sessionId": session_id}  # deliberately no iceConfig (see the module docstring)

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

    try:
        from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI

        app.mount("/client", PipecatPrebuiltUI)

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(url="/client/")
    except Exception as e:  # the prebuilt package is optional
        logger.warning(f"prebuilt client not mounted: {e}")

    return app
