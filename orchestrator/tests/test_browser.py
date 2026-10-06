"""The browser entry, model-free: our own WebRTC routes (no Google STUN handed out) and a whole turn through
the project's own page in a separate headless Chromium (Playwright, temporary profile, a fake microphone fed from
a WAV file), never the user's own Chrome. Speech models are the fakes, the LLM is the stub."""
from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest

from harness import free_port, rig

pytestmark = pytest.mark.needs_pi
CHROMIUM = Path.home() / "Library/Caches/ms-playwright/chromium-1243"


def mic_wav(path: Path) -> Path:
    """1.2 s of a loud tone, then 3 s of silence, at 48 kHz: one utterance per loop of Chromium's fake microphone."""
    rate = 48000
    t = np.arange(int(1.2 * rate)) / rate
    tone = 0.4 * np.sin(2 * np.pi * 300 * t)
    a = np.concatenate([tone, np.zeros(3 * rate)])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((a * 32767).astype("<i2").tobytes())
    return path


async def test_start_route_hands_out_no_ice_servers(tmp_path):
    bport = free_port()
    async with rig(tmp_path, overrides={"server": {"browser": {"enabled": True, "port": bport}}}) as r:
        async with httpx.AsyncClient() as h:
            res = await h.post(f"http://127.0.0.1:{bport}/start", json={"createDailyRoom": False,
                                                                        "enableDefaultIceServers": True,
                                                                        "transport": "webrtc"})
            body = res.json()
            assert list(body) == ["sessionId"]                      # no iceConfig: no stun.l.google.com
            page = await h.get(f"http://127.0.0.1:{bport}/")
            assert page.status_code == 200 and "iceServers: []" in page.text and "http" not in page.text.split("<script>")[1]
            assert (await h.get(f"http://127.0.0.1:{bport}/client/")).status_code == 404   # the prebuilt client is off


@pytest.mark.skipif(not CHROMIUM.exists(), reason="needs the cached Playwright Chromium 1243")
async def test_a_turn_through_the_page_in_headless_chromium(tmp_path):
    from playwright.async_api import async_playwright

    bport = free_port()
    wav = mic_wav(tmp_path / "mic.wav")
    async with rig(tmp_path, stt_script=["hello from the browser"], stt_text="again",
                   overrides={"server": {"browser": {"enabled": True, "port": bport}}}) as r:
        r.stub.script([{"text": "Hello browser, I hear you."}], default={"text": "Still here."})
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"])
            ctx = await browser.new_context(permissions=["microphone"])
            page = await ctx.new_page()
            requests: list[str] = []
            page.on("request", lambda req: requests.append(req.url))
            await page.goto(f"http://127.0.0.1:{bport}/")
            await page.locator("#connect").click()
            await page.wait_for_function("document.getElementById('connect').textContent === 'Disconnect'", timeout=15000)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and "Hello browser, I hear you." not in r.tts.spoken:
                await asyncio.sleep(0.2)
            sessions = list(r.orch.browser_sessions.values())
            await page.wait_for_function("document.querySelectorAll('.msg.user').length > 0", timeout=10000)
            shown = await page.locator(".msg.user").first.text_content()
            await browser.close()
        assert r.user_texts()[0] == "hello from the browser"
        assert "Hello browser, I hear you." in r.tts.spoken
        assert shown == "hello from the browser"                     # the RTVI transcript reached the page
        assert sessions and not sessions[0].protocol_v1
        assert requests and all(u.startswith(f"http://127.0.0.1:{bport}") for u in requests), requests   # all local
