"""The browser page (static/index.html) in a separate headless Chromium (Playwright, temporary profile, a fake microphone
from a WAV file, --disable-gpu), never the user's own Chrome; fake speech models, the stub LLM, a real Pi child.

- Captions: an early live test showed most sentences two or three times ("A. A. B. C. A. B. B. C. C.") while
  each was generated once (2026-10-05). The test counts, for a reply of three sentences around a tool call, how often
  each sentence reached the page in the messages the old page displayed (bot-transcription and every bot-output) and
  how often the page shows it now.
- The approval card (PROTOCOL.md "Approvals"): the summary, the file, the folder and the text to be written on screen,
  answered by its button (the file is written), or by voice (the card closes itself on confirm_cancel).
"""
from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from harness import free_port, rig

pytestmark = pytest.mark.needs_pi
CHROMIUM = Path.home() / "Library/Caches/ms-playwright/chromium-1243"


def one_utterance_wav(path: Path, silence_s: float = 60.0, at: tuple[float, ...] = (0.5,)) -> Path:
    """1.2 s of a loud tone at each of `at` seconds, then a long silence, at 48 kHz: that many utterances within the
    test (Chromium loops the file)."""
    rate = 48000
    t = np.arange(int(1.2 * rate)) / rate
    tone = 0.4 * np.sin(2 * np.pi * 300 * t)
    a = np.zeros(int((max(at) + 1.2 + silence_s) * rate))
    for start in at:
        a[int(start * rate):int(start * rate) + len(tone)] = tone
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((a * 32767).astype("<i2").tobytes())
    return path


async def open_page(p, bport: int, wav: Path):
    browser = await p.chromium.launch(headless=True, args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={wav}",
        "--autoplay-policy=no-user-gesture-required", "--disable-gpu"])
    ctx = await browser.new_context(permissions=["microphone"])
    page = await ctx.new_page()
    await page.goto(f"http://127.0.0.1:{bport}/")
    # keep every RTVI message the page receives, to count what the old page would have shown
    await page.evaluate("""() => { window.__rtvi = []; const orig = window.onRtvi;
        window.onRtvi = (m) => { window.__rtvi.push(m); orig(m); }; }""")
    await page.locator("#connect").click()
    await page.wait_for_function("document.getElementById('connect').textContent === 'Disconnect'", timeout=15000)
    return browser, page


@pytest.mark.skipif(not CHROMIUM.exists(), reason="needs the cached Playwright Chromium 1243")
async def test_each_sentence_is_captioned_once(tmp_path):
    from playwright.async_api import async_playwright

    bport = free_port()
    sentences = ["Let me check the folder first.", "There are two notes in it.", "Want me to read them?"]
    async with rig(tmp_path, stt_script=["what is in the folder"], stt_text="",
                   overrides={"server": {"browser": {"enabled": True, "port": bport}}}) as r:
        r.stub.script([{"text": sentences[0], "tool_calls": [{"name": "ls", "arguments": {"path": "."}}]},
                       {"text": " ".join(sentences[1:])}])
        async with async_playwright() as p:
            browser, page = await open_page(p, bport, one_utterance_wav(tmp_path / "mic.wav"))
            # the messages come as the reply's audio plays out: wait for the last sentence on the page, then a second
            # more for any copy of it still on its way
            await page.wait_for_function(f"document.body.innerText.includes({sentences[-1]!r})", timeout=20000)
            await asyncio.sleep(1.0)
            shown = await page.evaluate("Array.from(document.querySelectorAll('.msg.bot')).map(e => e.textContent)")
            rtvi = await page.evaluate("window.__rtvi")
            await browser.close()
    old_page = [m["data"].get("text", "") for m in rtvi if m.get("type") in ("bot-transcription", "bot-output")]
    before = {s: sum(s in t for t in old_page) for s in sentences}
    after = {s: sum(b.count(s) for b in shown) for s in sentences}
    print(f"\ncaptions per sentence, old page: {before}; now: {after}; bubbles: {shown}")
    assert all(n >= 2 for n in before.values()), before        # the fault, reproduced: two or three copies each
    assert after == {s: 1 for s in sentences}, (after, shown)
    assert len(shown) == 1, shown                               # one reply, one bubble ("Let me look." + its answer)


def card_rig(tmp_path, bport: int, script: list[str]):
    return rig(tmp_path, stt_script=script, stt_text="", spaces={"tools": ["read", "ls"], "act_tools": ["write"]},
               overrides={"server": {"browser": {"enabled": True, "port": bport}},
                          "agent": {"approvals": {"card_wait_s": 30.0, "reask_after_s": 0}}})


WRITE = {"tool_calls": [{"name": "write", "arguments": {"path": "Hello.txt", "content": "Hello world from the voice agent\n"}}]}


@pytest.mark.skipif(not CHROMIUM.exists(), reason="needs the cached Playwright Chromium 1243")
async def test_the_card_shows_exactly_what_will_happen_and_its_button_answers(tmp_path):
    from playwright.async_api import async_playwright

    bport = free_port()
    async with card_rig(tmp_path, bport, ["make a file called hello"]) as r:
        await r.orch.hub.set_mode("act")
        r.stub.script([WRITE, {"text": "Done, it is there."}])
        async with async_playwright() as p:
            browser, page = await open_page(p, bport, one_utterance_wav(tmp_path / "mic.wav"))
            card = page.locator("#cards .card")
            await card.wait_for(timeout=20000)
            text = await card.inner_text()
            buttons = await card.locator("button").all_inner_texts()
            await page.screenshot(path=str(tmp_path / "card.png"))
            await card.locator("button", has_text="Do it").click()
            await page.wait_for_function("document.querySelector('#log .card.closed') !== null", timeout=5000)
            closed = await page.locator("#log .card.closed .wait").inner_text()
            for _ in range(50):
                if (r.workdir / "Hello.txt").exists():
                    break
                await asyncio.sleep(0.1)
            await browser.close()
    assert "Create a new file Hello.txt in the scratch folder." in text
    assert str(r.workdir / "Hello.txt") in text and "Hello world from the voice agent" in text
    assert "write" in text and "create" in text and "act mode" in text and "or say yes or no" in text
    assert buttons == ["Do it", "Allow writing files in the scratch folder for the rest of this session", "Don't"]
    assert closed == "You chose: Do it"
    assert (r.workdir / "Hello.txt").read_text() == "Hello world from the voice agent\n"


@pytest.mark.skipif(not CHROMIUM.exists(), reason="needs the cached Playwright Chromium 1243")
async def test_a_card_answered_by_voice_closes_itself(tmp_path):
    from playwright.async_api import async_playwright

    bport = free_port()
    async with card_rig(tmp_path, bport, ["make a file called hello", "yes"]) as r:
        await r.orch.hub.set_mode("act")
        r.stub.script([WRITE, {"text": "Done."}])
        async with async_playwright() as p:
            browser, page = await open_page(p, bport, one_utterance_wav(tmp_path / "mic.wav", at=(0.5, 9.0)))
            await page.locator("#cards .card").wait_for(timeout=20000)
            await page.wait_for_function("document.querySelector('#log .card.closed') !== null", timeout=20000)
            closed = await page.locator("#log .card.closed .wait").inner_text()
            await browser.close()
    assert closed == "Answered by voice."
    assert (r.workdir / "Hello.txt").read_text() == "Hello world from the voice agent\n"
