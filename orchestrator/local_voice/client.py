"""A protocol v1 test client (PROTOCOL.md): streams 16 kHz PCM at real-time pace, plays the replies on a simulated
clock, answers `interrupt` with `played_ms`, and records every message with its arrival time.

The e2e tests drive it with `say`-generated speech (machine ground truth: the text is known), the model-free tests
with tones. As a command it runs one spoken turn against a running orchestrator:

    python -m local_voice.client --say "what does my knowledge base say about the hold gate?"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import websockets

IN_RATE = 16000
OUT_RATE = 24000


def say_pcm(text: str, voice: str = "Samantha", rate: int = IN_RATE) -> bytes:
    """`say` to 16 kHz mono PCM16 (the measurement clips were made the same way: say, then afconvert)."""
    with tempfile.TemporaryDirectory(prefix="lv-say-") as d:
        aiff, wav = Path(d) / "s.aiff", Path(d) / "s.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1", str(aiff), str(wav)], check=True)
        with wave.open(str(wav)) as w:
            assert w.getframerate() == rate and w.getnchannels() == 1 and w.getsampwidth() == 2
            return w.readframes(w.getnframes())


def tone_pcm(seconds: float, freq: float = 300.0, amp: float = 0.3, rate: int = IN_RATE) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2").tobytes()


def speech_end_s(pcm: bytes, rate: int = IN_RATE, threshold: float = 0.01) -> float:
    """Where the last audible sample of a clip is (its end of speech), in seconds."""
    a = np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
    loud = np.flatnonzero(a > threshold)
    return (loud[-1] + 1) / rate if loud.size else 0.0


@dataclass
class Reply:
    id: str
    started_at: float | None = None        # audio_start received
    first_audio_at: float | None = None    # first binary frame
    first_loud_at: float | None = None     # first frame with audible content
    bytes: int = 0
    pcm: bytearray = field(default_factory=bytearray)
    text: list[str] = field(default_factory=list)
    ended_at: float | None = None
    interrupted_at: float | None = None
    played_ms: float | None = None
    # A simulated player: each frame plays as soon as it arrives and the one before has finished, so a pause in the
    # server's sending (bargein.py) leaves a gap instead of counting as played. (start, end) of each stretch played.
    segments: list[list[float]] = field(default_factory=list)

    @property
    def audio_s(self) -> float:
        return self.bytes / 2 / OUT_RATE

    def arrived(self, at: float, nbytes: int) -> None:
        d = nbytes / 2 / OUT_RATE
        if self.segments and self.segments[-1][1] >= at:
            self.segments[-1][1] += d
        else:
            self.segments.append([at, at + d])

    def played_s(self, at: float) -> float:
        """Seconds of this reply's audio the simulated player had played by `at` (a flush at `at` stops it there)."""
        return sum(max(0.0, min(e, at) - s) for s, e in self.segments)

    def silent_from(self, since: float) -> float | None:
        """The first moment at or after `since` when the player had nothing to play (a gap or the end of the audio
        received so far), or None while it is still playing."""
        for s, e in self.segments:
            if e >= since and s <= since:
                return e
            if s > since:
                return since
        return since if self.segments else None


class V1Client:
    def __init__(self, url: str, *, device: str = "test-client", client: str = "test", mic: str = "vad",
                 frame_ms: int = 20):
        self.url = url
        self.hello = {"t": "hello", "v": 1, "client": client, "device": device, "mic": mic}
        self.frame_ms = frame_ms
        self.ws: Any = None
        self.messages: list[tuple[float, dict]] = []
        self.replies: dict[str, Reply] = {}
        self.current: Reply | None = None
        self.welcome: dict | None = None
        self.close_code: int | None = None
        self._reader: asyncio.Task | None = None
        self._event = asyncio.Event()
        self.t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self.t0

    async def connect(self, **hello: Any) -> dict:
        self.ws = await websockets.connect(self.url, max_size=None)
        try:
            await self.ws.send(json.dumps({**self.hello, **hello}))
        except websockets.ConnectionClosed as e:   # refused at once (4403)
            self.close_code = e.rcvd.code if e.rcvd else None
            raise ConnectionError(f"closed with {self.close_code}") from e
        self._reader = asyncio.create_task(self._read())
        self.welcome = await self.wait_for(lambda m: m.get("t") == "welcome", timeout=10)
        return self.welcome

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
        if self._reader:
            self._reader.cancel()

    async def send(self, msg: dict) -> None:
        await self.ws.send(json.dumps(msg))

    # -- receiving

    async def _read(self) -> None:
        try:
            async for data in self.ws:
                at = self.now()
                if isinstance(data, bytes):
                    r = self.current
                    if r is None:
                        r = self.current = self.replies.setdefault("?", Reply("?"))
                    if r.interrupted_at is None:
                        if r.first_audio_at is None:
                            r.first_audio_at = at
                        a = np.frombuffer(data, dtype="<i2")
                        if r.first_loud_at is None and a.size and np.abs(a).max() > 330:   # about -40 dBFS
                            r.first_loud_at = at
                        r.bytes += len(data)
                        r.pcm += data
                        r.arrived(at, len(data))
                    continue
                msg = json.loads(data)
                self.messages.append((at, msg))
                await self._on_message(at, msg)
                self._event.set()
        except websockets.ConnectionClosed as e:
            self.close_code = e.rcvd.code if e.rcvd else None
        finally:
            if self.close_code is None and self.ws is not None and self.ws.close_code is not None:
                self.close_code = self.ws.close_code
            self._event.set()

    def played_ms(self, r: Reply, at: float) -> float:
        """What a client playing each frame as it arrives had played by `at`."""
        return 1000 * r.played_s(at)

    async def _on_message(self, at: float, msg: dict) -> None:
        t = msg.get("t")
        if t == "audio_start":
            self.current = self.replies.setdefault(msg["reply_id"], Reply(msg["reply_id"]))
            self.current.started_at = at
        elif t == "reply_text" and msg.get("reply_id") in self.replies:
            self.replies[msg["reply_id"]].text.append(msg.get("delta", ""))
        elif t == "interrupt":
            r = self.replies.get(msg.get("reply_id")) or self.current
            if r is not None and r.interrupted_at is None:
                r.interrupted_at = at
                r.played_ms = self.played_ms(r, at)
                await self.send({"t": "played_ms", "reply_id": r.id, "ms": round(r.played_ms)})
        elif t == "end_of_turn":
            r = self.replies.get(msg.get("reply_id"))
            if r is not None and r.ended_at is None:
                r.ended_at = at
                if r.interrupted_at is None and r.bytes:
                    r.played_ms = r.audio_s * 1000
                    await self.send({"t": "played_ms", "reply_id": r.id, "ms": round(r.played_ms)})
            if self.current is r:
                self.current = None

    async def wait_for(self, pred, timeout: float = 30.0, since: float = 0.0) -> dict:
        """The first message (received at or after `since`) that pred accepts."""
        deadline = time.monotonic() + timeout
        while True:
            for at, m in self.messages:
                if at >= since and pred(m):
                    return m
            if self.close_code is not None or (self.ws is not None and self.ws.close_code is not None):
                raise ConnectionError(f"closed with {self.close_code or self.ws.close_code}")
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no matching message in {timeout} s; last: {[m for _, m in self.messages[-5:]]}")
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), min(left, 0.5))
            except TimeoutError:
                pass

    def of_type(self, t: str, since: float = 0.0) -> list[dict]:
        return [m for at, m in self.messages if at >= since and m.get("t") == t]

    # -- sending audio

    async def stream(self, pcm: bytes, *, realtime: bool = True) -> float:
        """Send PCM16 at 16 kHz in frame_ms frames, paced like a microphone. Returns the time the last frame left."""
        step = IN_RATE * 2 * self.frame_ms // 1000
        start = time.monotonic()
        for i, off in enumerate(range(0, len(pcm), step)):
            await self.ws.send(pcm[off:off + step])
            if realtime:
                due = start + (i + 1) * self.frame_ms / 1000
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        return self.now()

    async def silence(self, seconds: float) -> float:
        return await self.stream(bytes(int(IN_RATE * seconds) * 2))

    async def speak(self, pcm: bytes, *, tail_s: float = 0.0) -> float:
        """Stream an utterance, then `tail_s` of silence. Returns the client time at which the speech ended."""
        start = self.now()
        await self.stream(pcm)
        end = start + speech_end_s(pcm)
        if tail_s:
            await self.silence(tail_s)
        return end

    def reply_after(self, since: float) -> Reply | None:
        rs = [r for r in self.replies.values() if r.started_at is not None and r.started_at >= since]
        return min(rs, key=lambda r: r.started_at) if rs else None


async def _cli(args) -> int:
    c = V1Client(args.url, device=args.device, mic="vad")
    w = await c.connect()
    print("welcome", w)
    pcm = say_pcm(args.say)
    eos = await c.speak(pcm)
    tail = asyncio.create_task(c.silence(args.wait))
    try:
        r = None
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            r = c.reply_after(eos)
            if r is not None and r.ended_at is not None:
                break
            await asyncio.sleep(0.1)
        for at, m in c.messages:
            if m.get("t") not in ("pong",):
                print(f"{at:7.3f} {m}")
        if r and r.first_audio_at is not None:
            print(f"end of speech -> first audio {1000 * (r.first_audio_at - eos):.0f} ms; first audible "
                  f"{1000 * ((r.first_loud_at or r.first_audio_at) - eos):.0f} ms; reply {r.audio_s:.1f} s")
    finally:
        tail.cancel()
        await c.close()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8770/v1/voice")
    ap.add_argument("--device", default="cli-test")
    ap.add_argument("--say", required=True)
    ap.add_argument("--wait", type=float, default=40.0)
    raise SystemExit(asyncio.run(_cli(ap.parse_args())))


if __name__ == "__main__":
    main()
