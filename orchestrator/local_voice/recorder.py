"""Opt-in session recording: both audio streams and every turn-taking event on one clock, local only (2026-10-05).

Why. In an early live test (the browser page on the laptop's speakers) 5 of 19 voice
turns were the agent's own words (tools/echo_turns.py): the last ~10 s of one reply heard again 6-59 s after the
replies ended, two of them cutting a reply. Nothing showed what played them (the person may have read the lines
aloud), and the voice laughed, swung in pitch and slurred, but production kept no audio. A short recorded live session
has to give the real echo mechanism and real voice audio to score, so this records everything tools/session_report.py needs. Off by default (config.yaml `record:`); `./run.sh --record` turns it on.
Nothing leaves the Mac: the files go under record.dir (../state/recordings), gitignored with the rest of state/.

Per connection (one pipeline), in <record.dir>/<session id>/ (a resumed protocol v1 session gets -2, -3...):

    mic.wav       16 kHz mono PCM16: what the input transport delivered, continuous on the recording's clock; silence
                  where no frames came (the test client sends only while it speaks; a stalled network). From the
                  browser page that is after Chrome's echo canceller, noise suppression and gain control (index.html
                  asks for all three: a steady tone fell from -3 to -21 dBFS within 1.2 s in the headless test); from
                  the apps, after Apple's voice processing.
    playback.wav  24 kHz mono PCM16: what the output transport sent (bargein.PausableWebsocketOutput after its pause
                  gate; any other output, the browser's SmallWebRTC one included, at its write_audio_frame), each chunk
                  where it was sent, silence between replies.
    events.jsonl  one object per line: {"t": seconds on the tracks' clock, "mono": time.monotonic(), "wall": epoch
                  seconds, "ev": kind, ...}. VAD start/stop, transcripts as pushed ("stt") and the echo guard's verdict
                  on each, before and after ("guard"), the turn-start decisions, user turn start/stop, bot started/
                  stopped speaking, interruptions, the reply pause, every TTS generation with its text, times, engine
                  chunk seams and a fingerprint of its first audio ("tts"), the client's played_ms and the protocol v1
                  reply markers, and the stretches of playback actually sent ("sent").
    meta.json     the config as loaded (pi.env values redacted), the client, versions (Python, Pipecat, numpy,
                  mlx-audio, the repo's commit), and at the end the recording's own counts.

The clock. t0 is time.monotonic() when the recording starts: mic sample k is at t0 + k / 16000, playback sample j at
t0 + j / 24000, an event at t0 + t. A microphone frame ends where it arrived (Pipecat's push timestamp from the input
transport, so a sample sits at its capture time plus the uplink). The first frame sits where it arrived and the rest
go back to back (the client's own sample clock: arrival jitter does not move them), with two corrections forward:
a frame more than GAP_S late means frames were missing, and silence fills to it; every frame of the last second
DRIFT_TOL_S late or more means a few were missing (or the stream paused briefly, as the test client does between its
calls), and silence takes the track up to the earliest of them. So a steady stream is placed to its first frame's
jitter, and a stream with short holes to within DRIFT_TOL_S. A burst after a stall is appended (frames arriving
early are never moved back; the most any was early is in meta.json). A playback chunk sits where it was sent or
right after the chunk before, whichever is later: Pipecat's WebSocket output sends one chunk ahead of real time and a
WebRTC track plays its queue in order, so a client playing each chunk as it comes hears it there. The echo delay
session_report measures (mic against playback) is therefore the round trip as the server sees it: downlink, the
client's playout, the room, the microphone, the uplink.

Off the hot path, and never in the way. On the event loop each hook only works out a position and puts (track,
offset, bytes) or an event dict on a bounded queue (put_nowait): when the queue is full the item is dropped and
counted, which leaves a hole of silence but never shifts the tracks (positions come from the clock, not from the
data written). A writer thread does the files and patches the WAV headers twice a second, so a crash leaves valid
files up to the last patch. Every hook catches its own errors and turns the recording off with one log line; the
session goes on. It stops at record.max_minutes (one log line), when the connection's worker cleans up its observers,
or at process exit.
"""
from __future__ import annotations

import atexit
import json
import platform
import queue
import struct
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

MIC_RATE = 16000
PLAY_RATE = 24000
GAP_S = 0.25            # a microphone frame this much later than the track's end: frames were missing, fill silence
DRIFT_TOL_S = 0.02      # ... and every frame of the last DRIFT_WIN_S this much late or more: a few frames were missing
DRIFT_WIN_S = 1.0
RUN_GAP_S = 0.05        # a playback chunk this much later than the last one's end starts a new sent stretch
QUEUE_ITEMS = 4000      # about 50 s of both tracks' chunks waiting for the writer before items are dropped
PATCH_S = 0.5           # how often the writer makes the WAV headers match the data written
REPO = Path(__file__).resolve().parents[2]
_open: weakref.WeakSet = weakref.WeakSet()


class Wav:
    """A mono PCM16 WAV written in order, with silence where the next write starts past the end, and a header patched
    to the data written (patch()), so a reader sees a valid file at any time after a patch."""

    def __init__(self, path: Path, rate: int):
        self.path, self.rate = path, rate
        self.f = open(path, "wb")
        self.samples = 0                  # samples in the file
        self.f.write(self._header(0))

    def _header(self, n: int) -> bytes:
        data = 2 * n
        return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE" + b"fmt " +
                struct.pack("<IHHIIHH", 16, 1, 1, self.rate, 2 * self.rate, 2, 16) + b"data" + struct.pack("<I", data))

    def write_at(self, offset: int, pcm: bytes) -> int:
        """Write `pcm` so its first sample is sample `offset`; returns the samples skipped because they overlapped what
        is already written (the placement never overlaps, so this stays 0)."""
        skip = 0
        if offset > self.samples:
            gap = offset - self.samples
            while gap:
                k = min(gap, 1 << 20)
                self.f.write(bytes(2 * k))
                gap -= k
            self.samples = offset
        elif offset < self.samples:
            skip = min(self.samples - offset, len(pcm) // 2)
            pcm = pcm[2 * skip:]
        if pcm:
            self.f.write(pcm[: len(pcm) // 2 * 2])
            self.samples += len(pcm) // 2
        return skip

    def patch(self) -> None:
        end = self.f.tell()
        self.f.seek(0)
        self.f.write(self._header(self.samples))
        self.f.seek(end)
        self.f.flush()

    def close(self) -> None:
        if not self.f.closed:
            self.patch()
            self.f.close()


def _versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("pipecat-ai", "numpy", "mlx-audio", "mlx", "websockets", "aiortc"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    out["commit"] = _git_commit(REPO)
    return out


def _git_commit(repo: Path) -> str | None:
    """HEAD's commit read from .git directly (no subprocess on a session's start)."""
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        p = repo / ".git" / ref
        if p.exists():
            return p.read_text().strip()
        packed = repo / ".git" / "packed-refs"
        for line in packed.read_text().splitlines() if packed.exists() else []:
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def config_snapshot(raw: dict) -> dict:
    """The loaded config (config.yaml with config.local.yaml and overrides merged), pi.env's values redacted: it is
    the extra environment of every Pi child and may hold a token."""
    snap = json.loads(json.dumps(raw, default=str))
    env = (snap.get("pi") or {}).get("env")
    if isinstance(env, dict):
        snap["pi"]["env"] = {k: "(redacted)" for k in env}
    return snap


def unique_folder(base: Path, session_id: str) -> Path:
    """<base>/<session id>, or -2, -3... when a resumed session (the same id, a new pipeline) recorded there already."""
    p = Path(base) / session_id
    n = 2
    while p.exists():
        p = Path(base) / f"{session_id}-{n}"
        n += 1
    return p


class SessionRecorder:
    """One connection's recording (module docstring). The hooks (mic, playback, event and the named ones) are called on
    the event loop and never raise or block."""

    def __init__(self, folder: Path, *, session_id: str, meta: dict[str, Any], max_s: float,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 queue_items: int = QUEUE_ITEMS):
        self.folder = Path(folder)
        self.session_id = session_id
        self.max_s = max_s
        self._clock = clock
        self.t0 = clock()
        self.wall0 = wall()
        self._meta = {**meta, "t0_mono": self.t0, "wall0": self.wall0}
        self._q: queue.Queue = queue.Queue(maxsize=queue_items)
        self._closing = threading.Event()
        self.active = True
        self.stop_reason: str | None = None
        self.failed: str | None = None
        # placement state (event loop only)
        self._mic_end: int | None = None  # mic samples placed (None until the first frame anchors the track)
        self._late: deque[tuple[float, int]] = deque()   # (arrival, how late it was in samples) over DRIFT_WIN_S
        self._play_end = 0                # playback samples placed
        self._run_start: int | None = None
        # counts (meta.json at the end)
        self.stats = {"mic_frames": 0, "mic_gaps": 0, "mic_gap_s": 0.0, "mic_resyncs": 0, "mic_resync_s": 0.0,
                      "mic_ahead_max_s": 0.0, "mic_skipped": 0,
                      "play_chunks": 0, "play_skipped": 0, "events": 0, "dropped_items": 0, "overlap_samples": 0}
        self._thread = threading.Thread(target=self._writer, name=f"recorder-{session_id}", daemon=True)

    # -------------------------------------------------------------------------------------------- lifecycle

    def start(self) -> SessionRecorder:
        """Make the folder and start the writer (it writes meta.json first: set every meta field before this)."""
        self.folder.mkdir(parents=True, exist_ok=False)
        self._thread.start()
        _open.add(self)
        return self

    def close(self, reason: str = "the session ended") -> None:
        """Stop recording; the writer writes what is queued, finishes the files and exits (nothing waits for it)."""
        if not self.active:
            return
        self._finish_run()
        self.active = False
        self.stop_reason = reason
        self._closing.set()
        self._put(("wake",))

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the writer to finish (tests, process exit)."""
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _fail(self, where: str, e: BaseException) -> None:
        if self.failed is None:
            self.failed = f"{where}: {type(e).__name__}: {e}"
            logger.error(f"recording {self.session_id}: stopped, {self.failed} (the session goes on)")
        self.close("failed")

    def _due(self, at: float) -> bool:
        """Still recording at `at`; past max_s the recording stops (once, with a log line)."""
        if not self.active:
            return False
        if at - self.t0 > self.max_s:
            logger.info(f"recording {self.session_id}: stopped at {self.max_s / 60:g} min (record.max_minutes); "
                        f"files in {self.folder}")
            self.close(f"max_minutes ({self.max_s / 60:g})")
            return False
        return True

    def _put(self, item: tuple) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            if item[0] != "wake":
                self.stats["dropped_items"] += 1

    # ------------------------------------------------------------------------------------------------ hooks

    def mic(self, pcm: bytes, at: float | None = None, rate: int = MIC_RATE, channels: int = 1) -> None:
        """One input frame, `at` the moment the transport delivered it (its end on the track)."""
        try:
            at = self._clock() if at is None else at
            if not self._due(at) or not pcm:
                return
            if rate != MIC_RATE or channels != 1:
                self.stats["mic_skipped"] += 1
                return
            n = len(pcm) // 2
            start = max(0, int(round((at - self.t0) * MIC_RATE)) - n)       # where it would sit if it ended on arrival
            if self._mic_end is None:
                self._mic_end = start
            late = start - self._mic_end
            if late > int(GAP_S * MIC_RATE):
                self.stats["mic_gaps"] += 1
                self.stats["mic_gap_s"] += late / MIC_RATE
                self._mic_end, late = start, 0
                self._late.clear()
            elif late < 0:
                self.stats["mic_ahead_max_s"] = max(self.stats["mic_ahead_max_s"], -late / MIC_RATE)
            self._late.append((at, late))
            while self._late and at - self._late[0][0] > DRIFT_WIN_S:
                self._late.popleft()
            if self._late and at - self._late[0][0] >= 0.9 * DRIFT_WIN_S:
                least = min(x for _, x in self._late)
                if least >= int(DRIFT_TOL_S * MIC_RATE):
                    self.stats["mic_resyncs"] += 1
                    self.stats["mic_resync_s"] += least / MIC_RATE
                    self._mic_end += least
                    self._late.clear()
            self._put(("mic", self._mic_end, bytes(pcm)))
            self._mic_end += n
            self.stats["mic_frames"] += 1
        except Exception as e:  # noqa: BLE001 - never into the session
            self._fail("mic", e)

    def playback(self, pcm: bytes, rate: int = PLAY_RATE, at: float | None = None) -> None:
        """One chunk the output transport sent, `at` the moment it went out."""
        try:
            at = self._clock() if at is None else at
            if not self._due(at) or not pcm:
                return
            if rate != PLAY_RATE:
                self.stats["play_skipped"] += 1
                return
            n = len(pcm) // 2
            sent = int(round((at - self.t0) * PLAY_RATE))
            if sent > self._play_end + int(RUN_GAP_S * PLAY_RATE):
                self._finish_run()
            pos = max(sent, self._play_end)
            if self._run_start is None:
                self._run_start = pos
            self._put(("play", pos, bytes(pcm)))
            self._play_end = pos + n
            self.stats["play_chunks"] += 1
        except Exception as e:  # noqa: BLE001
            self._fail("playback", e)

    def _finish_run(self) -> None:
        """The stretch of playback sent since the last gap, as a "sent" event (session_report cuts clips inside it)."""
        if self._run_start is not None:
            a, b = self._run_start, self._play_end
            self._run_start = None
            self._put(("ev", self._line("sent", self.t0 + a / PLAY_RATE, {"from_s": round(a / PLAY_RATE, 4),
                                                                          "to_s": round(b / PLAY_RATE, 4),
                                                                          "samples": [a, b]})))
            self.stats["events"] += 1

    def _line(self, ev: str, at: float, fields: dict[str, Any]) -> dict[str, Any]:
        return {"t": round(at - self.t0, 4), "mono": round(at, 4), "wall": round(self.wall0 + (at - self.t0), 4),
                "ev": ev, **fields}

    def event(self, ev: str, at: float | None = None, **fields: Any) -> None:
        try:
            at = self._clock() if at is None else at
            if not self._due(at):
                return
            self._put(("ev", self._line(ev, at, fields)))
            self.stats["events"] += 1
        except Exception as e:  # noqa: BLE001
            self._fail("event", e)

    # the named hooks the pipeline's parts call (attach_recorder wires them)

    def verdict(self, record: dict[str, Any]) -> None:
        """The echo guard's verdict on one transcript (echo_guard.EchoFilter.gate)."""
        self.event("guard", **record)

    def decision(self, d) -> None:
        """A turn-start decision (turn_start.BusyHoldStartStrategy)."""
        self.event("decision", action=d.action, reason=d.reason, text=d.text, final=d.final, held_s=d.held_s)

    def pause(self, ev) -> None:
        """A reply pause event (bargein.ReplyPause)."""
        self.event("pause", at=ev.at, kind=ev.kind, held_ms=None if ev.held_ms is None else round(ev.held_ms, 1))

    # ----------------------------------------------------------------------------------------------- writer

    def _writer(self) -> None:
        mic = play = events = None
        last_patch = time.monotonic()
        try:
            mic = Wav(self.folder / "mic.wav", MIC_RATE)
            play = Wav(self.folder / "playback.wav", PLAY_RATE)
            events = open(self.folder / "events.jsonl", "w", encoding="utf-8")
            self._meta.update(versions=_versions())
            self._write_meta()
            done = False
            while not done:
                try:
                    batch = [self._q.get(timeout=PATCH_S)]
                except queue.Empty:
                    batch = []
                while len(batch) < 2000:
                    try:
                        batch.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                for item in batch:
                    kind = item[0]
                    if kind == "mic":
                        self.stats["overlap_samples"] += mic.write_at(item[1], item[2])
                    elif kind == "play":
                        self.stats["overlap_samples"] += play.write_at(item[1], item[2])
                    elif kind == "ev":
                        events.write(json.dumps(item[1], ensure_ascii=False, default=str) + "\n")
                done = self._closing.is_set() and self._q.empty()
                if done or time.monotonic() - last_patch >= PATCH_S:
                    mic.patch()
                    play.patch()
                    events.flush()
                    last_patch = time.monotonic()
                if batch and not done:
                    time.sleep(0.05)      # let a few frames gather: ~20 wake-ups a second, not one per frame
        except Exception as e:  # noqa: BLE001 - the writer dies alone
            self.failed = self.failed or f"writer: {type(e).__name__}: {e}"
            self.active = False
            logger.error(f"recording {self.session_id}: stopped, {self.failed} (the session goes on)")
        finally:
            for f in (mic, play):
                try:
                    if f is not None:
                        f.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                if events is not None:
                    events.close()
                self._meta.update(ended_wall=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                                  duration_s=round(self._clock() - self.t0, 3), stop_reason=self.stop_reason,
                                  failed=self.failed, mic_s=round((mic.samples if mic else 0) / MIC_RATE, 3),
                                  playback_s=round((play.samples if play else 0) / PLAY_RATE, 3),
                                  stats=dict(self.stats))
                self._write_meta()
            except Exception:  # noqa: BLE001
                pass
            _open.discard(self)

    def _write_meta(self) -> None:
        tmp = self.folder / "meta.json.tmp"
        tmp.write_text(json.dumps(self._meta, indent=1, ensure_ascii=False, default=str))
        tmp.replace(self.folder / "meta.json")


@atexit.register
def _close_all() -> None:
    for rec in list(_open):
        rec.close("process exit")
        rec.join(2.0)


# --------------------------------------------------------------------------------------- the pipeline's side

def recording_observer(rec: SessionRecorder, input_processor=None):
    """A Pipecat observer for the microphone (each InputAudioRawFrame the input transport pushes) and the turn-taking
    events (module docstring), each at its push time. It sees a frame's first push only, and one copy of a frame
    broadcast both ways. Pipecat runs observers from their own queue (worker_observer.py 1.12), so nothing here
    holds up a frame; its cleanup, when the connection's worker ends, closes the recording."""
    from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame,
                                       InterimTranscriptionFrame, InterruptionFrame, OutputTransportMessageFrame,
                                       OutputTransportMessageUrgentFrame, TranscriptionFrame,
                                       UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                                       VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
    from pipecat.observers.base_observer import BaseObserver

    from .echo_guard import JUDGED

    V1_KEPT = {"audio_start", "audio_end", "end_of_turn", "interrupt"}

    class RecordingObserver(BaseObserver):
        def __init__(self):
            super().__init__(observe_every_push=False)
            self.rec = rec
            self._clock0: float | None = None      # the pipeline clock's start on time.monotonic()
            self._seen: deque[int] = deque(maxlen=256)

        def _mono(self, data) -> float:
            if not data.timestamp:
                return time.monotonic()
            if self._clock0 is None:
                self._clock0 = time.monotonic() - data.source.get_clock().get_time() / 1e9
            return self._clock0 + data.timestamp / 1e9

        async def on_push_frame(self, data) -> None:
            try:
                f = data.frame
                if isinstance(f, InputAudioRawFrame):
                    if input_processor is None or data.source is input_processor:
                        rec.mic(f.audio, self._mono(data), f.sample_rate, f.num_channels)
                    return
                if f.broadcast_sibling_id is not None and f.broadcast_sibling_id in self._seen:
                    return
                ev: dict[str, Any] | None = None
                if isinstance(f, VADUserStartedSpeakingFrame):
                    ev = {"ev": "vad", "on": True, "start_secs": f.start_secs}
                elif isinstance(f, VADUserStoppedSpeakingFrame):
                    ev = {"ev": "vad", "on": False, "stop_secs": f.stop_secs}
                elif isinstance(f, UserStartedSpeakingFrame):
                    ev = {"ev": "turn", "on": True}
                elif isinstance(f, UserStoppedSpeakingFrame):
                    ev = {"ev": "turn", "on": False}
                elif isinstance(f, BotStartedSpeakingFrame):
                    ev = {"ev": "bot", "on": True}
                elif isinstance(f, BotStoppedSpeakingFrame):
                    ev = {"ev": "bot", "on": False}
                elif isinstance(f, InterruptionFrame):
                    ev = {"ev": "interruption", "by": type(data.source).__name__}
                elif isinstance(f, (TranscriptionFrame, InterimTranscriptionFrame)):
                    ev = {"ev": "stt", "final": isinstance(f, TranscriptionFrame), "text": f.text,
                          "judged": bool(f.metadata.get(JUDGED)), "by": type(data.source).__name__}
                elif isinstance(f, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
                    m = f.message if isinstance(f.message, dict) else {}
                    if m.get("t") in V1_KEPT:
                        ev = {"ev": "v1", "msg": m}
                elif type(f).__name__ == "ClientMessageFrame":       # protocol_v1.py: played_ms and the rest
                    m = getattr(f, "message", None) or {}
                    if m.get("t") == "played_ms":
                        ev = {"ev": "played_ms", "reply_id": m.get("reply_id"), "ms": m.get("ms")}
                    elif m.get("t") not in (None, "ping", "pong"):
                        ev = {"ev": "client", "t": m.get("t")}
                if ev is not None:
                    self._seen.append(f.id)
                    kind = ev.pop("ev")
                    rec.event(kind, at=self._mono(data), **ev)
            except Exception as e:  # noqa: BLE001
                rec._fail("observer", e)

        async def cleanup(self) -> None:
            await super().cleanup()
            rec.close()

    return RecordingObserver()


def tap_output(out, rec: SessionRecorder) -> str:
    """Hear what an output transport sends. bargein.PausableWebsocketOutput reports each chunk itself, after its pause
    gate (on_audio_sent); any other output (the browser's SmallWebRTCOutputTransport) gets its write_audio_frame
    wrapped on this instance only: Pipecat's media sender calls it through the instance (base_output.py 1.12,
    _internal_write_audio_frame), and SmallWebRTC's write returns once its track has taken the chunk."""
    if hasattr(out, "on_audio_sent"):
        out.on_audio_sent = rec.playback
        return "on_audio_sent"
    orig = out.write_audio_frame

    async def write_audio_frame(frame):
        at = time.monotonic()
        ok = await orig(frame)
        if ok:
            rec.playback(frame.audio, frame.sample_rate, at)
        return ok

    out.write_audio_frame = write_audio_frame
    return "write_audio_frame"


def attach_recorder(session, *, transport, cfg) -> SessionRecorder | None:
    """Record this connection when config.yaml `record.enabled` is on: the recorder, its observer on the session's
    worker, the output tap, and the hooks of the TTS, the echo guard, the turn start and the reply pause. Returns None
    (and logs why) when off or when the recording cannot start; the session never depends on it."""
    rc = getattr(cfg, "record", None) or {}
    if not rc.get("enabled"):
        return None
    try:
        from .echo_guard import EchoFilter

        folder = unique_folder(Path(rc["dir"]), session.id)
        meta = {"session": session.id, "folder": str(folder), "client": session.client, "device": session.device,
                "mic": session.mic, "protocol_v1": session.protocol_v1, "transport": type(transport).__name__,
                "started_wall": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "mic_rate": MIC_RATE, "playback_rate": PLAY_RATE, "max_minutes": rc["max_minutes"],
                "clock": "t is seconds from t0 (time.monotonic() at start); mic sample k at t0 + k/16000, playback "
                         "sample j at t0 + j/24000 (recorder.py)",
                "config_path": str(getattr(cfg, "path", "")), "config": config_snapshot(getattr(cfg, "raw", {}) or {})}
        meta["playback_tap"] = "on_audio_sent" if hasattr(transport.output(), "on_audio_sent") else "write_audio_frame"
        rec = SessionRecorder(folder, session_id=session.id, meta=meta, max_s=60.0 * float(rc["max_minutes"]))
        rec.start()
        session.worker.add_observer(recording_observer(rec, transport.input()))
        tap_output(transport.output(), rec)
        if hasattr(session.tts, "recorder"):
            session.tts.recorder = rec
        for p in getattr(session.pipeline, "processors", []):
            if isinstance(p, EchoFilter):
                p.record = rec.verdict
        if session.turn_start is not None:
            session.turn_start.on_decision = rec.decision
        pause = session.reply_pause
        if pause is not None:
            before = pause.on_event

            def on_event(ev, _before=before):
                if _before is not None:
                    _before(ev)
                rec.pause(ev)
            pause.on_event = on_event
        session.extra["recorder"] = rec
        logger.info(f"recording {session.id} ({session.client or 'unknown'}) to {folder}: mic.wav, playback.wav, "
                    f"events.jsonl, meta.json; local only, stops after {rc['max_minutes']:g} min")
        return rec
    except Exception as e:  # noqa: BLE001 - a recording that cannot start must not stop the session
        logger.error(f"recording {getattr(session, 'id', '?')}: not recording: {type(e).__name__}: {e}")
        return None


__all__ = ["SessionRecorder", "attach_recorder", "recording_observer", "tap_output", "unique_folder", "Wav",
           "config_snapshot", "MIC_RATE", "PLAY_RATE"]

if sys.version_info < (3, 10):   # pragma: no cover
    raise RuntimeError("recorder.py needs Python 3.10+")
