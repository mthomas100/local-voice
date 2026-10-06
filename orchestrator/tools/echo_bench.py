"""The planted-echo bench: what the agent's own voice, coming back into the microphone, does to turns and barge-in, with
the echo guard off and on, measured by machine (2026-10-05; built model-free, run on the GPU when it is free).

Why. In an early live test (the browser page in Chrome on the laptop's speakers) 5 of
19 voice turns were the agent's own words: replays of one reply's last ~10 s, 6 to 59 s after the replies ended, and
two of them cut a reply; during the replies themselves Chrome's canceller held the live echo (local_voice/echo_guard.py
has the log evidence). The server-side guard (echo_guard.py, config `echo:`) drops words matching the agent's last
5 minutes of speech while it is busy (they never barge in) and lets them through with a mark while it is idle (the
person may read the agent's lines aloud to ask about them); `echo.hold_for_words` makes a VAD start while the agent is
busy wait for words (turn_start.py). A project rule: never ask anyone to listen, rate or label, so every
number here comes from planted signals whose content is known.

How. A protocol v1 test client (EchoClient, a local_voice.client.V1Client) streams a microphone that never stops: a
-60 dBFS pink-noise room, `say` speech planted at known moments, and the echo of whatever its simulated player plays
(tools/echo_mixer.py: resampled to 16 kHz, delayed, attenuated, optionally through a synthetic room). Each scenario runs
on its own connection against a scratch server, under two configs:

    off: echo {guard: false, hold_for_words: false}
    on:  echo {guard: true, hold_for_words: all}         (`all`, so protocol v1 exercises the hold too)

Scenarios (SCENARIOS): control (no echo); live echo of a long reply at -30, -20 and -10 dB, 150 ms, dry and through
the room, plus 300 ms at -20 dB in the room; a true barge-in ("Stop. What is the capital of Italy?") 2.0 s into the
reply over -20 dB room echo; the early live test's case, a delayed replay (no live echo, then 6 s after the reply ends its last
8 s come back at full level); and a sentence of the reply read aloud in another voice 2 s after it ends, which must go
through as a turn. The request is always "Tell me a long story about a lighthouse keeper and his cat with lots of
detail." (tests/e2e/test_turn.py's barge-in request: "story" lifts the spoken cap to 150 words, and no comma, at which
Smart Turn once split it).

The LLM. By default the stub (tests/stub_llm.py, no model): the scratch config's `pi.models_source` points at a copy
of the configured models.json whose providers' baseUrl is the stub, so the Pi children talk to it while STT, TTS, Silero
and Smart Turn are the real ones; it answers by the person's own words (route_reply): the story request gets
test_turn.py's 84-word LONG_REPLY (about 32 s of Qwen3-TTS "Ryan" at 2.6 words/s), "capital of Italy" gets "The capital
of Italy is Rome.", anything else a 26-word "I'm not sure what you meant..." (long enough to overlap the rest of a
replay). Deterministic replies, and no LLM traffic: gpu_clear.sh's LLM check is not disturbed. `--llm real` keeps
production's provider (qwen38 through the hold gate): the same request, the reply as written, and the read-aloud
sentence picked from it and rendered after it.

The server. `./run.sh --scratch DIR --port 8771 --browser-port 7861` (local_voice/scratch.py: state, turn log, brain
state, a kb clone and the spaces' roots on clones under DIR, loopback only). make_scratch rewrites DIR/config.yaml at
every start, but load_config merges DIR/config.local.yaml over it, and that file it never touches: the bench writes the
scenario config there (write_local_config): the `echo:` keys, the browser page off (the bench is protocol v1 only, which
frees 7861, the second port this work may use, for the stub LLM), and the stub's models.json. A config is read at
start, so `run` starts the server once per config and stops it after (SIGINT, as Ctrl-C). `--attach` uses a server
started by hand instead (`prepare` writes the file first).

Measured per scenario (metrics()), from the client's messages, the turn log (DIR/turns/<day>.jsonl, this
connection's session) and the orchestrator log (DIR/state/logs, the scenario's time window): user turns with their
text, each classed as the planted speech it carries, echo (echo_guard.best_match against what the agent said before,
>= 3 words and >= 60% of the turn, or 1-2 words all from its last 3 sentences) or other, and whether it began during a
reply or how long after one (live echo lags the speaker, so the echo of a reply's last words begins after the server's
"Bot stopped speaking", when the agent is idle: the harness showed it with the guard on (2026-10-05), the VAD
start 137 ms after it, never held, let through); `interrupt` messages, each a barge-in by planted speech (between its
onset and 2 s after its end, cutting a reply it overlapped) or false; whether the reply played to its end, and its
pauses (gaps in the simulated playout: bargein.py's reply pause); the barge-in's interrupt and audio stop after its
onset, its transcript (WER) and whether it was answered (Rome); replies cut after the replay began; whether the
read-aloud sentence became a turn and was answered; the log's lines (the guard's drops and idle passes, turn_start.py's
holds and how each ended, the reply pause's paused/resumed/confirmed, Pipecat's user turn starts). Written as
results.jsonl (one line per config and scenario) and report.md (before/after) under
~/repos/local-voice/state/echo-bench/<stamp>/.

GPU. `run` refuses unless ../measure/bench/gpu_clear.sh exits 0, and between scenarios stops on a GPU job or a closed
hold gate (e2e_fixtures.gpu_still_ours's rule). The plan runs about 18 minutes, two server starts included (`plan`
prints the estimate); an echo loop (no guard, -10 dB) can stretch a scenario to its cap.

    .venv/bin/python tools/echo_bench.py plan                     # scenarios, configs, GPU-time estimate (no model)
    .venv/bin/python tools/echo_bench.py run                      # GPU: both configs, all scenarios (~18 min)
    .venv/bin/python tools/echo_bench.py run --configs on --only live-20-room,bargein-20-room
    .venv/bin/python tools/echo_bench.py prepare --config on      # then ./run.sh --scratch ... and run --attach
    .venv/bin/python tools/echo_bench.py report ../state/echo-bench/<stamp>    # rebuild report.md (no model)
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ORCH = Path(__file__).resolve().parents[1]
REPO = ORCH.parent
sys.path.insert(0, str(ORCH))
sys.path.insert(0, str(ORCH / "tools"))

from echo_mixer import (  # noqa: E402
    FRAME,
    MIC_RATE,
    SPK_RATE,
    EchoMixer,
    EchoSettings,
    Planted,
    pcm16_to_float,
    played,
    resample_24k_to_16k,
)
from local_voice.client import V1Client  # noqa: E402
from local_voice.echo_guard import best_match, words  # noqa: E402

GPU_CLEAR = str(REPO / "measure" / "bench" / "gpu_clear.sh")
RUN_SH = ORCH / "run.sh"               # what ScratchServer starts (the tests start a fake in its place)
OUT_ROOT = REPO / "state" / "echo-bench"
SCRATCH = OUT_ROOT / "scratch"         # shared by runs: make_scratch keeps its clones (kb, spaces) from run to run
PORT, BROWSER_PORT, STUB_PORT = 8771, 7861, 7861   # the two ports this work may use (lead, 2026-10-05)

# --------------------------------------------------------------------------------------------- the words

REQUEST = "Tell me a long story about a lighthouse keeper and his cat with lots of detail."
# tests/e2e/test_turn.py's LONG_REPLY, word for word, so the planted controls there and the echo here hear one reply
LONG_REPLY = ("The lighthouse stood at the end of a long stone pier, and every evening the keeper climbed its spiral "
              "stairs to light the lamp. His cat followed him all the way up, step by step, and sat by the window "
              "while the beam swept across the dark water. Ships far out at sea saw the light and knew where the rocks "
              "were. In the morning the keeper polished the glass, wound the clockwork, and wrote the weather in his "
              "logbook before he slept.")
BARGEIN = "Stop. What is the capital of Italy?"
BARGEIN_REPLY = "The capital of Italy is Rome."
DEFAULT_REPLY = ("I'm not sure what you meant by that. Could you say it again in your own words, so I can help you "
                 "with exactly what you need?")
READ_ALOUD = ("His cat followed him all the way up, step by step, and sat by the window while the beam swept across "
              "the dark water.")
PERSON_VOICE, READER_VOICE = "Samantha", "Daniel"    # `say` voices the other benches use (tools/bargein_bench.py)

# --------------------------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class Scenario:
    """One planted situation. level_db None: no live echo (Chrome's canceller held it in early live tests)."""
    name: str
    kind: str                          # control | live | bargein | replay | read_aloud
    level_db: float | None = None      # the live echo, relative to the reply's own level
    delay_ms: float = 150.0
    reverb: bool = False
    bargein_after_s: float = 2.0       # bargein: speech starts this long after the reply's first audio
    replay_after_s: float = 6.0        # replay: this long after the reply's playout ended ...
    replay_s: float = 8.0              # ... its last this-many seconds come back ...
    replay_db: float = 0.0             # ... at this level relative to the reply (full level)
    read_after_s: float = 2.0          # read_aloud: this long after the reply ended, a sentence of it is read aloud
    max_s: float = 90.0                # cap from the request's end: an echo feeding itself ends here

    def echo(self) -> EchoSettings:
        return EchoSettings(level_db=self.level_db, delay_ms=self.delay_ms, reverb=self.reverb)


SCENARIOS: list[Scenario] = [
    Scenario("control", "control"),
    Scenario("live-30-dry", "live", -30.0),
    Scenario("live-30-room", "live", -30.0, reverb=True),
    Scenario("live-20-dry", "live", -20.0),
    Scenario("live-20-room", "live", -20.0, reverb=True),
    Scenario("live-10-dry", "live", -10.0),
    Scenario("live-10-room", "live", -10.0, reverb=True),
    Scenario("live-20-room-300ms", "live", -20.0, delay_ms=300.0, reverb=True),
    Scenario("bargein-20-room", "bargein", -20.0, reverb=True, max_s=60.0),
    Scenario("replay", "replay", max_s=120.0),
    Scenario("read-aloud", "read_aloud"),
]
# hold_for_words: off | browser | all (config.py since 53c27a4; false reads as off, true as all). `guard` is opt-in
# (--configs off,guard,on): production's own setting for the protocol v1 apps, whose `browser` hold leaves them
# unheld, so a VAD start interrupts before any words are known.
CONFIGS: dict[str, dict[str, Any]] = {"off": {"guard": False, "hold_for_words": False},
                                      "guard": {"guard": True, "hold_for_words": False},
                                      "on": {"guard": True, "hold_for_words": "all"}}
DEFAULT_CONFIGS = ("off", "on")

# The estimate's assumptions. Speech rates: `say` about 2.9 words/s at its default rate; Qwen3-TTS "Ryan" 2.6 words/s
# (protocol_v1.WORDS_PER_MS). Turn end about 1 s after speech (Silero's stop plus Smart Turn), first audio about 0.6 s
# after that. A server start: 4-5 s of model loading (startup.json of the e2e runs, 2026-10-05 16:22-17:36), the run.sh
# start and the first Pi child: 15 s.
SAY_WPS, TTS_WPS, TURN_S, FIRST_AUDIO_S = 2.9, 2.6, 1.0, 0.6
QUIET_S, SETTLE_S, LEAD_S, SERVER_START_S = 4.0, 2.5, 1.0, 15.0


def spoken_s(text: str, wps: float) -> float:
    return len(words(text)) / wps


def expected_s(sc: Scenario) -> float:
    """Seconds one scenario should take when nothing goes wrong (stub replies)."""
    request = LEAD_S + spoken_s(REQUEST, SAY_WPS) + TURN_S + FIRST_AUDIO_S
    reply = spoken_s(LONG_REPLY, TTS_WPS)
    after = QUIET_S + SETTLE_S
    answer = TURN_S + FIRST_AUDIO_S + spoken_s(DEFAULT_REPLY, TTS_WPS)
    if sc.kind == "bargein":
        return (request + sc.bargein_after_s + spoken_s(BARGEIN, SAY_WPS) + TURN_S + FIRST_AUDIO_S
                + spoken_s(BARGEIN_REPLY, TTS_WPS) + after)
    if sc.kind == "replay":
        return request + reply + sc.replay_after_s + sc.replay_s + answer + after
    if sc.kind == "read_aloud":
        return request + reply + sc.read_after_s + spoken_s(READ_ALOUD, SAY_WPS) + answer + after
    return request + reply + after


def plan(configs: Iterable[str] = DEFAULT_CONFIGS, only: Iterable[str] | None = None) -> list[tuple[str, Scenario]]:
    """The run order: every chosen scenario under one config, then under the next (one server start per config)."""
    names = set(only) if only else None
    unknown = (names or set()) - {s.name for s in SCENARIOS}
    if unknown:
        raise ValueError(f"no scenario {', '.join(sorted(unknown))}; known: {', '.join(s.name for s in SCENARIOS)}")
    for c in configs:
        if c not in CONFIGS:
            raise ValueError(f"no config {c!r}; known: {', '.join(CONFIGS)}")
    return [(c, s) for c in configs for s in SCENARIOS if names is None or s.name in names]


def estimate(pairs: list[tuple[str, Scenario]]) -> dict[str, float]:
    configs = {c for c, _ in pairs}
    exp = sum(expected_s(s) for _, s in pairs) + SERVER_START_S * len(configs)
    cap = (sum(s.max_s + LEAD_S + spoken_s(REQUEST, SAY_WPS) + SETTLE_S for _, s in pairs)
           + SERVER_START_S * len(configs))
    return {"scenarios": len(pairs), "configs": len(configs), "expected_min": round(exp / 60, 1),
            "worst_case_min": round(cap / 60, 1)}


# --------------------------------------------------------------------------------------------- the clips


@dataclass
class Clips:
    """The person's speech, rendered once before any timing (rendering blocks for a second or two: test_turn.said)."""
    request: np.ndarray
    bargein: np.ndarray
    read_aloud: np.ndarray | None       # None: picked from the reply and rendered after it (--llm real)
    texts: dict[str, str] = field(default_factory=lambda: {"request": REQUEST, "barge-in": BARGEIN,
                                                           "read-aloud": READ_ALOUD})


def say16(text: str, voice: str = PERSON_VOICE) -> np.ndarray:
    from local_voice.client import say_pcm
    return pcm16_to_float(say_pcm(text, voice=voice))


def render_clips(read_aloud: bool = True) -> Clips:
    return Clips(request=say16(REQUEST), bargein=say16(BARGEIN),
                 read_aloud=say16(READ_ALOUD, READER_VOICE) if read_aloud else None)


def pick_sentence(text: str) -> str:
    """A sentence of a reply to read aloud: the first of 8-25 words, else the longest."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    fit = [s for s in sents if 8 <= len(words(s)) <= 25]
    return fit[0] if fit else max(sents, key=lambda s: len(words(s)), default="")


# --------------------------------------------------------------------------------------------- the client


class EchoClient(V1Client):
    """A protocol v1 test client whose microphone never stops: from start_mic() on it sends one 20 ms frame of
    EchoMixer output every 20 ms, each once its last sample is due (a real microphone cannot send sooner), instead of
    the clips and silences the e2e tests stream one after another. The mixer hears what this client's simulated player
    plays (Reply.segments, cut at interrupted_at). `plant` puts a clip into the next frame not yet rendered."""

    def __init__(self, url: str, *, echo: EchoSettings, **kw: Any):
        super().__init__(url, **kw)
        self.wall0 = time.time() - self.now()        # the wall clock of client time 0 (turn log, server log)
        self.mic_t0: float | None = None
        self.mixer = EchoMixer(echo, self._speaker)
        self.frames_rendered = 0
        self.mic_error: str | None = None
        self._mic: asyncio.Task | None = None

    def _speaker(self, j0: int, n: int) -> np.ndarray:
        return played(list(self.replies.values()), self.mic_t0, j0, n)

    async def start_mic(self) -> None:
        self.mic_t0 = self.now()
        self._mic = asyncio.create_task(self._run_mic(), name=f"mic-{self.hello.get('device')}")

    async def _run_mic(self) -> None:
        import websockets

        start = self.t0 + self.mic_t0
        i = 0
        try:
            while True:
                delay = start + (i + 1) * FRAME / MIC_RATE - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                pcm = self.mixer.frame(i)
                self.frames_rendered = i + 1
                await self.ws.send(pcm)
                i += 1
        except websockets.ConnectionClosed:
            pass
        except Exception as e:  # noqa: BLE001 - a dead microphone must show in the record, not vanish with its task
            self.mic_error = f"{type(e).__name__}: {e}"

    async def stop_mic(self) -> None:
        if self._mic is not None:
            self._mic.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._mic

    def mic_time(self, k: int) -> float:
        return self.mic_t0 + k / MIC_RATE

    def plant(self, samples: np.ndarray, *, label: str, kind: str, text: str = "", gain_db: float = 0.0) -> Planted:
        return self.mixer.plant(samples, self.frames_rendered * FRAME, label=label, kind=kind, text=text,
                                gain_db=gain_db)

    # -- waiting on the conversation

    def playout_end(self, r) -> float | None:
        if r.interrupted_at is not None:
            return r.interrupted_at
        return r.segments[-1][1] if r.segments else r.ended_at

    def over(self, r) -> bool:
        if r.interrupted_at is not None:
            return True
        end = self.playout_end(r)
        return r.ended_at is not None and end is not None and self.now() >= end

    def playing(self) -> bool:
        now = self.now()
        return any(r.started_at is not None and r.interrupted_at is None
                   and (r.ended_at is None or (r.segments and r.segments[-1][1] > now))
                   for r in self.replies.values())

    def last_activity(self) -> float:
        ts = [t for t, m in self.messages if m.get("t") != "pong"]
        ts += [r.segments[-1][1] for r in self.replies.values() if r.segments and r.interrupted_at is None]
        ts += [self.mic_time(p.end) for p in self.mixer.planted]
        return max(ts, default=0.0)

    def thinking(self) -> bool:
        """The last state the server sent was `thinking`, lately: a turn is on its way to a reply. A turn whose words
        were all dropped may never be followed by `listening`, so it counts for 8 s at most."""
        st = [(t, m.get("v")) for t, m in self.messages if m.get("t") == "state"]
        return bool(st) and st[-1][1] == "thinking" and self.now() - st[-1][0] < 8.0

    async def wait_until(self, pred: Callable[[], bool], deadline: float) -> bool:
        while self.now() < deadline:
            if pred():
                return True
            if self.close_code is not None:
                raise ConnectionError(f"closed with {self.close_code}")
            await asyncio.sleep(0.02)
        return pred()

    async def wait_reply(self, since: float, deadline: float):
        found: list = []

        def pred() -> bool:
            rs = [r for r in self.replies.values() if r.started_at is not None and r.started_at >= since
                  and r.first_audio_at is not None]
            if rs:
                found.append(min(rs, key=lambda r: r.started_at))
            return bool(rs)
        return found[-1] if await self.wait_until(pred, deadline) else None

    async def wait_quiet(self, after: float, quiet_s: float, deadline: float) -> bool:
        return await self.wait_until(lambda: self.now() >= after and not self.playing() and not self.thinking()
                                     and self.now() - self.last_activity() >= quiet_s, deadline)


def reply_json(r) -> dict:
    return {"id": r.id, "started_at": _r4(r.started_at), "first_audio_at": _r4(r.first_audio_at),
            "first_loud_at": _r4(r.first_loud_at), "ended_at": _r4(r.ended_at),
            "interrupted_at": _r4(r.interrupted_at), "played_ms": None if r.played_ms is None else round(r.played_ms),
            "audio_s": round(r.audio_s, 3), "segments": [[round(s, 4), round(e, 4)] for s, e in r.segments],
            "text": "".join(r.text), "sentences": [s.strip() for s in r.text if s.strip()]}


def _r4(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def replay_clip(r, seconds: float) -> np.ndarray:
    """The last `seconds` of what the client played of reply r, at 16 kHz: what something outside the server played
    again in an early live test."""
    x = pcm16_to_float(r.pcm)
    if r.played_ms is not None:
        x = x[:int(round(r.played_ms * SPK_RATE / 1000))]
    return resample_24k_to_16k(x[-int(seconds * SPK_RATE):])


async def run_scenario(url: str, sc: Scenario, clips: Clips, *, config: str, device: str,
                       quiet_s: float = QUIET_S, settle_s: float = SETTLE_S, lead_s: float = LEAD_S,
                       save_audio: Path | None = None, log: Callable[[str], None] = print) -> dict:
    """One scenario on its own connection; returns everything the client saw (metrics() reads it with the server's turn
    log and log). Never raises on a scenario that goes wrong: a missing reply or a timeout is a note in the record."""
    c = EchoClient(url, echo=sc.echo(), device=device, client="test")
    notes: list[str] = []
    main = None
    await c.connect()
    if (c.welcome or {}).get("state") != "listening":
        notes.append(f"welcome state {c.welcome.get('state')!r}: the server is held, nothing was measured")
    await c.start_mic()
    t_begin = c.now()
    try:
        await asyncio.sleep(lead_s)                   # the room before the person speaks
        req = c.plant(clips.request, label="request", kind="speech", text=clips.texts["request"])
        deadline = c.mic_time(req.end) + sc.max_s
        main = await c.wait_reply(since=c.mic_time(req.onset), deadline=deadline)
        if main is None:
            notes.append(f"no reply within {sc.max_s:.0f} s of the request")
        elif sc.kind == "bargein":
            await c.wait_until(lambda: c.now() >= main.first_audio_at + sc.bargein_after_s, deadline)
            if c.over(main):
                notes.append("the reply was over before the barge-in")
            b = c.plant(clips.bargein, label="barge-in", kind="speech", text=clips.texts["barge-in"])
            if not await c.wait_quiet(c.mic_time(b.end), quiet_s, deadline):
                notes.append("still busy at the cap")
        elif sc.kind in ("replay", "read_aloud"):
            await c.wait_until(lambda: c.over(main), deadline)
            end = c.playout_end(main) or c.now()
            if sc.kind == "replay":
                await c.wait_until(lambda: c.now() >= end + sc.replay_after_s, deadline)
                p = c.plant(replay_clip(main, sc.replay_s), label="replay", kind="echo", text="".join(main.text),
                            gain_db=sc.replay_db)
            else:
                clip, text = clips.read_aloud, clips.texts["read-aloud"]
                if clip is None:                      # --llm real: a sentence of the reply as it was written
                    text = pick_sentence("".join(main.text))
                    clip = await asyncio.to_thread(say16, text, READER_VOICE)   # the mic keeps running meanwhile
                await c.wait_until(lambda: c.now() >= end + sc.read_after_s, deadline)
                p = c.plant(clip, label="read-aloud", kind="speech", text=text)
            if not await c.wait_quiet(c.mic_time(p.end), quiet_s, deadline):
                notes.append("still busy at the cap")
        else:
            await c.wait_until(lambda: c.over(main), deadline)
            if not await c.wait_quiet(c.playout_end(main) or c.now(), quiet_s, deadline):
                notes.append("still busy at the cap")
        # the turn log: an interrupted turn waits up to brain.heard_wait_s (1.5 s) for played_ms before it is written
        await asyncio.sleep(settle_s)
    except (ConnectionError, OSError) as e:
        notes.append(f"connection: {e}")
    finally:
        t_end = c.now()
        await c.stop_mic()
        await c.close()
    if c.mic_error:
        notes.append(f"the microphone stopped: {c.mic_error}")
    run = {"config": config, "scenario": asdict(sc), "echo": sc.echo().label(), "device": device,
           "session": (c.welcome or {}).get("session"), "wall0": c.wall0, "mic_t0": c.mic_t0,
           "t_begin": round(t_begin, 4), "t_end": round(t_end, 4), "frames_sent": c.frames_rendered,
           "messages": [[round(t, 4), m] for t, m in c.messages if m.get("t") != "pong"],
           "replies": [reply_json(r) for r in sorted(c.replies.values(), key=lambda r: r.started_at or 1e9)],
           "planted": [p.to_json(c.mic_t0) for p in c.mixer.planted], "main_reply": main.id if main else None,
           "notes": notes}
    if save_audio is not None:
        save_audio.mkdir(parents=True, exist_ok=True)
        n = c.frames_rendered * FRAME
        write_wav(save_audio / f"{config}-{sc.name}-mic.wav", c.mixer.render(0, n), MIC_RATE)
        write_wav(save_audio / f"{config}-{sc.name}-speaker.wav", c._speaker(0, n * 3 // 2), SPK_RATE)
    n_int = sum(m.get("t") == "interrupt" for _, m in run["messages"])
    tail = "; " + "; ".join(notes) if notes else ""
    log(f"{config} {sc.name}: {len(run['replies'])} replies, {n_int} interrupts{tail}")
    return run


def write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.round(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


# --------------------------------------------------------------------------------------------- the server's side

_LOG_T = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")


def read_turns(turn_dir: Path, session: str | None) -> list[dict]:
    """The turn log's records of one session (brain/TURN_LOG.md §1), in order."""
    out = []
    for p in sorted(Path(turn_dir).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("type") == "turn" and rec.get("session") == session:
                out.append(rec)
    return sorted(out, key=lambda r: (r.get("t_start") or "", r.get("turn") or 0))


def read_log(log_dir: Path, wall_from: float, wall_to: float) -> list[tuple[float, str]]:
    """The orchestrator log's lines between two wall-clock times (loguru's local, offset-less times; a line without a
    time continues the one before)."""
    out: list[tuple[float, str]] = []
    for p in sorted(Path(log_dir).glob("orchestrator-*.log")):
        if p.stat().st_mtime < wall_from - 1:
            continue
        t = None
        for line in p.read_text(errors="replace").splitlines():
            m = _LOG_T.match(line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
            if t is not None and wall_from <= t <= wall_to:
                out.append((t, line))
    return sorted(out, key=lambda x: x[0])


# What the guard and the turn taking say in the log (2026-10-05 wording). turn_start.py (53c27a4) logs "<strategy>:
# <action> (<why>) after holding <s> s ...": `hold` when a VAD start while the agent is busy waits for words, then
# `resume` (echo, a fragment, a backchannel: the reply carries on) or `start` (the turn starts, which interrupts); a
# start because the agent is idle or a yes/no is pending was never held. `speech with no words for 1.5 s` is the hold's
# own timeout (max_hold_s), the one a continuous echo with no transcript ends in.
_TS = r"local_voice\.turn_start.* - \S+: "
LOG_KINDS = {
    "dropped": re.compile(r"echo of the agent's own speech"),
    "passed": re.compile(r"matching the agent's own speech while it was idle"),
    "turn_start": re.compile(r"local_voice\.turn_start"),
    "held": re.compile(_TS + r"hold \("),
    "held_resumed": re.compile(_TS + r"resume \("),
    "held_started": re.compile(_TS + r"start \((?!the agent is idle\)|a spoken yes/no)"),
    "held_timed_out": re.compile(_TS + r"start \(speech with no words"),
    "paused": re.compile(r"barge-in .*: speech cue, reply paused"),
    "resumed": re.compile(r"barge-in .*: resumed after"),
    "confirmed": re.compile(r"barge-in .*: confirmed after"),
    "user_started": re.compile(r"User started speaking"),
}
LOG_KEEP = re.compile(r"local_voice\.(echo_guard|turn_start|bargein)|User started speaking|interrupted mid-run|"
                      r"barge-in mid-generation|silent run")


# --------------------------------------------------------------------------------------------- metrics


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return round(d[len(h)] / max(1, len(r)), 3)


def _epoch(iso: str | None) -> float | None:
    try:
        return datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError):
        return None


def classify(text: str, t: float, speech: list[dict], said_before: list[tuple[str, float]],
             last_sentences: list[str] | None = None) -> tuple[str, Any]:
    """A user turn is the planted speech it carries (its words align with >= 3 of the clip's, and it starts between
    5 s before the clip's onset, when echo may already have opened the turn, and 1 s after its end), else echo
    (best_match against the agent's speech before it, >= 3 words and >= 60% of the turn: echo_guard's rule; or a
    fragment, 1-2 words all among the words of its last 3 sentences: EchoGuard.fragment_is_echo's rule, since the
    reply pause cuts live echo to a word or two, and the echo of a reply's last word lands after the reply ends), else
    other. The read-aloud sentence is the agent's own words too: the time decides, and that scenario has no live
    echo; no planted speech is 1-2 words of the reply, so a fragment here is always echo."""
    for p in speech:
        if p["onset_t"] - 5.0 <= t <= p["end_t"] + 1.0:
            m = best_match(text, [(p["text"], p["onset_t"])])
            if m is not None and m.matched >= min(3, len(words(p["text"]))):
                return f"speech:{p['label']}", m
    m = best_match(text, said_before) if said_before else None
    if m is not None and m.is_echo():
        return "echo", m
    w = words(text)
    if 1 <= len(w) <= 2 and last_sentences and set(w) <= set(words(" ".join(last_sentences[-3:]))):
        return "echo", "fragment"
    return "other", m


def playout_end(r: dict) -> float | None:
    """When the client's player stopped playing reply r (its flush, or the end of its last segment)."""
    if r["interrupted_at"] is not None:
        return r["interrupted_at"]
    return r["segments"][-1][1] if r["segments"] else r["ended_at"]


def gaps(segments: list[list[float]], min_s: float = 0.03) -> list[float]:
    return [b[0] - a[1] for a, b in zip(segments, segments[1:]) if b[0] - a[1] >= min_s]


def silent_from(segments: list[list[float]], since: float) -> float | None:
    """local_voice.client.Reply.silent_from on recorded segments."""
    for s, e in segments:
        if s <= since <= e:
            return e
        if s > since:
            return since
    return since if segments else None


def metrics(run: dict, turns: list[dict], log: list[tuple[float, str]] | None = None) -> dict:
    """Everything the report compares, from one scenario's client record, its session's turn-log records and the
    orchestrator log's lines of its time window (wall clock). Times in the output are seconds on the client's clock or
    milliseconds after a planted onset."""
    wall0 = run["wall0"]
    replies = run["replies"]
    planted = run["planted"]
    speech = [p for p in planted if p["kind"] == "speech"]
    replay = next((p for p in planted if p["label"] == "replay"), None)
    main = next((r for r in replies if r["id"] == run.get("main_reply")), None)

    def before(t: float) -> list[dict]:
        return [r for r in replies if r["text"] and r["started_at"] is not None and r["started_at"] < t]

    user_turns = []
    for rec in turns:
        ts = (_epoch(rec.get("t_start")) or wall0) - wall0
        text = rec.get("user_text") or ""
        said = before(ts)
        sentences = [s for r in said for s in (r.get("sentences") or [r["text"]])]
        kind, m = classify(text, ts, speech, [(r["text"], r["started_at"]) for r in said], sentences)
        # where the turn began against the playout: during a reply, or this long after the last one stopped (live
        # echo of a reply's last words starts after the server's "Bot stopped speaking", when the agent is idle)
        heard = [r for r in replies if r["segments"]]
        during = any(r["segments"][0][0] <= ts <= (playout_end(r) or ts) for r in heard)
        ends = [e for e in (playout_end(r) for r in heard) if e is not None and e <= ts]
        user_turns.append({"turn": rec.get("turn"), "t": round(ts, 3), "text": text, "kind": kind,
                           "matched": m if isinstance(m, str) or m is None else f"{m.matched}/{m.heard_words}",
                           "interrupted": bool(rec.get("interrupted")), "reply_text": rec.get("reply_text") or "",
                           "during_reply": during,
                           "after_reply_s": None if during or not ends else round(ts - max(ends), 3),
                           "during_replay": bool(replay and replay["onset_t"] - 0.5 <= ts <= replay["end_t"] + 3.0),
                           **({"echo_mark": rec["echo"]} if "echo" in rec else {})})

    by_id = {r["id"]: r for r in replies}
    interrupts = []
    for t, msg in run["messages"]:
        if msg.get("t") != "interrupt":
            continue
        cut = by_id.get(msg.get("reply_id")) or {}
        heard_from = cut.get("first_audio_at") or cut.get("started_at")
        cause = "false"
        for p in speech:
            # speech can only cut a reply it overlapped: the reply to the request starts after it, so an echo barging
            # into that reply's first second is not the request's doing
            if p["onset_t"] <= t <= p["end_t"] + 2.0 and (heard_from is None or heard_from < p["end_t"]):
                cause = f"speech:{p['label']}"
        if cause == "false" and replay and replay["onset_t"] <= t <= replay["end_t"] + 3.0:
            cause = "replay"
        interrupts.append({"t": round(t, 3), "reply_id": msg.get("reply_id"), "cause": cause})

    finals = [{"t": round(t, 3), "text": m.get("text", "")} for t, m in run["messages"]
              if m.get("t") == "transcript" and m.get("final")]
    out: dict[str, Any] = {
        "user_turns": user_turns, "transcripts": finals, "interrupts": interrupts,
        "n_user_turns": len(user_turns), "n_echo_turns": sum(u["kind"] == "echo" for u in user_turns),
        # echo turns that began within 2 s of a reply's end: the tail of the reply, heard when the agent is idle
        "n_tail_echo_turns": sum(u["kind"] == "echo" and u["after_reply_s"] is not None and u["after_reply_s"] <= 2.0
                                 for u in user_turns),
        "n_other_turns": sum(u["kind"] == "other" for u in user_turns),
        "false_barge_ins": sum(not i["cause"].startswith("speech:") for i in interrupts),
        "replies": len(replies), "replies_cut": sum(r["interrupted_at"] is not None for r in replies),
        "notes": run.get("notes", []),
    }
    if main is not None:
        g = gaps(main["segments"])
        out["main_reply"] = {"id": main["id"], "audio_s": main["audio_s"],
                             "played_s": round((main["played_ms"] or 0) / 1000, 2),
                             "complete": main["interrupted_at"] is None and main["ended_at"] is not None,
                             "pauses": len(g), "paused_ms": round(1000 * sum(g)), "text": main["text"]}
    else:
        out["main_reply"] = None

    def turn_for(label: str) -> dict | None:
        return next((u for u in user_turns if u["kind"] == f"speech:{label}"), None)

    for p in speech:
        u = turn_for(p["label"])
        after = [r for r in replies if r["started_at"] is not None and r["started_at"] >= p["onset_t"]]
        entry: dict[str, Any] = {"onset_t": p["onset_t"], "turn": u is not None, "text": u["text"] if u else None,
                                 "wer": wer(p["text"], u["text"]) if u else None}
        if p["label"] == "request":
            entry["answered"] = main is not None
        elif p["label"] == "barge-in":
            first = next((i for i in interrupts if i["t"] >= p["onset_t"]), None)
            entry["interrupt_ms"] = None if first is None else round(1000 * (first["t"] - p["onset_t"]))
            playing = next((r for r in replies if r["segments"] and r["segments"][0][0] <= p["onset_t"]
                            and (r["interrupted_at"] is None or r["interrupted_at"] >= p["onset_t"])
                            and r["segments"][-1][1] >= p["onset_t"]), None)
            stop = silent_from(playing["segments"], p["onset_t"]) if playing else None
            if playing and playing["interrupted_at"] is not None and stop is not None:
                stop = min(stop, playing["interrupted_at"])
            entry["reply_playing"] = playing is not None
            entry["audio_stop_ms"] = None if stop is None else round(1000 * (stop - p["onset_t"]))
            entry["answered"] = any("rome" in r["text"].lower() for r in after)
        else:
            entry["answered"] = bool(u) and any(r["started_at"] >= u["t"] for r in after)
        out[p["label"]] = entry
    if replay is not None:
        after = [r for r in replies if r["started_at"] is not None and r["started_at"] >= replay["onset_t"]]
        out["replay"] = {"onset_t": replay["onset_t"], "end_t": replay["end_t"],
                         "turns": sum(u["during_replay"] for u in user_turns),
                         "echo_turns": sum(u["during_replay"] and u["kind"] == "echo" for u in user_turns),
                         "replies_after": len(after),
                         "replies_cut": sum(r["interrupted_at"] is not None for r in after),
                         "interrupts": sum(i["t"] >= replay["onset_t"] for i in interrupts)}
    lines = [(t - wall0, s) for t, s in (log or []) if run["t_begin"] - 0.5 <= t - wall0 <= run["t_end"] + 0.5]
    out["log"] = {k: sum(bool(rx.search(s)) for _, s in lines) for k, rx in LOG_KINDS.items()}
    out["log_lines"] = [f"{t:8.3f} {s}" for t, s in lines if LOG_KEEP.search(s)][:200]
    return out


# --------------------------------------------------------------------------------------------- the report


def _yes(v: Any) -> str:
    return "–" if v is None else ("yes" if v else "no")


def cell(row: dict | None, fn: Callable[[dict], Any]) -> str:
    if row is None:
        return "–"
    try:
        v = fn(row["metrics"])
    except (KeyError, TypeError):
        return "–"
    return "–" if v is None else str(v)


def _speech_cell(m: dict) -> str | None:
    if "barge-in" in m:
        b = m["barge-in"]
        ms = "no interrupt" if b["interrupt_ms"] is None else f"interrupt {b['interrupt_ms']} ms"
        stop = "" if b.get("audio_stop_ms") is None else f", stop {b['audio_stop_ms']} ms"
        return f"{ms}{stop}, Rome {_yes(b['answered'])}"
    if "read-aloud" in m:
        r = m["read-aloud"]
        return f"turn {_yes(r['turn'])}, answered {_yes(r['answered'])}"
    if "replay" in m:
        r = m["replay"]
        return f"{r['replies_cut']} of {r['replies_after']} replies cut, {r['turns']} turns"
    req = m.get("request") or {}
    return f"request WER {req.get('wer')}, answered {_yes(req.get('answered'))}"


def _reply_cell(m: dict) -> str | None:
    mr = m.get("main_reply")
    if not mr:
        return "no reply"
    return f"{_yes(mr['complete'])} ({mr['pauses']} pauses, {mr['paused_ms']} ms)"


def _turns_cell(m: dict) -> str:
    tail = m.get("n_tail_echo_turns") or 0
    return f"{m['n_echo_turns']}/{m['n_user_turns']}" + (f" ({tail} at a reply's end)" if tail else "")


COLUMNS: list[tuple[str, Callable[[dict], Any]]] = [
    ("echo turns / user turns", _turns_cell),
    ("false barge-ins", lambda m: m["false_barge_ins"]),
    ("reply played to the end", _reply_cell),
    ("the planted speech", _speech_cell),
    ("log: dropped / passed / held: resumed, started / paused",
     lambda m: "{dropped} / {passed} / {held_resumed}, {held_started} / {paused}".format(**m["log"])),
]


def report_md(rows: list[dict], header: dict | None = None) -> str:
    """report.md: one before/after table (each cell `off → on`), then each scenario's turns, interrupts and log
    lines."""
    header = header or {}
    by = {(r["config"], r["scenario"]["name"]): r for r in rows}
    names = list(dict.fromkeys(r["scenario"]["name"] for r in rows))
    configs = [c for c in CONFIGS if any(r["config"] == c for r in rows)]
    out = [f"# Echo bench {header.get('stamp', '')}".rstrip(), ""]
    for k in ("llm", "server", "git", "started", "gpu_clear", "load1"):
        if header.get(k) is not None:
            out.append(f"- {k}: {header[k]}")
    for c in configs:
        seen = header.get("server_echo", {}).get(c)
        out.append(f"- config `{c}`: echo {json.dumps(CONFIGS[c])}" + (f"; the server read {json.dumps(seen)}"
                                                                       if seen is not None else ""))
    out += ["", f"Each cell: {' → '.join(configs)}. Echo turns: user turns whose words are the agent's earlier "
            "speech (echo_guard.best_match). False barge-ins: `interrupt` with no planted speech cutting a reply it "
            "overlapped. Pauses: gaps in the simulated playout (the reply pause). Log counts: the guard's dropped and "
            "passed-while-idle lines, turn_start.py's held VAD starts that resumed the reply or started the turn, the "
            "reply pause's pauses.", ""]
    out.append("| scenario | live echo | " + " | ".join(n for n, _ in COLUMNS) + " |")
    out.append("|" + "---|" * (len(COLUMNS) + 2))
    for name in names:
        any_row = next(by[(c, name)] for c in configs if (c, name) in by)
        cells = [" → ".join(cell(by.get((c, name)), fn) for c in configs) for _, fn in COLUMNS]
        out.append(f"| {name} | {any_row['echo']} | " + " | ".join(cells) + " |")
    out.append("")
    for name in names:
        out.append(f"## {name}")
        for c in configs:
            r = by.get((c, name))
            if r is None:
                continue
            m = r["metrics"]
            out.append(f"### {c}")
            for n in m.get("notes") or []:
                out.append(f"- note: {n}")
            for u in m["user_turns"]:
                where = (" during a reply" if u.get("during_reply") else "" if u.get("after_reply_s") is None
                         else f" {u['after_reply_s']:.1f} s after a reply")
                out.append(f"- turn {u['turn']} at {u['t']:.1f} s{where}, {u['kind']} ({u['matched']})"
                           f"{', interrupted' if u['interrupted'] else ''}: {u['text']!r}")
            for i in m["interrupts"]:
                out.append(f"- interrupt at {i['t']:.2f} s ({i['cause']})")
            if m["log_lines"]:
                out.append("")
                out.append("```")
                out += [s[:220] for s in m["log_lines"][:40]]
                out.append("```")
            out.append("")
    return "\n".join(out) + "\n"


def write_report(run_dir: Path) -> Path:
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    hp = run_dir / "run.json"
    header = json.loads(hp.read_text()) if hp.exists() else {}
    out = run_dir / "report.md"
    out.write_text(report_md(rows, header))
    return out


# --------------------------------------------------------------------------------------------- the stub LLM and config


def route_reply(prompt: str) -> dict:
    """The stub's answer, by the person's own words: the prompt's last paragraph (the orchestrator's notes, such as
    pi_rpc.interrupted_note, come first and end with a blank line). Streamed 2 words per delta every 20 ms (the 84-word
    story in under a second, like qwen38), so the reply is still being written when its audio starts."""
    said = (prompt or "").strip().split("\n\n")[-1].lower()
    if "capital" in said or "italy" in said:
        text = BARGEIN_REPLY
    elif "story" in said:
        text = LONG_REPLY
    else:
        text = DEFAULT_REPLY
    return {"text": text, "chunk_words": 2, "delay_ms": 20}


def last_user_text(body: dict) -> str:
    msgs = [m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
    if not msgs:
        return ""
    c = msgs[-1].get("content")
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def start_stub(port: int, log_dir: Path, route: Callable[[str], dict] = route_reply):
    """tests/stub_llm.py's StubLLM with each chat request's script chosen by route() from the request itself (the
    stub's own queue only goes in order, and a false turn would shift it). One request is answered at a time."""
    sys.path.insert(0, str(ORCH / "tests"))
    from stub_llm import StubLLM, _handler, _Server

    stub = StubLLM(port, log_dir)
    base = _handler(stub.state)
    lock = threading.Lock()

    class Routed(base):
        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return super().do_POST()
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            orig, self.rfile = self.rfile, io.BytesIO(raw)    # the stub reads the body again
            try:
                with lock:
                    stub.script([route(last_user_text(body))])
                    super().do_POST()
            finally:
                self.rfile = orig                              # keep-alive: the next request comes on the socket

    stub.server = _Server(("127.0.0.1", port), Routed, stub.state)
    stub.port = stub.server.server_address[1]
    threading.Thread(target=stub.server.serve_forever, daemon=True, name="echo-bench-stub").start()
    stub.state.log(f"echo-bench routed stub on 127.0.0.1:{stub.port}")
    return stub


def stub_models(src: Path, dest: Path, base_url: str) -> Path:
    """A `pi.models_source` whose model calls go to the stub: the configured models.json (only read) with every
    provider's baseUrl replaced. agent_dir.derive_agent_dir copies the providers spaces.yaml names from it, so Pi's own
    provider settings (compat flags, model ids, token caps) stay production's."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    for p in (d.get("providers") or {}).values():
        if isinstance(p, dict):
            p["baseUrl"] = base_url
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(d, indent=1))
    return dest


def write_local_config(scratch: Path, config: str, *, models: Path | None = None) -> Path:
    """DIR/config.local.yaml for one config: load_config merges it over DIR/config.yaml, which make_scratch rewrites at
    every `./run.sh --scratch DIR` start; this file it never touches."""
    data: dict[str, Any] = {"echo": dict(CONFIGS[config]), "server": {"browser": {"enabled": False}}}
    if models is not None:
        data["pi"] = {"models_source": str(models)}
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "config.local.yaml"
    out.write_text(f"# tools/echo_bench.py, config {config!r} ({time.strftime('%Y-%m-%d %H:%M:%S')}): merged by "
                   "load_config over config.yaml, which ./run.sh --scratch rewrites; the browser page is off (the "
                   "bench is protocol v1 only) and port 7861 carries the stub LLM\n"
                   + yaml.safe_dump(data, sort_keys=False))
    return out


def server_echo(scratch: Path) -> dict | None:
    """The echo settings a server started on this scratch dir reads (config.yaml with config.local.yaml)."""
    from local_voice.config import ConfigError, load_config
    p = scratch / "config.yaml"
    try:
        return load_config(p).echo if p.exists() else None
    except ConfigError as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------------------------- the GPU and the server


def gpu_check(cmd: str | None) -> tuple[bool, str]:
    if not cmd:
        return True, "no check"
    r = subprocess.run([cmd], capture_output=True, text=True, timeout=30)
    return r.returncode == 0, (r.stdout or r.stderr).strip()


def gpu_still_ours(cmd: str | None) -> str | None:
    """None while no GPU job has started and the gate is open (tests/e2e/e2e_fixtures.py's rule: the LLM-idle field
    cannot apply mid-run)."""
    if not cmd:
        return None
    _, line = gpu_check(cmd)
    jobs = re.search(r"gpu_jobs=(\d+)", line)
    hold = re.search(r"hold=(\w+)", line)
    if jobs and jobs.group(1) != "0":
        return f"a GPU job started: {line}"
    if hold and hold.group(1) not in ("open", "absent"):
        return f"the hold gate is {hold.group(1)}: {line}"
    return None


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class ScratchServer:
    """`./run.sh --scratch DIR --port P --browser-port B` as a child process, its output in `log_path`."""

    def __init__(self, scratch: Path, log_path: Path, *, port: int = PORT, browser_port: int = BROWSER_PORT):
        self.scratch, self.log_path, self.port, self.browser_port = scratch, log_path, port, browser_port
        self.proc: subprocess.Popen | None = None

    def start(self, timeout_s: float = 600.0) -> dict:
        if port_open(self.port):
            raise RuntimeError(f"something already listens on 127.0.0.1:{self.port}; stop it, or use --attach")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log_path, "ab")
        self.proc = subprocess.Popen([str(RUN_SH), "--scratch", str(self.scratch), "--port", str(self.port),
                                      "--browser-port", str(self.browser_port)], cwd=ORCH, stdout=fh,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout_s       # the first start clones the kb and the spaces' roots
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                why = " (75: the hold gate is held or draining; no model may load)" if code == 75 else ""
                raise RuntimeError(f"run.sh exited with {code}{why}; see {self.log_path}")
            with contextlib.suppress(OSError, ValueError):
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/status", timeout=2) as r:
                    return json.loads(r.read())
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"the server did not answer /v1/status within {timeout_s:.0f} s; see {self.log_path}")

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            with contextlib.suppress(ProcessLookupError):
                p.send_signal(sig)               # SIGINT first: uvicorn and amain stop the children and models cleanly
            try:
                p.wait(wait)
                return
            except subprocess.TimeoutExpired:
                continue


# --------------------------------------------------------------------------------------------- the run


def git_head() -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def result_row(run: dict, turn_dir: Path, log_dir: Path) -> dict:
    """One results.jsonl line: the scenario, its metrics, and the client's record (for a later look)."""
    turns = read_turns(turn_dir, run["session"])
    lines = read_log(log_dir, run["wall0"] + run["t_begin"] - 1, run["wall0"] + run["t_end"] + 1)
    return {"config": run["config"], "scenario": run["scenario"], "echo": run["echo"],
            "metrics": metrics(run, turns, lines), "run": {k: v for k, v in run.items() if k != "scenario"}}


async def run_config(url: str, config: str, scenarios: list[Scenario], clips: Clips, *, scratch: Path, out_dir: Path,
                     stamp: str, gpu_clear: str | None, save_audio: bool, log=print) -> tuple[list[dict], str | None]:
    """Every scenario of one config against the server at `url`; (rows, why it stopped early or None)."""
    rows = []
    for sc in scenarios:
        why = gpu_still_ours(gpu_clear)
        if why:
            log(f"stopping before {config} {sc.name}: {why}")
            return rows, why
        try:
            run = await run_scenario(url, sc, clips, config=config, device=f"echo-{config}-{sc.name}-{stamp}",
                                     save_audio=out_dir / "audio" if save_audio else None, log=log)
        except Exception as e:  # noqa: BLE001 - refused, a handshake error, or a bug: stop with the server stopped
            log(f"stopping at {config} {sc.name}: {type(e).__name__}: {e}")
            return rows, f"{config} {sc.name}: {type(e).__name__}: {e}"
        row = result_row(run, scratch / "turns", scratch / "state" / "logs")
        with open(out_dir / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
    return rows, None


def cmd_run(a: argparse.Namespace) -> int:
    pairs = plan(a.configs.split(","), a.only.split(",") if a.only else None)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(a.out or OUT_ROOT / stamp).expanduser().resolve()
    scratch = Path(a.scratch).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "bench.log", "a")

    def log(s: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    ok, line = gpu_check(a.gpu_clear)
    log(f"gpu_clear: {line}")
    if not ok:
        log("refusing to start: gpu_clear.sh did not exit 0 (no model load without it)")
        return 75
    configs = list(dict.fromkeys(c for c, _ in pairs))
    if a.attach and len(configs) != 1:
        log("--attach runs one config, the one the running server was started with")
        return 2
    header = {"stamp": stamp, "llm": a.llm, "git": git_head(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gpu_clear": line, "load1": round(os.getloadavg()[0], 2), "scratch": str(scratch),
              "server": f"./run.sh --scratch {scratch} --port {a.port} --browser-port {a.browser_port}"
                        + (" (attached)" if a.attach else ""),
              "scenarios": [asdict(s) for _, s in pairs], "estimate": estimate(pairs), "server_echo": {}}
    (out_dir / "run.json").write_text(json.dumps(header, indent=1))
    log(f"plan: {len(pairs)} scenario runs, about {header['estimate']['expected_min']} min "
        f"(cap {header['estimate']['worst_case_min']} min); results in {out_dir}")
    clips = render_clips(read_aloud=a.llm == "stub")
    stub = None
    models = None
    if a.llm == "stub":
        if port_open(a.stub_port):
            log(f"refusing to start: something already listens on 127.0.0.1:{a.stub_port} (the stub's port)")
            return 2
        stub = start_stub(a.stub_port, out_dir / "stub")
        from local_voice.config import load_config
        models = stub_models(load_config().pi_models_source, scratch / "stub-models.json", stub.base_url)
    rows: list[dict] = []
    url = f"ws://127.0.0.1:{a.port}/v1/voice"
    try:
        for config in configs:
            scenarios = [s for c, s in pairs if c == config]
            server = None
            if not a.attach:
                write_local_config(scratch, config, models=models)
                server = ScratchServer(scratch, out_dir / f"server-{config}.log", port=a.port,
                                       browser_port=a.browser_port)
                log(f"{config}: starting the scratch server")
                try:
                    server.start()
                except RuntimeError as e:             # 75 (a hold), a port in use, no answer in time
                    log(f"{config}: {e}")
                    header["stopped"] = f"{config}: {e}"
                    (out_dir / "run.json").write_text(json.dumps(header, indent=1))
                    break
            seen = server_echo(scratch)
            header["server_echo"][config] = seen
            (out_dir / "run.json").write_text(json.dumps(header, indent=1))
            want_guard = bool(CONFIGS[config]["guard"])
            if seen is not None and bool((seen or {}).get("guard")) != want_guard:
                log(f"{config}: the server's echo config {seen} is not {CONFIGS[config]}; stopping")
                if server:
                    server.stop()
                break
            try:
                done, stopped = asyncio.run(run_config(url, config, scenarios, clips, scratch=scratch, out_dir=out_dir,
                                                       stamp=stamp, gpu_clear=a.gpu_clear, save_audio=a.save_audio,
                                                       log=log))
                rows += done
            finally:
                if server:
                    log(f"{config}: stopping the scratch server")
                    server.stop()
            if stopped:
                header["stopped"] = stopped
                (out_dir / "run.json").write_text(json.dumps(header, indent=1))
                break                                   # a GPU job, a hold, or a dead server: no next config
    finally:
        if stub is not None:
            stub.stop()
    if rows:
        log(f"report: {write_report(out_dir)}")
    return 0 if rows else 1


def cmd_plan(a: argparse.Namespace) -> int:
    pairs = plan(a.configs.split(","), a.only.split(",") if a.only else None)
    if a.json:
        print(json.dumps({"configs": {c: CONFIGS[c] for c in dict.fromkeys(c for c, _ in pairs)},
                          "scenarios": [asdict(s) for s in dict.fromkeys(s for _, s in pairs)],
                          "estimate": estimate(pairs)}, indent=1))
        return 0
    for c in dict.fromkeys(c for c, _ in pairs):
        print(f"config {c}: echo {json.dumps(CONFIGS[c])}")
    for s in dict.fromkeys(s for _, s in pairs):
        extra = {"bargein": f", {BARGEIN!r} {s.bargein_after_s} s into the reply",
                 "replay": f", the reply's last {s.replay_s:.0f} s at {s.replay_db:+.0f} dB "
                           f"{s.replay_after_s:.0f} s after it",
                 "read_aloud": f", {READ_ALOUD[:40]!r}... in {READER_VOICE}'s voice {s.read_after_s:.0f} s after it"}
        print(f"  {s.name:20s} live echo {s.echo().label():22s} about {expected_s(s):5.1f} s, cap {s.max_s:.0f} s"
              f"{extra.get(s.kind, '')}")
    e = estimate(pairs)
    print(f"{e['scenarios']} scenario runs over {e['configs']} server starts: about {e['expected_min']} min of GPU "
          f"time (at most {e['worst_case_min']} min if every scenario runs to its cap); the first start also clones "
          "the kb and the spaces' roots into the scratch dir")
    return 0


def cmd_prepare(a: argparse.Namespace) -> int:
    scratch = Path(a.scratch).expanduser().resolve()
    models = None
    if a.llm == "stub":
        from local_voice.config import load_config
        models = stub_models(load_config().pi_models_source, scratch / "stub-models.json",
                             f"http://127.0.0.1:{a.stub_port}/v1")
    p = write_local_config(scratch, a.config, models=models)
    print(f"wrote {p}\nthen: ../measure/bench/gpu_clear.sh && ./run.sh --scratch {scratch} --port {a.port} "
          f"--browser-port {a.browser_port}\n"
          f"and:  .venv/bin/python tools/echo_bench.py run --attach --configs {a.config}"
          + (f" (it starts the stub LLM on 127.0.0.1:{a.stub_port})" if models else ""))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    print(write_report(Path(a.run_dir).expanduser().resolve()))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--scratch", default=str(SCRATCH))
        p.add_argument("--port", type=int, default=PORT)
        p.add_argument("--browser-port", type=int, default=BROWSER_PORT)
        p.add_argument("--stub-port", type=int, default=STUB_PORT)
        p.add_argument("--llm", choices=("stub", "real"), default="stub")
        if name == "prepare":
            p.add_argument("--config", choices=list(CONFIGS), required=True)
        else:
            p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                           help=f"comma-separated, of {', '.join(CONFIGS)}")
            p.add_argument("--only", default=None, help="scenario names, comma-separated")
        if name == "plan":
            p.add_argument("--json", action="store_true")
        if name == "run":
            p.add_argument("--out", default=None, help="the run's folder (default ../state/echo-bench/<stamp>)")
            p.add_argument("--attach", action="store_true", help="use a running server (started after `prepare`)")
            p.add_argument("--gpu-clear", default=GPU_CLEAR, help="the check that must exit 0 first")
            p.add_argument("--save-audio", action="store_true", help="write each scenario's mic and speaker WAVs")
    p = sub.add_parser("report")
    p.add_argument("run_dir")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "prepare": cmd_prepare, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
