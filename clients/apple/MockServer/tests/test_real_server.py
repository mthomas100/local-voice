"""Protocol v1 against the real orchestrator: the Swift client core (lvclient) over a real WebSocket to the server that
runs the models. Marker `real`; plain `pytest` skips it.

Run it only when nothing else is using the server. A real turn runs speech and language models on the GPU, and qwen38 keeps one
conversation warm at a time, so a turn from here would collide with the orchestrator's own measurements.

    LV_REAL_URL=ws://127.0.0.1:8770/v1/voice ../build.sh real    # the running server (cd orchestrator && ./run.sh)
    LV_REAL_URL=mock ../build.sh real                             # a dry run of this harness against the mock

GPU rule: against a real server nothing runs unless measure/bench/gpu_clear.sh exits 0 when the
module starts; before each test its gpu_jobs and hold fields are checked again (its LLM-idle field cannot be, since
these tests are LLM traffic), as the orchestrator's own e2e does.

Machine ground truth only: the questions are `say -v Samantha` recordings (the voice the orchestrator's e2e streams),
so each transcript is scored against the known words, and each question has a one-word answer the reply must contain.
Every turn is timed on the client, from the JSONL log lvclient writes:
- push-to-talk: release (the key or button let go) to `audio_start`, and to the first reply audio going to the player;
- open microphone: end of speech (the clip's last audible sample leaving the synthetic microphone) to the same two.
Between `audio_start` and the player the client adds its pre-roll: 60 ms here, the Mac app's setting on loopback.

What these turns leave on the server: they are real turns in the active space's Pi session and in its turn log, from
client `test`, device `apple-real`, so they can be told apart from a person's own turns.

Evidence: MockServer/runs/<stamp>/<test>/ (client JSONL, the reply audio as played) and
MockServer/runs/<stamp>/real-results.json (one row per turn, the server's /v1/status after each).
"""

from __future__ import annotations

import array
import json
import os
import pathlib
import re
import statistics
import subprocess
import time
import urllib.request
import wave
from dataclasses import dataclass

import pytest

from conftest import APPLE, RUNS

REAL = os.environ.get("LV_REAL_URL", "")
DRY = REAL == "mock"
REPO = APPLE.parents[1]
GPU_CLEAR = REPO / "measure" / "bench" / "gpu_clear.sh"
DEVICE = "apple-real"
PREROLL_MS = 60  # the Mac app's pre-roll on loopback (LocalVoiceUI AppSettings)

pytestmark = [pytest.mark.real,
              pytest.mark.skipif(not REAL, reason="set LV_REAL_URL to the server's ws:// URL, or to mock for a dry run")]

# The orchestrator's e2e questions (orchestrator/tests/e2e/test_turn.py): one-word answers, and no comma inside a
# request (Smart Turn ended a turn at a comma's pause in its first run).
FRANCE = "What is the capital of France?"
MOO = "Which animal says moo?"
COLD = "What is the opposite of cold?"
STORY = "Tell me a long story about a lighthouse keeper and his cat with lots of detail."
STOP_ITALY = "Stop. What is the capital of Italy?"
ITALY = "What is the capital of Italy?"

LONG_REPLY = ("The lighthouse stood at the end of a long stone pier, and every evening the keeper climbed its spiral "
              "stairs to light the lamp. His cat followed him all the way up, step by step, and sat by the window "
              "while the beam swept across the dark water. Ships far out at sea saw the light and knew where the rocks "
              "were. In the morning the keeper polished the glass, wound the clockwork, and wrote the weather in his "
              "logbook before he slept.")

# A dry run's mock answers each user turn of a test with these, in order (mock_server.py --turns).
DRY_TURNS = {
    "test_push_to_talk_turn": [{"transcript": FRANCE, "reply": "Paris is the capital of France."}],
    "test_hands_free_turn": [{"transcript": MOO, "reply": "A cow says moo."}],
    "test_typed_turn": [{"reply": "The opposite of cold is hot."}],
    "test_barge_in_during_a_long_answer": [{"transcript": STORY, "reply": LONG_REPLY},
                                           {"transcript": STOP_ITALY, "reply": "Rome is the capital of Italy."}],
    "test_push_to_talk_barge_in": [{"transcript": STORY, "reply": LONG_REPLY},
                                   {"transcript": ITALY, "reply": "Rome is the capital of Italy."}],
    "test_status_endpoint": [],
}

RESULTS: list[dict] = []


def record(test: str, **data):
    RESULTS.append({"test": test, "target": "mock (dry run)" if DRY else REAL, "at": time.strftime("%H:%M:%S"), **data})
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "real-results.json").write_text(json.dumps(RESULTS, indent=1))


def ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1000)


# MARK: speech


@dataclass
class Clip:
    text: str
    path: pathlib.Path
    seconds: float
    speech_start: float
    speech_end: float

    @property
    def spec(self) -> str:
        return f"file:{self.path}"


def render(text: str, folder: pathlib.Path) -> Clip:
    """`say -v Samantha` to AIFF (what lvclient plays into its synthetic microphone), and where its speech starts and
    ends: the first and last sample above -40 dBFS at 16 kHz, as the orchestrator's speech_end_s finds them."""
    name = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]
    aiff, wav = folder / f"{name}.aiff", folder / f"{name}.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
    with wave.open(str(wav)) as w:
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
    loud = [i for i, x in enumerate(a) if abs(x) > 327]
    return Clip(text, aiff, len(a) / 16000, loud[0] / 16000, (loud[-1] + 1) / 16000)


@pytest.fixture(scope="module")
def speech() -> dict[str, Clip]:
    folder = RUNS / "real-speech"
    folder.mkdir(parents=True, exist_ok=True)
    return {t: render(t, folder) for t in (FRANCE, MOO, STORY, STOP_ITALY, ITALY)}


# MARK: the GPU and the target


def gpu_line() -> tuple[int, str]:
    r = subprocess.run([str(GPU_CLEAR)], capture_output=True, text=True, timeout=30)
    return r.returncode, (r.stdout or r.stderr).strip()


def gpu_log(line: str):
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "gpu.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


@pytest.fixture(scope="module")
def gpu_go() -> str:
    """Against a real server the GPU must be clear when the module starts (gpu_clear.sh exits 0)."""
    if DRY:
        return "dry run: the mock runs no models"
    code, line = gpu_line()
    gpu_log(f"start: {line}")
    if code != 0:
        pytest.skip(f"GPU not clear, nothing run: {line}")
    return line


def gpu_still_ours() -> str | None:
    """None while no GPU job or hold has appeared since the module started; otherwise why to stop."""
    _, line = gpu_line()
    gpu_log(line)
    jobs = re.search(r"gpu_jobs=(\d+)", line)
    hold = re.search(r"hold=(\w+)", line)
    if jobs and jobs.group(1) != "0":
        return f"a GPU job started: {line}"
    if hold and hold.group(1) not in ("open", "absent"):
        return f"the hold gate is {hold.group(1)}: {line}"
    return None


@dataclass
class Target:
    url: str
    status_url: str


def status_url(ws_url: str) -> str:
    scheme, rest = ws_url.split("://", 1)
    return f"{'https' if scheme == 'wss' else 'http'}://{rest.split('/', 1)[0]}/v1/status"


@pytest.fixture
def target(request, gpu_go, mock, run_dir) -> Target:
    if DRY:
        turns = run_dir / "turns.json"
        turns.write_text(json.dumps(DRY_TURNS[request.node.originalname]))
        # 800 ms of silence ends a mock turn: the pause after "Stop." must not split the question.
        m = mock("--reply-voice", "say", "--turns", str(turns), "--think-ms", "300", "--vad-end-ms", "800")
        return Target(m.url, status_url(m.url))
    why = gpu_still_ours()
    if why:
        pytest.skip(f"stopping: {why}")
    return Target(REAL, status_url(REAL))


def get_status(t: Target) -> dict:
    with urllib.request.urlopen(t.status_url, timeout=5) as r:
        return json.loads(r.read())


def talk(client, t: Target, script: str, *, mic: str):
    c = client(t.url, script, mic=mic, device=DEVICE, timeout=240, extra=["--preroll-ms", str(PREROLL_MS)])
    assert c.proc.returncode == 0, f"{c.proc.stdout}\n{c.proc.stderr}"
    return c


# MARK: reading the client's log


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = [[i + j if i * j == 0 else 0 for j in range(len(h) + 1)] for i in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return round(d[len(r)][len(h)] / max(1, len(r)), 3)


def first(records, since: float, pred):
    return next((r for r in records if r["t"] >= since and pred(r)), None)


def recv(t: str):
    return lambda r: r["event"] == "recv" and r["msg"]["t"] == t


def step_time(records, step: str) -> float:
    return next(r["t"] for r in records if r["event"] == "step" and r["step"] == step)


def releases(records) -> list[float]:
    return [r["t"] for r in records if r["event"] == "talk" and r["phase"] == "releasing"]


def clip_eos(records, clip: Clip) -> float:
    """When the clip's speech ended: its last sample left the microphone, less the clip's own trailing silence."""
    done = next(r["t"] for r in records if r["event"] == "clip" and r["name"] == clip.spec)
    return done - (clip.seconds - clip.speech_end)


def measure(records, *, said: str | None, since: float, heard_since: float | None = None) -> dict:
    """The first reply whose `audio_start` came after `since`, as the client saw it; the final transcripts between
    `heard_since` (default `since`) and that reply are what the server heard."""
    start = first(records, since, recv("audio_start"))
    assert start, f"no reply audio after {since:.2f} s"
    rid = start["msg"]["reply_id"]
    playing = first(records, start["t"], lambda r: r["event"] == "playback" and r["kind"] == "started"
                    and r.get("reply") == rid)
    heard_from = since if heard_since is None else heard_since
    finals = [r["msg"]["text"] for r in records if heard_from <= r["t"] <= start["t"] and r["event"] == "recv"
              and r["msg"]["t"] == "transcript" and r["msg"].get("final")]
    deltas = [r["msg"]["delta"] for r in records if r["event"] == "recv" and r["msg"]["t"] == "reply_text"
              and r["msg"]["reply_id"] == rid]
    audio = next((r for r in records if r["event"] == "reply_audio" and r["reply"] == rid), None)
    played = next((r["msg"]["ms"] for r in records if r["event"] == "sent" and r["msg"]["t"] == "played_ms"
                   and r["msg"]["reply_id"] == rid), None)
    underruns = [r for r in records if r["event"] == "playback" and r["kind"] == "underrun" and r.get("reply") == rid]
    lat = next((r for r in records if r["event"] == "latency" and r.get("reply") == rid), None)
    end = first(records, start["t"], lambda r: r["event"] == "recv" and r["msg"]["t"] == "end_of_turn"
                and r["msg"].get("reply_id") == rid)
    transcript = " ".join(finals)
    return {
        "said": said, "transcript": transcript,
        "transcript_wer": wer(said, transcript) if said and finals else None,
        "reply_id": rid, "reply_text": " ".join(" ".join(deltas).split()),
        "reply_audio_ms": audio["bytes"] // 48 if audio else None,  # 24 kHz PCM16: 48 bytes a millisecond
        "played_ms": played, "underruns": len(underruns),
        # Each time the player ran dry: for how long, and whether the reply's audio came back (late audio grows the
        # pre-roll) or the reply was cut or ended first (the server's pause, not the network; PlaybackController).
        "underrun_gaps": [{"gap_ms": r.get("gap_ms"), "resumed": r.get("resumed")} for r in underruns],
        # What was buffered when the reply began to play: the pre-roll rounded up to whole 40 ms chunks.
        "preroll_ms_at_start": playing.get("preroll_ms") if playing else None,
        "stop_to_playback_ms": lat["stop_to_playback_ms"] if lat else None,
        "tools": [r["msg"]["name"] for r in records if since <= r["t"] <= (end["t"] if end else 1e9)
                  and r["event"] == "recv" and r["msg"]["t"] == "tool" and r["msg"]["phase"] == "start"],
        "audio_start_t": start["t"], "playback_t": playing["t"] if playing else None,
    }


def timed(row: dict, origin: float | None, name: str) -> dict:
    """Adds `<name>_to_audio_start_ms` and `<name>_to_playback_ms` from `origin` (log seconds)."""
    row[f"{name}_to_audio_start_ms"] = ms(row["audio_start_t"] - origin) if origin is not None else None
    row[f"{name}_to_playback_ms"] = ms(row["playback_t"] - origin) if origin is not None and row["playback_t"] else None
    return row


def last_sound(summary: dict, start: float, before: float | None) -> float | None:
    """The end of the last audible stretch of output that began at `start` or later and ended before `before`."""
    ends = [e for s, e in summary["audible"] if s >= start - 0.05 and e <= (before if before is not None else 1e9)]
    return max(ends) if ends else None


def sounding_after(cut: float, quiet: float | None) -> int:
    """How long the output kept sounding after a cut: 0 when it was already quiet (the cut fell in a pause between
    words, or the server had stopped sending), which a tone-based check never sees."""
    return max(0, ms(quiet - cut)) if quiet is not None else 0


def interrupts_outside_replies(records) -> list[dict]:
    """PROTOCOL.md: `interrupt` comes only while a reply is active, between its `audio_start` and the client's
    `played_ms` for it. Returns every one that broke that."""
    active: set[str] = set()
    bad = []
    for r in records:
        if r["event"] == "recv" and r["msg"]["t"] == "audio_start":
            active.add(r["msg"]["reply_id"])
        elif r["event"] == "sent" and r["msg"]["t"] == "played_ms":
            active.discard(r["msg"]["reply_id"])
        elif r["event"] == "recv" and r["msg"]["t"] == "interrupt":
            rid = r["msg"].get("reply_id")
            if (rid is None and not active) or (rid is not None and rid not in active):
                bad.append(r)
    return bad


def connected(c) -> dict:
    """What the server said on connect: its welcome, and the `space` it sends next (the orchestrator does)."""
    welcome = c.recv("welcome")[0]
    space = c.recv("space")
    return {"welcome": welcome, "space_after_welcome": space[0] if space else None}


def check_answer(row: dict, expect: str, *, max_wer: float = 0.25):
    """The transcript is close to what was said, and the reply has the answer ("cow" in "Cows say moo.")."""
    if row["said"] is not None:
        assert row["transcript_wer"] is not None and row["transcript_wer"] <= max_wer, (row["said"], row["transcript"])
    assert any(w.startswith(expect) for w in words(row["reply_text"])), f"no {expect!r} in: {row['reply_text']!r}"
    assert row["reply_audio_ms"] and row["reply_audio_ms"] >= 300, row


def check_played_to_the_end(row: dict):
    """A reply nobody interrupted: `played_ms` is all the audio that came, within one 40 ms chunk and a buffer."""
    assert row["played_ms"] is not None and abs(row["played_ms"] - row["reply_audio_ms"]) <= 120, row


# MARK: tests


def test_push_to_talk_turn(target, client, speech):
    """Hold, ask, let go: the transcript round-trips, the answer is spoken and played to the end."""
    q = speech[FRANCE]
    c = talk(client, target, f"connect; wait:ready; sleep:500; press; say-wait:{q.spec}; release; "
                             "wait:end_of_turn:90000; wait:playback-finished:30000; sleep:300", mic="ptt")
    recs = c.records
    release = releases(recs)[0]
    row = timed(timed(measure(recs, said=FRANCE, since=release), release, "release"), clip_eos(recs, q), "eos")
    record("push-to-talk turn", kind="turn", mic="ptt", **row, **connected(c),
           server_last_turn=get_status(target).get("last_turn"))
    check_answer(row, "paris")
    check_played_to_the_end(row)
    assert 0 < row["release_to_audio_start_ms"] < 20_000, row
    assert not interrupts_outside_replies(recs)


def test_hands_free_turn(target, client, speech):
    """Open microphone: the server finds the end of the turn itself; the client's gate arms after the reply."""
    q = speech[MOO]
    c = talk(client, target, f"handsfree-on; wait:ready; sleep:1000; say-wait:{q.spec}; "
                             "wait:end_of_turn:90000; wait:playback-finished:30000; sleep:800", mic="vad")
    recs = c.records
    row = timed(measure(recs, said=MOO, since=step_time(recs, f"say-wait:{q.spec}")), clip_eos(recs, q), "eos")
    record("hands-free turn", kind="turn", mic="vad", **row, **connected(c),
           server_last_turn=get_status(target).get("last_turn"))
    check_answer(row, "cow")
    check_played_to_the_end(row)
    assert 0 < row["eos_to_audio_start_ms"] < 20_000, row
    assert any(r["event"] == "gate" and r["armed"] for r in recs), "the mic gate armed after the reply"
    assert not interrupts_outside_replies(recs)


def test_typed_turn(target, client):
    """Typed words are a turn too; the server sends no transcript for them (the apps show them themselves)."""
    c = talk(client, target, f"connect; wait:ready; text:{COLD}; wait:end_of_turn:90000; "
                             "wait:playback-finished:30000; sleep:300", mic="ptt")
    recs = c.records
    sent = first(recs, 0, lambda r: r["event"] == "sent" and r["msg"]["t"] == "text")
    row = timed(measure(recs, said=None, since=sent["t"]), sent["t"], "sent")
    record("typed turn", kind="turn", mic="text", **row, transcripts=[m["text"] for m in c.recv("transcript")],
           server_last_turn=get_status(target).get("last_turn"))
    check_answer(row, "hot")
    check_played_to_the_end(row)
    assert not interrupts_outside_replies(recs)


def test_barge_in_during_a_long_answer(target, client, speech):
    """Talking over a long answer on an open microphone: the server stops it (`interrupt` while the reply is active),
    the client flushes at once and reports what was heard (`played_ms`), and the question asked over it is answered."""
    story, stop = speech[STORY], speech[STOP_ITALY]
    c = talk(client, target,
             f"handsfree-on; wait:ready; sleep:1000; say-wait:{story.spec}; wait:playback-started:90000; sleep:2500; "
             f"say:{stop.spec}; wait:interrupt:15000; wait:sent:played_ms:5000; wait:clip-finished:15000; "
             "next:end_of_turn:120000; wait:playback-finished:30000; sleep:500", mic="vad")
    recs, summary = c.records, c.summary
    told = timed(measure(recs, said=STORY, since=step_time(recs, f"say-wait:{story.spec}")), clip_eos(recs, story), "eos")
    spoke = step_time(recs, f"say:{stop.spec}") + stop.speech_start
    intr = first(recs, spoke, recv("interrupt"))
    assert intr, "no interrupt after the user spoke over the reply"
    # The answer is the reply after the whole utterance ended: if the pause after "Stop." splits the turn, a reply to
    # "Stop." can start and be cut by the rest of the question (the orchestrator's e2e measures it the same way).
    eos = clip_eos(recs, stop)
    answer = timed(measure(recs, said=STOP_ITALY, since=eos, heard_since=intr["t"]), eos, "eos")
    # When the story went quiet: the end of its last audible stretch before the answer began to play (the server may
    # stop sending at the first sound of speech, before its interrupt; orchestrator bargein.py).
    quiet = last_sound(summary, told["playback_t"], answer["playback_t"])
    after_cut = sounding_after(intr["t"], quiet)
    record("barge-in, open microphone", kind="barge-in", mic="vad",
           interrupt_after_speech_ms=ms(intr["t"] - spoke), audio_quiet_after_speech_ms=ms(quiet - spoke) if quiet else None,
           sounding_after_interrupt_ms=after_cut, flush_tails=summary["flush_tails"], story=told, answer=answer,
           server_last_turn=get_status(target).get("last_turn"))
    assert intr["msg"].get("reply_id") in (None, told["reply_id"]), "the interrupt names the story"
    assert told["played_ms"] is not None and 2000 <= told["played_ms"] <= told["reply_audio_ms"] + 50, told
    assert after_cut < 30, f"the story kept sounding {after_cut} ms after the interrupt arrived"
    check_answer(answer, "rome", max_wer=0.5)
    check_played_to_the_end(answer)
    assert not interrupts_outside_replies(recs)


def test_push_to_talk_barge_in(target, client, speech):
    """Pressing talk over a long answer: the client stops it itself (flush, `interrupt`, `played_ms`, then `start`),
    the server sends no interrupt of its own, and the next question is answered."""
    story, q = speech[STORY], speech[ITALY]
    c = talk(client, target,
             f"connect; wait:ready; sleep:500; press; say-wait:{story.spec}; release; wait:playback-started:90000; "
             f"sleep:2500; press; say-wait:{q.spec}; release; next:end_of_turn:120000; "
             "wait:playback-finished:30000; sleep:500", mic="ptt")
    recs, summary = c.records, c.summary
    first_release, second_release = releases(recs)[:2]
    told = timed(measure(recs, said=STORY, since=first_release), first_release, "release")
    answer = timed(timed(measure(recs, said=ITALY, since=second_release), second_release, "release"),
                   clip_eos(recs, q), "eos")
    order = [r["msg"]["t"] for r in recs if r["event"] == "sent"
             and r["msg"]["t"] in ("start", "stop", "interrupt", "played_ms")]
    cut = first(recs, 0, lambda r: r["event"] == "sent" and r["msg"]["t"] == "interrupt")
    after_cut = sounding_after(cut["t"], last_sound(summary, told["playback_t"], answer["playback_t"])) if cut else None
    record("barge-in, push-to-talk", kind="barge-in", mic="ptt", wire_order=order, sounding_after_press_ms=after_cut,
           flush_tails=summary["flush_tails"], story=told, answer=answer, server_interrupts=c.recv("interrupt"),
           server_last_turn=get_status(target).get("last_turn"))
    assert order == ["start", "stop", "interrupt", "played_ms", "start", "stop", "played_ms"], order
    assert not c.recv("interrupt"), "the server does not interrupt a reply the client already stopped"
    assert told["played_ms"] is not None and told["played_ms"] >= 2000, told
    assert after_cut is not None and after_cut < 30, f"the story kept sounding {after_cut} ms after the press"
    check_answer(answer, "rome")
    check_played_to_the_end(answer)
    assert not interrupts_outside_replies(recs)


def test_status_endpoint(target):
    """GET /v1/status has PROTOCOL.md's fields (the apps' status view reads them)."""
    s = get_status(target)
    record("status", kind="status", status=s)
    for key in ("v", "state", "space", "mode", "tier", "model", "hold", "tool", "last_turn", "clients"):
        assert key in s, f"/v1/status has no {key}: {sorted(s)}"
    assert s["v"] == 1 and isinstance(s["hold"], dict) and "phase" in s["hold"]
    for key in ("eos_to_first_audio_ms", "stt_ms", "llm_ttft_ms", "tts_first_audio_ms"):
        assert key in s["last_turn"], s["last_turn"]


@pytest.fixture(scope="module", autouse=True)
def summary_row():
    yield
    turns = [r for r in RESULTS if r.get("kind") == "turn"]
    if not turns:
        return
    ptt = [r["release_to_audio_start_ms"] for r in turns if r.get("release_to_audio_start_ms") is not None]
    eos = [r["eos_to_audio_start_ms"] for r in turns if r.get("eos_to_audio_start_ms") is not None]
    record("summary", kind="summary", turns=len(turns),
           release_to_audio_start_ms=ptt, eos_to_audio_start_ms=eos,
           eos_to_audio_start_median_ms=statistics.median(eos) if eos else None,
           client_vs_server=[versus(r) for r in RESULTS if r.get("kind") in ("turn", "barge-in")])


def versus(r: dict) -> dict:
    """The client's time to the first sound of a turn's answer next to the server's own eos_to_first_audio_ms, each
    from the moment that side knows the turn ended: the `stop` sent (push-to-talk), the `text` sent (typed), or the end
    of speech in the clip (open microphone, where the server finds the end itself, so the two differ by its VAD)."""
    row = r["answer"] if r["kind"] == "barge-in" else r
    origin, client_ms = next(((name, row.get(key)) for name, key in
                              (("stop sent", "stop_to_playback_ms"), ("text sent", "sent_to_playback_ms"),
                               ("end of speech", "eos_to_playback_ms")) if row.get(key) is not None), (None, None))
    server_ms = (r.get("server_last_turn") or {}).get("eos_to_first_audio_ms")
    return {"test": r["test"], "from": origin, "client_to_first_sound_ms": client_ms,
            "server_eos_to_first_audio_ms": server_ms,
            "client_minus_server_ms": round(client_ms - server_ms) if client_ms is not None and server_ms else None,
            "preroll_ms_at_start": row.get("preroll_ms_at_start"),
            "server_llm_ttft_ms": (r.get("server_last_turn") or {}).get("llm_ttft_ms")}
