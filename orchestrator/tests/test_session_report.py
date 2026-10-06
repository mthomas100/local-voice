"""tools/session_report.py on planted recordings, model-free (2026-10-05): every property the live session's analysis
rests on is measured here on signals whose truth is known, never by listening. The agent's voice and the person's are
tests/recording_fakes.speech_like (seeded noise under a syllable envelope: one voice per seed, the same words in
another voice are other audio); the room is tools/echo_mixer.py's (resampling, delay, level, pink noise)."""
from __future__ import annotations

import base64
import json
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ORCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH / "tools"))

import session_report as sr  # noqa: E402
from echo_mixer import EchoMixer, EchoSettings, PinkNoise, db, float_to_pcm16, resample_24k_to_16k, rms  # noqa: E402
from recording_fakes import speech_like  # noqa: E402

PLAY, MIC = 24000, 16000
WALL0 = 1_801_000_000.0
SENTENCES = ["His cat followed him all the way up, step by step.", "Ships far out at sea saw the light.",
             "In the morning the keeper polished the glass and wound the clockwork.",
             "Then he wrote the weather in his logbook before he slept."]


def voice(text: str, who: str = "agent", lead_s: float = 0.0) -> np.ndarray:
    return speech_like(text, PLAY, seconds_per_char=0.06, level=0.1, voice=who, lead_s=lead_s)


def place(track: np.ndarray, x: np.ndarray, at_s: float, rate: int) -> int:
    k = int(round(at_s * rate))
    track[k:k + len(x)] += x[: max(0, len(track) - k)]
    return k


def write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(float_to_pcm16(x) if x.dtype != np.int16 else x.astype("<i2").tobytes())


def recording(folder: Path, play: np.ndarray, mic: np.ndarray, events: list[dict], *, turn_dir: Path | None = None) -> Path:
    """A folder as local_voice/recorder.py writes it (its clock: t from t0, wall = WALL0 + t)."""
    folder.mkdir(parents=True)
    write_wav(folder / "playback.wav", play, PLAY)
    write_wav(folder / "mic.wav", mic, MIC)
    with (folder / "events.jsonl").open("w") as fh:
        for e in events:
            fh.write(json.dumps({"mono": 5000.0 + e["t"], "wall": WALL0 + e["t"], **e}) + "\n")
    conf = {"echo": {"guard": True, "hold_for_words": "browser", "window_s": 300, "min_words": 3, "min_share": 0.6,
                     "tail_s": 1.0},
            "tts": {"adapter": "qwen3", "adapters": {"qwen3": {"impl": "x:Y", "model": "m", "voice": "Ryan"}}},
            "brain": {"turn_log_dir": str(turn_dir) if turn_dir else None}}
    (folder / "meta.json").write_text(json.dumps({
        "session": "s-test", "client": "browser", "transport": "SmallWebRTCTransport", "mic": "vad", "wall0": WALL0,
        "t0_mono": 5000.0, "duration_s": len(mic) / MIC, "config": conf, "config_path": str(ORCH / "config.yaml"),
        "stop_reason": "the session ended", "stats": {}}))
    return folder


def reply(play: np.ndarray, at_s: float, sentences: list[str], gap_s: float = 0.3) -> tuple[list[dict], float]:
    """The agent's sentences into the playback from at_s, with bot on/off events; returns (events, end)."""
    t, ev = at_s, [{"t": at_s, "ev": "bot", "on": True}]
    for s in sentences:
        x = voice(s)
        place(play, x, t, PLAY)
        ev.append({"t": t - 0.5, "ev": "tts", "text": s})
        t += len(x) / PLAY + gap_s
    ev.append({"t": t - gap_s, "ev": "bot", "on": False})
    return ev, t - gap_s


def noise_floor(n: int) -> np.ndarray:
    return PinkNoise(-60.0).take(0, n)


# ------------------------------------------------------------------------------------------- (b) live echo

@pytest.mark.parametrize("level_db,delay_ms", [(-25.0, 150.0), (-40.0, 320.0)])
def test_a_planted_live_echo_is_measured_within_2_db_and_10_ms(tmp_path, level_db, delay_ms):
    play = np.zeros(14 * PLAY, np.float32)
    ev, _ = reply(play, 2.0, SENTENCES)
    mixer = EchoMixer(EchoSettings(level_db=level_db, delay_ms=delay_ms, noise_dbfs=-60.0),
                      lambda j0, n: play[max(j0, 0):max(j0, 0) + n] if j0 >= 0 else
                      np.concatenate([np.zeros(-j0, np.float32), play[: n + j0]]))
    mic = mixer.render(0, 14 * MIC)
    rec = sr.load(recording(tmp_path / "rec", play, mic, ev))
    (r,) = sr.live_echo(rec)
    assert r["found"] and abs(r["delay_ms"] - delay_ms) <= 10 and abs(r["level_db"] - level_db) <= 2, r
    # no echo at all: nothing is found
    rec = sr.load(recording(tmp_path / "quiet", play, noise_floor(14 * MIC), ev))
    (r,) = sr.live_echo(rec)
    assert not r["found"], r


# ---------------------------------------------------------------------------------- (c) replays, (a) turns

def test_a_replay_is_found_at_its_lag_and_reading_aloud_is_not_one(tmp_path):
    """An early live test's case and its alternative in one recording. A reply plays from 5 s; 60 s after its last 8 s were
    played they come back into the microphone at -6 dB (something replays them); 30 s later the person reads the same
    sentences aloud in their own voice. Both are echo turns by their words; only the first is a replay by its audio."""
    play = np.zeros(130 * PLAY, np.float32)
    ev, end = reply(play, 5.0, SENTENCES)
    mic = noise_floor(130 * MIC)
    src_from = end - 8.0
    seg = resample_24k_to_16k(play[int(src_from * PLAY):int(end * PLAY)]) * np.float32(10 ** (-6 / 20))
    place(mic, seg, src_from + 60.0, MIC)
    aloud = voice(SENTENCES[2], who="person")              # the same words in the person's voice: other audio
    place(mic, resample_24k_to_16k(aloud), 100.0, MIC)
    turns = tmp_path / "turns"
    turns.mkdir()
    day = datetime.fromtimestamp(WALL0 + 60).strftime("%Y-%m-%d")
    iso = lambda t: datetime.fromtimestamp(WALL0 + t).astimezone().isoformat(timespec="milliseconds")   # noqa: E731
    rows = [("tell me a story about a lighthouse", 1.0, 3.0),
            (SENTENCES[3].lower().rstrip("."), src_from + 60.0 + 3.0, src_from + 68.0),
            (SENTENCES[2].lower().rstrip("."), 100.2, 105.0)]
    (turns / f"{day}.jsonl").write_text("".join(json.dumps({
        "v": 1, "type": "turn", "session": "s-test", "turn": i + 1, "t_start": iso(a), "t_end": iso(b),
        "input": "voice", "user_text": u, "reply_text": ""}) + "\n" for i, (u, a, b) in enumerate(rows)))
    folder = recording(tmp_path / "rec", play, mic, ev, turn_dir=turns)
    a = sr.analyse(folder, clips=False)
    (r,) = a["replays"]
    assert abs(r["lag_s"] - 60.0) <= 0.1, r                                    # the lag, to the fingerprint's hop
    assert abs(r["source_from_s"] - src_from) <= 0.5 and abs(r["source_to_s"] - end) <= 0.5, r   # its 0.5 s frames
    assert abs(r["mic_from_s"] - (src_from + 60.0)) <= 0.5 and abs(r["mic_to_s"] - (end + 60.0)) <= 0.5, r
    echo = {t["turn"]: t for t in a["echo_turns"]}
    assert not echo[1]["echo"] and echo[2]["echo"] and echo[3]["echo"]
    assert echo[2]["replay"] == r and echo[3]["replay"] is None          # a replay, and words said anew
    md = (folder / "report.md").read_text()
    assert "2 of 3 voice turns" in md and "60.0 s earlier" in md and "said anew" in md


def test_a_replay_through_a_room_is_still_found(tmp_path):
    """The replay through tools/echo_mixer.py's synthetic room (RT60 0.3 s) at -10 dB over the -60 dBFS floor."""
    play = np.zeros(80 * PLAY, np.float32)
    ev, end = reply(play, 2.0, SENTENCES)
    src = play[int((end - 8.0) * PLAY):int(end * PLAY)]
    m = EchoMixer(EchoSettings(level_db=-10.0, delay_ms=20.0, reverb=True, noise_dbfs=-60.0),
                  lambda j0, n: np.concatenate([np.zeros(max(0, -j0), np.float32),
                                                src[max(j0, 0):max(j0, 0) + n - max(0, -j0)],
                                                np.zeros(max(0, j0 + n - len(src)), np.float32)])[:n])
    mic = noise_floor(80 * MIC)
    place(mic, m.render(0, int(9 * MIC)), end - 8.0 + 45.0, MIC)
    (r,) = sr.find_replays(mic, resample_24k_to_16k(play))
    assert abs(r["lag_s"] - 45.02) <= 0.1 and r["ber"] < 0.25, r


# ------------------------------------------------------------------------------------------------ (d) clips

def test_clips_are_cut_where_each_generation_starts_and_hold_all_it_sent(tmp_path):
    """Four generations in playback at known places, the third with a 0.4 s hole in the middle (a voice slower than
    real time) and the fourth cut short by an interruption. Each clip starts on its generation's first sample and
    holds exactly the audio of it that went out, its text alongside, in the voice bench's layout."""
    import tts_quality_bench as bench

    play = np.zeros(40 * PLAY, np.int16)
    gens = [voice(s, lead_s=0.08 if i else 0.03) for i, s in enumerate(SENTENCES)]
    pcm = [np.round(np.clip(g, -1, 1) * 32767).astype(np.int16) for g in gens]
    at = [2.0, 2.0 + len(gens[0]) / PLAY, 9.0, 18.0]
    ev, sent, starts = [], [], []
    for i, (g, t) in enumerate(zip(pcm, at)):
        k = int(round(t * PLAY))
        starts.append(k)
        if i == 2:                                                       # a hole of 0.4 s after 1 s
            h = PLAY
            play[k:k + h] = g[:h]
            play[k + h + int(0.4 * PLAY):k + int(0.4 * PLAY) + len(g)] = g[h:]
            sent += [[k, k + h], [k + h + int(0.4 * PLAY), k + int(0.4 * PLAY) + len(g)]]
        elif i == 3:                                                     # interrupted after 1.5 s
            play[k:k + int(1.5 * PLAY)] = g[:int(1.5 * PLAY)]
            sent.append([k, k + int(1.5 * PLAY)])
        else:
            play[k:k + len(g)] = g
            sent.append([k, k + len(g)])
        onset = int(np.flatnonzero(np.abs(g.astype(np.int32)) >= 328)[0])
        fp_at = max(0, onset - 240)
        ev.append({"t": t - 0.4, "ev": "tts", "gen": i + 1, "text": SENTENCES[i], "samples": len(g),
                   "status": "cut" if i == 3 else "done", "seams": [1920 * 4, 1920 * 9], "first_audio_s": 0.1,
                   "fp_at": fp_at, "fp": base64.b64encode(g[fp_at:fp_at + 2400].tobytes()).decode()})
    sent[0:2] = [[sent[0][0], sent[1][1]]]                               # the first two went out back to back
    ev += [{"t": a / PLAY, "ev": "sent", "samples": [a, b], "from_s": a / PLAY, "to_s": b / PLAY} for a, b in sent]
    folder = recording(tmp_path / "rec", play, np.zeros(40 * MIC, np.float32), ev)
    a = sr.analyse(folder)
    clips = {c["gen"]: c for c in a["clips"]}
    assert [round(clips[i + 1]["t_play_s"] * PLAY) for i in range(4)] == starts
    for i in range(4):
        x, rate = sr.read_wav(folder / "clips" / clips[i + 1]["clip"])
        want = pcm[i] if i < 3 else pcm[i][:int(1.5 * PLAY)]
        assert rate == PLAY and np.array_equal(x, want), i                # exactly what it sent, hole left out
    assert [clips[i + 1]["complete"] for i in range(4)] == [True, True, True, False]
    side = json.loads((folder / "clips" / "session" / "g0001-0.json").read_text())
    assert side["text"] == SENTENCES[0] and side["seams"] == [1920 * 4, 1920 * 9] and side["controls"]
    assert [p.name for p in bench.scorers.clip_sidecars(folder / "clips")][:2] == ["g0001-0.json", "g0002-0.json"]
    manifest = json.loads((folder / "clips" / "run.json").read_text())
    assert [i["id"] for i in manifest["items"]] == ["g0001", "g0002", "g0003", "g0004"]
    assert "4 generations cut from playback.wav" in (folder / "report.md").read_text()
