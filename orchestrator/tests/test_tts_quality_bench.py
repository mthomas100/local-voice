"""Model-free tests of the voice-quality bench: tools/tts_quality_bench.py and the numpy half of
tools/tts_bench_scorers.py (2026-10-05).

Nothing here loads a model or touches the GPU: the render stage runs the project's FakeSynthesizer, gpu_clear.sh is
replaced by `true` and `false`, the asr stage gets a stand-in transcriber, and the Praat checks run in the scorer venv
(tools/.venv-bench), skipped when that venv is absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ORCH = Path(__file__).resolve().parents[1]
TOOLS = ORCH / "tools"
sys.path.insert(0, str(TOOLS))

import speech_gaps  # noqa: E402
import tts_bench_scorers as scorers  # noqa: E402
import tts_quality_bench as bench  # noqa: E402
from local_voice.engines.base import float_to_pcm16  # noqa: E402
from local_voice.engines.fakes import FakeSynthesizer  # noqa: E402

SCORER_PY = TOOLS / ".venv-bench" / "bin" / "python"
needs_scorer_venv = pytest.mark.skipif(not SCORER_PY.exists(), reason="no scorer venv at tools/.venv-bench")
RATE = 24000
CHUNK = int(0.08 * RATE)      # FakeSynthesizer's chunk
quiet = lambda m: None        # noqa: E731


def tts_conf() -> dict:
    return bench.read_config()["tts"]


def fake_tts(lead_s: float = 0.24) -> dict:
    tts = json.loads(json.dumps(tts_conf()))
    tts["adapter"] = "fake"
    tts["adapters"]["fake"] = {"impl": "local_voice.engines.fakes:FakeSynthesizer", "seconds_per_char": 0.03,
                               "lead_s": lead_s}
    return tts


def tone(seconds: float, f: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * RATE)), np.float32)


def chunks_of(x: np.ndarray, n: int = CHUNK) -> list[np.ndarray]:
    return [x[i:i + n] for i in range(0, len(x), n)]


# ---------------------------------------------------------------------------------------------------- grid

def test_grid_expands_one_factor_at_a_time():
    tts = tts_conf()
    grid = {s.name: s for s in bench.expand_grid(tts)}
    assert list(grid) == ["baseline", "temp0.7", "temp0.5", "minwords4", "maxsent2", "maxsent3", "interval0.64",
                          "interval1.0", "nostream", "instruct", "qwen0.6b", "kokoro"]
    prod = dict(tts["adapters"][tts["adapter"]])
    base = grid["baseline"]
    assert base.impl == prod.pop("impl") and base.settings == prod
    assert base.stream and (base.min_words, base.max_sentences) == (0, 1)

    def diff(s):
        return {k: v for k, v in s.settings.items() if base.settings.get(k) != v}

    assert diff(grid["temp0.7"]) == {"temperature": 0.7}
    assert diff(grid["temp0.5"]) == {"temperature": 0.5}
    assert diff(grid["interval0.64"]) == {"streaming_interval": 0.64}
    assert diff(grid["interval1.0"]) == {"streaming_interval": 1.0}
    assert diff(grid["instruct"]) == {"instruct": "Speak in a calm, natural, steady voice."}
    assert diff(grid["qwen0.6b"]) == {"model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}
    assert not grid["nostream"].stream and diff(grid["nostream"]) == {}
    for name, mw, ms in (("minwords4", 4, 1), ("maxsent2", 0, 2), ("maxsent3", 0, 3)):
        s = grid[name]
        assert (s.min_words, s.max_sentences) == (mw, ms) and s.engine_key() == base.engine_key()
    k = grid["kokoro"]
    assert k.adapter == "kokoro" and k.impl == "local_voice.engines.kokoro:KokoroEngine"
    assert k.settings["voice"] == "am_michael" and k.settings["model"] == "mlx-community/Kokoro-82M-bf16"


def test_grid_selection_and_combinations():
    tts = tts_conf()
    assert [s.name for s in bench.expand_grid(tts, ["kokoro", "temp0.5"])] == ["baseline", "kokoro", "temp0.5"]
    calm = bench.expand_grid(tts, ["calm"], ["calm=instruct+temp0.7+maxsent2"])[1]
    assert calm.name == "calm" and calm.settings["instruct"] == bench.STYLE and calm.settings["temperature"] == 0.7
    assert calm.max_sentences == 2 and calm.adapter == "qwen3"
    names = [s.name for s in bench.expand_grid(tts, None, ["calm=instruct+temp0.7"])]
    assert names[-1] == "calm" and len(names) == len(bench.GRID) + 1
    for bad in (dict(names=["nope"]), dict(combos=["x=kokoro+nope"]), dict(combos=["temp0.7=temp0.5"]),
                dict(combos=["x"])):
        with pytest.raises(ValueError):
            bench.expand_grid(tts, bad.get("names"), bad.get("combos", ()))


def test_nostream_changes_only_the_stream_flag():
    from local_voice.engines.qwen3_tts import Qwen3TTSEngine

    mlx_before = "mlx.core" in sys.modules
    grid = {s.name: s for s in bench.expand_grid(tts_conf(), ["nostream"])}
    streamed, whole = bench.build_engine(grid["baseline"]), bench.build_engine(grid["nostream"])
    assert isinstance(whole, Qwen3TTSEngine)
    a, b = streamed._kwargs("Hello there."), whole._kwargs("Hello there.")
    assert a.pop("stream") is True and b.pop("stream") is False
    assert a.pop("streaming_interval") == 0.32 and "streaming_interval" not in b
    assert a == b
    assert ("mlx.core" in sys.modules) == mlx_before        # building an adapter loads nothing
    with pytest.raises(ValueError):
        bench.build_engine(bench.make_setting("x", {"adapter": "kokoro", "stream": False}, tts_conf()))


# ---------------------------------------------------------------------------------------------------- texts

def test_text_set_loads_and_splits():
    items = bench.load_texts()
    by_id = {i.id: i for i in items}
    assert 30 <= len(items) <= 40 and len(by_id) == len(items)
    assert {i.kind for i in items} == set(bench.KINDS)
    for t in ("Hey!", "Six.", "Saved.", "Done.", "Yes.", "Okay.", "Rome."):
        assert any(i.kind == "short" and i.sentences == [t] for i in items), t
    assert any("—" in i.text for i in items if i.kind == "dash")
    logged = [i for i in items if i.logged]
    assert len(logged) == 17
    # the two places where the reply items differ from Pipecat's aggregator are the spoken cap's mid-path release
    # (t06) and a TTSSpeakFrame said as one generation (t19), as the YAML says
    assert {i.id for i in logged if bench.aggregator_split(i.text) != i.sentences} == {"t06-path", "t19-approval"}
    assert bench.aggregator_split(by_id["t06-path"].text)[1].startswith("I keep them in ~/.notes/garden: one file")
    assert by_id["t06-path"].sentences[1:3] == ["I keep them in ~/.", by_id["t06-path"].sentences[2]]
    assert all(i.sentences == bench.aggregator_split(i.text) for i in items if not i.logged)
    assert bench.aggregator_split("Hey! What can I help you with? Okay.") == ["Hey!", "What can I help you with?",
                                                                              "Okay."]


def test_session_replies_reads_a_log_and_its_turns(tmp_path):
    turns = tmp_path / "turns.jsonl"
    turns.write_text("".join(json.dumps({"turn": n, "t_start": t}) + "\n" for n, t in
                             ((1, "2026-01-01T10:00:00-08:00"), (2, "2026-01-01T10:00:20-08:00"))))
    log = tmp_path / "orchestrator.log"
    log.write_text("2026-01-01 10:00:01.100 | DEBUG | x - Generating TTS [Hey!]\n"
                   "2026-01-01 10:00:01.500 | DEBUG | x - Generating TTS [What can I help you with?]\n"
                   "2026-01-01 10:00:21.000 | DEBUG | x - Generating TTS [Six.]\n"
                   "2026-01-01 11:00:00.000 | DEBUG | x - Generating TTS [Out of the window.]\n")
    got = bench.session_replies(log, turns, ("2026-01-01 09:59:00", "2026-01-01 10:30:00"))
    assert got == [{"turn": 1, "sentences": ["Hey!", "What can I help you with?"]}, {"turn": 2, "sentences": ["Six."]}]


def test_grouping_filter_and_llm_characters():
    grid = {s.name: s for s in bench.expand_grid(tts_conf())}
    t01 = next(i for i in bench.load_texts() if i.id == "t01-hey")
    assert bench.generations(t01, grid["baseline"]) == ["Hey!", "What can I help you with?"]
    for name in ("minwords4", "maxsent2", "maxsent3"):
        assert bench.generations(t01, grid[name]) == ["Hey! What can I help you with?"]
    assert bench.llm_chars(t01, grid["baseline"]) == len("Hey!") + 2
    assert bench.llm_chars(t01, grid["minwords4"]) == len("Hey! What can I help you with?")
    md = bench.Item("md", "reply", "x", ["**Done.**", "---", "Saved to `notes.md`."])
    assert bench.generations(md, grid["baseline"]) == ["Done.", "Saved to notes.md."]


def test_plan_counts_reuse():
    settings = bench.expand_grid(tts_conf(), ["minwords4"])
    items = bench.load_texts()
    changed = [i for i in items if bench.generations(i, settings[0]) != bench.generations(i, settings[1])]
    p = {r["setting"]: r for r in bench.plan(settings, items, 3)}
    assert p["baseline"]["renders"] == 3 * len(items) and p["baseline"]["reused"] == 0
    assert p["minwords4"]["renders"] == 3 * len(changed) and p["minwords4"]["reused"] == 3 * (len(items) - len(changed))
    assert 0 < len(changed) < 17


# ---------------------------------------------------------------------------------------------------- trim, render

def test_trim_finds_a_short_silence_and_misses_a_long_one():
    trim = bench.production_trim(tts_conf())
    assert (trim.enabled, trim.keep_ms, trim.max_scan_s) == (True, 30.0, 0.8)
    short = bench.service_trim(chunks_of(np.concatenate([silence(0.4), tone(1.0)])), RATE, trim, True)
    assert short["onset_found"] and short["start"] / RATE == pytest.approx(0.4 - 0.03, abs=0.006)
    # an early live test's case: more silence than the 0.8 s scan, so nothing is trimmed and all of it is heard
    long = bench.service_trim(chunks_of(np.concatenate([silence(1.04), tone(1.0)])), RATE, trim, True)
    assert long["onset_found"] is False and long["start"] == 0 and long["scanned_s"] >= 0.8
    later = bench.service_trim(chunks_of(np.concatenate([silence(0.4), tone(1.0)])), RATE, trim, False)
    assert later == {"start": 0, "out_chunk": 0, "onset_found": None, "scanned_s": 0.0}


async def test_trim_replay_matches_the_production_service():
    """The bench's replay against services/tts.py itself: the same FakeSynthesizer through MLXTTSService.run_tts gives
    byte-identical audio, with a silence the trim finds (0.24, 0.4 s) and one it gives up on (1.04 s)."""
    from local_voice.mlx_worker import MLXWorker
    from local_voice.services.tts import MLXTTSService
    from pipecat.frames.frames import TTSAudioRawFrame

    trim = bench.production_trim(tts_conf())
    text = "Hello there, this is a test of the leading silence trim."
    setting = bench.expand_grid(fake_tts())[0]
    for lead in (0.24, 0.4, 1.04):
        svc = MLXTTSService(engine=FakeSynthesizer({"seconds_per_char": 0.03, "lead_s": lead}),
                            worker=MLXWorker(use_mlx=False), sample_rate=RATE, keep_before_onset_ms=trim.keep_ms)
        svc._sample_rate = RATE   # set by the StartFrame in a pipeline
        heard = b"".join([f.audio async for f in svc.run_tts(text, "ctx") if isinstance(f, TTSAudioRawFrame)])
        audio, side = bench.render_clip(FakeSynthesizer({"seconds_per_char": 0.03, "lead_s": lead}),
                                        bench.Item("x", "reply", text, [text]), setting, 0, trim)
        assert float_to_pcm16(audio) == heard, lead
        assert side["lead_raw_s"] == pytest.approx(lead, abs=0.01)
        assert side["lead_heard_s"] == pytest.approx(0.03 if lead < 0.8 else lead, abs=0.01)


def test_render_with_the_fake_engine(tmp_path, monkeypatch):
    tts = fake_tts()
    settings = bench.expand_grid(tts, ["maxsent2"])
    items = [bench.Item("two", "reply", "Hey! What can I help you with?", ["Hey!", "What can I help you with?"]),
             bench.Item("one", "short", "Done.", ["Done."])]
    trim = bench.production_trim(tts)
    spoken: list[str] = []
    stream = FakeSynthesizer.stream
    monkeypatch.setattr(FakeSynthesizer, "stream", lambda self, text: (spoken.append(text), stream(self, text))[1])
    run = lambda **kw: bench.render(tmp_path, settings, items, 2, trim=trim, log=quiet, **{"gpu_clear": "true", **kw})  # noqa: E731

    assert run() == bench.EXIT_OK
    assert sorted(spoken) == sorted(["Hey!", "What can I help you with?", "Done."] * 2
                                    + ["Hey! What can I help you with?"] * 2)
    for s in ("baseline", "maxsent2"):
        for iid in ("two", "one"):
            for r in (0, 1):
                with wave.open(str(tmp_path / s / f"{iid}-{r}.wav")) as w:
                    assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2)
                    frames = w.getnframes()
                side = json.loads((tmp_path / s / f"{iid}-{r}.json").read_text())
                assert side["samples"] == frames and side["setting"] == s and side["r"] == r
    assert json.loads((tmp_path / "maxsent2" / "one-0.json").read_text())["reused_from"] == "baseline"
    assert "reused_from" not in json.loads((tmp_path / "maxsent2" / "two-0.json").read_text())
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert [s["name"] for s in manifest["settings"]] == ["baseline", "maxsent2"] and manifest["reps"] == 2

    side = json.loads((tmp_path / "baseline" / "two-0.json").read_text())
    g0, g1 = side["generations"]
    assert [g["text"] for g in side["generations"]] == ["Hey!", "What can I help you with?"]
    assert g0["lead_raw_s"] == pytest.approx(0.24, abs=0.01) and g0["trim_onset_found"] is True
    assert g0["trimmed_s"] == pytest.approx(0.21, abs=0.01) and side["lead_heard_s"] == pytest.approx(0.03, abs=0.002)
    assert g1["trimmed_s"] == 0 and g1["lead_raw_s"] == pytest.approx(0.24, abs=0.01)   # later ones keep their pause
    assert g1["start"] == g0["samples"] and side["samples"] == g0["samples"] + g1["samples"]
    shift = round(g0["trimmed_s"] * RATE)
    assert side["seams"][0] == 3 * CHUNK - shift            # FakeSynthesizer's 80 ms chunks, moved by the trim
    assert side["seams"][1:] == [g1["start"] + k * CHUNK for k in range(1, len(g1["chunk_samples"]))]
    assert all(0 < c < side["samples"] for c in side["controls"]) and len(side["controls"]) == len(side["seams"]) + 2
    assert side["hit_cap"] is False and side["seed"] == bench.stable_seed("two", 0) and side["seeded"] is False
    x, rate = scorers.read_wav(tmp_path / "baseline" / "two-0.wav")
    assert np.abs(x[: int(0.025 * rate)]).max() == 0 and np.abs(x[int(0.035 * rate): int(0.1 * rate)]).max() > 0.1
    meta = json.loads((tmp_path / "baseline" / "setting.json").read_text())
    assert meta["runs"][0]["rendered"] == 4 and len(meta["runs"][0]["loadavg_start"]) == 3

    spoken.clear()
    assert run() == bench.EXIT_OK and spoken == []          # resume: everything is there
    (tmp_path / "baseline" / "two-1.json").unlink()
    assert run() == bench.EXIT_OK and spoken == ["Hey!", "What can I help you with?"]
    spoken.clear()
    (tmp_path / "baseline" / "one-0.json").unlink()
    assert run(gpu_clear="false") == bench.EXIT_BUSY        # BUSY: refuses to start, renders nothing
    assert spoken == [] and not (tmp_path / "baseline" / "one-0.json").exists()
    bigger = bench.expand_grid(tts, ["maxsent2", "maxsent3"])
    assert bench.render(tmp_path, bigger, items, 2, trim=trim, gpu_clear="true", log=quiet) == bench.EXIT_OK
    with pytest.raises(ValueError):                         # a setting may not change under a run
        bench.render(tmp_path, bench.expand_grid(fake_tts(lead_s=0.4)), items, 2, trim=trim, gpu_clear="true",
                     log=quiet)


def test_asr_stage_with_a_stand_in_transcriber(tmp_path):
    tts = fake_tts()
    items = [bench.Item("two", "reply", "Hey! What can I help you with?", ["Hey!", "What can I help you with?"])]
    trim = bench.production_trim(tts)
    assert bench.render(tmp_path, bench.expand_grid(tts, ["baseline"]), items, 1, trim=trim, gpu_clear="true",
                        log=quiet) == bench.EXIT_OK
    seen = []

    def transcribe(a16):
        seen.append(len(a16))
        return "Hey, what can I help you?", [{"w": "Hey,", "start": 0.03, "end": 0.1}]

    assert bench.asr(tmp_path, {}, transcribe=transcribe, log=quiet) == bench.EXIT_OK
    side = json.loads((tmp_path / "baseline" / "two-0.json").read_text())
    res = json.loads((tmp_path / "baseline" / "two-0.asr.json").read_text())
    assert seen == [pytest.approx(side["samples"] * 16000 / RATE, abs=2)]       # resampled to 16 kHz
    assert res["del"] == ["with"] and res["sub"] == [] and res["wer"] == pytest.approx(1 / 7)
    assert bench.asr(tmp_path, {}, transcribe=transcribe, log=quiet) == bench.EXIT_OK and len(seen) == 1


@needs_scorer_venv
def test_score_stage_in_the_scorer_venv(tmp_path):
    """The score stage end to end without the neural models (Praat, flux, pauses): every clip gets a score, a clip
    the render stage copied gets its source's score copied, and the report reads them."""
    tts = fake_tts()
    items = [bench.Item("two", "reply", "Hey! What can I help you with?", ["Hey!", "What can I help you with?"]),
             bench.Item("one", "short", "Done.", ["Done."])]
    assert bench.render(tmp_path, bench.expand_grid(tts, ["maxsent2"]), items, 1, trim=bench.production_trim(tts),
                        gpu_clear="true", log=quiet) == bench.EXIT_OK
    assert bench.score(tmp_path, neural=False, gpu_clear=None, log=quiet) == 0
    scores = {p.parent.name + "/" + p.name: json.loads(p.read_text()) for p in tmp_path.glob("*/*.score.json")}
    assert len(scores) == 4 and not any(s["neural"] for s in scores.values())
    assert scores["maxsent2/one-0.score.json"]["copied_from"] == "baseline"
    two = scores["baseline/two-0.score.json"]
    assert two["seams_n"] > 0 and two["seam_ratio_median"] > 10        # the fake restarts its sine at every chunk
    assert two["lead_energy_s"] < 0.05 and two["jumps_over6"] == 0 and two["f0_median_st"] is not None
    report, _ = bench.write_report(tmp_path)
    assert "scored: 4" in report.read_text()


# ---------------------------------------------------------------------------------------------------- WER

def test_normalise_and_wer():
    assert bench.normalise("I'm your voice assistant — on this Mac.") == ["i'm", "your", "voice", "assistant", "on",
                                                                          "this", "mac"]
    assert bench.normalise("Six.") == bench.normalise("6") == ["six"]
    assert bench.normalise("a qwen-27b variant") == ["a", "qwen", "twenty", "seven", "b", "variant"]
    assert bench.normalise("atlas/journal/2026-10-05.md") == ["atlas", "journal", "two", "thousand", "twenty", "six",
                                                              "ten", "five", "md"]
    assert bench.normalise("OK, 12,000 files didn’t") == ["okay", "twelve", "thousand", "files", "didn't"]
    w = bench.wer(bench.normalise("Let me look in the atlas folder."),
                  bench.normalise("Let me look at the atlas holder, please."))
    assert w["sub"] == [["in", "at"], ["folder", "holder"]] and w["ins"] == ["please"] and w["del"] == []
    assert w["wer"] == pytest.approx(3 / 7) and w["errors"] == 3
    assert bench.wer(["a", "b"], [])["del"] == ["a", "b"] and bench.wer([], [])["wer"] == 0.0
    assert bench.wer([], ["uh"])["wer"] == 1.0
    words = bench.tokens_to_words([(" Let", 0.0, 0.2), (" me", 0.2, 0.3), (" lo", 0.3, 0.4), ("ok", 0.4, 0.55),
                                   (".", 0.55, 0.6)])
    assert [w["w"] for w in words] == ["Let", "me", "look."] and (words[2]["start"], words[2]["end"]) == (0.3, 0.6)


# ---------------------------------------------------------------------------------------------------- scorers

def test_pitch_metrics_on_planted_tracks():
    n = 200
    steady = np.full(n, 120.0)
    glide = 100 * 2 ** (np.arange(n) / n)                                   # an octave over 2 s, smooth
    jumps = np.where((np.arange(n) // 25) % 2 == 0, 110.0, 185.0)           # seven 9 st jumps
    paused = steady.copy()
    paused[50:60], paused[60:] = 0, 240                                     # an octave up across a pause: no jump
    s, g, j, p = (scorers.f0_metrics(x) for x in (steady, glide, jumps, paused))
    assert s["f0_range_st"] < 1e-9 and s["f0_sd_st"] < 1e-9 and s["jump_max_st"] < 1e-9 and s["jumps_over6"] == 0
    assert 10 < g["f0_range_st"] < 11.5 and g["jump_max_st"] < 0.1 and g["jumps_over6"] == 0
    assert j["jump_max_st"] == pytest.approx(12 * np.log2(185 / 110)) and j["jumps_over6"] == 7
    assert p["jumps_over6"] == 0 and p["voiced_share"] == pytest.approx(0.95) and p["f0_range_st"] > 11
    assert scorers.f0_metrics(np.zeros(50))["f0_range_st"] is None


@needs_scorer_venv
def test_praat_pitch_on_planted_signals():
    out = subprocess.run([str(SCORER_PY), str(TOOLS / "tts_bench_scorers.py"), "selftest-pitch"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    m = json.loads(out.stdout.strip().splitlines()[-1])
    assert m["steady"]["f0_range_st"] < 0.5 and m["steady"]["jump_max_st"] < 1 and m["steady"]["jumps_over6"] == 0
    assert m["glide"]["f0_range_st"] > 9 and m["glide"]["jump_max_st"] < 2 and m["glide"]["jumps_over6"] == 0
    assert m["jumps"]["jump_max_st"] > 8 and m["jumps"]["jumps_over6"] >= 5
    assert all(m[k]["voiced_share"] > 0.9 for k in m)


def test_seam_ratio_finds_planted_discontinuities():
    seams = [int(s * RATE) for s in (1.5, 3.0, 4.5)]
    controls = [int(s * RATE) for s in (0.75, 2.25, 3.75, 5.25)]
    t = np.arange(6 * RATE) / RATE
    phase = np.zeros(len(t))
    for k, b in enumerate(seams, 1):        # each chunk restarts a quarter period off, as a broken decoder state would
        phase[b:] = k * np.pi / 2
    clean = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    broken = (0.3 * np.sin(2 * np.pi * 220 * t + phase)).astype(np.float32)
    c, b = scorers.seam_stats(clean, RATE, seams, controls), scorers.seam_stats(broken, RATE, seams, controls)
    assert b["seam_ratio_median"] > 100 and b["seams_over_p99"] == 3        # 583x on this Mac
    assert c["seam_ratio_median"] < 3 and c["seams_over_p99"] == 0          # the window's ripple on a pure tone: 1.7x
    assert b["control_ratio_median"] < 3 and b["control_over_p99"] == 0 and b["control_n"] == 4


def test_pauses_and_leading_silence_on_planted_silences(tmp_path):
    x = np.concatenate([silence(0.4), tone(1.0), silence(0.5), tone(0.8), silence(0.2), tone(0.6), silence(0.3)])
    p = scorers.pauses(x, RATE)
    assert p["lead_s"] == pytest.approx(0.4) and p["tail_s"] == pytest.approx(0.3)
    assert p["gaps"] == [pytest.approx(0.5)] and p["longest_s"] == pytest.approx(0.5)   # 0.2 s is no pause
    assert p["speech_span_s"] == pytest.approx(3.1)
    bench.write_wav(tmp_path / "x.wav", x)
    y, _ = scorers.read_wav(tmp_path / "x.wav")
    q, g = scorers.pauses(y, RATE), speech_gaps.gaps(str(tmp_path / "x.wav"), 0.35)
    assert (g["lead_s"], g["tail_s"], g["gaps"], g["speech_span_s"]) == (
        round(q["lead_s"], 2), round(q["tail_s"], 2), [round(v, 2) for v in q["gaps"]], round(q["speech_span_s"], 2))
    assert bench.onset_s(x, RATE) == pytest.approx(0.4, abs=0.006)
    assert scorers.pauses(silence(0.5), RATE)["speech"] is False


def test_voiced_time_without_words():
    voiced = np.arange(0, 3.0, 0.01)
    words = [{"w": "hi", "start": 0.2, "end": 1.0}, {"w": "there", "start": 1.1, "end": 2.0}]
    assert scorers.uncovered_voiced_s(voiced, words) == pytest.approx(1.0, abs=0.03)
    assert scorers.uncovered_voiced_s(np.zeros(0), words) == 0.0


# ---------------------------------------------------------------------------------------------------- report

def test_report_aggregates_fake_scores(tmp_path):
    settings = bench.expand_grid(tts_conf(), ["temp0.5", "kokoro"])
    items = [bench.Item("a", "reply", "Hello there.", ["Hello there."]), bench.Item("b", "short", "Done.", ["Done."])]
    bench.write_manifest(tmp_path, settings, items, 2, bench.Trim())
    first_out = {"baseline": 0.15, "temp0.5": 0.25, "kokoro": 0.6}
    for s in ("baseline", "temp0.5", "kokoro"):
        (tmp_path / s).mkdir()
        for it in items:
            for r in (0, 1):
                stem = tmp_path / s / f"{it.id}-{r}"
                runaway = s == "kokoro" and it.id == "a" and r == 0
                side = {"item": it.id, "kind": it.kind, "r": r, "setting": s, "text": it.text, "n_words": 2,
                        "n_chars": len(it.text), "duration_s": 10.0 if runaway else 2.0, "first_chunk_s": 0.1,
                        "first_out_s": first_out[s], "lead_heard_s": 0.03, "lead_raw_s": 0.5 if s == "kokoro" else 0.1,
                        "hit_cap": False, "llm_chars_first": 14, "llm_chars_first_one_per_sentence": 14}
                stem.with_suffix(".json").write_text(json.dumps(side))
                bad_wer = s == "baseline" and it.id == "b" and r == 0
                stem.with_suffix(".asr.json").write_text(json.dumps(
                    {"hyp": "x", "wer": 0.5 if bad_wer else 0.0, "sub": [["done", "dun"]] if bad_wer else [],
                     "del": [], "ins": []}))
                laugh = 0.9 if (s == "baseline" and it.id == "a" and r == 1) else 0.02
                stem.with_suffix(".score.json").write_text(json.dumps(
                    {"laugh_max": laugh, "mos": 3.8 if s != "baseline" else 3.2, "jump_max_st": 2.0,
                     "f0_range_st": 6.0, "neural": True}))
    rows = bench.gather(tmp_path)
    assert len(rows) == 12
    summary = bench.summarise(rows, ["baseline", "temp0.5", "kokoro"])
    b, t, k = summary["baseline"], summary["temp0.5"], summary["kokoro"]
    assert (b["fail_any"], b["fail_wer"], b["fail_laugh"], b["fail_mos"], b["measured_mos"]) == (2, 1, 1, 0, 4)
    assert t["fail_any"] == 0 and k["fail_runaway"] == 1 and k["fail_lead"] == 4 and k["fail_any"] == 4
    assert b["kinds"]["short"] == (1, 2) and b["kinds"]["reply"] == (1, 2)
    assert b["first_audio_s"]["median"] == pytest.approx(0.18) and t["mos"]["median"] == pytest.approx(3.8)
    assert t["delta_first_audio"]["mean"] == pytest.approx(0.1)
    assert k["delta_first_audio"]["mean"] == pytest.approx(0.45)
    ranking = bench.rank(summary)
    assert [r["setting"] for r in ranking] == ["temp0.5", "baseline", "kokoro"]
    assert ranking[0]["within_budget"] and not ranking[-1]["within_budget"]
    v = {r["flag"]: r for r in bench.verdicts(summary, {s.name: s.to_json() for s in settings})}
    assert v["laugh>0.5"]["why"].startswith("a setting: temp0.5")
    assert v["lead>0.3s"]["why"] == "the baseline passes"
    report, results = bench.write_report(tmp_path)
    text = report.read_text()
    assert "**Best within the budget: temp0.5**" in text and "| kokoro |" in text
    assert len(results.read_text().splitlines()) == 12
