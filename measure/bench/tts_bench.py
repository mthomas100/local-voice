#!/usr/bin/env python
"""tts_bench.py — measure one mlx-audio TTS model: load time, warm-up, time-to-first-audio (streaming),
total generation time, real-time factor, MLX peak memory, process peak RSS. One model per process.

Writes one JSON object per line to --out; saves the first run of each (text, mode) as WAV under --audio-dir.
"""
import argparse, json, os, resource, sys, time
import numpy as np
import mlx.core as mx

SHORT = "The kettle is on, and the rain has finally stopped outside."  # 11 words
LONG = ("Before we start the meeting, let me summarize where things stand. The new build passed every test "
        "last night, the design review is scheduled for Thursday morning, and two customers have already asked "
        "when they can try the beta. I think we are in good shape.")  # 44 words
WARM = "Hello there, this is a warm-up sentence."


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS reports bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--instruct", default=None)
    ap.add_argument("--stream-intervals", default="2.0,0.5", help="comma list of streaming_interval values; Kokoro ignores")
    ap.add_argument("--kokoro", action="store_true", help="Kokoro: no stream kwarg; emulate streaming by sentence split")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-espeak-fallback", action="store_true", help="Kokoro: disable misaki EspeakFallback (espeak-ng data path crash)")
    args = ap.parse_args()
    os.makedirs(args.audio_dir, exist_ok=True)
    out = open(args.out, "a")

    def emit(**d):
        d.update(model=args.model, tag=args.tag, ts=time.strftime("%H:%M:%S"))
        out.write(json.dumps(d) + "\n"); out.flush()
        print(json.dumps(d), flush=True)

    import soundfile as sf
    from mlx_audio.tts.utils import load_model
    if args.no_espeak_fallback:
        import misaki.espeak as _esp
        class _Disabled:
            def __init__(self, *a, **k): raise RuntimeError("EspeakFallback disabled by bench (--no-espeak-fallback)")
        _esp.EspeakFallback = _Disabled

    t0 = time.perf_counter()
    model = load_model(args.model)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0
    emit(event="load", load_s=round(load_s, 3), active_gib=mx.get_active_memory() / 2**30,
         peak_gib=mx.get_peak_memory() / 2**30, rss_gib=rss_gib(), mlx_version=mx.__version__)

    kw = {}
    if args.voice: kw["voice"] = args.voice
    if args.lang: kw["lang_code"] = args.lang
    if args.instruct: kw["instruct"] = args.instruct

    def run_once(text, stream, interval, split_pattern=None):
        gk = dict(kw)
        if args.kokoro:
            if split_pattern: gk["split_pattern"] = split_pattern
        else:
            gk["stream"] = stream
            if stream: gk["streaming_interval"] = interval
        mx.reset_peak_memory()
        chunks, sr, first_s, first_dur, n = [], None, None, None, 0
        t = time.perf_counter()
        for r in model.generate(text=text, **gk):
            a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
            now = time.perf_counter()
            if first_s is None:
                first_s, first_dur = now - t, len(a) / r.sample_rate
            chunks.append(a); sr = r.sample_rate; n += 1
        total = time.perf_counter() - t
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        dur = len(audio) / sr if sr else 0.0
        return dict(ttfa_s=round(first_s, 3) if first_s is not None else None,
                    first_chunk_audio_s=round(first_dur, 3) if first_dur is not None else None,
                    total_s=round(total, 3), audio_s=round(dur, 3), rtf=round(total / dur, 3) if dur else None,
                    chunks=n, sample_rate=sr, peak_gib=round(mx.get_peak_memory() / 2**30, 3),
                    rms=round(float(np.sqrt(np.mean(audio**2))), 4) if len(audio) else 0.0), audio, sr

    # warm-up (first generate compiles Metal kernels)
    t = time.perf_counter()
    w, _, _ = run_once(WARM, False, None)
    emit(event="warmup", text=WARM, **w, wall_s=round(time.perf_counter() - t, 3))

    modes = [("nonstream", False, None)]
    if args.kokoro:
        modes.append(("sentence_split", False, None))
    else:
        for iv in [float(x) for x in args.stream_intervals.split(",") if x]:
            modes.append((f"stream_{iv}", True, iv))

    for tname, text in [("short", SHORT), ("long", LONG)]:
        for mname, stream, iv in modes:
            for i in range(args.runs):
                sp = r"(?<=[.!?])\s+" if mname == "sentence_split" else None
                res, audio, sr = run_once(text, stream, iv, split_pattern=sp)
                emit(event="gen", text_name=tname, words=len(text.split()), mode=mname, run=i + 1, **res,
                     rss_gib=round(rss_gib(), 3))
                if i == 0 and len(audio):
                    safe = args.model.replace("/", "_")
                    sf.write(os.path.join(args.audio_dir, f"{safe}_{tname}_{mname}.wav"), audio, sr)
    emit(event="done", peak_gib_process=round(mx.get_peak_memory() / 2**30, 3), rss_gib=round(rss_gib(), 3))


if __name__ == "__main__":
    main()
