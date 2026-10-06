#!/usr/bin/env python
"""stt_bench.py — Parakeet (mlx-audio) latency on a short clip: load, warm-up, 3 runs from path and from a
pre-loaded array, plus stream_generate() chunk timings. One process."""
import argparse, json, os, resource, sys, time
import numpy as np
import mlx.core as mx


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = open(args.out, "a")

    def emit(**d):
        d.update(model=args.model, ts=time.strftime("%H:%M:%S"))
        out.write(json.dumps(d) + "\n"); out.flush(); print(json.dumps(d), flush=True)

    from mlx_audio.stt.utils import load_model
    from mlx_audio.utils import load_audio
    t0 = time.perf_counter(); model = load_model(args.model); mx.eval(model.parameters())
    emit(event="load", load_s=round(time.perf_counter() - t0, 3), active_gib=round(mx.get_active_memory() / 2**30, 3),
         peak_gib=round(mx.get_peak_memory() / 2**30, 3), rss_gib=round(rss_gib(), 3))

    def text_of(r):
        return getattr(r, "text", None) or " ".join(s.text for s in getattr(r, "sentences", []))

    t = time.perf_counter(); r = model.generate(args.clip)
    emit(event="warmup", s=round(time.perf_counter() - t, 3), text=text_of(r))

    sr = model.preprocessor_config.sample_rate
    audio = load_audio(args.clip, sr)
    mx.eval(audio)
    clip_s = audio.shape[0] / sr
    for i in range(args.runs):
        mx.reset_peak_memory(); t = time.perf_counter(); r = model.generate(args.clip)
        emit(event="gen", source="path", run=i + 1, clip_s=round(clip_s, 3), s=round(time.perf_counter() - t, 3),
             text=text_of(r), peak_gib=round(mx.get_peak_memory() / 2**30, 3))
    for i in range(args.runs):
        mx.reset_peak_memory(); t = time.perf_counter(); r = model.generate(audio)
        emit(event="gen", source="array", run=i + 1, clip_s=round(clip_s, 3), s=round(time.perf_counter() - t, 3),
             text=text_of(r), peak_gib=round(mx.get_peak_memory() / 2**30, 3))
    # chunked "streaming" over the finished clip
    for cd, ov in [(5.0, 1.0), (1.0, 0.5)]:
        t = time.perf_counter(); events = []
        for sr_ in model.stream_generate(audio, chunk_duration=cd, overlap_duration=ov):
            events.append(dict(at_s=round(time.perf_counter() - t, 3), text=getattr(sr_, "text", ""), final=getattr(sr_, "is_final", None)))
        emit(event="stream_generate", chunk_duration=cd, overlap=ov, total_s=round(time.perf_counter() - t, 3), events=events)
    emit(event="done", rss_gib=round(rss_gib(), 3), peak_gib=round(mx.get_peak_memory() / 2**30, 3))


if __name__ == "__main__":
    main()
