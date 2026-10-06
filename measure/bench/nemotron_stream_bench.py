#!/usr/bin/env python
"""nemotron_stream_bench.py — Nemotron 3.5 ASR streaming via mlx-audio's live StreamingSession.
Feeds a 16 kHz clip in 320 ms chunks (5120 samples), paced in real time, and drives step() continuously
on the same thread. Records: load, per-chunk compute (time spent in step() per 320 ms of audio), every
partial delta with its wall time and how much audio had been fed, final transcript, peak memory."""
import argparse, json, resource, time
import numpy as np, mlx.core as mx


def gib(x): return round(x / 2**30, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--paced", action="store_true", help="feed chunks at real time; otherwise as fast as possible")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = open(a.out, "a")
    def emit(**d):
        d.update(ts=time.strftime("%H:%M:%S")); out.write(json.dumps(d) + "\n"); out.flush(); print(json.dumps(d), flush=True)
    from mlx_audio.stt.utils import load_model
    from mlx_audio.utils import load_audio
    t = time.perf_counter(); model = load_model(a.model); mx.eval(model.parameters())
    emit(event="load", model=a.model, load_s=round(time.perf_counter() - t, 3), active_gib=gib(mx.get_active_memory()), rss_gib=gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    sr = model.preprocessor_config.sample_rate
    sess = model.create_streaming_session()
    emit(event="session", input_sample_rate=sess.input_sample_rate, encoder_chunk_mel=sess._encoder.chunk_mel, hop=model.preprocessor_config.hop_length,
         native_chunk_s=round(sess._encoder.chunk_mel * model.preprocessor_config.hop_length / sr, 3), att_context=model.default_att_context_size)
    for ci, clip in enumerate(a.clips):
        audio = np.asarray(load_audio(clip, sr), dtype=np.float32)
        n = int(sr * a.chunk_ms / 1000)
        sess = model.create_streaming_session()
        mx.reset_peak_memory()
        deltas, chunk_costs, text = [], [], ""
        t0 = time.perf_counter(); fed = 0
        for start in range(0, len(audio), n):
            chunk = audio[start:start + n]
            if a.paced:
                target = t0 + (start + len(chunk)) / sr   # moment this chunk would have finished arriving from a mic
                while time.perf_counter() < target: time.sleep(0.001)
            sess.feed(chunk); fed = start + len(chunk)
            if start + n >= len(audio): sess.close()
            tc = time.perf_counter(); steps = 0
            while True:
                d = sess.step(max_decode_tokens=8); steps += 1
                now = time.perf_counter()
                for x in d:
                    text += x; deltas.append(dict(wall_s=round(now - t0, 3), audio_fed_s=round(fed / sr, 3), text=x))
                if sess.done or (not d and not sess._encoded and not sess._audio and not (sess._closed and not sess._flushed)):
                    break
                if steps > 2000: break
            chunk_costs.append(round(time.perf_counter() - tc, 4))
        # drain to completion
        td = time.perf_counter()
        while not sess.done:
            d = sess.step(max_decode_tokens=8); now = time.perf_counter()
            for x in d: text += x; deltas.append(dict(wall_s=round(now - t0, 3), audio_fed_s=round(fed / sr, 3), text=x))
            if time.perf_counter() - td > 10: break
        total = time.perf_counter() - t0
        cc = np.array(chunk_costs)
        emit(event="clip", clip=clip, clip_s=round(len(audio) / sr, 3), chunk_ms=a.chunk_ms, paced=a.paced, chunks=len(chunk_costs), total_wall_s=round(total, 3),
             step_cost_per_chunk_s=dict(p50=round(float(np.percentile(cc, 50)), 4), p95=round(float(np.percentile(cc, 95)), 4), max=round(float(cc.max()), 4), sum=round(float(cc.sum()), 3)),
             first_delta=deltas[0] if deltas else None, last_delta=deltas[-1] if deltas else None, n_deltas=len(deltas), text=text.strip(),
             peak_gib=gib(mx.get_peak_memory()), deltas=deltas)
    emit(event="done", rss_gib=gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), peak_gib=gib(mx.get_peak_memory()))


if __name__ == "__main__":
    main()
