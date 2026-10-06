#!/usr/bin/env python
"""vad_bench.py — Silero VAD (mlx-community/silero-vad) and Smart Turn v3 (mlx-community/smart-turn-v3) latency.
Silero: per-512-sample (32 ms) feed() latency over the whole clip (p50/p95), whole-clip predict_proba, speech timestamps.
Smart Turn: predict_endpoint() on the last 8 s window, 10 calls, p50/p95, plus the probability on the full clip and
on the clip cut mid-sentence (should be 'incomplete')."""
import argparse, json, resource, time
import numpy as np, mlx.core as mx


def gib(x): return round(x / 2**30, 3)
def pct(a): a = np.array(a); return dict(p50=round(float(np.percentile(a, 50)), 4), p95=round(float(np.percentile(a, 95)), 4), max=round(float(a.max()), 4), n=len(a))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = open(a.out, "a")
    def emit(**d):
        d.update(ts=time.strftime("%H:%M:%S")); out.write(json.dumps(d) + "\n"); out.flush(); print(json.dumps(d), flush=True)
    from mlx_audio.vad.utils import load_model
    from mlx_audio.utils import load_audio
    audio = np.asarray(load_audio(a.clip, 16000), dtype=np.float32); clip_s = len(audio) / 16000
    # --- Silero ---
    t = time.perf_counter(); vad = load_model("mlx-community/silero-vad"); mx.eval(vad.parameters()); load_s = time.perf_counter() - t
    emit(event="silero_load", load_s=round(load_s, 3), active_gib=gib(mx.get_active_memory()))
    cs = vad.config.branch_16k.chunk_size  # 512 samples at 16 kHz (silero_vad/config.py:18); ModelConfig has no chunk_size (fixed 2026-10-05)
    state = None; probs = []; lat = []
    # warm-up
    p, st = vad.feed(audio[:cs], None, 16000); mx.eval(p)
    state = None
    for i in range(0, len(audio) - cs + 1, cs):
        t = time.perf_counter(); p, state = vad.feed(audio[i:i + cs], state, 16000); v = float(p.reshape(-1)[0].item()); lat.append(time.perf_counter() - t); probs.append(v)
    probs = np.array(probs); speech = probs > 0.5
    emit(event="silero_stream", clip_s=round(clip_s, 3), chunk_samples=cs, chunk_ms=cs / 16, chunks=len(lat), feed_latency_s=pct(lat), speech_chunks=int(speech.sum()),
         first_speech_s=round(float(np.argmax(speech) * cs / 16000), 3) if speech.any() else None,
         last_speech_s=round(float((len(speech) - np.argmax(speech[::-1])) * cs / 16000), 3) if speech.any() else None, peak_gib=gib(mx.get_peak_memory()))
    lat2 = []
    for _ in range(5):
        t = time.perf_counter(); pp = vad.predict_proba(audio, sample_rate=16000); mx.eval(pp); lat2.append(time.perf_counter() - t)
    t = time.perf_counter(); ts_ = vad.get_speech_timestamps(audio, sample_rate=16000); ts_s = time.perf_counter() - t
    emit(event="silero_whole_clip", predict_proba_s=pct(lat2), get_speech_timestamps_s=round(ts_s, 4), timestamps=[{k: (round(v / 16000, 3) if isinstance(v, (int, float)) and v > 100 else v) for k, v in (d.items() if isinstance(d, dict) else d.__dict__.items())} for d in ts_][:8])
    # --- Smart Turn v3 ---
    t = time.perf_counter(); st_model = load_model("mlx-community/smart-turn-v3"); mx.eval(st_model.parameters()); load_s = time.perf_counter() - t
    emit(event="smart_turn_load", load_s=round(load_s, 3), active_gib=gib(mx.get_active_memory()), max_audio_s=st_model.config.processor_config.max_audio_seconds)
    r = st_model.predict_endpoint(audio, sample_rate=16000)  # warm-up
    lat = []
    for _ in range(10):
        t = time.perf_counter(); r = st_model.predict_endpoint(audio, sample_rate=16000); lat.append(time.perf_counter() - t)
    cut = audio[: int(len(audio) * 0.45)]
    lat_cut = []
    for _ in range(5):
        t = time.perf_counter(); rc = st_model.predict_endpoint(cut, sample_rate=16000); lat_cut.append(time.perf_counter() - t)
    emit(event="smart_turn", clip_s=round(clip_s, 3), full_clip=dict(prediction=int(r.prediction), probability=round(r.probability, 4), latency_s=pct(lat)),
         cut_at_s=round(len(cut) / 16000, 3), cut_clip=dict(prediction=int(rc.prediction), probability=round(rc.probability, 4), latency_s=pct(lat_cut)), peak_gib=gib(mx.get_peak_memory()))
    emit(event="done", active_gib_both=gib(mx.get_active_memory()), peak_gib=gib(mx.get_peak_memory()), rss_gib=gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    main()
