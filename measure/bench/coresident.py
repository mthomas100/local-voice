#!/usr/bin/env python
"""coresident.py — load Parakeet and one TTS model in ONE process, run each once, report combined MLX memory."""
import argparse, json, resource, time
import numpy as np
import mlx.core as mx


def gib(x): return round(x / 2**30, 3)
def rss(): return gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", default="mlx-community/parakeet-tdt-0.6b-v3")
    ap.add_argument("--tts", required=True)
    ap.add_argument("--voice", default=None); ap.add_argument("--lang", default=None); ap.add_argument("--kokoro", action="store_true")
    ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = open(a.out, "a")
    def emit(**d):
        d.update(ts=time.strftime("%H:%M:%S")); out.write(json.dumps(d) + "\n"); out.flush(); print(json.dumps(d), flush=True)
    from mlx_audio.stt.utils import load_model as load_stt
    from mlx_audio.tts.utils import load_model as load_tts
    emit(event="start", active_gib=gib(mx.get_active_memory()), rss_gib=rss())
    t = time.perf_counter(); stt = load_stt(a.stt); mx.eval(stt.parameters())
    emit(event="stt_loaded", model=a.stt, load_s=round(time.perf_counter() - t, 3), active_gib=gib(mx.get_active_memory()), peak_gib=gib(mx.get_peak_memory()), rss_gib=rss())
    t = time.perf_counter(); tts = load_tts(a.tts); mx.eval(tts.parameters())
    emit(event="tts_loaded", model=a.tts, load_s=round(time.perf_counter() - t, 3), active_gib=gib(mx.get_active_memory()), peak_gib=gib(mx.get_peak_memory()), rss_gib=rss())
    kw = {}
    if a.voice: kw["voice"] = a.voice
    if a.lang: kw["lang_code"] = a.lang
    if not a.kokoro: kw["stream"] = True; kw["streaming_interval"] = 0.5
    for i in range(2):
        t = time.perf_counter(); r = stt.generate(a.clip); stt_s = time.perf_counter() - t
        txt = getattr(r, "text", "")
        t = time.perf_counter(); n = 0; first = None
        for g in tts.generate(text="Sure, I can check the weather for tomorrow afternoon.", **kw):
            _ = np.asarray(g.audio); n += 1
            if first is None: first = time.perf_counter() - t
        tts_s = time.perf_counter() - t
        emit(event="round", run=i + 1, stt_s=round(stt_s, 3), stt_text=txt, tts_ttfa_s=round(first, 3) if first else None, tts_total_s=round(tts_s, 3), tts_chunks=n,
             active_gib=gib(mx.get_active_memory()), peak_gib=gib(mx.get_peak_memory()), cache_gib=gib(mx.get_cache_memory()), rss_gib=rss())
    emit(event="done", active_gib=gib(mx.get_active_memory()), peak_gib=gib(mx.get_peak_memory()), rss_gib=rss())


if __name__ == "__main__":
    main()
