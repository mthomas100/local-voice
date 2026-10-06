#!/usr/bin/env python
"""voicechat_bench.py — NVIDIA NemotronLabs VoiceChat 11B on MLX: load time, peak memory, per-80 ms-frame compute time
(p50/p95/max) over >= 60 s of synthetic conversation (say-generated clips separated by silence), the runtime's own
per-stage frame profile, assistant text, user transcript, function-channel text, and the output audio as a 16-bit WAV
(22.05 kHz). With --transcribe, VoiceChat is freed and Parakeet transcribes the output, so 'speech or not' is
machine-judged.

Runtimes (2026-10-05): the mlx-community checkpoints are in mlx-vlm's format (config.json has
mlx_runtime_config_version 2) and load only with mlx-vlm (0.7.4: mlx_vlm.load, model.create_session(processor),
.create_streaming_session(profile=True)). mlx-audio's sts loader (create_duplex_session) parses the NeMo layout and
finds none of its keys in them. --runtime auto picks by that config key. Frames are fed as fast as possible: the
per-frame time is compute, and real time needs p95 under 80 ms."""
import argparse, gc, json, os, resource, time, wave
import numpy as np, mlx.core as mx


def gib(x): return round(x / 2**30, 3)
def pct(a): a = np.array(a); return dict(p50=round(float(np.percentile(a, 50)), 4), p95=round(float(np.percentile(a, 95)), 4), max=round(float(a.max()), 4), mean=round(float(a.mean()), 4), n=len(a))
def rss(): return gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # macOS reports bytes


def detect_runtime(repo):
    from huggingface_hub import snapshot_download
    cfg = json.load(open(os.path.join(snapshot_download(repo, allow_patterns=["config.json"]), "config.json")))
    return "vlm" if "mlx_runtime_config_version" in cfg else "audio"


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/NemotronLabs-VoiceChat-11B-4bit")
    ap.add_argument("--runtime", choices=["auto", "vlm", "audio"], default="auto")
    ap.add_argument("--clips", nargs="+", required=True); ap.add_argument("--min-seconds", type=float, default=64)
    ap.add_argument("--out", required=True); ap.add_argument("--wav", required=True)
    ap.add_argument("--system-prompt", default="You are a helpful assistant. Be concise and answer in one sentence.")
    ap.add_argument("--transcribe", action="store_true", help="after the session, free VoiceChat and transcribe the output WAV with --stt")
    ap.add_argument("--stt", default="mlx-community/parakeet-tdt-0.6b-v3")
    a = ap.parse_args(); out = open(a.out, "a")
    def emit(**d):
        d.update(ts=time.strftime("%H:%M:%S")); out.write(json.dumps(d) + "\n"); out.flush(); print(json.dumps(d), flush=True)
    from mlx_audio.utils import load_audio
    runtime = a.runtime if a.runtime != "auto" else detect_runtime(a.model)
    t = time.perf_counter()
    if runtime == "vlm":
        from mlx_vlm import load as vlm_load
        model, processor = vlm_load(a.model); mx.eval(model.parameters())
        parent = model.create_session(processor)
        new_session = lambda: parent.create_streaming_session(system_prompt=a.system_prompt, seed=0, profile=True)
    else:
        from mlx_audio.sts import load
        model = load(a.model); mx.eval(model.parameters()); parent = None
        new_session = lambda: model.create_duplex_session(system_prompt=a.system_prompt, seed=0)
    load_s = time.perf_counter() - t
    emit(event="load", model=a.model, runtime=runtime, mlx_version=mx.__version__, load_s=round(load_s, 3), active_gib=gib(mx.get_active_memory()), peak_gib=gib(mx.get_peak_memory()), rss_gib=rss())
    sr = 16000
    clips = [np.asarray(load_audio(c, sr), dtype=np.float32).reshape(-1) for c in a.clips]
    parts, total, i, layout = [], 0.0, 0, []
    gaps = [4.0, 6.0, 5.0, 7.0]
    while total < a.min_seconds:
        c = clips[i % len(clips)]; g = gaps[i % len(gaps)]
        layout.append(dict(clip=os.path.basename(a.clips[i % len(clips)]), start_s=round(total, 2), speech_s=round(len(c) / sr, 2), then_silence_s=g))
        parts += [c, np.zeros(int(sr * g), np.float32)]; total += len(c) / sr + g; i += 1
    user_audio = np.concatenate(parts)
    t = time.perf_counter(); sess = new_session(); sess_s = time.perf_counter() - t
    fs = sess.frame_samples
    emit(event="session", create_s=round(sess_s, 3), frame_samples=fs, frame_ms=fs / sr * 1000, user_audio_s=round(len(user_audio) / sr, 2), layout=layout, active_gib=gib(mx.get_active_memory()))
    mx.reset_peak_memory()
    frame_t, audio_chunks, out_sr, a_text, u_text, f_text, n_audio_ev = [], [], None, "", "", "", 0
    first_audio_frame = first_text_frame = None; f_events = []; text_frames = []
    t0 = time.perf_counter()
    for k in range(0, len(user_audio) - fs + 1, fs):
        t = time.perf_counter(); evs = sess.push_audio(user_audio[k:k + fs], sample_rate=sr)
        for e in evs:
            if e.kind == "audio" and e.samples is not None:
                s = np.asarray(e.samples, dtype=np.float32).reshape(-1); audio_chunks.append(s); out_sr = e.sample_rate; n_audio_ev += 1
                if first_audio_frame is None and np.abs(s).max() > 0.01: first_audio_frame = k // fs
            elif e.kind == "assistant_text_delta":
                a_text += e.delta or ""; text_frames.append(k // fs)
                if first_text_frame is None: first_text_frame = k // fs
            elif e.kind == "user_transcript_delta": u_text += e.delta or ""
            elif e.kind == "function_delta":
                f_text += e.delta or ""; f_events.append(dict(frame=k // fs, token_id=e.token_id, delta=e.delta))
        frame_t.append(time.perf_counter() - t)
    t = time.perf_counter(); evs = sess.flush(); flush_s = time.perf_counter() - t
    for e in evs:
        if e.kind == "audio" and e.samples is not None: audio_chunks.append(np.asarray(e.samples, dtype=np.float32).reshape(-1)); out_sr = e.sample_rate
        elif e.kind == "assistant_text_delta": a_text += e.delta or ""
        elif e.kind == "user_transcript_delta": u_text += e.delta or ""
        elif e.kind == "function_delta": f_text += e.delta or ""; f_events.append(dict(frame="flush", token_id=e.token_id, delta=e.delta))
    wall = time.perf_counter() - t0
    out_sr = out_sr or getattr(sess, "output_sample_rate", 22050)
    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, np.float32)
    if len(audio): write_wav(a.wav, audio, out_sr)
    fr = int(out_sr * 0.02); n = len(audio) // fr
    voiced = float((np.abs(audio[:n * fr]).reshape(n, fr).max(axis=1) > 0.01).mean()) if n else 0.0
    prof = getattr(sess, "profile", None)
    emit(event="duplex", frames=len(frame_t), user_audio_s=round(len(user_audio) / sr, 2), wall_s=round(wall, 2), realtime_factor=round(wall / (len(user_audio) / sr), 3),
         frame_compute_s=pct(frame_t), frame_compute_after_10_s=pct(frame_t[10:]) if len(frame_t) > 20 else None, frames_over_80ms=int((np.array(frame_t) > 0.080).sum()), flush_s=round(flush_s, 3),
         first_text_at_s=round(first_text_frame * fs / sr, 2) if first_text_frame is not None else None, first_audio_at_s=round(first_audio_frame * fs / sr, 2) if first_audio_frame is not None else None,
         out_audio_s=round(len(audio) / out_sr, 2), out_sr=out_sr, voiced_fraction=round(voiced, 3), rms=round(float(np.sqrt(np.mean(audio**2))), 4) if len(audio) else 0,
         assistant_text=a_text.strip()[:2000], assistant_text_frames=text_frames[:400], user_transcript=u_text.strip()[:1000],
         function_text=f_text.strip()[:500], function_events=len(f_events), function_events_first=f_events[:20],
         runtime_profile=prof.summary(drop_first=0) if prof is not None and prof.frames else None,
         runtime_profile_after_10=prof.summary(drop_first=10) if prof is not None and len(prof.frames) > 20 else None,
         frame_times_ms=[round(x * 1000, 1) for x in frame_t],
         peak_gib=gib(mx.get_peak_memory()), active_gib=gib(mx.get_active_memory()), rss_gib=rss())
    if a.transcribe and len(audio):
        del sess, parent, model; gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
        from mlx_audio.stt.utils import load_model as load_stt
        stt = load_stt(a.stt); mx.eval(stt.parameters())
        x = mx.array(np.asarray(load_audio(a.wav, 16000), dtype=np.float32).reshape(-1))
        t = time.perf_counter(); r = stt.generate(x); st = time.perf_counter() - t
        txt = getattr(r, "text", "") or ""
        emit(event="speech_check", stt=a.stt, transcript=txt.strip()[:2000], words=len(txt.split()), stt_s=round(st, 3), out_audio_s=round(len(audio) / out_sr, 2))
    emit(event="done", peak_gib_since_last_reset=gib(mx.get_peak_memory()), rss_gib=rss())  # duplex peak is in the "duplex" event


if __name__ == "__main__":
    main()
