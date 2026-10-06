#!/bin/zsh
# setup.sh — build and fetch what run_remaining.sh needs, outside any GPU hold: network and CPU only, no model is loaded
# on the GPU (2026-10-05). Idempotent; run_remaining.sh calls it before taking its hold. Exit 0 when every phase is ready.
#   1. .venv-kokoro: the mlx-audio clone's interpreter (Python 3.11.15) plus requirements-kokoro.txt (misaki 0.9.4, spaCy,
#      en_core_web_sm), a .pth that puts the clone and its venv's site-packages on the path, and a sitecustomize.py that
#      disables misaki's EspeakFallback (espeakng-loader's baked-in data path aborts the process on this Mac).
#      Replaces an earlier scratch venv-kokoro under /private/tmp, which is lost at reboot.
#   2. .venv-vlm: Python 3.12 plus requirements-vlm.txt (mlx-vlm 0.7.4, mlx 0.32.3, mlx-audio 0.5.7), the package set of
#      ~/repos/local-decision/.venv. VoiceChat's MLX checkpoint loads only through mlx-vlm.
#   3. Auxiliary hub files the loaders fetch on first load (none was cached on 2026-10-05): the Mimi codec
#      (kyutai/moshiko-pytorch-bf16, 0.385 GB) plus the tokenizer and default voice prompt of
#      Marvis-AI/marvis-tts-250m-v0.2 for Marvis (sesame.py:470-474, 640), and the Qwen/Qwen2.5-0.5B tokenizer for
#      VibeVoice (vibevoice.py:328). Tokenizers come through AutoTokenizer, so no model weights of those repos are fetched.
#   4. Checks, with HF_HUB_OFFLINE=1 as the runner will have it: every model repo and auxiliary file resolves from the
#      cache, and the two test clips exist (regenerated with `say` + `afconvert` if missing; *.wav is gitignored).
here=${0:a:h}; SP=${SP:-${here:h}}
CLONE=$HOME/repos/mlx-audio; CPY=$CLONE/.venv/bin/python
command -v uv >/dev/null || { echo "setup: uv not found"; exit 1; }
[ -x $CPY ] || { echo "setup: the mlx-audio clone venv is missing ($CPY)"; exit 1; }
$CPY -c 'import importlib.util as u,sys; sys.exit(0 if u.find_spec("mlx_audio") else 1)' || { echo "setup: mlx_audio not importable from $CPY"; exit 1; }
fail=0; warn=0   # fail: a hard problem (exit 1, nothing starts); warn: a phase not ready (it will fail offline, others run)

# 1. Kokoro venv
PK=$here/.venv-kokoro/bin/python
kokoro_ok() { [ -x $PK ] && $PK -c 'import importlib.util as u,sys
import misaki.en, spacy; spacy.load("en_core_web_sm")
sys.exit(0 if u.find_spec("mlx_audio") and u.find_spec("mlx") else 1)' >/dev/null 2>&1; }
if kokoro_ok; then echo "setup: .venv-kokoro ready"; else
  echo "setup: building .venv-kokoro"
  rm -rf $here/.venv-kokoro
  uv venv --quiet --python $CPY $here/.venv-kokoro && uv pip install --quiet --python $PK -r $here/requirements-kokoro.txt || { echo "setup: .venv-kokoro install failed"; exit 1; }
  site=$($PK -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  printf '%s\n%s\n' $CLONE $($CPY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') > $site/zz_mlx_audio_clone.pth
  cat > $site/sitecustomize.py <<'PYEOF'
# .venv-kokoro only (bench/setup.sh): misaki's EspeakFallback aborts the process on this Mac (espeakng-loader's data path
# is baked in at build time: "Error processing file '<build-machine>/runner/work/.../phontab'"). Make its constructor raise, so
# KokoroPipeline takes its own except branch (pipeline.py:146-151) and runs with fallback=None (out-of-dictionary
# words are skipped). Same patch as the 2026-10-04 measurement venv.
try:
    import misaki.espeak as _esp
    class _DisabledEspeakFallback:
        def __init__(self, *a, **k):
            raise RuntimeError("EspeakFallback disabled by sitecustomize (espeak-ng data path crash)")
    _esp.EspeakFallback = _DisabledEspeakFallback
except Exception:
    pass
PYEOF
  kokoro_ok && echo "setup: .venv-kokoro built" || { echo "setup: .venv-kokoro built but its import check failed"; exit 1; }
fi

# 2. mlx-vlm venv
PV=$here/.venv-vlm/bin/python
vlm_ok() { [ -x $PV ] && $PV -c 'import importlib.metadata as m,importlib.util as u,sys
sys.exit(0 if u.find_spec("mlx_vlm") and m.version("mlx-vlm")=="0.7.4" else 1)' >/dev/null 2>&1; }
if vlm_ok; then echo "setup: .venv-vlm ready"; else
  echo "setup: building .venv-vlm"
  rm -rf $here/.venv-vlm
  uv venv --quiet --python 3.12 $here/.venv-vlm && uv pip install --quiet --python $PV -r $here/requirements-vlm.txt || { echo "setup: .venv-vlm install failed (fallback: VLM_PY=~/repos/local-decision/.venv/bin/python, read-only use)"; exit 1; }
  vlm_ok && echo "setup: .venv-vlm built: $($PV -c 'import importlib.metadata as m; print(*(f"{p} {m.version(p)}" for p in ("mlx-vlm","mlx","mlx-audio","transformers")), sep=", ")')" || { echo "setup: .venv-vlm import check failed"; exit 1; }
fi

# 3 and 4. auxiliary files, then the offline check
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE $CPY - <<'PYEOF' || warn=1
import os, time
from huggingface_hub import hf_hub_download, try_to_load_from_cache
files = [("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors", "marvis: Mimi codec"),
         ("Marvis-AI/marvis-tts-250m-v0.2", "prompts/conversational_a.wav", "marvis: default voice prompt"),
         ("Marvis-AI/marvis-tts-250m-v0.2", "prompts/conversational_a.txt", "marvis: default voice prompt text")]
for repo, fn, why in files:
    p = try_to_load_from_cache(repo, fn)
    if isinstance(p, str):
        print(f"setup: cached  {repo}/{fn} ({why})")
        continue
    t = time.time(); p = hf_hub_download(repo, fn)
    print(f"setup: fetched {repo}/{fn} ({why}): {os.path.getsize(p)/1e9:.3f} GB in {time.time()-t:.1f} s")
from transformers import AutoTokenizer
for repo, why in [("Qwen/Qwen2.5-0.5B", "vibevoice: text tokenizer"), ("Marvis-AI/marvis-tts-250m-v0.2", "marvis: text tokenizer")]:
    try:
        AutoTokenizer.from_pretrained(repo, local_files_only=True); print(f"setup: cached  {repo} tokenizer ({why})")
    except Exception:
        t = time.time(); AutoTokenizer.from_pretrained(repo); print(f"setup: fetched {repo} tokenizer ({why}) in {time.time()-t:.1f} s")
PYEOF
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $CPY - <<'PYEOF' || warn=1
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer
need = {
  "pocket": [("snap", "mlx-community/pocket-tts"), ("file", "kyutai/pocket-tts-without-voice-cloning", "tokenizer.model", "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"),  # gitleaks:allow (pinned HF file revision, not a secret)
             ("file", "kyutai/pocket-tts-without-voice-cloning", "embeddings/alba.safetensors", "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3")],
  "vibevoice": [("snap", "mlx-community/VibeVoice-Realtime-0.5B-8bit"), ("tok", "Qwen/Qwen2.5-0.5B")],
  "marvis": [("snap", "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"), ("tok", "Marvis-AI/marvis-tts-250m-v0.2"),
             ("file", "kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors", None),
             ("file", "Marvis-AI/marvis-tts-250m-v0.2", "prompts/conversational_a.wav", None)],
  "vad": [("snap", "mlx-community/silero-vad"), ("snap", "mlx-community/smart-turn-v3")],
  "nemotron": [("snap", "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")],
  "coresident": [("snap", "mlx-community/parakeet-tdt-0.6b-v3"), ("snap", "mlx-community/Kokoro-82M-bf16"), ("snap", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")],
  "voicechat": [("snap", "mlx-community/NemotronLabs-VoiceChat-11B-4bit"), ("snap", "mlx-community/parakeet-tdt-0.6b-v3")],
}
bad = 0
for phase, items in need.items():
    missing = []
    for it in items:
        try:
            if it[0] == "snap": snapshot_download(it[1], local_files_only=True)
            elif it[0] == "tok": AutoTokenizer.from_pretrained(it[1], local_files_only=True)
            else: hf_hub_download(it[1], it[2], revision=it[3], local_files_only=True)
        except Exception as e:
            missing.append(f"{it[1]} {it[2] if len(it) > 2 else ''}".strip() + f" ({type(e).__name__})")
    print(f"setup: offline check {phase:10s} " + ("READY" if not missing else "NOT READY: " + "; ".join(missing)))
    bad += bool(missing)
raise SystemExit(1 if bad else 0)
PYEOF

# clips (gitignored; synthetic `say` speech, texts as in 09b-phase2-report.md)
mkdir -p $SP/audio
mk() { [ -s $SP/audio/$1_16k.wav ] && return; say -v Samantha -o $SP/audio/$1.aiff "$2" && afconvert -f WAVE -d LEI16@16000 -c 1 $SP/audio/$1.aiff $SP/audio/$1_16k.wav && rm -f $SP/audio/$1.aiff && echo "setup: regenerated audio/$1_16k.wav"; }
mk clip3 "Could you check the weather for tomorrow afternoon?"
mk clip8 "I was thinking we could move the planning meeting to Thursday afternoon, since two people are travelling on Wednesday. Does that work for you, or would Friday morning be better?"
[ -s $SP/audio/clip3_16k.wav ] && [ -s $SP/audio/clip8_16k.wav ] || { echo "setup: test clips missing"; fail=1; }
if [ $fail != 0 ]; then echo "setup: FAILED (see above)"; elif [ $warn != 0 ]; then echo "setup: ready, except the phases marked NOT READY above (they will fail offline; the others run)"; else echo "setup: all phases ready"; fi
exit $fail
