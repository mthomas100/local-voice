#!/bin/zsh
# run.sh — start the local voice orchestrator (M1, 2026-10-05).
#
# Refuses to start while the hold gate (127.0.0.1:8090) is draining or held: nothing may load a model then (a rule of
# this Mac since 2026-10-04); exit code 75 means "try again when the GPU is free". Otherwise it loads and warms the STT and TTS
# adapters named in config.yaml on the MLX thread, Silero and Smart Turn on the CPU, renders the busy notice, and serves:
#   protocol v1   ws://127.0.0.1:8770/v1/voice and ws://<tailnet ip>:8770/v1/voice, status at /v1/status
#   browser       http://127.0.0.1:7860/ (the project's own WebRTC page; the Mac's own browser only)
# Options: --check (load, warm, render, then exit), --config <file>, --log-level DEBUG, --port N, --browser-port N,
# --scratch DIR (a scratch copy for client tests: state, turn log, brain state, a kb clone and the spaces' roots on
# clones under DIR, loopback only; local_voice/scratch.py), --record (record every connection's microphone, playback
# and events under config.yaml's record.dir for tools/session_report.py; local only; local_voice/recorder.py).
# Every model it uses is cached already; HF_HUB_OFFLINE keeps the live path from ever downloading.
set -e
here=${0:a:h}
cd $here
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ ! -x .venv/bin/python ]; then
  echo "run.sh: creating the venv (uv sync --extra kokoro)"
  uv sync --python 3.12 --extra kokoro
fi
exec .venv/bin/python -m local_voice.server "$@"
