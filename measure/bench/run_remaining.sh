#!/bin/zsh
# run_remaining.sh [phase ...] — run the remaining phase-2 measurements inside ONE exclusive GPU hold (2026-10-05).
#   phases (default: all, in this order): pocket vibevoice marvis vad nemotron coresident voicechat
# Outer part (what you run):
#   1. bench/gate.sh refuses to start unless the GPU is clear now: hold gate open, no render or GPU job process, the
#      LLM unloaded or idle for >= 120 s. Refusal means nothing was started and nothing is queued.
#   2. bench/setup.sh builds .venv-kokoro and .venv-vlm if missing and fetches the auxiliary hub files the loaders
#      need (tokenizers, the Marvis voice prompt, the Mimi codec). Network and CPU only, before any hold.
#   3. exec `hold run --kind bench --reason "voice benchmarks phase 2" -- run_remaining.sh --inside <stamp> <phases>`:
#      the gate queues the hold, drains model calls in flight, unloads the LLM, runs the inner part, and releases the
#      hold when it exits (also if it crashes or is interrupted). See docs/hold.md in the companion local-rig repo.
# Inner part (runs under the hold, HOLD_ID set by `hold run`):
#   - HF_HUB_OFFLINE=1: nothing may download while a model is timed (Pocket TTS's first run did, 2026-10-04).
#   - before every phase, bench/gate.sh with HOLD_ID: GO only while our hold is the granted one, no other hold is
#     queued (another job waiting means we yield), no GPU job runs and the LLM stays unloaded. The first refusal
#     stops the run cleanly; the summary lists the phases not run, which can be re-run by name.
#   - each step runs in its own process. Filtered output goes to a local log, raw/09-measurements-log.md (bench/log.sh; not published), JSON to
#     raw/phase2-<stamp>/, and the full output, WAVs and summary.md to bench/results/<stamp>/ (gitignored).
here=${0:a:h}
export SP=${SP:-${here:h}}                        # measure/ (bench/, raw/, audio/)
HOLD=${HOLD_BIN:-$HOME/.local/bin/hold}
ALL=(pocket vibevoice marvis vad nemotron coresident voicechat)

if [ "$1" != --inside ]; then
  phases=(${@:-$ALL})
  for ph in $phases; do (( ${ALL[(Ie)$ph]} )) || { echo "unknown phase '$ph' (phases: $ALL)"; exit 2; }; done
  # the pre-flight check ignores any hold this shell may have inherited: never start inside another job's hold
  env -u HOLD_ID $here/gate.sh "pre-flight: run_remaining.sh $phases" || { echo "refused: the GPU is not clear (see the gate block above); nothing started"; exit 2; }
  LOG=$SP/raw/09-measurements-log.md
  { echo; echo "### $(date +%H:%M:%S) setup.sh (before the hold; network and CPU only)"; echo '```'; } >> $LOG
  $here/setup.sh 2>&1 | tee -a $LOG; rc=${pipestatus[1]}; echo '```' >> $LOG
  [ $rc = 0 ] || { echo "setup.sh failed; nothing started (fix it, or run it by hand to see why)"; exit 3; }
  stamp=$(date +%Y%m%d-%H%M%S)
  echo "taking the hold: $HOLD run --kind bench --reason \"voice benchmarks phase 2\" (it waits for calls in flight, then unloads the LLM)"
  exec $HOLD run --kind bench --reason "voice benchmarks phase 2" -- $here/run_remaining.sh --inside $stamp $phases
fi

# ---------------------------------------------------------------- inner part, under the hold
stamp=$2; shift 2; phases=($@)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$HOME/repos/mlx-audio/.venv/bin/python        # mlx-audio clone 0.5.3+21 (ee0c65d), mlx 0.31.2: same as phase 1 and the clean window
PK=$here/.venv-kokoro/bin/python                 # PY's packages plus misaki/spaCy (setup.sh)
PV=${VLM_PY:-$here/.venv-vlm/bin/python}         # mlx-vlm 0.7.4, mlx 0.32.3: the only runtime that loads this VoiceChat checkpoint
A=$SP/audio; J=$SP/raw/phase2-$stamp; R=$here/results/$stamp
mkdir -p $J $R/audio
F='grep -vE "it/s\]|^\s*$" | grep -E "^\{|maximum resident|real|user|sys|peak memory footprint|Error|Traceback|rror"'
note() { echo "$(date +%H:%M:%S) $*" | tee -a $R/runner.log; }
typeset -A res          # per-phase result (not "status": a read-only special parameter in zsh)
# step <name> <command>: full output to results/<stamp>/<name>.out, filtered output to the measurement log
step() { local name=$1; shift
  $here/log.sh "$* 2>&1 | tee $R/$name.out | $F" > /dev/null
  if grep -q '"event": "done"' $R/$name.out; then note "  $name: OK"; else note "  $name: FAILED (see $R/$name.out)"; failed=1; fi; }
gate() { $here/gate.sh "clean, own hold $HOLD_ID: $1" > /dev/null; }

{ echo; echo "## $(date '+%Y-%m-%d %H:%M') phase-2 run $stamp under hold ${HOLD_ID:-?} (run_remaining.sh): $phases"; } >> $SP/raw/09-measurements-log.md
note "run $stamp, hold ${HOLD_ID:-?}, phases: $phases; JSON $J; outputs $R"
[ "${HOLD_ID:-none}" = none ] && note "WARNING: hold run could not reach the hold gate; running without a hold (the gate is down, so nothing can load through :8090)"
stopped=""
for ph in $phases; do
  case $ph in
    pocket) desc="Pocket TTS warm rerun, intervals 0.32 and 0.5";;
    vibevoice) desc="VibeVoice-Realtime 0.5B 8-bit, intervals 0.32 and 0.5";;
    marvis) desc="Marvis TTS 250M v0.2 8-bit (sesame loader), intervals 0.32 and 0.5";;
    vad) desc="Silero VAD + Smart Turn v3 on the 9.3 s and 2.5 s clips";;
    nemotron) desc="Nemotron 3.5 ASR streaming 0.6B 8-bit, 320 ms chunks, paced then unpaced";;
    coresident) desc="co-residency Parakeet + Kokoro, Parakeet + Qwen3-TTS 0.6B";;
    voicechat) desc="VoiceChat 11B 4-bit (mlx-vlm), >= 64 s duplex session, per-80 ms-frame timing";;
  esac
  if ! gate "$desc"; then stopped=$ph; note "STOP before $ph: the gate refused (block in the measurement log)"; break; fi
  note "phase $ph: $desc"; failed=0
  case $ph in
    pocket) step pocket "/usr/bin/time -l $PY $here/tts_bench.py --model mlx-community/pocket-tts --stream-intervals 0.32,0.5 --runs 3 --out $J/tts_pocket_warm.jsonl --audio-dir $R/audio --tag clean-pocket-warm";;
    vibevoice) step vibevoice "/usr/bin/time -l $PY $here/tts_bench.py --model mlx-community/VibeVoice-Realtime-0.5B-8bit --voice en-Carter_man --stream-intervals 0.32,0.5 --runs 3 --out $J/tts_vibevoice.jsonl --audio-dir $R/audio --tag clean-vibevoice";;
    marvis) step marvis "/usr/bin/time -l $PY $here/tts_bench.py --model Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit --stream-intervals 0.32,0.5 --runs 3 --out $J/tts_marvis.jsonl --audio-dir $R/audio --tag clean-marvis";;
    vad) step vad-clip8 "/usr/bin/time -l $PY $here/vad_bench.py --clip $A/clip8_16k.wav --out $J/vad.jsonl"
         gate "$desc (second clip)" || { stopped=$ph; res[$ph]="PARTIAL: first step $([ $failed = 0 ] && echo OK || echo FAILED), then the gate refused"; note "STOP inside vad: the gate refused"; break; }
         step vad-clip3 "/usr/bin/time -l $PY $here/vad_bench.py --clip $A/clip3_16k.wav --out $J/vad.jsonl";;
    nemotron) step nemotron-paced "/usr/bin/time -l $PY $here/nemotron_stream_bench.py --clips $A/clip3_16k.wav $A/clip8_16k.wav --chunk-ms 320 --paced --out $J/nemotron_stream.jsonl"
         gate "$desc (unpaced)" || { stopped=$ph; res[$ph]="PARTIAL: first step $([ $failed = 0 ] && echo OK || echo FAILED), then the gate refused"; note "STOP inside nemotron: the gate refused"; break; }
         step nemotron-unpaced "/usr/bin/time -l $PY $here/nemotron_stream_bench.py --clips $A/clip8_16k.wav --chunk-ms 320 --out $J/nemotron_stream.jsonl";;
    coresident) step coresident-kokoro "/usr/bin/time -l $PK $here/coresident.py --tts mlx-community/Kokoro-82M-bf16 --voice af_heart --lang a --kokoro --clip $A/clip3_16k.wav --out $J/coresident_kokoro.jsonl"
         gate "$desc (Qwen pair)" || { stopped=$ph; res[$ph]="PARTIAL: first step $([ $failed = 0 ] && echo OK || echo FAILED), then the gate refused"; note "STOP inside coresident: the gate refused"; break; }
         step coresident-qwen06 "/usr/bin/time -l $PY $here/coresident.py --tts Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --voice Ryan --lang english --clip $A/clip3_16k.wav --out $J/coresident_qwen06.jsonl";;
    voicechat) step voicechat "/usr/bin/time -l $PV $here/voicechat_bench.py --runtime vlm --clips $A/clip3_16k.wav $A/clip8_16k.wav --min-seconds 64 --out $J/voicechat.jsonl --wav $R/audio/voicechat_output.wav --transcribe";;
  esac
  res[$ph]=$([ $failed = 0 ] && echo OK || echo FAILED)
done

# ---------------------------------------------------------------- summary (the hold is released when this script exits)
{ echo "# phase-2 run $stamp"; echo
  echo "Hold ${HOLD_ID:-?}; started with phases: $phases. JSON: \`raw/phase2-$stamp/\`. Full output: \`bench/results/$stamp/\`."; echo
  echo "| phase | result |"; echo "|---|---|"
  for ph in $phases; do
    if [ -n "${res[$ph]}" ]; then echo "| $ph | ${res[$ph]} |"
    elif [ "$ph" = "$stopped" ]; then echo "| $ph | not run: the gate refused (see the measurement log) |"
    else echo "| $ph | not run |"; fi
  done
  tts=($J/tts_*.jsonl(N)); (( $#tts )) && { echo; echo '## TTS tables (bench/summarize.py)'; $PY $here/summarize.py $tts 2>&1; }
  for f in $J/vad.jsonl $J/nemotron_stream.jsonl $J/coresident_*.jsonl(N) $J/voicechat.jsonl; do
    [ -f $f ] || continue; echo; echo "## ${f:t}"; echo '```'; grep -vE '"event": "clip"' $f | cut -c1-1500; grep '"event": "clip"' $f | python3 -c 'import json,sys
for l in sys.stdin:
  d=json.loads(l); d.pop("deltas",None); print(json.dumps(d)[:1500])'; echo '```'; done
  echo; echo "LLM POSTs to llama-swap since this run started (the hold should keep this empty):"
  # match the date too (fixed 2026-10-05: comparing only the time of day printed other days' requests)
  awk -v mon="$(date '+%b')" -v day="$(date '+%-d')" -v since="${stamp[10,11]}:${stamp[12,13]}:${stamp[14,15]}" '$1 == mon && $2 == day && $3 >= since && /POST/ {print $3, $(NF)}' ~/.ds4/llama-swap.log | tr '\n' ' '; echo "(end)"
} > $R/summary.md
note "summary: $R/summary.md"; [ -n "$stopped" ] && note "phases not run or cut short: ${phases[${phases[(ie)$stopped]},-1]} (re-run them by name)"
cat $R/summary.md | head -20
