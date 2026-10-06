#!/bin/zsh
# wait_gpu_clear.sh [minutes_clear=20] [max_hours=16] — block until gpu_clear.sh has said CLEAR on every check for
# <minutes_clear> consecutive minutes (polled once a minute), so a short gap between two render steps does not count.
# Exit 0 when that happens, 1 when <max_hours> pass first. Prints only status changes.
here=${0:a:h}; need=${1:-20}; cap=$(( ${2:-16} * 60 )); streak=0; prev=""
for i in {1..$cap}; do
  line=$($here/gpu_clear.sh); st=${line##* }
  if [ "$st" = CLEAR ]; then streak=$((streak+1)); else streak=0; fi
  [ "$st" != "$prev" ] && echo "$line (streak $streak/$need)"; prev=$st
  [ $streak -ge $need ] && { echo "$line: CLEAR for $need consecutive minutes"; exit 0; }
  sleep 60
done
echo "gave up after ${2:-16} h: $(${here}/gpu_clear.sh)"; exit 1
