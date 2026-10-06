#!/bin/zsh
# gpu_clear.sh — quiet GPU check for the voice project (2026-10-05). Prints one status line; exit 0 only when:
#   the hold gate is open (or not installed), no GPU job process runs (renders, image/3D generation, the sibling video
#   and 3D rigs), and the local LLM is unloaded or has had no request for >= 120 s.
# Same rules as gate.sh, without writing to the measurement log, so a waiter can poll it.
HOLD=${HOLD_BIN:-$HOME/.local/bin/hold}
pat='ltx-2-mlx generat[e]|mflux-generat[e]|hy3d|run-queue[.]sh|story[.]sh|redo[.]sh|bin/vidgen|reel-studio|reel (animate|still|music|voice|cut|render)|m3d (generate|rig|animate)'
n=0; for p in $(pgrep -f "$pat"); do ps -o args= -p $p 2>/dev/null | grep -q 'shell-snapshots' || n=$((n+1)); done
hphase=$($HOLD status --json 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("phase","unknown"))
except Exception: print("absent")')
r=$(curl -s --max-time 3 127.0.0.1:8090/running)
last=$(grep 'Request 127.0.0.1 "POST' ~/.ds4/llama-swap.log 2>/dev/null | tail -1 | awk '{print $1" "$2" "$3}')
ago=$(python3 -c 'import sys,time,datetime as d
try: print(int(time.time()-d.datetime.strptime(sys.argv[1]+" "+str(d.date.today().year),"%b %d %H:%M:%S %Y").timestamp()))
except Exception: print(99999)' "$last")
ok=1; [ $n -eq 0 ] || ok=0; { [ "$hphase" = open ] || [ "$hphase" = absent ]; } || ok=0; { [ "$r" = '{"running":[]}' ] || [ "$ago" -ge 120 ]; } || ok=0
echo "$(date +%H:%M:%S) gpu_jobs=$n hold=$hphase llm_last_post_ago=${ago}s → $([ $ok = 1 ] && echo CLEAR || echo BUSY)"
[ $ok = 1 ]
