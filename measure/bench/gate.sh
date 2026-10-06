#!/bin/zsh
# gate.sh <phase name> — GPU gate for the voice benchmarks. Exit 0 (go) only if ALL hold:
#   1. the hold gate is open (or not installed): `hold status --json` phase is "open"; "held"/"draining" means another
#      job owns the GPU. (2026-10-05: added after a video render ran as a plain Python process that the job pattern
#      below did not match while the LLM was unloaded by its hold, which made the old gate say GO mid-render.)
#      Inside run_remaining.sh's own hold (2026-10-05): `hold run` puts the hold's id in HOLD_ID, so "held" is accepted
#      only when the first hold is that id and granted, no other hold is queued (another job waiting means we yield),
#      and the gate has detected no GPU job outside any hold. HOLD_ID=none (gate unreachable) counts as no hold.
#   2. no GPU job process: renders, image/3D generation, the sibling video and 3D rigs (Claude shell
#      wrappers excluded, since their command lines quote these patterns);
#   3. llama-swap /running is empty OR the LLM has had no POST for >= 120 s.
# Logs the state it saw to the measurement log either way.
LOG=${SP:-${0:a:h:h}}/raw/09-measurements-log.md
HOLD=${HOLD_BIN:-$HOME/.local/bin/hold}
own=${HOLD_ID:-}; [ "$own" = none ] && own=""
r=$(curl -s --max-time 3 127.0.0.1:8090/running)
pat='ltx-2-mlx generat[e]|mflux-generat[e]|hy3d|run-queue[.]sh|story[.]sh|redo[.]sh|bin/vidgen|reel-studio|reel (animate|still|music|voice|cut|render)|m3d (generate|rig|animate)'
jobs=""; for p in $(pgrep -f "$pat"); do a=$(ps -o args= -p $p 2>/dev/null | head -1); echo "$a" | grep -q 'shell-snapshots' || jobs="$jobs\n  JOB $p: ${a:0:100}"; done
last=$(grep 'Request 127.0.0.1 "POST' ~/.ds4/llama-swap.log 2>/dev/null | tail -1 | awk '{print $1" "$2" "$3}')
ago=$(python3 -c 'import sys,time,datetime as d
s=sys.argv[1]
try:
  t=d.datetime.strptime(s+" "+str(d.date.today().year),"%b %d %H:%M:%S %Y").timestamp(); print(int(time.time()-t))
except Exception: print(99999)' "$last")
# one read of the gate: phase | first hold id | its state | every other hold | count and first of jobs detected outside holds
hinfo=$($HOLD status --json 2>/dev/null | python3 -c 'import json,sys
own=sys.argv[1]
try: d=json.load(sys.stdin)
except Exception:
  print("absent|-|-|-|0|-"); sys.exit()
hs=d.get("holds") or []; det=d.get("detected") or []
first=hs[0] if hs else {}
others="; ".join("%s %s %s (%s)" % (h.get("id"), h.get("kind"), h.get("reason",""), h.get("state")) for h in hs if h.get("id") != own)
d1=("%s pid %s" % (det[0].get("name"), det[0].get("pid"))) if det else "-"
print("|".join(str(x).replace("|","/") for x in (d.get("phase","unknown"), first.get("id","-"), first.get("state","-"), others or "-", len(det), d1)))' "${own:-none}")
IFS='|' read -r hphase hfirst hstate hothers ndet det1 <<< "$hinfo"
hold=$($HOLD status 2>&1 | head -1)
why=()
if [ -n "$own" ]; then
  if [ "$hphase" != absent ]; then
    { [ "$hphase" = held ] && [ "$hfirst" = "$own" ] && [ "$hstate" = granted ]; } || why+=("own hold $own is not the granted hold (phase $hphase, first $hfirst $hstate)")
    [ "$hothers" = "-" ] || why+=("another hold is queued or held: $hothers (yield)")
    [ "${ndet:-0}" -eq 0 ] || why+=("gate detects a GPU job outside any hold: $det1")
  fi
else
  { [ "$hphase" = open ] || [ "$hphase" = absent ]; } || why+=("hold gate is $hphase")
fi
[ -z "$jobs" ] || why+=("GPU job process running")
{ [ "$r" = '{"running":[]}' ] || [ "$ago" -ge 120 ]; } || why+=("LLM loaded and its last POST was ${ago}s ago")
if [ ${#why} -eq 0 ]; then verdict=GO; else verdict=SKIP; fi
{ echo; echo "### gate $(date +%H:%M:%S) [$1] → $verdict"; echo '```'; echo "hold phase: $hphase ($hold)"
  [ -n "$own" ] && echo "own hold: $own (first hold $hfirst, $hstate; others: $hothers; detected outside holds: $ndet)"
  echo "running: ${r:0:90}"; echo "gpu jobs:${jobs:-" none"}"; echo "last LLM POST: $last (${ago}s ago)"
  [ ${#why} -eq 0 ] || echo "refused: ${(j:; :)why}"; echo '```'; } | tee -a "$LOG"
[ "$verdict" = GO ]
