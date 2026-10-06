#!/bin/zsh
# fetch_esc50.sh — download the ESC-50 recordings the barge-in bench and e2e use as planted controls (2026-10-05).
#
# ESC-50 (Piczak 2015, CC BY-NC 3.0, github.com/karolpiczak/ESC-50): 5 s environmental clips at 44.1 kHz. Only the
# categories a person or a room makes while a reply plays: coughs, sneezes, breathing, laughter, knocks, typing, clicks,
# claps, footsteps, sipping, a clock, a can, a dog, a cat, a vacuum cleaner, rain (260 files, about 110 MB). Local test
# data only, never committed: it lands in <repo>/state/bargein/esc50 (gitignored). Read by tools/bargein_bench.py and
# tests/e2e/test_turn.py::test_planted_controls_over_a_reply.
set -e
out=${0:a:h}/../../state/bargein/esc50
mkdir -p $out/audio
cd $out
[[ -s esc50.csv ]] || curl -sSfL -m 30 -o esc50.csv https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv
typeset -A n=(coughing 40 sneezing 40 breathing 20 laughing 20 door_wood_knock 20 keyboard_typing 20 mouse_click 10
              clapping 20 footsteps 10 drinking_sipping 10 clock_tick 10 can_opening 10 dog 10 cat 10 vacuum_cleaner 5 rain 5)
for cat in ${(k)n}; do
  grep ",$cat," esc50.csv | head -${n[$cat]} | cut -d, -f1 | while read f; do
    [[ -s audio/$f ]] || curl -sSfL -m 60 -o audio/$f "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/$f"
  done
done
echo "$(ls audio | wc -l | tr -d ' ') files in $out/audio"
