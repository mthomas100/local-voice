#!/bin/zsh
# brain/run.sh: the background brain's entry point (M4, 2026-10-05). With no arguments it is one heartbeat, which is
# what launchd runs every 30 minutes; anything else passes through:
#   run.sh status | run.sh run --day 2026-10-04 [--force] [--dry-run] | run.sh retry 2026-10-04
# launchd starts jobs with a bare PATH, so it is set here: uv and python3.12 from Homebrew (kb runs under uv), kb and
# hold from ~/.local/bin, curl/pgrep/ioreg from the system. Pi's node comes from llm.pi_bin's own directory.
here=${0:a:h}
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
py=$here/.venv/bin/python
[ -x $py ] || py=/opt/homebrew/bin/python3.12
[ $# -eq 0 ] && set -- heartbeat
cd $here && exec $py -m local_voice_brain "$@"
