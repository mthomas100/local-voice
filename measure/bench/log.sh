#!/bin/zsh
# log.sh "<command>" — run a command, print its output, and append command + full output to the raw log.
LOG=${SP:-${0:a:h:h}}/raw/09-measurements-log.md
{ echo; echo "### $(date +%H:%M:%S)"; echo '```'; echo "\$ $*"; eval "$@" 2>&1; echo "(exit $?)"; echo '```'; } | tee -a "$LOG"
