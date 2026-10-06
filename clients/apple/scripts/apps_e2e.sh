#!/bin/zsh
# Builds both apps (ad hoc signed) and runs them unattended against the mock server: the Mac app from its build
# folder, the iPhone app in the simulator (MockServer/tests/test_apps.py). Evidence: MockServer/runs/<stamp>/.
set -euo pipefail
cd "$(dirname "$0")/.."
./build.sh mac
./build.sh ios
cd MockServer && .venv/bin/python -m pytest -m "apps and not real" -q "$@"
