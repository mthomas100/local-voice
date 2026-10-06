#!/bin/zsh
# run_tests.sh: the bridge's pytest suite against the stub LLM (no model, no GPU, nothing on :8090).
# Optional: ATLAS_REPO=<an Atlas git repo or clone> adds the Atlas test (it clones; never writes to the source).
# pytest and pytest-asyncio come from uv's cache; nothing is installed into the repo.
cd ${0:a:h} && exec uv run --offline --no-project --python 3.12 --with pytest --with pytest-asyncio python -m pytest -q "$@"
