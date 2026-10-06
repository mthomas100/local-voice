#!/usr/bin/env python3
"""The router (local_voice/router.py) on a set of phrasings with planted negatives (tests/fixtures/router_phrasings.yaml),
against the real spaces.yaml. CPU only, no model. Prints every miss and the totals: switches found (right space, and the
right words passed on), questions asked, negatives left alone.

    .venv/bin/python tools/router_bench.py [--router path/to/router.py]

--router evaluates another version of router.py (for a before/after: `git show <rev>:orchestrator/local_voice/router.py
> /tmp/old_router.py`)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from local_voice.config import load_config  # noqa: E402
from local_voice.spaces import load_spaces  # noqa: E402

PHRASINGS = HERE / "tests/fixtures/router_phrasings.yaml"


def load_router(path: Path | None):
    if path is None:
        from local_voice import router
        return router
    spec = importlib.util.spec_from_file_location("local_voice._router_under_test", path,
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "local_voice"
    sys.modules[spec.name] = mod          # its dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


def evaluate(router, spaces, data: dict) -> dict:
    out = {"switch_ok": 0, "switch_n": 0, "rest_ok": 0, "negative_ok": 0, "negative_n": 0, "misses": []}
    for case in data["switches"]:
        r = router.route(case["say"], spaces)
        out["switch_n"] += 1
        got = "ask" if r.kind == "ask" else (r.space if r.kind == "switch" else "none")
        if got == case["want"]:
            out["switch_ok"] += 1
            if case["want"] == "ask" or r.rest == case.get("rest", ""):
                out["rest_ok"] += 1
            else:
                out["misses"].append(f"rest   {case['say']!r}: passed on {r.rest!r}, want {case['rest']!r}")
        else:
            out["misses"].append(f"switch {case['say']!r}: got {got}, want {case['want']}")
    for case in data["negatives"]:
        r = router.route(case["say"], spaces)
        out["negative_n"] += 1
        if r.kind in ("switch", "ask"):
            out["misses"].append(f"false  {case['say']!r}: {r.kind} {r.space or r.candidates}")
        else:
            out["negative_ok"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, help="another router.py to evaluate")
    args = ap.parse_args()
    spaces = load_spaces(load_config(), check_paths=False)
    res = evaluate(load_router(args.router), spaces, yaml.safe_load(PHRASINGS.read_text()))
    for m in res["misses"]:
        print(m)
    print(f"switches {res['switch_ok']}/{res['switch_n']} (words passed on right in {res['rest_ok']}), "
          f"negatives left alone {res['negative_ok']}/{res['negative_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
