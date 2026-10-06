"""CLI: `python -m local_voice_brain <command>` (brain/run.sh wraps it for launchd).

  heartbeat            what launchd runs every 30 min: digest the oldest due day if the Mac is idle
  run --day D          digest one day now (still gated; --force skips all gates but gpu_clear; --dry-run writes nothing)
  status               the ledger, the outbox, the model the next run would use, and the gates as they stand now
  retry D              forget a day's failures so the next heartbeat tries it again

Exit codes: 0 done or nothing to do, 75 deferred (gates or the lock), 1 failed, 78 bad configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from . import gates, outbox, state
from .config import ConfigError, load
from .job import heartbeat
from .llm import choose_model


def _now(s: str | None) -> datetime:
    if not s:
        return datetime.now().astimezone()
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise SystemExit("--now needs a UTC offset")
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="local_voice_brain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file (default brain/config.toml)")
    ap.add_argument("--now", help=argparse.SUPPRESS)  # tests: pretend it is this time (ISO 8601 with offset)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("heartbeat")
    p = sub.add_parser("run")
    p.add_argument("--day", required=True)
    p.add_argument("--force", action="store_true", help="skip the quiet, user-idle and orchestrator gates")
    p.add_argument("--dry-run", action="store_true", help="ask the model, write nothing (kb, Atlas, outbox, ledger)")
    sub.add_parser("status")
    p = sub.add_parser("retry")
    p.add_argument("day")
    a = ap.parse_args(argv)
    try:
        cfg = load(a.config)
    except ConfigError as e:
        print(f"brain: {e}", file=sys.stderr)
        return 78
    now = _now(a.now)
    if a.cmd in ("heartbeat", "run"):
        out = heartbeat(cfg, now=now, day=getattr(a, "day", None), force=getattr(a, "force", False),
                        dry_run=getattr(a, "dry_run", False), log=lambda s: print(f"  {s}", file=sys.stderr))
        print(f"{now.isoformat(timespec='seconds')} {out.kind}" + (f" day={out.day}" if out.day else "")
              + f": {out.detail}")
        return out.code
    if a.cmd == "retry":
        led = state.Ledger.load(cfg.state_dir)
        if a.day not in led.data["days"]:
            print(f"{a.day} is not in the ledger")
            return 1
        led.data["days"][a.day].update(status=state.RETRY, attempts=0, last_error=None)
        led.save()
        print(f"{a.day}: will be tried again at the next heartbeat")
        return 0
    led = state.Ledger.load(cfg.state_dir)
    model, note = choose_model(cfg.model, cfg.fallback_model, cfg.agent_dir)
    print(json.dumps({
        "days": led.data["days"],
        "outbox": [{"day": d.day, "text": d.text, "expires": d.expires.isoformat()} for d in outbox.pending(cfg.state_dir, now)],
        "model": model, "model_note": note,
        "gates": [str(g) for g in (gates.quiet(cfg.turns_dir, now, cfg.quiet_minutes),
                                   gates.user_idle(cfg.user_idle_minutes),
                                   gates.orchestrator(cfg.orchestrator_status, cfg.busy_states),
                                   gates.gpu_clear(cfg.gpu_clear))],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
