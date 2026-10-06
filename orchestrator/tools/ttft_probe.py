"""LLM time-to-first-token probe (2026-10-05): why a voice turn waits for its first word.

Runs the orchestrator's exact home-space Pi child (config.yaml, spaces.yaml, persona, extensions, derived agent dir)
with short typed turns, through a logging proxy on a scratch port in front of the hold gate. No speech model loads.
Per request it keeps the body, when it reached the proxy, the first content delta, usage, and whether the request
re-sends the previous one plus the reply as streamed (a request that does not would make ds4 rewind). After the run it
copies ds4-server's own lines for the run from llama-swap's upstream log (prefill time per request), and with
--keepalive it first watches ds4's GPU-queue keepalive (ioreg lastSubmittedTime) while ds4 is idle.

What it found (2026-10-05): every continued turn extends ds4's live state exactly, and a
short continuation takes about 0.2 s right after a request but 1-2.5 s after a few seconds idle, because ds4's
keepalive, meant to tick every second, ticks every 4-5 s under the darwinbg class llama-swap's LaunchAgent gives it.

GPU rule: this makes real LLM calls through the gate. Run measure/bench/gpu_clear.sh first; never during a hold.
Pi's kb.ts writes a session digest, so the child gets a fresh kb clone (KB_HOME) under OUT_DIR, or --kb.

usage:
  .venv/bin/python tools/ttft_probe.py OUT_DIR [--gaps 0.3,4,10] [--turns 13] [--keepalive 30] [--kb DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

PROMPTS = ["What is the capital of France?", "Say hello in Spanish.", "What color is the sky on a clear day?",
           "Name a fruit that is yellow.", "What is the opposite of cold?", "Which animal says moo?",
           "What is two plus two?", "Name a colour of the rainbow."]
LLAMA_SWAP = "http://127.0.0.1:8091"           # read only: /logs/stream/upstream (ds4-server's own lines)
DS4_LINE = re.compile(r"^\d{4} (\d\d:\d\d:\d\d) ds4-server: chat ctx=(\d+)\.\.(\d+):(\d+) (?:TOOLS )?prompt done ([\d.]+)s")


class Proxy:
    """Forwards everything to the gate unchanged; records chat requests and their streamed replies."""

    def __init__(self, upstream: str, out: Path):
        self.upstream = upstream.rstrip("/")
        self.out = out
        self.n = 0
        self.rows: list[dict] = []
        self.http: ClientSession | None = None

    async def handle(self, req: web.Request) -> web.StreamResponse:
        body = await req.read()
        t_wall, m0 = time.time(), time.monotonic()
        self.n += 1
        n = self.n
        chat = req.method == "POST" and req.path.endswith("/chat/completions")
        if chat:
            (self.out / f"req-{n:03d}.json").write_bytes(body)
        headers = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length", "accept-encoding")}
        assert self.http is not None
        first_content = None
        content, usage, buf, gone = "", None, b"", False
        async with self.http.request(req.method, self.upstream + req.path_qs, data=body, headers=headers) as up:
            resp = web.StreamResponse(status=up.status, headers={
                k: v for k, v in up.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})
            await resp.prepare(req)
            async for chunk in up.content.iter_any():
                if not gone:
                    try:
                        await resp.write(chunk)
                    except (ConnectionError, RuntimeError):
                        gone = True     # Pi hung up after the last event; keep reading for usage
                if not chat:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                        continue
                    try:
                        ev = json.loads(line[5:])
                    except ValueError:
                        continue
                    usage = ev.get("usage") or usage
                    for ch in ev.get("choices") or []:
                        text = (ch.get("delta") or {}).get("content")
                        if text:
                            content += text
                            first_content = first_content or time.monotonic()
            if not gone:
                try:
                    await resp.write_eof()
                except (ConnectionError, RuntimeError):
                    pass
        if chat:
            row = {"n": n, "t": time.strftime("%H:%M:%S", time.localtime(t_wall)),
                   "first_content_ms": None if first_content is None else round((first_content - m0) * 1000, 1),
                   "end_ms": round((time.monotonic() - m0) * 1000, 1), "usage": usage, "content": content}
            self.rows.append(row)
            with (self.out / "proxy-rows.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        return resp


def extends_previous(prev: dict, now: dict) -> dict:
    """Does `now` re-send `prev` unchanged (same parameters, tools, system and history) and add to it?"""
    pm, m = prev.get("messages") or [], now.get("messages") or []
    diff = next((i for i, (a, b) in enumerate(zip(pm, m)) if a != b), None)
    return {"params_changed": sorted(k for k in set(prev) | set(now) if k != "messages" and prev.get(k) != now.get(k)),
            "history_resent_unchanged": diff is None and len(m) > len(pm), "first_diff_index": diff}


def ds4_pid() -> str | None:
    out = subprocess.run(["pgrep", "-f", "repos/ds4/ds4-server"], capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def keepalive_period(seconds: float) -> list[float]:
    """Seconds between ds4-server's GPU submissions while it is idle, read from the driver (ioreg, read only).
    ds4's queue keepalive (ds4_metal.m) means this to be about 1 s; the driver drops an idle queue's mappings after
    about 3 s."""
    pid = ds4_pid()
    if pid is None:
        return []
    pat = re.compile(r'"lastSubmittedTime"=(\d+),"accumulatedGPUTime"=\d+\}\)\s*\n\s*"IOUserClientCreator" = "pid '
                     + pid + r', ds4-server"')
    seen: list[tuple[float, int]] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        m = pat.search(subprocess.run(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-l"],
                                      capture_output=True, text=True).stdout)
        if m and (not seen or int(m.group(1)) != seen[-1][1]):
            seen.append((time.monotonic(), int(m.group(1))))
        time.sleep(0.1)
    return [round((b[1] - a[1]) / 1e9, 2) for a, b in zip(seen, seen[1:])]


def ds4_prefills(since: str) -> list[dict]:
    """ds4-server's own 'prompt done' lines at or after HH:MM:SS today, from llama-swap's upstream log buffer."""
    try:
        text = subprocess.run(["curl", "-s", "-m", "3", f"{LLAMA_SWAP}/logs/stream/upstream"],
                              capture_output=True, text=True).stdout
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        m = DS4_LINE.match(line)
        if m and m.group(1) >= since:
            rows.append({"t": m.group(1), "start": int(m.group(2)), "new_tokens": int(m.group(4)),
                         "prompt_s": float(m.group(5))})
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--gate", default="http://127.0.0.1:8090")
    ap.add_argument("--port", type=int, default=18093, help="the proxy's scratch port on 127.0.0.1")
    ap.add_argument("--gaps", default="0.3,4,10", help="seconds between turns, cycled")
    ap.add_argument("--turns", type=int, default=13)
    ap.add_argument("--keepalive", type=float, default=0, help="first watch ds4's idle GPU submissions this long")
    ap.add_argument("--kb", default=None, help="an existing kb clone to reuse")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from local_voice.agent_dir import derive_agent_dir
    from local_voice.config import load_config
    from local_voice.pi_rpc import PiChild, Settled, TextDelta
    from local_voice.spaces import load_spaces, pi_space_config

    summary: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "gaps": a.gaps}
    if a.keepalive:
        periods = keepalive_period(a.keepalive)
        summary["keepalive_periods_s"] = periods
        print(f"ds4 keepalive: {periods} s between idle GPU submissions (meant to be about 1 s)", flush=True)

    kb = Path(a.kb) if a.kb else out / "kb"
    if not kb.exists():
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(Path.home() / "kb"), str(kb)], check=True)
        subprocess.run(["git", "-C", str(kb), "remote", "set-url", "--push", "origin", "DISABLED-probe-clone"], check=True)
    assert kb.resolve() != (Path.home() / "kb").resolve(), "KB_HOME must be a copy"
    cfg = load_config(overrides={"state_dir": str(out / "state"),
                                 "pi": {"env": {"KB_HOME": str(kb), "KB_SEARCH_BACKEND": "rg", "UV_OFFLINE": "1"}},
                                 "brain": {"turn_log_dir": str(out / "turns"), "state_dir": str(out / "brain")}})
    spaces = load_spaces(cfg)
    agent_dir = derive_agent_dir(cfg, spaces, out / "pi-agent", base_url=f"http://127.0.0.1:{a.port}/v1")
    sc = pi_space_config(cfg, spaces.spaces["home"], agent_dir=agent_dir, state_dir=out / "state")

    proxy = Proxy(a.gate, out)
    proxy.http = ClientSession(timeout=ClientTimeout(total=600))
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", a.port).start()

    gaps = [float(x) for x in a.gaps.split(",")]
    child = PiChild(sc, transcript=out / "pi-transcript.jsonl")
    await child.start()
    turns = []
    try:
        for i, text in enumerate((PROMPTS * (a.turns // len(PROMPTS) + 1))[: a.turns]):
            gap = gaps[(i - 1) % len(gaps)] if i else None
            if gap:
                await asyncio.sleep(gap)
            n0, t0 = proxy.n, time.monotonic()
            load1 = float(subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.split()[1])
            first = None
            async for ev in child.turn(text, settle_timeout=10.0):
                if isinstance(ev, TextDelta) and ev.text and first is None:
                    first = time.monotonic()
                elif isinstance(ev, Settled):
                    break
            reqs = [r for r in proxy.rows if r["n"] > n0]
            row = {"turn": i + 1, "gap_s": gap, "load1": load1,
                   "pi_first_text_ms": None if first is None else round((first - t0) * 1000, 1),
                   "server_first_content_ms": [r["first_content_ms"] for r in reqs],
                   "cached_tokens": [((r["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens") for r in reqs],
                   "prompt_tokens": [(r["usage"] or {}).get("prompt_tokens") for r in reqs]}
            turns.append(row)
            print(f"turn {i + 1:2d} gap {gap!s:>4} s: first content {row['server_first_content_ms']} ms, cached "
                  f"{row['cached_tokens']} of {row['prompt_tokens']} tokens, load1 {load1}", flush=True)
    finally:
        await child.close()
        await runner.cleanup()
        await proxy.http.close()

    bodies = sorted(out.glob("req-*.json"))
    summary["requests_extend_previous"] = [extends_previous(json.loads(p.read_text()), json.loads(c.read_text()))
                                           for p, c in zip(bodies, bodies[1:])]
    summary["ds4_prefills"] = ds4_prefills(summary["started"][11:])
    by_gap: dict[str, list[float]] = {}
    for t in turns[1:]:
        if t["server_first_content_ms"]:
            by_gap.setdefault(str(t["gap_s"]), []).append(t["server_first_content_ms"][0])
    summary["first_content_ms_by_gap"] = {g: {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
                                          for g, v in by_gap.items()}
    summary["turns"] = turns
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    clean = all(x["history_resent_unchanged"] and not x["params_changed"] for x in summary["requests_extend_previous"])
    print(f"every request re-sends the previous one unchanged and adds to it: {clean}")
    for g, s in summary["first_content_ms_by_gap"].items():
        print(f"gap {g} s: first content median {s['median']} ms (min {s['min']}, max {s['max']}, n {s['n']})")
    print(f"ds4 prefills: {[(p['t'], p['new_tokens'], p['prompt_s']) for p in summary['ds4_prefills']]}")
    print(f"written: {out / 'summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
