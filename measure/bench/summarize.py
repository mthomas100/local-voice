#!/usr/bin/env python
"""summarize.py <jsonl>... — markdown tables (median + all runs) from tts_bench / stt_bench output."""
import json, statistics as st, sys
from collections import defaultdict
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    load = [r for r in rows if r["event"] == "load"]
    warm = [r for r in rows if r["event"] == "warmup"]
    print(f"\n### {path.split('/')[-1]}")
    for r in load:
        print(f"load: {r['load_s']} s; MLX active after load {r['active_gib']:.2f} GiB; RSS {r['rss_gib']:.2f} GiB")
    for r in warm:
        print(f"warm-up (first generate, kernel compile): {r.get('total_s', r.get('s'))} s")
    gens = [r for r in rows if r["event"] == "gen"]
    if gens and "mode" in gens[0]:
        groups = defaultdict(list)
        for r in gens: groups[(r["text_name"], r["words"], r["mode"])].append(r)
        print("| text | mode | TTFA s median (runs) | total s median (runs) | audio s (runs) | RTF median | first chunk audio s | MLX peak GiB |")
        print("|---|---|---|---|---|---|---|---|")
        for (t, w, m), rs in groups.items():
            def f(k):
                vals = [r[k] for r in rs]
                return "%.3f (%s)" % (st.median(vals), ", ".join("%.3f" % v for v in vals))
            av = [r["audio_s"] for r in rs]
            fa = "%.2f (%s)" % (st.median(av), ", ".join("%.2f" % v for v in av))
            print(f"| {t} ({w} w) | {m} | {f('ttfa_s')} | {f('total_s')} | {fa} | {st.median([r['rtf'] for r in rs]):.3f} | {rs[0]['first_chunk_audio_s']} | {max(r['peak_gib'] for r in rs):.2f} |")
    elif gens:
        groups = defaultdict(list)
        for r in gens: groups[r["source"]].append(r)
        print("| input | clip s | latency s median (runs) | text | MLX peak GiB |")
        print("|---|---|---|---|---|")
        for s, rs in groups.items():
            vals = [r["s"] for r in rs]
            print("| %s | %s | %.3f (%s) | %s | %.2f |" % (s, rs[0]["clip_s"], st.median(vals), ", ".join("%.3f" % v for v in vals), rs[0]["text"], max(r["peak_gib"] for r in rs)))
        for r in [r for r in rows if r["event"] == "stream_generate"]:
            print(f"stream_generate chunk_duration={r['chunk_duration']} overlap={r['overlap']}: total {r['total_s']} s; events {r['events']}")
    for r in [r for r in rows if r["event"] == "done"]:
        print(f"done: RSS {r['rss_gib']} GiB; MLX peak {r.get('peak_gib_process', r.get('peak_gib'))} GiB")
