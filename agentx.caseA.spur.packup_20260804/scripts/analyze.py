#!/usr/bin/env python3
"""Pull the headline SLIs out of an aiperf artifact dir."""
import csv, json, sys, os

def load(csvp):
    with open(csvp) as f:
        rd = list(csv.reader(f))
    hdr = [h.strip() for h in rd[0]]
    d = {}
    for r in rd[1:]:
        if r and r[0].strip():
            d[r[0].strip()] = r[1:]
    return hdr, d

def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None

def show(art, label):
    csvp = os.path.join(art, "profile_export_aiperf.csv")
    if not os.path.exists(csvp):
        print(f"{label}: NO CSV"); return
    hdr, d = load(csvp)
    col = {h: i - 1 for i, h in enumerate(hdr)}   # -1: row lists exclude "Metric"
    def val(m, stat):
        r = d.get(m)
        i = col.get(stat)
        if r is None or i is None or i >= len(r): return None
        return num(r[i])
    print(f"\n=== {label} ===")
    for m in ["Time to First Token (ms)", "Request Latency (ms)", "Inter Token Latency (ms)",
              "Input Sequence Length (tokens)", "Output Sequence Length (tokens)"]:
        if m in d:
            print(f"  {m:38} avg={val(m,'avg')}  p50={val(m,'p50')}  p90={val(m,'p90')}  p99={val(m,'p99')}")
    for m in ["Output Token Throughput (tokens/sec)", "Input Token Throughput (tokens/sec)",
              "Request Throughput (requests/sec)", "Request Count",
              "Theoretical Prefix Cache Hit (%)"]:
        if m in d:
            print(f"  {m:38} {val(m,'avg')}")
    p = val("Usage Prompt Tokens (tokens)", "avg")
    c = val("Usage Prompt Cache Read Tokens (tokens)", "avg")
    if p and c:
        print(f"  {'prompt tok avg':38} {p:,.0f}")
        print(f"  {'cache-read tok avg':38} {c:,.0f}")
        print(f"  {'SERVER-MEASURED cache read/prompt':38} {100*c/p:.1f} %")
    jp = os.path.join(art, "profile_export_aiperf.json")
    if os.path.exists(jp):
        try:
            md = json.load(open(jp)).get("metadata", {})
            print(f"  submission_valid={md.get('submission_valid')} reasons={md.get('submission_invalid_reasons')}")
        except Exception:
            pass

for a in sys.argv[1:]:
    show(a, os.path.basename(a.rstrip("/")))
