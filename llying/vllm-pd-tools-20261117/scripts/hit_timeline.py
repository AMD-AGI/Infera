"""Prefill prompt-token hit rate per window from metrics_sampler output.
   hit_timeline.py SAMPLES.jsonl PREFILL_URL [WINDOW_S=300]
Columns: window start (UTC), prompt tok/s, local-cache share, CPU-tier (external) share, computed share."""
import datetime
import json
import sys

path, purl = sys.argv[1:3]
win = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
keys = ("vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total", "vllm:external_prefix_cache_hits_total")
pts = []
for line in open(path):
    d = json.loads(line)
    if d.get("url") != purl:
        continue
    m = (d.get("m") or {}).get("0") or {}
    if all(k in m for k in keys):
        pts.append((d["t"], *(m[k] for k in keys)))
pts.sort()
if not pts:
    sys.exit("no prefill samples")
print("window | prompt tok/s | local | CPU tier | computed")
i = 0
while i < len(pts):
    j = i
    while j + 1 < len(pts) and pts[j + 1][0] - pts[i][0] <= win:
        j += 1
    if j == i:
        break
    dt = pts[j][0] - pts[i][0]
    dp, dl, de = (pts[j][k] - pts[i][k] for k in (1, 2, 3))
    if dp > 0:
        t = datetime.datetime.utcfromtimestamp(pts[i][0]).strftime("%H:%M")
        print(f"{t} | {dp / dt:,.0f} | {dl / dp:.3f} | {de / dp:.3f} | {1 - (dl + de) / dp:.3f}")
    i = j
