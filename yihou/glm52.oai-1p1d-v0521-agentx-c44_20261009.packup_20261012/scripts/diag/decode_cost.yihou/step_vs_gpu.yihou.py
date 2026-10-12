# Matched-load decode step residual vs GPU clocks/power/temp (hostsample gpu_ts.tsv).
# usage: step_vs_gpu.yihou.py <decode_log> <gpu_ts.tsv> <HH:MM:SS start> [HH:MM:SS end] [ref=e2c40]
# Residual = step_ms(tick) - e2c40 median step_ms at the same max-DP running-req (ref table from e2c40 run).
import collections, csv, datetime as dt, re, subprocess, sys
import numpy as np

dlog, gts, ws = sys.argv[1], sys.argv[2], sys.argv[3]
we = sys.argv[4] if len(sys.argv) > 4 else "23:59:59"
REF = {2: 30.5, 3: 33.9, 4: 36.0, 5: 37.9, 6: 43.0, 7: 45.1, 8: 46.2, 9: 51.1, 10: 53.1}  # e2c40 (R041)
pat = re.compile(r"(\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)\.\d+)Z \[[^\]]*DP(\d) TP\d\] Decode batch, #running-req: (\d+), #token: (\d+)")
out = subprocess.run(["strings", dlog], capture_output=True, text=True).stdout
rows = []
for m in pat.finditer(out):
    if ws <= m.group(2) <= we:
        rows.append((dt.datetime.fromisoformat(m.group(1)[:26]).replace(tzinfo=dt.timezone.utc).timestamp(),
                     int(m.group(3)), int(m.group(4))))
rows.sort()
ticks, cur = [], []
for r in rows:
    if cur and r[0] - cur[0][0] > 1.0:
        ticks.append(cur); cur = []
    cur.append(r)
if cur: ticks.append(cur)
ticks = [t for t in ticks if len({x[1] for x in t}) == 4 and len(t) == 4]
T = []
for a, b in zip(ticks, ticks[1:]):
    d = np.mean([x[0] for x in b]) - np.mean([x[0] for x in a])
    if 0 < d <= 8:
        mr = max(x[2] for x in b)
        T.append((np.mean([x[0] for x in b]), mr, d / 40 * 1e3))
print(f"ticks={len(T)} window {ws}-{we}")
tab = collections.defaultdict(list)
for t, mr, s in T:
    tab[mr].append(s)
print("matched-load table (median ms (n)) vs e2c40:")
print("  " + " ".join(f"{k}:{np.median(v):.1f}({len(v)})[ref {REF.get(k, float('nan')):.1f}]" for k, v in sorted(tab.items()) if len(v) >= 5))
res = [(t, s - REF[mr]) for t, mr, s in T if mr in REF]
if res:
    r = np.array([x[1] for x in res])
    print(f"residual vs e2c40: mean {r.mean():+.1f} ms, median {np.median(r):+.1f} ms, n={len(r)}")
# GPU time series: decode GPUs 4-7 (HIP_VISIBLE_DEVICES=4,5,6,7), prefill 0-3
g = collections.defaultdict(list)
day = dt.datetime.fromtimestamp(T[0][0], dt.timezone.utc).date() if T else dt.date.today()
for row in csv.DictReader(open(gts), delimiter="\t"):
    try:
        ts = dt.datetime.combine(day, dt.time.fromisoformat(row["t"]), dt.timezone.utc).timestamp()
        g[int(row["gpu"])].append((ts, *(float(row[k]) if row[k] not in ("None", "N/A", "") else np.nan
                                        for k in ("power_W", "cap_W", "gfx_clk", "mem_clk", "gfx_pct", "umc_pct", "hotspot_C", "mem_C"))))
    except (ValueError, KeyError):
        continue
names = ("power", "cap", "gfx_clk", "mem_clk", "gfx%", "umc%", "hotspot", "memT")
for side, gpus in (("decode GPU4-7", (4, 5, 6, 7)), ("prefill GPU0-3", (0, 1, 2, 3))):
    print(f"== {side}: mean (min..max) over window")
    for gi in gpus:
        a = np.array([x[1:] for x in g[gi] if x[0] >= (T[0][0] if T else 0)])
        if len(a):
            print(f"  GPU{gi}: " + " | ".join(f"{n} {np.nanmean(a[:, i]):.0f}({np.nanmin(a[:, i]):.0f}..{np.nanmax(a[:, i]):.0f})" for i, n in enumerate(names)))
# correlation of residual with decode-GPU mean metrics at nearest sample
if res and g[4]:
    feats = []
    for t, rr in res:
        vals = []
        for gi in (4, 5, 6, 7):
            arr = g[gi]; j = min(range(len(arr)), key=lambda k: abs(arr[k][0] - t))
            vals.append(arr[j][1:] if abs(arr[j][0] - t) < 30 else [np.nan] * 8)
        feats.append(np.nanmean(np.array(vals, dtype=float), axis=0).tolist() + [rr])
    F = np.array(feats); F = F[~np.isnan(F).any(axis=1)]
    print(f"== corr(residual, decode-GPU mean metric), n={len(F)}  (Spearman-ish: ranks)")
    rk = lambda x: np.argsort(np.argsort(x))
    for i, n in enumerate(names if len(F) > 5 else ()):
        col = F[:, i]
        c = np.corrcoef(rk(col), rk(F[:, -1]))[0, 1] if col.std() > 0 else float("nan")
        print(f"  {n:8s} r={c:+.2f}  (metric range {col.min():.0f}..{col.max():.0f})")
