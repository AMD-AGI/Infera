# Empirical decode step cost model from decode-log lines (one line per DP per 40 decode steps).
import re, subprocess, sys, datetime as dt
import numpy as np
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
runs = {"e2c40": ("e2c", "e2c/agentx-c40-e2cstack-flushed"), "e2d": ("e2d", "e2d/agentx-c40"),
        "e2e": ("e2e", "e2e/agentx-c40"), "e2f": ("e2f", "e2f/agentx-c40")}
# Extra runs from argv: name=<decode_log>:<benchmark_log>  (absolute paths). e2c40 is always kept as reference.
LOGS = {k: (f"{R}/{sd}/server-logs/decode-0.a1.log", f"{R}/{bd}/benchmark.log") for k, (sd, bd) in runs.items()}
if len(sys.argv) > 1:
    LOGS = {"e2c40": LOGS["e2c40"]}
    for a in sys.argv[1:]:
        name, paths = a.split("=", 1); dl, bl = paths.split(":", 1)
        LOGS[name] = (dl, bl)
runs = LOGS
def window(bl):
    s = e = None
    for line in open(bl, errors="ignore"):
        if "Phase profiling (profiling) started" in line: s = line[:8]
        if "Phase profiling" in line and "complete" in line: e = line[:8]
    return s, e
pat = re.compile(r"(\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)\.\d+)Z \[[^\]]*DP(\d) TP\d\] Decode batch, #running-req: (\d+), #token: (\d+),.*?accept len: ([\d.]+)")
INTERVAL = 40
allX = []; allY = []
for k, (dlog, blog) in runs.items():
    s, e = window(blog)
    out = subprocess.run(["strings", dlog], capture_output=True, text=True).stdout
    rows = []
    for m in pat.finditer(out):
        if s and not (s <= m.group(2) <= e): continue
        t = dt.datetime.fromisoformat(m.group(1)[:26]).timestamp()
        rows.append((t, int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6))))
    rows.sort()
    # cluster into ticks of 4 DPs
    ticks = []; cur = []
    for r in rows:
        if cur and r[0] - cur[0][0] > 1.0:
            ticks.append(cur); cur = []
        cur.append(r)
    if cur: ticks.append(cur)
    ticks = [t for t in ticks if len({x[1] for x in t}) == 4 and len(t) == 4]
    X = []; Y = []
    for a, b in zip(ticks, ticks[1:]):
        dtw = np.mean([x[0] for x in b]) - np.mean([x[0] for x in a])
        if dtw <= 0 or dtw > 8: continue
        run = [x[2] for x in b]; tok = [x[3] for x in b]
        X.append([1.0, max(run), sum(run), max(tok) / 1e6, sum(tok) / 1e6]); Y.append(dtw / INTERVAL * 1e3)
    X = np.array(X); Y = np.array(Y)
    allX.append(X); allY.append(Y)
    print(f"{k}: win {s}-{e} ticks={len(Y)} step_ms mean={Y.mean():.1f} p50={np.median(Y):.1f} | maxrun={X[:,1].mean():.2f} sumrun={X[:,2].mean():.2f} maxtokM={X[:,3].mean():.3f} sumtokM={X[:,4].mean():.3f} tok/req={(X[:,4]/np.maximum(X[:,2],1)).mean()*1e3:.0f}k")
X = np.vstack(allX); Y = np.concatenate(allY)
for name, cols in [("maxrun", [0, 1]), ("maxtok", [0, 3]), ("maxrun+maxtok", [0, 1, 3]), ("sumrun+sumtok", [0, 2, 4]), ("all", [0, 1, 2, 3, 4])]:
    A = X[:, cols]; c, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ c; r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    print(f"fit {name:14s} R2={r2:.3f} coef={np.round(c, 2).tolist()}")
# binned table by max running
print("maxrun -> step_ms median (n), maxtokM median")
for v in sorted(set(X[:, 1].astype(int))):
    m = X[:, 1] == v
    if m.sum() >= 20: print(f"  {v:3d}: {np.median(Y[m]):6.1f} ({m.sum()}) tokM={np.median(X[m,3]):.2f}")
print("per-run median step_ms by maxrun (n)")
for (k, X_, Y_) in zip(runs, allX, allY):
    s = []
    for v in range(2, 11):
        m = X_[:, 1] == v
        s.append(f"{v}:{np.median(Y_[m]):.1f}({m.sum()})" if m.sum() >= 10 else f"{v}:-")
    print(f"  {k:6s} " + " ".join(s))
