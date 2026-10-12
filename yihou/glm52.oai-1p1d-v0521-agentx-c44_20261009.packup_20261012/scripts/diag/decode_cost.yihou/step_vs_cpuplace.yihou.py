# Decode step residual (vs e2c40 matched load) vs scheduler-thread CPU placement/migration (cpuplace tsv, 1 s).
# usage: step_vs_cpuplace.yihou.py <decode_log> <cpuplace.decode.tsv> <HH:MM:SS start> <HH:MM:SS end>
import collections, datetime as dt, re, subprocess, sys
import numpy as np

dlog, cp, ws, we = sys.argv[1:5]
REF = {2: 30.5, 3: 33.9, 4: 36.0, 5: 37.9, 6: 43.0, 7: 45.1, 8: 46.2, 9: 51.1, 10: 53.1}
node = lambda c: 0 if c <= 117 else 1
# cpuplace: t, title, pid, tid, cpu, ticks
th = collections.defaultdict(list)  # (title, is_main) -> [(t, cpu, ticks)]
for l in open(cp):
    p = l.rstrip("\n").split("\t")
    if len(p) != 6 or "scheduler_DP" not in p[1]:
        continue
    th[(p[1], p[2] == p[3])].append((float(p[0]), int(p[4]), int(p[5])))
for k in th:
    th[k].sort()
# per-thread summary
print("thread: n_samples | node0% | migrations(cpu change)/min | node switches/min | util%")
for (title, main), v in sorted(th.items()):
    a = np.array(v)
    if len(a) < 10: continue
    dur = (a[-1, 0] - a[0, 0]) / 60
    cpus = a[:, 1].astype(int); nodes = np.array([node(c) for c in cpus])
    util = np.diff(a[:, 2]) / np.diff(a[:, 0])  # ticks/s (100 = 1 core)
    print(f"  {title[-14:]} {'main ' if main else 'helper'}: {len(a)} | {100*(nodes==0).mean():5.1f} | {(np.diff(cpus)!=0).sum()/dur:6.1f} | {(np.diff(nodes)!=0).sum()/dur:5.2f} | {util.mean():5.1f}")
# decode step windows (per-DP lines -> 4-DP ticks)
pat = re.compile(r"(\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)\.\d+)Z \[[^\]]*DP(\d) TP\d\] Decode batch, #running-req: (\d+)")
rows = []
for m in pat.finditer(subprocess.run(["strings", dlog], capture_output=True, text=True).stdout):
    if ws <= m.group(2) <= we:
        rows.append((dt.datetime.fromisoformat(m.group(1)[:26]).replace(tzinfo=dt.timezone.utc).timestamp(), int(m.group(3)), int(m.group(4))))
rows.sort(); ticks, cur = [], []
for r in rows:
    if cur and r[0] - cur[0][0] > 1.0: ticks.append(cur); cur = []
    cur.append(r)
if cur: ticks.append(cur)
ticks = [t for t in ticks if len({x[1] for x in t}) == 4 and len(t) == 4]
W = []
for a, b in zip(ticks, ticks[1:]):
    t0, t1 = np.mean([x[0] for x in a]), np.mean([x[0] for x in b])
    mr = max(x[2] for x in b)
    if 0 < t1 - t0 <= 8 and mr in REF:
        W.append((t0, t1, (t1 - t0) / 40 * 1e3 - REF[mr]))
print(f"windows={len(W)} residual median {np.median([w[2] for w in W]):+.1f} ms")
# features per window: total main-thread node switches, cpu migrations, min main util, helper on same SMT core as main
mains = {t: np.array(v) for (t, m), v in th.items() if m}
F = []
for t0, t1, r in W:
    sw = mig = 0; utils = []; cross = 0
    for t, a in mains.items():
        sel = a[(a[:, 0] >= t0 - 0.5) & (a[:, 0] <= t1 + 0.5)]
        if len(sel) < 2: continue
        c = sel[:, 1].astype(int)
        mig += (np.diff(c) != 0).sum(); sw += (np.diff([node(x) for x in c]) != 0).sum()
        utils.append(((sel[-1, 2] - sel[0, 2]) / (sel[-1, 0] - sel[0, 0])))
    if utils:
        F.append((r, mig, sw, min(utils), np.mean(utils)))
F = np.array(F)
rk = lambda x: np.argsort(np.argsort(x))
if len(F) > 10:
    for i, n in enumerate(("cpu migrations (4 mains)", "node switches", "min main util", "mean main util"), start=1):
        c = np.corrcoef(rk(F[:, 0]), rk(F[:, i]))[0, 1] if F[:, i].std() > 0 else float("nan")
        print(f"  r(residual, {n}) = {c:+.2f}   range {F[:, i].min():.1f}..{F[:, i].max():.1f}")
    hi = F[F[:, 2] > 0, 0]; lo = F[F[:, 2] == 0, 0]
    if len(hi) and len(lo):
        print(f"  residual median: windows with node switch {np.median(hi):+.1f} (n={len(hi)}) vs none {np.median(lo):+.1f} (n={len(lo)})")
# SMT co-location: optional 5th arg = cpuplace.prefill.tsv. For each 1-s sample, does any OTHER busy thread
# (decode helper or any prefill busy thread) sit on the same physical core (cpu//2; siblings are adjacent) as a decode main?
if len(sys.argv) > 5:
    others = collections.defaultdict(list)  # rounded t -> [cores]
    for fn, keep_main in ((cp, False), (sys.argv[5], True)):
        for l in open(fn):
            p = l.rstrip("\n").split("\t")
            if len(p) != 6: continue
            is_dec_main = fn == cp and "scheduler_DP" in p[1] and p[2] == p[3]
            if is_dec_main: continue
            others[round(float(p[0]))].append(int(p[4]) // 2)
    co = []  # (t, n_mains_colocated)
    for t, a in mains.items():
        for row in a:
            ts = round(row[0]); core = int(row[1]) // 2
            near = others.get(ts, []) + others.get(ts - 1, []) + others.get(ts + 1, [])
            co.append((row[0], t, core in near))
    cot = collections.defaultdict(int)
    for tt, t, c in co:
        cot[round(tt)] += c
    print(f"  SMT co-location: {100*np.mean([c for _, _, c in co]):.1f}% of main-thread samples share a physical core with another busy thread")
    G = []
    for t0, t1, r in W:
        n = sum(v for k, v in cot.items() if t0 - 0.5 <= k <= t1 + 0.5)
        G.append((r, n))
    G = np.array(G)
    if len(G) > 10 and G[:, 1].std() > 0:
        print(f"  r(residual, co-located main-seconds) = {np.corrcoef(rk(G[:,0]), rk(G[:,1]))[0,1]:+.2f}; residual median with co-loc {np.median(G[G[:,1]>0,0]):+.1f} (n={(G[:,1]>0).sum()}) vs none {np.median(G[G[:,1]==0,0]) if (G[:,1]==0).any() else float('nan'):+.1f} (n={(G[:,1]==0).sum()})")
