# Summarize a run_hostsample.yihou.sh output dir; also writes gpu_ts.tsv / host_ts.tsv time series.
import csv, collections, datetime as dt, io, os, re, statistics as st, sys

out = sys.argv[1]
# 1) per-thread CPU (sidecars)
f = os.path.join(out, "threads.tsv")
if os.path.exists(f):
    agg = collections.defaultdict(list); nsamp = collections.defaultdict(set)
    for row in csv.DictReader(open(f), delimiter="\t"):
        if row["t_end"] == "t_end" or row.get("cpu_pct") is None:
            continue
        nsamp[row["container"]].add(row["t_end"])
        agg[(row["container"], row["pid"], row["proc_title"][:60], row["tid"], row["comm"])].append(float(row["cpu_pct"]))
    n = max((len(v) for v in nsamp.values()), default=1)
    print(f"== per-thread CPU (mean % over {n} intervals; threads >=1% shown)")
    per_proc = collections.defaultdict(float)
    for (c, pid, t, tid, comm), v in agg.items():
        per_proc[(c, pid, t)] += sum(v) / n
    for mean, mx, (c, pid, t, tid, comm) in sorted(((sum(v) / n, max(v), k) for k, v in agg.items()), reverse=True)[:25]:
        print(f"  {mean:6.1f}% (max {mx:5.1f})  {c[-24:]:24s} pid={pid:>7s} tid={tid:>7s} {comm:16s} {t}")
    print("== per-process CPU (sum of threads, mean %)")
    for (c, pid, t), v in sorted(per_proc.items(), key=lambda x: -x[1])[:16]:
        print(f"  {v:6.1f}%  {c[-24:]:24s} pid={pid:>7s} {t}")

# 2) ticks.log: sections per tick
f = os.path.join(out, "ticks.log")
if not os.path.exists(f):
    sys.exit(0)
ticks = re.split(r"^=== ", open(f, errors="replace").read(), flags=re.M)[1:]
gpu_rows, host_rows, foreign = [], [], collections.Counter()
nfs_rows, log_rows, nfsop_rows = [], [], []
for tk in ticks:
    ts = tk.split("\n", 1)[0].strip()
    secs = dict(re.findall(r"^--- (\w+)\n(.*?)(?=^--- |\Z)", tk, flags=re.M | re.S))
    if "load" in secs:
        ll = secs["load"].splitlines()
        la = float(ll[0].split()[0]) if ll and ll[0][:1].isdigit() else None
        idle = next((float(l.split()[-1]) for l in ll if l.startswith("Average:")), None)
        host_rows.append((ts, la, idle))
    for l in secs.get("foreign", "").splitlines():
        if l.strip():
            foreign[l.split("|")[0] + " | " + (l.split("|")[1] if "|" in l else "")] += 1
    m = re.search(r"write_fsync_us=(\d+) stat_us=(\d+)", secs.get("nfs", ""))
    if m:
        nfs_rows.append((ts, int(m.group(1)), int(m.group(2))))
    nst = {}
    for l in secs.get("nfsstats", "").splitlines():
        q = l.split()
        if len(q) == 5:
            nst[q[0].rstrip(":")] = tuple(int(x) for x in q[1:])
    if nst:
        nfsop_rows.append((ts, nst))
    for l in secs.get("lograte", "").splitlines():
        mm = re.match(r"(\S+) bytes_10s=(\d+)", l)
        if mm:
            log_rows.append((ts, mm.group(1), int(mm.group(2))))
    caps = {}
    for r in csv.DictReader(io.StringIO(secs.get("powercap", ""))):
        caps[r.get("gpu")] = r.get("max_power")
    for r in csv.DictReader(io.StringIO(secs.get("amdsmi", ""))):
        g = r.get("gpu")
        if g is None or not g.isdigit():
            continue
        gpu_rows.append((ts, g, r.get("socket_power"), caps.get(g), r.get("gfx_0_clk"), r.get("mem_0_clk"),
                         r.get("gfx_activity"), r.get("umc_activity"), r.get("hotspot"), r.get("mem"),
                         r.get("throttle_status")))
with open(os.path.join(out, "gpu_ts.tsv"), "w") as w:
    w.write("t\tgpu\tpower_W\tcap_W\tgfx_clk\tmem_clk\tgfx_pct\tumc_pct\thotspot_C\tmem_C\tthrottle\n")
    for r in gpu_rows:
        w.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(out, "host_ts.tsv"), "w") as w:
    w.write("t\tloadavg1\tidle_pct\n")
    for r in host_rows:
        w.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(out, "nfs_ts.tsv"), "w") as w:
    w.write("t\twrite_fsync_us\tstat_us\n")
    for r in nfs_rows:
        w.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(out, "lograte_ts.tsv"), "w") as w:
    w.write("t\tcontainer\tbytes_10s\n")
    for r in log_rows:
        w.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(out, "nfsops_ts.tsv"), "w") as w:
    w.write("t\top\tops_per_s\tmean_rtt_ms\tmean_exec_ms\n")
    for (ta, a), (tb, b) in zip(nfsop_rows, nfsop_rows[1:]):
        try:
            dts = (dt.datetime.strptime(tb, "%H:%M:%S") - dt.datetime.strptime(ta, "%H:%M:%S")).total_seconds()
        except ValueError:
            continue
        if dts <= 0:
            continue
        for op in b:
            if op in a:
                dn = b[op][0] - a[op][0]
                if dn > 0:
                    w.write(f"{tb}\t{op}\t{dn/dts:.1f}\t{(b[op][2]-a[op][2])/dn:.3f}\t{(b[op][3]-a[op][3])/dn:.3f}\n")
if nfs_rows:
    wf = sorted(r[1] for r in nfs_rows); sf = sorted(r[2] for r in nfs_rows)
    print(f"== NFS probe ({len(nfs_rows)}): write+fsync p50 {wf[len(wf)//2]/1e3:.1f} ms max {wf[-1]/1e3:.1f} | stat p50 {sf[len(sf)//2]/1e3:.1f} ms max {sf[-1]/1e3:.1f}")
fl = lambda v: float(v) if v not in (None, "", "N/A") else None
ok = [r for r in host_rows if r[1] is not None and r[2] is not None]
if ok:
    print(f"== host ({len(host_rows)} ticks): loadavg1 mean {st.mean(r[1] for r in ok):.1f} max {max(r[1] for r in ok):.1f}; "
          f"mpstat idle% mean {st.mean(r[2] for r in ok):.1f} min {min(r[2] for r in ok):.1f}")
print("== per GPU mean (min..max): power W | cap W | gfx clk MHz | mem clk | gfx% | umc% | hotspot C | mem C | throttle values")
per = collections.defaultdict(lambda: collections.defaultdict(list))
for r in gpu_rows:
    for i, k in enumerate(("power", "cap", "gclk", "mclk", "gfx", "umc", "hot", "memt"), start=2):
        v = fl(r[i])
        if v is not None:
            per[r[1]][k].append(v)
    per[r[1]]["thr"].append(r[10])
for g in sorted(per, key=int):
    d = per[g]
    m = lambda k: (f"{st.mean(d[k]):5.0f}({min(d[k]):.0f}..{max(d[k]):.0f})" if d[k] else "-")
    print(f"  GPU{g}: {m('power')} | {m('cap')} | {m('gclk')} | {m('mclk')} | {m('gfx')} | {m('umc')} | {m('hot')} | {m('memt')} | {sorted(set(d['thr']))}")
print("== non-yihou containers seen (ticks present)")
for k, v in foreign.most_common(20):
    print(f"  {v:4d}  {k}")
if not foreign:
    print("  none")
