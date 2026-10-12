# GPU throttle/residency accumulators sampler (R050 follow-up). Runs on the login node and polls via the
# spur ssh shim; read-only `amd-smi metric -v --json` (+ -c --json for per-XCD deep-sleep state).
# Per interval and GPU: residency % of PPT, LOW_UTILIZATION (gfx clk below host limit due to low
# utilization = idle down-clock / deep sleep), thermal, prochot = d(acc_X)/d(accumulation_counter).
# usage: gpuacc_sampler.yihou.py <node> <job> <duration_s> <interval_s> <out_tsv>
import json, os, subprocess, sys, time

node, job, dur, itv, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
env = dict(os.environ, PATH="/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:" + os.environ["PATH"],
           SPUR_SSH_JOBS=f"{node}:{job}")


def read():
    r = subprocess.run(["ssh", node, "amd-smi metric -v --json 2>/dev/null; echo '@@'; amd-smi metric -c --json 2>/dev/null"],
                       capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=60)
    a, _, b = r.stdout.partition("@@")
    v = json.loads(a); c = json.loads(b) if b.strip() else []
    v = v if isinstance(v, list) else v.get("gpu_data", v)
    c = c if isinstance(c, list) else c.get("gpu_data", c)
    res = {}
    for g in v:
        t = g["throttle"]
        lu = t.get("low_utilization_accumulated", {})
        lu = lu.get("xcp_0", []) if isinstance(lu, dict) else []
        res[g["gpu"]] = dict(acc=t["accumulation_counter"], ppt=t["ppt_accumulated"], prochot=t["prochot_accumulated"],
                             sock=t["socket_thermal_accumulated"], hbm=t["hbm_thermal_accumulated"],
                             lowutil=sum(x for x in lu if isinstance(x, int)), nxcd=len(lu),
                             ppt_active=t.get("ppt_violation_status"))
    for g in c:
        clk = g.get("clock", {})
        ds = [v2.get("deep_sleep") for k2, v2 in clk.items() if k2.startswith("gfx_") and isinstance(v2, dict)]
        res.setdefault(g["gpu"], {})["ds_enabled_xcds"] = sum(1 for x in ds if str(x).upper() == "ENABLED")
    return time.time(), res


new = not os.path.exists(out)
f = open(out, "a")
if new:
    f.write("t_utc\tgpu\tppt_pct\tlowutil_pct\tprochot_pct\tsock_thm_pct\thbm_thm_pct\tppt_status\tds_enabled_xcds\n")
t_end = time.time() + dur
t0, prev = read()
while time.time() < t_end:
    time.sleep(max(0.0, itv - (time.time() - t0)))
    try:
        t1, cur = read()
    except Exception as e:  # keep sampling through transient ssh/json errors
        print("read error", e, file=sys.stderr); t0 = time.time(); continue
    ts = time.strftime("%H:%M:%S", time.gmtime(t1))
    for gpu, x in sorted(cur.items()):
        p = prev.get(gpu)
        if not p or "acc" not in x or "acc" not in p:
            continue
        da = x["acc"] - p["acc"]
        if da <= 0:
            continue
        pct = lambda k, n=1: 100.0 * (x[k] - p[k]) / (da * n)
        f.write(f"{ts}\t{gpu}\t{pct('ppt'):.2f}\t{pct('lowutil', max(x['nxcd'], 1)):.2f}\t{pct('prochot'):.2f}\t"
                f"{pct('sock'):.2f}\t{pct('hbm'):.2f}\t{x.get('ppt_active')}\t{x.get('ds_enabled_xcds')}\n")
    f.flush()
    t0, prev = t1, cur
