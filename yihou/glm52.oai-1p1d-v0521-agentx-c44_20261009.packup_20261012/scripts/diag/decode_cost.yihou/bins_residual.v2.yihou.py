# 10-min-bin matched-load residual (vs e2c40 table) per run + stack age (decode log first timestamp).
import runpy, sys, io, contextlib, subprocess, re, datetime as dt
import numpy as np
R = "/shared_nfs/yihou_glm52/v0521dbg_20261010/runs"
RUNS = [("e1c40", "e1c", "08:19:31", "09:19:31"), ("e2b4-C44", "e2b4", "10:04:41", "11:04:41"),
        ("e2c-C44", "e2c", "12:12:35", "13:12:35"), ("e2c40", "e2c", "16:15:52", "17:15:52"),
        ("e2d", "e2d", "18:16:58", "19:16:58"), ("e2e", "e2e", "20:49:36", "21:49:36"),
        ("e2f", "e2f", "22:59:00", "23:59:00"), ("e2gA1", "e2g", "06:11:25", "07:11:25")]
extra = [a.split(",") for a in sys.argv[1:]]  # name,rundir,start,end
for name, rd, ws, we in RUNS + [tuple(e) for e in extra]:
    rdp = rd if rd.startswith("/") else f"{R}/{rd}"
    import glob
    dl = sorted(glob.glob(f"{rdp}/server-logs/decode-0.a*.log"))[-1]
    sys.argv = ["x", dl, "/dev/null", ws, we]
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ns = runpy.run_path("step_vs_gpu.yihou.py")
    T, REF = ns["T"], ns["REF"]
    if not T:
        print(name, "no ticks"); continue
    first = subprocess.run(f"strings {dl} | grep -a -m1 -oE '^[0-9]{{4}}-[0-9-]+T[0-9:]+'", shell=True, capture_output=True, text=True).stdout.strip()
    age = (T[0][0] - dt.datetime.fromisoformat(first).replace(tzinfo=dt.timezone.utc).timestamp()) / 60 if first else float("nan")
    t0 = T[0][0]; b = {}
    for t, mr, s in T:
        if mr in REF:
            b.setdefault(int((t - t0) // 600), []).append(s - REF[mr])
    allr = [x for v in b.values() for x in v]
    print(f"{name:9s} stack_age@start {age:5.0f} min | all {np.median(allr):+.1f} | " +
          " ".join(f"{k*10}-{k*10+10}m:{np.median(v):+.1f}({len(v)})" for k, v in sorted(b.items()) if k < 6))
