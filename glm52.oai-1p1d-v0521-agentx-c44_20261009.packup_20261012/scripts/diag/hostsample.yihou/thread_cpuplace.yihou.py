# Every 1 s: for each sglang scheduler process in this pid ns, log (time, pid, tid, comm, last CPU, utime+stime ticks) for its busy threads.
import os, sys, time
out, dur = sys.argv[1], float(sys.argv[2])
f = open(out, "a"); t_end = time.time() + dur
while time.time() < t_end:
    ts = f"{time.time():.1f}"
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            title = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")[0].decode(errors="replace")
        except OSError:
            continue
        if not title.startswith("sglang::"): continue
        try: tids = os.listdir(f"/proc/{p}/task")
        except OSError: continue
        for tid in tids:
            try:
                s = open(f"/proc/{p}/task/{tid}/stat").read()
                x = s[s.rindex(")") + 2:].split()
                tk = int(x[11]) + int(x[12])
                if tk > 100000:
                    f.write(f"{ts}\t{title}\t{p}\t{tid}\t{x[36]}\t{tk}\n")
            except (OSError, ValueError, IndexError):
                pass
    f.flush(); time.sleep(1)
