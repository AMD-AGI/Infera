# Per-thread host CPU sampler (runs in a CPU-only sidecar with --pid host; reads /proc only).
# usage: python3 thread_cpu.yihou.py <out_dir> <duration_s> <interval_s> <container_map_file>
# container_map_file: 'ALL:<name>' when run with --pid container:<name>, else lines "<full_container_id> <name>" (from `docker ps --no-trunc`).
# Output: <out_dir>/threads.tsv  (t_end, container, pid, proc_title, tid, comm, cpu_pct over the interval)
import os, re, sys, time

out_dir, duration, interval, cmap_file = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
HZ = os.sysconf("SC_CLK_TCK")
cmap = {}
ALL = cmap_file[4:] if cmap_file.startswith("ALL:") else None  # sidecar in one container's pid ns
for line in ([] if ALL else open(cmap_file)):
    p = line.split()
    if len(p) >= 2:
        cmap[p[0]] = p[1]
ID_RE = re.compile(r"([0-9a-f]{64})")


def container_of(pid):
    if ALL:
        return ALL if pid != str(os.getpid()) else None
    try:
        for m in ID_RE.finditer(open(f"/proc/{pid}/cgroup").read()):
            if m.group(1) in cmap:
                return cmap[m.group(1)]
    except OSError:
        pass
    return None


def title(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").replace("\t", " ").replace("\n", " ")[:120].strip()
    except OSError:
        return "?"


def snapshot(pids):
    snap = {}
    for pid, cname in pids.items():
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                s = open(f"/proc/{pid}/task/{tid}/stat").read()
                comm = s[s.index("(") + 1:s.rindex(")")]
                f = s[s.rindex(")") + 2:].split()
                snap[(pid, tid)] = (comm, int(f[11]) + int(f[12]))  # utime + stime
            except (OSError, ValueError, IndexError):
                continue
    return snap


def find_pids():
    pids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            c = container_of(d)
            if c:
                pids[d] = c
    return pids


os.makedirs(out_dir, exist_ok=True)
out = open(os.path.join(out_dir, "threads.tsv"), "a")
out.write("t_end\tcontainer\tpid\tproc_title\ttid\tcomm\tcpu_pct\n")
t_stop = time.time() + duration
pids = find_pids()
titles = {p: title(p) for p in pids}
prev, t_prev = snapshot(pids), time.time()
while time.time() < t_stop:
    time.sleep(interval)
    pids = find_pids()
    for p in pids:
        titles.setdefault(p, title(p))
    cur, t_cur = snapshot(pids), time.time()
    dt = t_cur - t_prev
    ts = time.strftime("%H:%M:%S", time.gmtime(t_cur))
    for k, (comm, ticks) in cur.items():
        if k in prev:
            pct = 100.0 * (ticks - prev[k][1]) / HZ / dt
            if pct >= 1.0:
                pid, tid = k
                out.write(f"{ts}\t{pids[pid]}\t{pid}\t{titles[pid]}\t{tid}\t{comm}\t{pct:.1f}\n")
    out.flush()
    prev, t_prev = cur, t_cur
