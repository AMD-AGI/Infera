#!/usr/bin/env python3
"""One-screen health view of the running tuning round (read-only; run from the login host).

Usage: tune_watch.py [ROUND] [MINUTES]
Shows the queue driver state, the current round's client progress, and over the last
MINUTES (default 5) of server logs: prefill compute / cache share / queues per instance,
decode running / KV usage / pre-allocation / transfer queues per DP rank, plus new fatal
log patterns since the round started, router WARN/ERROR lines and GPU / host memory.
"""

import collections
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
ROUND = sys.argv[1] if len(sys.argv) > 1 else "r2"
MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
RUNS = KIT / "runtime" / "runs" / ROUND
LOGS = RUNS / "launch" / "server-logs"
TUNE = RUNS / "tune"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-n"]
CONTROL = "smci355-ccs-aus-n01-29"
NODES = {"smci355-ccs-aus-n01-29": "0,1,2,3,4,5,6,7", "smci355-ccs-aus-n02-33": "0,1,2,3,4,5,6,7"}
FATAL = re.compile(r"Memory access fault|out of memory|OutOfMemory|Traceback \(most recent call last\)|"
                   r"exited with code|scheduler died|transfer failed|Process failed for slice|"
                   r"retry counter exceeded|Session \S+ failed\.|hipIpc\w* failed|NCCL WARN")
PREFILL = re.compile(
    r"^(\S+)Z \[[^\]]*DP(\d+) TP\d+\] Prefill batch, #new-seq: \d+, #new-token: (\d+), "
    r"#cached-token: (\d+), token usage: ([0-9.]+), #running-req: \d+, #queue-req: (\d+), "
    r"#pending-token: \d+, #bootstrap-req: (\d+), #inflight-req: (\d+)")
DECODE = re.compile(
    r"^(\S+)Z \[[^\]]*DP(\d+) TP\d+\] Decode batch, #running-req: (\d+), #token: \d+, "
    r"token usage: ([0-9.]+),.*?#prealloc-req: (\d+), #transfer-req: (\d+), #retracted-req: (\d+)"
    r".*?gen throughput \(token/s\): ([0-9.]+), #queue-req: (\d+)")


def tail_lines(path: Path, nbytes: int = 64 << 20):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        data = f.read().decode(errors="replace")
    return data.splitlines()[1:]


def sh(cmd, timeout=25) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def current_round():
    status = TUNE / "status.tsv"
    if not status.exists():
        return None, []
    rows = [l.split("\t") for l in status.read_text().splitlines() if l.strip()]
    return (rows[-1][1] if rows else None), rows[-6:]


def queue_state():
    pid = (TUNE / "queue.pid").read_text().strip() if (TUNE / "queue.pid").exists() else ""
    alive = sh(SSH + [CONTROL, f"kill -0 {pid} 2>/dev/null && echo alive || echo dead"]).strip() if pid else "no pid"
    done = (TUNE / "queue.done").read_text().strip() if (TUNE / "queue.done").exists() else "0"
    return pid, alive, done


def client_progress(name):
    logs = sorted((TUNE / name).glob("agentx-*/runner.log")) if name else []
    if not logs:
        return "no runner.log yet"
    last = [l for l in tail_lines(logs[-1], 4 << 20) if "progress |" in l or "Phase " in l]
    return last[-1].strip()[:200] if last else "no progress line yet"


def window_lines(path: Path, since: datetime):
    s = since.strftime("%Y-%m-%dT%H:%M:%S")
    return [l for l in tail_lines(path) if l[:19] >= s]


def excluded() -> set:
    f = KIT / "config" / f"round.{ROUND}.sh"
    names = set()
    if f.exists():
        for line in f.read_text().splitlines():
            m = re.match(r'\s*EXCLUDE_INSTANCES="?([^"#]*)"?', line)
            if m:
                names = set(m.group(1).replace(",", " ").split())
    return names


def prefill_view(since: datetime):
    print(f"prefill (last {MINUTES:g} min): instance  new tok/s  cache%  queue(mean/max)  bootstrap  inflight  usage(max)")
    for log in sorted(LOGS.glob("prefill-[0-9].log")):
        if log.stem in excluded():
            continue
        new = cached = 0
        q, b, inf, use = [], [], [], []
        for l in window_lines(log, since):
            m = PREFILL.match(l)
            if m:
                new += int(m.group(3)); cached += int(m.group(4)); use.append(float(m.group(5)))
                q.append(int(m.group(6))); b.append(int(m.group(7))); inf.append(int(m.group(8)))
        if not q:
            print(f"  {log.stem:10} no batches")
            continue
        share = 100 * cached / max(1, new + cached)
        print(f"  {log.stem:10} {new / (MINUTES * 60):9.0f}  {share:5.1f}%  {sum(q) / len(q):5.1f}/{max(q):<4d}"
              f"  {sum(b) / len(b):9.2f}  {sum(inf) / len(inf):8.2f}  {max(use):.2f}")


def decode_view(since: datetime, decode_log: str):
    last = {}
    agg = collections.defaultdict(list)
    for l in window_lines(LOGS / decode_log, since):
        m = DECODE.match(l)
        if m:
            dp = int(m.group(2))
            vals = (int(m.group(3)), float(m.group(4)), int(m.group(5)), int(m.group(6)),
                    int(m.group(7)), float(m.group(8)), int(m.group(9)))
            last[dp] = vals
            agg[dp].append(vals)
    if not agg:
        print("decode: no batches in window")
        return
    print(f"decode (last {MINUTES:g} min, per DP: mean running / mean usage / max usage / mean prealloc / mean transfer)")
    cells = []
    for dp in sorted(agg):
        a = agg[dp]
        cells.append(f"DP{dp} {sum(x[0] for x in a) / len(a):4.1f}/{sum(x[1] for x in a) / len(a):.2f}/"
                     f"{max(x[1] for x in a):.2f}/{sum(x[2] for x in a) / len(a):.1f}/{sum(x[3] for x in a) / len(a):.1f}")
    for i in range(0, len(cells), 4):
        print("  " + "   ".join(cells[i:i + 4]))
    run = sum(v[0] for v in last.values())
    retr = sum(v[4] for v in last.values())
    gen = sum(v[5] for v in last.values())
    print(f"  latest: running {run}, retracted {retr}, gen {gen:.0f} tok/s")


def fatal_since(start: str):
    hits = collections.Counter()
    for log in sorted(LOGS.glob("*.log")):
        for l in tail_lines(log, 256 << 20):
            if l[:19] >= start and FATAL.search(l) and "Ignore import error" not in l:
                hits[(log.stem, re.sub(r"^\S+ ", "", l)[:150])] += 1
    return hits


def memory():
    out = []
    for node, gpus in NODES.items():
        txt = sh(SSH + [node, "rocm-smi --showmeminfo vram --csv 2>/dev/null; free -g | sed -n 2p"])
        free_gib = []
        for line in txt.splitlines():
            m = re.match(r"card(\d+),(\d+),(\d+)", line)
            if m and m.group(1) in gpus.split(","):
                free_gib.append((int(m.group(2)) - int(m.group(3))) / 2**30)
        avail = txt.strip().splitlines()[-1].split()[-1] if txt.strip() else "?"
        out.append(f"{node[-3:]}: GPU free min {min(free_gib):.1f} GiB, host avail {avail} GB"
                   if free_gib else f"{node[-3:]}: ?")
    return "; ".join(out)


def main() -> int:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    name, recent = current_round()
    pid, alive, done = queue_state()
    print(f"== {now:%Y-%m-%dT%H:%M:%SZ}  queue pid={pid} {alive}, steps done={done}, round={name}")
    for r in recent:
        print("  " + " | ".join(r)[:220])
    print("client: " + client_progress(name))
    env = {}
    if name and (TUNE / name / "round.env").exists():
        env = dict(l.split("=", 1) for l in (TUNE / name / "round.env").read_text().splitlines() if "=" in l)
    since = now - timedelta(minutes=MINUTES)
    prefill_view(since)
    decode_view(since, env.get("DECODE_LOG", "decode-0.after-realacc.log"))
    start_file = TUNE / name / "bench.start" if name else None
    if start_file and start_file.exists():
        start = start_file.read_text().strip()
        hits = fatal_since(start)
        print(f"fatal patterns since {start}: {sum(hits.values())}")
        for (log, text), c in hits.most_common(6):
            print(f"  {c:4d} {log}: {text}")
        early = TUNE / name / "flush_check.early.txt"
        if early.exists():
            print("flush check: " + early.read_text().strip().splitlines()[-1])
        rw = sh(SSH + [CONTROL, f"docker logs --since {start}Z glm52-llying-3p1d-{ROUND}-router 2>&1 | "
                                "sed -E 's/\\x1b\\[[0-9;]*m//g' | grep -E ' (WARN|ERROR) ' | wc -l"]).strip()
        print(f"router WARN/ERROR since start: {rw}")
    print("memory: " + memory())
    st = __import__("os").statvfs(RUNS)
    live = [LOGS / f"prefill-{i}.log" for i in range(3) if f"prefill-{i}" not in excluded()] + \
        [LOGS / env.get("DECODE_LOG", "decode-0.after-realacc.log")]
    ages = {p.stem: (now - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).replace(tzinfo=None)).total_seconds() for p in live if p.exists()}
    stale = max(ages, key=ages.get)
    print(f"/home free {st.f_bavail * st.f_frsize / 1e9:.0f} GB; stalest server log {stale} ({ages[stale]:.0f} s since last write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
