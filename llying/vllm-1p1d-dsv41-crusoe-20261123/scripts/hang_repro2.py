"""Stability repro for both open problems, with the GPUs busy most of the time (a GPU-utilization watchdog cancels jobs
that stay under 25% for 45 min; hang_repro.py idled ~85% of each cycle).

A cycle:
  1. idle IDLE_S seconds (every DP2 hang so far came on the first step after both ranks of a leg had idled for minutes);
  2. wake: one 7,500-word request alone;
  3. surge: SURGE requests sent at once, each a cycle-wide shared prefix of SURGE_WORDS words plus a unique tail,
     max_tokens=1 -- like AgentX's warmup phase switch, where ~1,000 requests reach the router within 2 s and some
     connections get reset. TCP counters (/proc/net/netstat TcpExt) and the router's listen queue are read before the
     surge, right after the last send, and when it has drained; every failed request is logged with its error;
  4. load: LOAD_S seconds with LOAD_CONC requests always in flight (random 1k-64k words, no shared prefix, max_tokens
     32-256), then wait for all.
The loop stops when the engine logs show a fatal error, or when the time budget ends.

usage: hang_repro2.py ROUTER_URL MODEL LOG_DIR [BUDGET_S=14400] [IDLE_S=150] [LOAD_S=300] [LOAD_CONC=48] [SURGE=480]
       [SURGE_WORDS=16384]
Prints one line per cycle; the last line is "RESULT cycles=N hangs=H resets=R ...".
"""
import concurrent.futures as cf
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

FATAL = ("engine core reported fatal failure", "LLM became permanently unhealthy")
TCP_KEYS = ("ListenOverflows", "ListenDrops", "TCPAbortOnData", "TCPAbortOnClose", "TCPAbortOnMemory",
            "TCPAbortOnTimeout", "TCPAbortOnLinger", "TCPAbortFailed", "TCPReqQFullDrop", "TCPReqQFullDoCookies",
            "SyncookiesSent", "SyncookiesRecv", "SyncookiesFailed", "TCPBacklogDrop", "TCPRcvQDrop", "TCPMemoryPressures",
            "EmbryonicRsts", "TCPOFODrop", "TCPZeroWindowDrop", "TCPToZeroWindowAdv")


def words(n, seed):
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(1 << 20):x}" for _ in range(n))


def request(url, model, content, max_tokens, timeout=900):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status == 200, time.time() - t, "" if r.status == 200 else f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, time.time() - t, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t, f"{type(e).__name__}: {e}"[:160]


def tcp_counters():
    with open("/proc/net/netstat") as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith("TcpExt:"):
            return dict(zip(lines[i].split()[1:], map(int, lines[i + 1].split()[1:])))
    return {}


def listen_queue(port):
    out = subprocess.run(["ss", "-Hltn", f"sport = :{port}"], capture_output=True, text=True).stdout.split()
    return f"recvq={out[1]} backlog={out[2]}" if len(out) > 2 else "?"


def delta(a, b):
    return " ".join(f"{k}+{b.get(k, 0) - a.get(k, 0)}" for k in TCP_KEYS if b.get(k, 0) != a.get(k, 0)) or "none"


def engine_died(log_dir):
    died = []
    for leg in ("prefill", "decode"):
        try:
            with open(os.path.join(log_dir, f"{leg}.log"), errors="replace") as f:
                if any(m in f.read() for m in FATAL):
                    died.append(leg)
        except OSError:
            pass
    return died


def surge(url, model, port, n, n_words, cycle, log):
    prefix = words(n_words, cycle * 104729)
    bodies = [f"{prefix}\n\nRequest {i} of cycle {cycle}: {words(64, cycle * 100003 + i)}" for i in range(n)]
    c0, q0 = tcp_counters(), listen_queue(port)
    go = threading.Barrier(n + 1)
    res = [None] * n

    def one(i):
        go.wait()
        res[i] = (time.time(),) + request(url, model, bodies[i], 1)

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    go.wait()
    t0 = time.time()
    time.sleep(2)
    c1, q1 = tcp_counters(), listen_queue(port)
    for t in threads:
        t.join()
    c2 = tcp_counters()
    bad = [(r[0] + r[2] - t0, r[3]) for r in res if not r[1]]
    for dt, err in bad:
        log.write(f"surge cycle {cycle}: failed at +{dt:.2f}s: {err}\n")
    log.flush()
    errs = sorted({e.split(":")[0] for _, e in bad})
    return (len(bad), f"surge {n - len(bad)}/{n} ok in {time.time() - t0:.0f}s{' errors ' + ','.join(errs) if errs else ''}; "
            f"listen {q0} -> {q1}; tcp +2s: {delta(c0, c1)}; tcp end: {delta(c1, c2)}")


def load(url, model, seconds, conc, cycle, log_dir):
    rng = random.Random(cycle * 7)
    stop = [time.time() + seconds]
    lock, stats = threading.Lock(), {"ok": 0, "fail": 0, "slowest": 0.0, "errs": set()}

    def lane(j):
        k = 0
        while time.time() < stop[0]:
            k += 1
            n = rng.choice((1024, 2048, 4096, 8192, 16384, 32768, 65536))
            ok, dt, err = request(url, model, words(n, cycle * 1000003 + j * 1009 + k), rng.randint(32, 256))
            with lock:
                stats["ok" if ok else "fail"] += 1
                stats["slowest"] = max(stats["slowest"], dt)
                if err:
                    stats["errs"].add(err.split(":")[0])
            if not ok:  # a dead engine fails requests at once; do not spin
                time.sleep(5)
                if engine_died(log_dir):
                    stop[0] = 0

    with cf.ThreadPoolExecutor(conc) as ex:
        list(ex.map(lane, range(conc)))
    return stats


def main():
    base = sys.argv[1].rstrip("/")
    url = base + "/v1/chat/completions"
    port = urllib.parse.urlparse(base).port
    model, log_dir = sys.argv[2], sys.argv[3]
    a = [float(x) for x in sys.argv[4:]] + [None] * 6
    budget = a[0] or 14400
    idle = a[1] if a[1] is not None else 150
    load_s = a[2] if a[2] is not None else 300
    load_conc, n_surge, surge_words = int(a[3] or 48), int(a[4] if a[4] is not None else 480), int(a[5] or 16384)
    socket.setdefaulttimeout(900)
    log = open(os.path.join(log_dir, "surge-errors.log"), "a")
    t0, cycle, resets = time.time(), 0, 0
    while time.time() - t0 + idle + load_s + 300 < budget:
        cycle += 1
        time.sleep(idle)
        tc = time.time()
        ok1, dt1, _ = request(url, model, words(7500, cycle * 7919), 32)
        nbad, s = surge(url, model, port, n_surge, surge_words, cycle, log) if n_surge else (0, "no surge")
        resets += nbad
        st = load(url, model, load_s, load_conc, cycle, log_dir)
        died = engine_died(log_dir)
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} cycle {cycle}: single ok={ok1} {dt1:.1f}s; {s}; "
              f"load {st['ok']} ok {st['fail']} failed{' (' + ','.join(sorted(st['errs'])) + ')' if st['errs'] else ''}, "
              f"slowest {st['slowest']:.0f}s; cycle {time.time() - tc:.0f}s"
              f"{'; ENGINE DIED: ' + ' '.join(died) if died else ''}", flush=True)
        if died:
            print(f"RESULT cycles={cycle} hangs=1 resets={resets} died={'+'.join(died)} elapsed={time.time() - t0:.0f}s",
                  flush=True)
            return
    print(f"RESULT cycles={cycle} hangs=0 resets={resets} elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
