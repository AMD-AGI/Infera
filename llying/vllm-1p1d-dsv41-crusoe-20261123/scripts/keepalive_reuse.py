"""Do idle pooled keep-alive connections to the router get reset when reused all at once?

AgentX's warmup failures (EXPERIMENT 4.2): at the phase switch the first ~75 requests sent were all reset, the ~1,000
sent after them were fine. AIPerf keeps idle connections pooled for up to 300 s (KEEPALIVE_TIMEOUT) with TCP
keepalive idle 60 s / interval 30 s / count 1 and TCP_USER_TIMEOUT 30 s, and about that many first-wave requests had
returned in the last 300 s before the switch, so the reset requests were likely the ones reusing pooled connections.
This opens N connections with AIPerf's socket options, sends one streaming request on each, leaves them idle IDLE
seconds, then sends on all of them at once and reports every failure.

usage: keepalive_reuse.py HOST PORT MODEL [N=80] [IDLES=30,90,150,240,290] [WORDS=2000]
"""
import http.client
import json
import random
import socket
import sys
import threading
import time


def sockopts(sock):
    sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, 60)
    sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, 30)
    sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10485760)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10485760)
    sock.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, 1)
    sock.setsockopt(socket.SOL_TCP, socket.TCP_USER_TIMEOUT, 30000)


class Conn:
    def __init__(self, host, port):
        self.c = http.client.HTTPConnection(host, port, timeout=600)
        self.c.connect()
        sockopts(self.c.sock)
        self.local = self.c.sock.getsockname()[1]

    def post(self, model, words):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": words}], "max_tokens": 1,
                           "stream": True}).encode()
        t = time.time()
        try:
            self.c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            r = self.c.getresponse()
            data = r.read()
            return r.status == 200 and b"[DONE]" in data, time.time() - t, "" if r.status == 200 else f"HTTP {r.status}"
        except Exception as e:  # noqa: BLE001
            return False, time.time() - t, f"{type(e).__name__}: {e}"[:120]


def words(n, seed):
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(1 << 20):x}" for _ in range(n))


def all_at_once(conns, model, nwords, tag):
    go = threading.Barrier(len(conns) + 1)
    res = [None] * len(conns)

    def one(i):
        go.wait()
        res[i] = conns[i].post(model, words(nwords, hash((tag, i)) & 0xFFFFFFFF))

    th = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(conns))]
    for t in th:
        t.start()
    go.wait()
    for t in th:
        t.join()
    return res


def main():
    host, port, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    idles = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "30,90,150,240,290").split(",")]
    nwords = int(sys.argv[6]) if len(sys.argv) > 6 else 2000
    for idle in idles:
        conns = [Conn(host, port) for _ in range(n)]
        first = all_at_once(conns, model, nwords, ("first", idle))
        bad1 = [r[2] for r in first if not r[0]]
        time.sleep(idle)
        t0 = time.time()
        second = all_at_once(conns, model, nwords, ("reuse", idle))
        bad = [(conns[i].local, r[1], r[2]) for i, r in enumerate(second) if not r[0]]
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} idle {idle:g}s: first {n - len(bad1)}/{n} ok; "
              f"reuse {n - len(bad)}/{n} ok in {time.time() - t0:.1f}s"
              f"{'; errors: ' + ', '.join(sorted({b[2].split(':')[0] for b in bad})) if bad else ''}", flush=True)
        for port_, dt, err in bad[:5]:
            print(f"    local port {port_} failed after {dt:.2f}s: {err}", flush=True)
        for c in conns:
            c.c.close()


if __name__ == "__main__":
    main()
