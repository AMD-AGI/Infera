"""Subscribe to every kv-event port over BOTH loopback and the node IP at once.

Two sequential single-address runs cannot tell an address problem apart from
having simply missed the publish window, because frames only appear while a
request is in flight. Listening on both addresses in one process during one
request removes that ambiguity: whatever the engine sends is offered to both
sockets at the same instant.
"""

import json
import sys
import threading
import time
import urllib.request

import zmq

HOSTS = (sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1,10.32.17.210").split(",")
BASE_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 32760
RANKS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
WINDOW = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
ROUTER = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/wekafs/models/GLM-5.2-FP8"

ctx = zmq.Context()
poller = zmq.Poller()
owner = {}
for host in HOSTS:
    for r in range(RANKS):
        s = ctx.socket(zmq.SUB)
        s.connect(f"tcp://{host}:{BASE_PORT + r}")
        s.subscribe(b"")
        owner[s] = (host, r)
        poller.register(s, zmq.POLLIN)

time.sleep(3)

done = threading.Event()


def fire():
    salt = f"Run {time.time():.6f}\n"
    body = salt + "\n".join(
        f"Line {i}: unique {i * 37 % 97} slot {i % 13}." for i in range(900)
    )
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": body + "\n\nReply OK."}],
            "max_tokens": 64,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        ROUTER, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=WINDOW) as resp:
            u = json.loads(resp.read()).get("usage", {})
            print(f"[req] prompt={u.get('prompt_tokens')} done at {time.strftime('%H:%M:%S')}", flush=True)
    except Exception as exc:
        print(f"[req] failed: {exc}", flush=True)
    finally:
        done.set()


threading.Thread(target=fire, daemon=True).start()

counts = {(h, r): 0 for h in HOSTS for r in range(RANKS)}
deadline = time.time() + WINDOW
# Keep listening for a while after the response so a late publish still lands.
grace_until = None
while time.time() < deadline:
    for sock, _ in poller.poll(timeout=1000):
        host, rank = owner[sock]
        frames = sock.recv_multipart()
        counts[(host, rank)] += 1
        print(
            f"[frame] {time.strftime('%H:%M:%S')} {host} dp{rank} "
            f"topic={frames[0]!r} bytes={len(frames[-1])}",
            flush=True,
        )
    if done.is_set():
        if grace_until is None:
            grace_until = time.time() + 25
        elif time.time() > grace_until:
            break

print("\nper-host totals:")
for host in HOSTS:
    per_rank = {r: counts[(host, r)] for r in range(RANKS) if counts[(host, r)]}
    total = sum(counts[(host, r)] for r in range(RANKS))
    print(f"  {host:<15} frames={total} {per_rank}")
