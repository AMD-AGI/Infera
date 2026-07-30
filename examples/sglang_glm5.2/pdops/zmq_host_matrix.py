"""Compare ZMQ PUB->SUB delivery over loopback vs the node IP, small vs large.

The engine binds tcp://*:PORT and the router connects via the node IP. Frames
arrive on loopback but not on the node IP, so this isolates the variable pair
(address, payload size) with no engine involved.
"""

import sys
import time

import zmq

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 45900
HOSTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["127.0.0.1", "10.32.17.210"]
SIZES = [64, 4096, 65536, 100_000]

ctx = zmq.Context()

for host in HOSTS:
    for size in SIZES:
        port = PORT
        PORT += 1
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://*:{port}")
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://{host}:{port}")
        sub.subscribe(b"")
        # PUB drops until the subscription has propagated back to the publisher.
        time.sleep(1.5)

        pub.send_multipart([b"kv-events", b"x" * size])
        got = sub.poll(timeout=4000)
        payload = sub.recv_multipart()[1] if got else b""
        print(
            f"{host:<15} size={size:<7} "
            f"{'OK ' + str(len(payload)) + 'B' if got else 'LOST'}",
            flush=True,
        )
        sub.close()
        pub.close()
