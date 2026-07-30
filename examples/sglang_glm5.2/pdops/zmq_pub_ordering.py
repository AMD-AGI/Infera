"""Reproduce the engine's ordering: publisher (and its thread) exist long before
any subscriber shows up, then a batch is published.

The earlier isolated test connected the SUB *before* the thread ever touched the
socket, which is the opposite of what the scheduler does — the publisher thread
starts during engine init and spins on an empty queue for minutes while weights
load, and the router only subscribes later.
"""

import sys
import time

import zmq

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.disaggregation.kv_events import (  # noqa: E402
    BlockStored,
    EventPublisherFactory,
    KVEventBatch,
)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 42780
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

pub = EventPublisherFactory.create(
    '{"publisher": "zmq", "endpoint": "tcp://*:%d", "topic": "kv-events"}' % PORT, 0
)
print(f"publisher up on {getattr(pub, '_endpoint', '?')}")

# Let the publisher thread spin on the empty queue first, the way it does while
# the engine loads weights.
print(f"idling {DELAY}s before any subscriber connects...")
time.sleep(DELAY)

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(f"tcp://127.0.0.1:{PORT}")
sub.subscribe(b"")
print("subscriber connected")
time.sleep(2)

batch = KVEventBatch(
    ts=time.time(),
    events=[
        BlockStored(
            block_hashes=[i],
            parent_block_hash=None,
            token_ids=list(range(64)),
            block_size=64,
            lora_id=None,
        )
        for i in range(217)
    ],
)
pub.publish(batch)
print("published a 217-event batch")

got = 0
deadline = time.time() + 15
while time.time() < deadline and got == 0:
    if sub.poll(timeout=500):
        sub.recv_multipart()
        got += 1

print("RECEIVED" if got else "NO FRAMES — reproduced the engine's symptom")
