"""Drive SGLang's own ZmqEventPublisher in isolation and see if a SUB gets frames.

If this passes, the publisher class is fine and the fault is in how the real
deployment wires it up. If it fails, the class itself is broken.
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

CONFIG = '{"publisher": "zmq", "endpoint": "tcp://*:42760", "topic": "kv-events"}'

pub = EventPublisherFactory.create(CONFIG, 0)
print(f"publisher: {type(pub).__name__}")
print(f"endpoint:  {getattr(pub, '_endpoint', '?')}")
print(f"topic:     {getattr(pub, '_topic_bytes', b'?')!r}")

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:42760")
sub.subscribe(b"")
time.sleep(1)

batch = KVEventBatch(
    ts=time.time(),
    events=[
        BlockStored(
            block_hashes=[12345],
            parent_block_hash=None,
            token_ids=list(range(64)),
            block_size=64,
            lora_id=None,
        )
    ],
)
pub.publish(batch)
print("published one batch")

got = []
deadline = time.time() + 10
while time.time() < deadline:
    if sub.poll(timeout=500):
        got.append(sub.recv_multipart())
        break

if got:
    print(f"RECEIVED {len(got)} frame set(s):")
    for f in got[0]:
        print(f"  {f[:60]!r}")
else:
    print("NO FRAMES — the publisher class itself does not deliver.")
