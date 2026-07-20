###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the MoRIIO write/push is_producer fix inside an engine container.

Workaround for AMD-AGI/Infera#67: in WRITE (push) mode the decode sends the
block-allocate notification with get_peer_zmq_from_request_id(..., is_producer=True),
so it addresses ITSELF; the consumer-side handler asserts get_role()==PRODUCER
("Only prefill can get block messages"), the moriio-notify thread dies, and the
request hangs (HTTP 000). The decode must address the PREFILL -> is_producer=False.
Upstream fixed this on vLLM main / v0.22.1rc0 SOURCE, but NO AMD ROCm image ships it
(v0.20.2 / v0.21.0 / v0.22.0 / v0.22.1 all still hardcode is_producer=True). pull
(read) mode never hits this branch, so it does NOT need this patch.

Idempotent. Run inside the container:
    docker exec <ctr> python3 <repo>/pd/patch_moriio_write.py
"""

import os
import sys

import vllm

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
)
if not os.path.exists(f):
    print(f"moriio-guard: {f} not found (no moriio connector in this image) — skipping")
    sys.exit(0)

src = open(f).read()
old = "request.request_id, is_producer=True"
new = "request.request_id, is_producer=False"

if new in src:
    print("moriio-guard: already patched")
    sys.exit(0)

n = src.count(old)
if n != 1:
    print(
        f"moriio-guard: expected 1 occurrence of is_producer=True, found {n} — moriio layout changed?"
    )
    sys.exit(1)

open(f, "w").write(src.replace(old, new))
print("moriio-guard: patched", f)
