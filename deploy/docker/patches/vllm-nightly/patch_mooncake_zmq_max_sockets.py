###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Mooncake connector: raise libzmq's per-context socket cap.

Baked into the engine image by Dockerfile.vllm.dsv41. Idempotent; exits non-zero
when its anchor is missing.

Why: a decode worker holds one DEALER socket per pending pull and producer rank
until the prefill answers. With hundreds of requests queued on the prefill that
exceeds libzmq's default of 1023 sockets per context, and every new pull fails
with EMFILE ("Too many open files"). Seen on DeepSeek-V4.1-Flash P/D under
AgentX at concurrency 768. VLLM_MOONCAKE_ZMQ_MAX_SOCKETS overrides the new cap
(default 65536); the process also needs a matching RLIMIT_NOFILE.
"""

import os
import sys

import vllm

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py",
)
src = open(f).read()
MARKER = "VLLM_MOONCAKE_ZMQ_MAX_SOCKETS"
if MARKER in src:
    print("mooncake-zmq-max-sockets: already patched")
    sys.exit(0)
if "\nimport os\n" not in src:
    print(f"mooncake-zmq-max-sockets: {f} does not import os")
    sys.exit(1)

old = "        self.async_zmq_ctx = zmq.asyncio.Context()\n"
new = (
    old + "        # One DEALER socket per pending pull and producer rank: hundreds of\n"
    "        # queued requests exceed libzmq's default of 1023 per context.\n"
    "        self.async_zmq_ctx.set(\n"
    '            zmq.MAX_SOCKETS, int(os.environ.get("VLLM_MOONCAKE_ZMQ_MAX_SOCKETS", "65536"))\n'
    "        )\n"
)
n = src.count(old)
if n != 1:
    print(f"mooncake-zmq-max-sockets: expected 1 anchor in {f}, found {n}")
    sys.exit(1)
open(f, "w").write(src.replace(old, new))
print("mooncake-zmq-max-sockets: patched", f)
