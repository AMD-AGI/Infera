###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Mooncake producer: hand over the KV of a prefill that finishes STOPPED.

Baked into the engine image by Dockerfile.vllm.dsv41. Idempotent; exits non-zero
when its anchor is missing so a base-image bump fails the build instead of
silently shipping an unpatched connector.

Why: MooncakeConnectorScheduler.request_finished only sends a producer request's
KV when it finished FINISHED_LENGTH_CAPPED. A prefill leg runs with max_tokens=1,
but its one sampled token can be EOS or a stop token, and check_stop tests those
before max_tokens, so the request finishes FINISHED_STOPPED. The producer then
frees the blocks and marks the transfer not processed, while the decoder's pull
waits until VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT: the request hangs. Producers
that drop the last prompt token (DeepSeek-V4.1 SWA bounded replay, Mamba) sample
a guess at that token, which is often a stop token: on DeepSeek-V4.1-Flash 1P1D
29 of 128 prefill legs finished STOPPED and the same 29 requests hung.

The fix treats FINISHED_STOPPED like FINISHED_LENGTH_CAPPED, as the NIXL
connector already does. Upstream: llying-001/vllm#3 (not in vllm-project/vllm
main as of 2026-11-20). Drop this patch once the base carries it.
"""

import os
import sys

import vllm

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py",
)
src = open(f).read()
MARKER = "RequestStatus.FINISHED_STOPPED,\n        ):"
if MARKER in src:
    print("mooncake-send-kv-on-stopped: already patched")
    sys.exit(0)

old = "        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:\n"
new = (
    "        # A completed prefill can also stop on its one sampled token (EOS or a\n"
    "        # stop token); the decoder never reads that token, so its KV is handed\n"
    "        # over all the same, or the decoder's pull waits for a send that never\n"
    "        # comes.\n"
    "        if request.status not in (\n"
    "            RequestStatus.FINISHED_LENGTH_CAPPED,\n"
    "            RequestStatus.FINISHED_STOPPED,\n"
    "        ):\n"
)
n = src.count(old)
if n != 1:
    print(f"mooncake-send-kv-on-stopped: expected 1 anchor in {f}, found {n}")
    sys.exit(1)
open(f, "w").write(src.replace(old, new))
print("mooncake-send-kv-on-stopped: patched", f)
