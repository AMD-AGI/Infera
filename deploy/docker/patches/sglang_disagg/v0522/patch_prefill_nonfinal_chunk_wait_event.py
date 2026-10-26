#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The prefill.py half of ../patch_mooncake_early_send_wait_event.py, for v0.5.21+.

That script's header is the record for the defect (chunked prefill over mooncake
hands non-final chunks to the RDMA reader while the forward may still be writing
them; prompts longer than one chunk come back partially wrong, silently).

WHAT UPSTREAM NOW CARRIES. sglang#41395 (c683190b, in the v0.5.21 base) landed
the mooncake consumer: `transfer_worker` synchronizes `kv_chunk.wait_event`, and
`MooncakeKVSender.send()` / `add_transfer_request()` carry
`_early_send_wait_event` through. It records an event only on the CACHED-PREFIX
early-send path, and only under overlap. The overlap send of a NON-FINAL chunk in
`process_batch_result_disagg_prefill` still records none. The parent script can no
longer run here -- its mooncake/conn.py anchors are gone because they are upstream
-- so this script carries just the remaining edit.

OPEN, not established: whether this edit is still load-bearing on v0.5.21/HIP.
`process_batch_result_disagg_prefill` already does `copy_done.synchronize()`
before the send, and on HIP `copy_to_cpu` runs on forward_stream, which may
already order the chunk's KV writes before the read. The proven v0519 image had
that sync too and carried the edit anyway. Kept because it is cheap (an event on
an already-finished forward) and removing it was never measured.

PRECONDITION, checked: the upstream consumer must be present, otherwise the event
recorded here would be dropped on the floor and the image would look fixed while
it is not. Exits non-zero if it is missing or if the anchor drifted.

Self-locating and idempotent. Marker: `_infera_nonfinal_chunk_wait_event`.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

_TAG = "[mc-nonfinal-wait-event]"
_MARKER = "_infera_nonfinal_chunk_wait_event"

_CONSUMER = {
    "disaggregation/mooncake/conn.py": (
        "kv_chunk.wait_event.synchronize()",
        'getattr(self, "_early_send_wait_event", None)',
    ),
}

_OLD = """                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
"""
_NEW = f"""                    # Non-final chunks are handed over while later chunks are
                    # already running on forward_stream, and the transfer worker
                    # reads device memory outside the CUDA stream. Gate the read
                    # on this chunk's writes completing.
                    {_MARKER} = torch.cuda.Event()
                    {_MARKER}.record(self.forward_stream)
                    req.disagg_kv_sender._early_send_wait_event = {_MARKER}
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
"""


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        print(f"{_TAG} cannot locate sglang")
        return 1
    srt = Path(spec.origin).parent / "srt"

    for rel, needles in _CONSUMER.items():
        text = (srt / rel).read_text()
        missing = [n for n in needles if n not in text]
        if missing:
            print(f"{_TAG} upstream mooncake wait_event consumer missing in {rel}: {missing}")
            print(f"{_TAG} this base needs the full ../patch_mooncake_early_send_wait_event.py")
            return 1

    path = srt / "disaggregation" / "prefill.py"
    src = path.read_text()
    if _MARKER in src:
        print(f"{_TAG} already present — skipping")
        return 0
    if src.count(_OLD) != 1:
        print(f"{_TAG} anchor found {src.count(_OLD)}x, want 1x in {path} — nothing written")
        return 1
    if "\nimport torch\n" not in src:
        print(f"{_TAG} prefill.py no longer imports torch at module level — nothing written")
        return 1
    path.write_text(src.replace(_OLD, _NEW, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"{_TAG} non-final chunk send now records a forward_stream completion event")
    return 0


if __name__ == "__main__":
    sys.exit(main())
