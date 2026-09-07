#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Make sglang's mooncake PD transport wait on the forward that wrote the KV.

WHAT: with chunked prefill over the mooncake KV transport, every chunk except the
last can be RDMA-read while the forward that writes those pages is still running,
so the decode leg gets half-written KV. It does not crash — prompts longer than one
prefill chunk come back PARTIALLY wrong. On GLM-5.2 DSA a needle read returns the
first digits of the needle and then repeats `</think>`:

    want=2183762  got='2183</think>2183</think>218</think>218</think> the'

The corruption boundary lands exactly on the chunk boundary (a needle in the final
chunk is retrieved correctly), and the same model, same DSA backends, same chunk
size in an aggregated single-node server passes 9/9.

ROOT CAUSE: `prefill.py`'s early-send path already records a completion event as
the barrier (`req.disagg_kv_sender._early_send_wait_event`), but only `mori/conn.py`
ever reads it — `mooncake/conn.py` has no `wait_event` / `synchronize()` anywhere,
so on mooncake the barrier has never taken effect. And the overlap-scheduling path
that moves non-final chunks (`process_batch_result_disagg_prefill`) does not even
record an event. The final chunk is always correct because it goes through the
sampling path, which already has a real `copy_done.synchronize()`.

FIX (three files, all mirroring what `mori` already does):
  disaggregation/common/utils.py   `TransferKVChunk` carries a `wait_event`, so the
                                   barrier travels with the work item.
  disaggregation/mooncake/conn.py  `send()` picks the event up off the sender,
                                   `add_transfer_request()` forwards it, and
                                   `transfer_worker` synchronizes on it BEFORE it
                                   reads device memory.
  disaggregation/prefill.py        the overlap non-final-chunk send, which had no
                                   barrier at all, records one on `forward_stream`.

Not DSA-specific: any PD deployment running chunked prefill over mooncake with
overlap scheduling is affected — DSA's sparse retrieval only makes it conspicuous
("retrieved half the digits") instead of a quiet quality drop.

VERIFIED: 2x 8xMI325X (gfx942), ROCm 7.2.0, sglang v0.5.16, GLM-5.2-FP8 1P1D over
mooncake RDMA, overlap scheduling ON, `chunked-prefill-size 131072` with
`--enable-dp-attention`: needle 5/9 -> 9/9, a 29k depth sweep 4/9 -> 9/9, and the
logs confirm the failing prompt is still really split into 4 chunks afterwards. The
anchors below were re-checked against both supported bases on 2026-09-02:
v0.5.16 has no `Set` in mooncake/conn.py's typing import, while v0.5.18 adds it.
The functional anchors and the defect are otherwise unchanged, so the import
edit accepts exactly those two source shapes.

UPSTREAM: sgl-project/sglang#33970 carries this fix and was still OPEN on
2026-09-02. The closest older report, #25583 (GLM-5-FP8 + NSA + 70k prompt,
identical symptom), was auto-closed with no follow-up; the aggregated-vs-PD A/B
above is what it was missing. The new `synchronize()` blocks the transfer worker,
trading some transfer overlap for correctness. DROP THIS PATCH once the pinned
base waits on the event in mooncake — this script then reports "already present"
and no-ops.

Self-locating and idempotent. All three files or none: a half-patched tree still
corrupts long prompts, so an anchor that is missing or no longer unique writes
NOTHING and fails (exit 1) instead of leaving the image silently broken.
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[mc-wait-event]"

# Path under sglang/srt -> [(anchor, anchor + our addition, expected occurrences)].
# The anchor must occur exactly that many times and every one is replaced; the
# replacement doubles as the already-applied marker.
#
# The three add_transfer_request call sites are anchored on the TAIL of their
# argument list rather than on a named argument: upstream keeps adding arguments
# there (num_kv_tokens landed between v0.5.16 and v0.5.17), and the tail is what
# this patch actually appends to. Verified to occur 1x / 2x on both bases.
_TYPING_IMPORT_VARIANTS: tuple[tuple[str, str], ...] = (
    (
        "from typing import List, Optional, Tuple, Union",
        "from typing import Any, List, Optional, Tuple, Union",
    ),
    (
        "from typing import List, Optional, Set, Tuple, Union",
        "from typing import Any, List, Optional, Set, Tuple, Union",
    ),
)

_EDITS: dict[str, list[tuple[str, str, int]]] = {
    "disaggregation/common/utils.py": [
        (
            "from typing import List, Optional, Tuple, Union",
            "from typing import Any, List, Optional, Tuple, Union",
            1,
        ),
        (
            """    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )
""",
            """    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )
    # Completion event for the forward that wrote these pages. The transfer
    # worker reads device memory outside the CUDA stream, so it must wait on
    # this before the RDMA read or it can observe half-written KV.
    wait_event: Optional[Any] = None
""",
            1,
        ),
    ],
    "disaggregation/mooncake/conn.py": [
        (
            """                kv_chunk: TransferKVChunk = queue.get()
""",
            """                kv_chunk: TransferKVChunk = queue.get()
                if kv_chunk.wait_event is not None:
                    # The RDMA read bypasses the CUDA stream, so without this the
                    # read can race the forward still writing these pages.
                    kv_chunk.wait_event.synchronize()
                    kv_chunk.wait_event = None
""",
            1,
        ),
        (
            """        trace_ctx: Optional[Union[TraceReqContext, TraceNullContext]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
""",
            """        trace_ctx: Optional[Union[TraceReqContext, TraceNullContext]] = None,
        wait_event: Optional[Any] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
""",
            1,
        ),
        (
            """                trace_ctx=trace_ctx,
            )
""",
            """                trace_ctx=trace_ctx,
                wait_event=wait_event,
            )
""",
            1,
        ),
        (
            """        if should_skip:
            return

        if not is_last_chunk:
""",
            """        if should_skip:
            return

        # Pages handed over before their forward is known to be complete carry a
        # completion event; the transfer worker waits on it before reading.
        wait_event = getattr(self, "_early_send_wait_event", None)
        self._early_send_wait_event = None

        if not is_last_chunk:
""",
            1,
        ),
        (
            """                trace_ctx=self.trace_ctx.copy_for_thread(),
            )
""",
            """                trace_ctx=self.trace_ctx.copy_for_thread(),
                wait_event=wait_event,
            )
""",
            2,
        ),
    ],
    "disaggregation/prefill.py": [
        (
            """                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
""",
            """                    ), f"Req {req.rid} does not have metadata buffer allocated"
                    # Non-final chunks are handed over while later chunks are
                    # already running on forward_stream, and the transfer worker
                    # reads device memory outside the CUDA stream. Gate the read
                    # on this chunk's writes completing.
                    ev = torch.cuda.Event()
                    ev.record(self.forward_stream)
                    req.disagg_kv_sender._early_send_wait_event = ev
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
""",
            1,
        ),
    ],
}


def _srt_dir():
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent / "srt"
    return d if d.is_dir() else None


def main():
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0

    edited: list[tuple[Path, str]] = []
    for rel, edits in _EDITS.items():
        f = srt / rel
        if not f.is_file():
            print(f"{_TAG} {f} is missing — sglang layout changed, re-anchor the patch")
            return 1
        src = out = f.read_text()
        if rel == "disaggregation/mooncake/conn.py":
            if not any(new in out for _, new in _TYPING_IMPORT_VARIANTS):
                matches = [
                    (old, new) for old, new in _TYPING_IMPORT_VARIANTS if out.count(old) == 1
                ]
                if len(matches) != 1:
                    counts = ", ".join(
                        f"{old!r}={out.count(old)}" for old, _ in _TYPING_IMPORT_VARIANTS
                    )
                    print(
                        f"{_TAG} expected exactly one supported typing import "
                        f"in {rel}; found {len(matches)} ({counts})"
                    )
                    print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
                    return 1
                old, new = matches[0]
                out = out.replace(old, new, 1)
        for old, new, want in edits:
            if new in out:
                continue  # this edit is already in the tree
            found = out.count(old)
            if found != want:
                print(
                    f"{_TAG} anchor found {found}x, want {want}x in {rel}: "
                    f"{old.strip().splitlines()[0]!r}"
                )
                print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
                return 1
            out = out.replace(old, new)
        if out != src:
            edited.append((f, out))

    if not edited:
        print(f"{_TAG} already present — skipping")
        return 0
    for f, out in edited:
        f.write_text(out)
        print(f"{_TAG} patched {f}")
    print(f"{_TAG} mooncake now waits on the prefill write event before the RDMA read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
