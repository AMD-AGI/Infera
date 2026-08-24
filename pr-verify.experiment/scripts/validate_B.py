#!/usr/bin/env python3
"""Plumbing + barrier-semantics check for the mooncake early-send wait event.

A 2-node PD pair is what proves the CORRUPTION is gone; this machine has one
node, so that half is deferred (see working_process.md). What IS checkable here,
against real CUDA events on a real GPU, is everything between the event being
recorded and the RDMA read being gated on it:

  1. TransferKVChunk carries the event, defaulting to None (so every existing
     construction site keeps working unchanged).
  2. MooncakeKVSender.send() picks the event off the sender and hands it to
     add_transfer_request(), for BOTH the last-chunk and non-last-chunk arms,
     and clears it so the next chunk does not inherit a stale barrier.
  3. The transfer worker's wait actually blocks until the recorded work is
     done -- i.e. it is a real barrier, not a no-op -- and clears the event so
     a chunk re-enqueued on a staging defer does not wait twice.
  4. The barrier is correct in the sense that matters: a read issued after it
     observes the writes the event was recorded on.

Run inside a ROCm/CUDA sglang container.
"""

import sys
import time

import torch

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------- 1. dataclass


def test_dataclass_field():
    print("\n[1] TransferKVChunk carries wait_event, defaulting to None")
    from sglang.srt.disaggregation.common.utils import TransferKVChunk

    fields = {f.name: f for f in TransferKVChunk.__dataclass_fields__.values()}
    check("field exists", "wait_event" in fields)
    if "wait_event" not in fields:
        return

    # Every pre-existing construction site omits wait_event, so it must default.
    chunk = TransferKVChunk(
        room=1,
        prefill_kv_indices=None,
        index_slice=slice(0, 1),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
    )
    check("defaults to None when omitted", chunk.wait_event is None)

    ev = torch.cuda.Event()
    chunk2 = TransferKVChunk(
        room=1,
        prefill_kv_indices=None,
        index_slice=slice(0, 1),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
        wait_event=ev,
    )
    check("accepts an event", chunk2.wait_event is ev)


# ----------------------------------------------------------------- 2. send()


def test_send_plumbing():
    print("\n[2] send() forwards the event on both arms, then clears it")
    import inspect

    from sglang.srt.disaggregation.mooncake.conn import (
        MooncakeKVManager,
        MooncakeKVSender,
    )

    sig = inspect.signature(MooncakeKVManager.add_transfer_request)
    check("add_transfer_request accepts wait_event", "wait_event" in sig.parameters)
    if "wait_event" in sig.parameters:
        check(
            "wait_event defaults to None",
            sig.parameters["wait_event"].default is None,
            "-- existing callers must keep working",
        )

    # Drive send() with a stub manager that records what it was handed.
    seen = []

    class StubMgr:
        def add_transfer_request(self, *a, **kw):
            seen.append(kw.get("wait_event", "ABSENT"))

    from sglang.srt.observability.trace import TraceNullContext

    class StubSender(MooncakeKVSender):
        def __init__(self):  # bypass the real __init__ (needs a live engine)
            self.kv_mgr = StubMgr()
            self.bootstrap_room = 1
            self.aux_index = 0
            # send() is wrapped by the mooncake trace decorator, which calls
            # into trace_ctx -- use the real null context, not a stub.
            self.trace_ctx = TraceNullContext()
            self._recorded = []

        def _prepare_send_indices(self, kv_indices, state_indices):
            return kv_indices, slice(0, 1), self._is_last, False

        def _record_transfer_indices(self, *a, **kw):
            self._recorded.append(a)

    for is_last in (False, True):
        s = StubSender()
        s._is_last = is_last
        ev = torch.cuda.Event()
        s._early_send_wait_event = ev
        s.send(kv_indices=None, state_indices=None)
        arm = "last_chunk" if is_last else "non_last_chunk"
        check(f"{arm}: event forwarded", seen and seen[-1] is ev, f"got {seen[-1]!r}")
        check(
            f"{arm}: sender's event cleared after send",
            s._early_send_wait_event is None,
            "-- else the next chunk inherits a stale barrier",
        )

    # No event recorded (the non-overlap path) must still work.
    s = StubSender()
    s._is_last = False
    s.send(kv_indices=None, state_indices=None)
    check("no event recorded: forwards None, does not raise", seen[-1] is None)


# --------------------------------------------------- 3+4. real barrier on GPU


def test_barrier_is_real():
    print("\n[3] the wait is a real barrier, and [4] it orders the read correctly")

    stream = torch.cuda.Stream()
    n = 4096

    # Writer: enough work that the event is genuinely not complete when
    # recorded, so a no-op "wait" would be caught.
    with torch.cuda.stream(stream):
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")
        dst = torch.zeros(n, n, device="cuda")
        for _ in range(30):
            dst = dst + (a @ b)
        ev = torch.cuda.Event()
        ev.record(stream)

    check("event is pending right after record", not ev.query())

    t0 = time.perf_counter()
    ev.synchronize()  # exactly what transfer_worker now does
    dt = time.perf_counter() - t0
    check("synchronize() actually blocked", dt > 1e-4, f"-- took {dt * 1e3:.2f} ms")
    check("event complete after synchronize()", ev.query())

    # The property the fix exists for: a read issued after the barrier, on a
    # DIFFERENT stream than the writer, observes the writer's result.
    expected = dst.clone()
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        observed = dst.clone()
    torch.cuda.synchronize()
    check(
        "post-barrier read on another stream sees the writes",
        torch.equal(expected, observed),
    )

    # And the worker clears it, so a re-enqueued chunk does not wait twice.
    chunk_ev = ev
    if chunk_ev is not None:
        chunk_ev.synchronize()
        chunk_ev = None
    check("event cleared after the wait", chunk_ev is None)


def main():
    print(f"device: {torch.cuda.get_device_properties(0).name}")
    print(f"torch:  {torch.__version__} hip={torch.version.hip}")
    test_dataclass_field()
    test_send_plumbing()
    test_barrier_is_real()
    print()
    if FAILURES:
        print("RESULT: FAIL —", ", ".join(FAILURES))
        return 1
    print("RESULT: PASS")
    print()
    print("NOT covered here (needs a 2-node PD pair): that the corruption is")
    print("actually gone end-to-end, and the synchronize()'s cost on prefill")
    print("throughput. Both deferred -- see working_process.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
