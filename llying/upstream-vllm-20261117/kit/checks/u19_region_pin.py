#!/usr/bin/env python3
"""U19 measurement: 4 worker processes (one GPU each) join one SharedOffloadRegion laid out like GLM-5.2 PP4 under
TieringOffloadingSpec (row = 4 slots of 796,416 B, page-rounded to 3,186,688 B) and pin it with vLLM's own
pin_mmap_region(worker_slot_only=True):
  full   the current rule: each row saves 2.39 MB < 3 MiB by pinning only the worker's slot, so every worker registers
         the whole region (4 x the region in total); on the P4D4 node two of four 1.25 TB registrations failed
  slots  the same with MIN_HOST_REGISTER_SAVED_BYTES = 0: each worker registers only its own slots, one call per row
Prints one JSON line per worker (pinned?, registrations, seconds) and the host's Shmem before / after.
  u19_region_pin.py GB MODE
"""

import json
import multiprocessing as mp
import sys
import time

GB = float(sys.argv[1])
MODE = sys.argv[2]
WORKERS, SLOT = 4, 796416
ROW = -(-WORKERS * SLOT // 4096) * 4096
NUM_CHUNKS = int(GB * 1e9) // ROW


def shmem_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("Shmem:"):
            return round(int(line.split()[1]) / 1048576, 1)
    return -1.0


def worker(rank: int, barrier, results) -> None:
    import torch

    torch.cuda.set_device(rank)
    torch.cuda.init()
    from vllm.v1.kv_offload.cpu import gpu_worker as gw
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    if MODE == "slots":
        gw.MIN_HOST_REGISTER_SAVED_BYTES = 0
    region = SharedOffloadRegion(
        engine_id=f"u19probe-{MODE}",
        num_chunks=NUM_CHUNKS,
        rank=rank,
        kv_bytes_per_chunk=ROW,
        cpu_page_size=SLOT,
        barrier=barrier.wait,
    )
    t0 = time.monotonic()
    gw.pin_mmap_region(region, worker_slot_only=True)
    seconds = time.monotonic() - t0
    results.put(
        {
            "mode": MODE,
            "rank": rank,
            "pinned": region.is_pinned,
            "registrations": len(region.pinned_addresses),
            "seconds": round(seconds, 1),
            "region_gb": round(region.total_size_bytes / 1e9, 1),
        }
    )
    barrier.wait()
    region.cleanup()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    barrier = mp.Barrier(WORKERS)
    results = mp.Queue()
    before = shmem_gib()
    procs = [mp.Process(target=worker, args=(r, barrier, results)) for r in range(WORKERS)]
    for p in procs:
        p.start()
    rows = [results.get(timeout=3600) for _ in procs]
    for p in procs:
        p.join(600)
    for r in sorted(rows, key=lambda r: r["rank"]):
        print(json.dumps(r), flush=True)
    print(json.dumps({"mode": MODE, "row_bytes": ROW, "slot_bytes": SLOT, "num_chunks": NUM_CHUNKS,
                      "shmem_gib_before": before, "shmem_gib_after": shmem_gib(),
                      "all_pinned": all(r["pinned"] for r in rows)}), flush=True)
