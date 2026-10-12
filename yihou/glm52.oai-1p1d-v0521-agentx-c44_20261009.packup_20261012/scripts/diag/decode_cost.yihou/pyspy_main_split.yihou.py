# Summarize py-spy raw (collapsed) records: main-thread time split (sync-wait vs other), top leaf frames.
# usage: pyspy_main_split.yihou.py <raw.txt> [...]
import collections, re, sys

for fn in sys.argv[1:]:
    tot = 0; cats = collections.Counter(); leaf = collections.Counter()
    for line in open(fn, errors="replace"):
        line = line.rstrip("\n")
        if not line or " " not in line:
            continue
        stack, _, cnt = line.rpartition(" ")
        try:
            n = int(cnt)
        except ValueError:
            continue
        if "run_scheduler_process" not in stack:  # main thread only
            continue
        tot += n
        frames = stack.split(";")
        s = stack
        if "synchronize (torch/cuda/streams.py" in s or "synchronize (torch/cuda/" in s:
            c = "sync:" + next((f for f in reversed(frames) if "overlap_utils" in f or "scheduler.py" in f or "decode.py" in f), "?")
        elif "all_gather" in s or "mlp_sync" in s or "prepare_mlp_sync_batch" in s:
            c = "dp_sync(gather)"
        elif "process_batch_result" in s:
            c = "process_batch_result"
        elif "process_decode_queue" in s or "pop_transferred" in s or "pop_preallocated" in s:
            c = "disagg_queues"
        elif "run_batch" in s or "forward_batch_generation" in s:
            c = "run_batch(launch)"
        elif "recv_requests" in s or "ingest_requests" in s:
            c = "recv"
        else:
            c = "other"
        cats[c] += n
        leaf[frames[-1]] += n
    print(f"== {fn}  main-thread samples={tot}")
    for c, n in cats.most_common(12):
        print(f"  {100*n/max(tot,1):5.1f}%  {c}")
    print("  top leaf frames:")
    for f, n in leaf.most_common(10):
        print(f"    {100*n/max(tot,1):5.1f}%  {f}")
