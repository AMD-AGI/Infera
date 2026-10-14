#!/usr/bin/env python3
"""TTFT stage breakdown and per-process CPU for one window, from sampler.py output.

Usage: sample_window.py SAMPLES_JSONL START_ISO END_ISO
Takes the first sample at/after START and the last at/before END and reports, from the deltas:
  prefill (all instances pooled): mean queue time, bootstrap wait, KV transfer time and size,
  arrival-to-done latency; decode: mean KV alloc wait, queue time, TTFT, ITL, e2e;
  CPU % per process (main = HTTP + TokenizerManager, detokenizer, data-parallel controller,
  schedulers mean / max) per engine container.
"""

import collections
import json
import sys


def load(path, start, end):
    first = last = None
    for line in open(path):
        rec = json.loads(line)
        if rec["ts"] >= start and first is None:
            first = rec
        if rec["ts"] <= end:
            last = rec
    return first, last


def kind(comm):
    if comm.startswith("sglang::schedul"):
        return "scheduler"
    if comm.startswith("sglang::detoken"):
        return "detokenizer"
    if comm.startswith("sglang::data_pa"):
        return "dp_controller"
    if comm.startswith("sglang::tokeniz"):
        return "tokenizer_worker"
    if comm.startswith("python3"):
        return "main"
    return None


def main():
    path, start, end = sys.argv[1:4]
    a, b = load(path, start, end)
    if not a or not b or a["ts"] >= b["ts"]:
        print("not enough samples in the window")
        return 1
    print(f"samples {a['ts']} .. {b['ts']}")
    pooled = {"prefill": collections.Counter(), "decode": collections.Counter()}
    for inst, eb in b["engines"].items():
        ea = a["engines"].get(inst, {})
        for k, v in eb.items():
            if k != "role" and isinstance(v, float) and k in ea:
                pooled[eb["role"]][k] += v - ea[k]

    def mean(role, h, scale=1.0):
        c = pooled[role].get(h + "_count", 0)
        return (pooled[role].get(h + "_sum", 0) / c * scale, int(c)) if c else (float("nan"), 0)

    p = pooled
    q, n = mean("prefill", "queue_time_seconds")
    print(f"prefill  n={n}: queue {q:.2f} s, bootstrap {mean('prefill', 'kv_transfer_bootstrap_ms', 1e-3)[0]:.2f} s,"
          f" transfer {mean('prefill', 'kv_transfer_latency_ms', 1e-3)[0]:.2f} s"
          f" ({mean('prefill', 'kv_transfer_total_mb', 1e-3)[0]:.2f} GB), arrival->done"
          f" {mean('prefill', 'e2e_request_latency_seconds')[0]:.2f} s")
    t, n = mean("decode", "time_to_first_token_seconds")
    print(f"decode   n={n}: KV alloc wait {mean('decode', 'kv_transfer_alloc_ms', 1e-3)[0]:.2f} s,"
          f" queue {mean('decode', 'queue_time_seconds')[0]:.2f} s, TTFT {t:.2f} s,"
          f" ITL {mean('decode', 'inter_token_latency_seconds', 1e3)[0]:.1f} ms,"
          f" e2e {mean('decode', 'e2e_request_latency_seconds')[0]:.2f} s")
    seconds = (__import__("datetime").datetime.fromisoformat(b["ts"]) -
               __import__("datetime").datetime.fromisoformat(a["ts"])).total_seconds()
    print("CPU % over the window (100 = one core):")
    for container, procs in sorted(b["cpu"].items()):
        before = {pid: t for pid, _, t in a["cpu"].get(container, [])}
        groups = collections.defaultdict(list)
        for pid, comm, t in procs:
            k = kind(comm)
            if k and pid in before:
                groups[k].append(100 * (t - before[pid]) / seconds)
        cells = []
        if groups["main"]:
            m = groups["main"]
            cells.append(f"main(HTTP/tokenizer) max {max(m):.0f} / sum {sum(m):.0f} (x{len(m)})")
        if groups["tokenizer_worker"]:
            t = groups["tokenizer_worker"]
            cells.append(f"tokenizer workers max {max(t):.0f} / sum {sum(t):.0f} (x{len(t)})")
        for k in ("detokenizer", "dp_controller"):
            if groups[k]:
                cells.append(f"{k} {max(groups[k]):.0f}")
        if groups["scheduler"]:
            s = groups["scheduler"]
            cells.append(f"scheduler mean {sum(s) / len(s):.0f} / max {max(s):.0f} (x{len(s)})")
        print(f"  {container.split('-r2-')[-1]:10} " + ", ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
