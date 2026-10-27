"""Where a prefill PP rank's scheduler thread spends its time between forward launches, from a torch-profiler trace:
  - blocking HIP runtime calls (hipEventSynchronize, hipStreamSynchronize, hipMemcpy*, ...) on the scheduler thread,
    totalled by their innermost sglang frames,
  - wall time of selected scheduler-loop phases (get_new_batch_prefill, eviction, load-back, PP send / recv, result
    processing, forward launch), summed over the profiled steps.
  python3 prefill_sched_cpu.py TRACE.json.gz [--frames 4]"""

import argparse
import bisect
import collections
import gzip
import json
import re

PHASES = [
    ("forward launch (run_batch)", r"scheduler\.py\(\d+\): run_batch$"),
    ("get_new_batch_prefill", r"scheduler\.py\(\d+\): get_new_batch_prefill$"),
    ("  evict_for_alloc", r"unified_radix_cache\.py\(\d+\): evict_for_alloc$"),
    ("  writing_check", r"unified_radix_cache\.py\(\d+\): writing_check$"),
    ("  init_load_back", r"unified_radix_cache\.py\(\d+\): init_load_back$"),
    ("  prepare_for_extend", r"schedule_batch\.py\(\d+\): prepare_for_extend$"),
    ("PP recv_tensor_dict", r"parallel_state\.py\(\d+\): recv_tensor_dict$"),
    ("PP send_tensor_dict", r"parallel_state\.py\(\d+\): send_tensor_dict$"),
    ("_pp_commit_comm_work", r"scheduler_pp_mixin\.py\(\d+\): _pp_commit_comm_work$"),
    ("process_batch_result", r"scheduler\.py\(\d+\): process_batch_result$"),
    ("check_hicache_events", r"unified_radix_cache\.py\(\d+\): check_hicache_events$"),
    ("kv_to_page_indices", r"mem_cache/common\.py\(\d+\): kv_to_page_indices$"),
]

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--frames", type=int, default=4)
a = ap.parse_args()
ev = json.load(gzip.open(a.trace, "rt") if a.trace.endswith(".gz") else open(a.trace))["traceEvents"]
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")), key=lambda e: e["ts"])
tid = steps[0]["tid"]
t0, t1 = steps[0]["ts"], steps[-1]["ts"] + steps[-1]["dur"]
py = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "python_function" and e.get("tid") == tid), key=lambda e: e["ts"])
py_ts = [e["ts"] for e in py]


def frames_at(ts, n):
    i = bisect.bisect_right(py_ts, ts)
    enc = sorted((e for e in py[max(0, i - 3000):i] if e["ts"] + e["dur"] >= ts), key=lambda e: e["ts"])
    out = []
    for e in enc:
        m = re.match(r"(.*?)\((\d+)\): (.*)", e["name"])
        if m and "sglang/" in m.group(1):
            out.append(f"{re.sub(r'.*sglang/(srt/)?', '', m.group(1))}:{m.group(2)} {m.group(3)}")
    return " > ".join(out[-n:])


span = (t1 - t0) / 1e3
print(f"== {a.trace}\nscheduler thread {tid}, {len(steps)} steps, {span:.0f} ms from first step start to last step end")
blk = collections.Counter()
cnt = collections.Counter()
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and e.get("tid") == tid and t0 <= e["ts"] <= t1:
        if re.search(r"Synchronize|hipMemcpy(WithStream)?$|hipMemcpyDtoH|hipMemcpyHtoD|hipFree|hipMalloc|hipHostMalloc|hipHostRegister", e["name"]) and e["dur"] > 0.2e3:
            key = f"{e['name']} @ {frames_at(e['ts'], a.frames)}"
            blk[key] += e["dur"]
            cnt[key] += 1
tot = sum(blk.values())
print(f"-- blocking runtime calls > 0.2 ms: {tot / 1e3:.0f} ms ({100 * tot / 1e3 / span:.0f}% of the span)")
for k, v in blk.most_common(15):
    print(f"   {v / 1e3:8.1f} ms x{cnt[k]:<4d} {k[:260]}")
print("-- scheduler-loop phases (wall, summed; nested phases are indented and included in their parent)")
rx = [(n, re.compile(r)) for n, r in PHASES]
ph = collections.Counter()
phn = collections.Counter()
for e in py:
    if t0 <= e["ts"] <= t1:
        for n, r in rx:
            if r.search(e["name"]):
                ph[n] += e["dur"]
                phn[n] += 1
for n, _ in PHASES:
    if phn[n]:
        print(f"   {ph[n] / 1e3:8.1f} ms x{phn[n]:<4d} ({100 * ph[n] / 1e3 / span:4.1f}%)  {n}")
