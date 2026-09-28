#!/usr/bin/env python3
"""Where the decode scheduler's main thread spends its time, from py-spy raw output.

Input: raw collapsed stacks from `py-spy record --idle --threads --format raw`
(one line per stack: "thread (0x..): NAME;frame;frame count").
Usage: pyspy_summary.py [--top N] [--focus REGEX] RAW [RAW ...]
Prints, over all given files (MainThread only): inclusive share of each function
matching --focus, the top-N functions by inclusive share, and the top-N leaves.
"""
import argparse
import collections
import re


def frames(stack):
    return [re.sub(r" \(.*\)$", "", f) for f in stack.split(";")[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--focus", default=(
        r"check_hicache_events|_sync_hicache_ready_counts|_all_reduce|all_reduce|writing_check|"
        r"loading_check|write_backup|evict_host|evict|_evict_write_through|init_load_back|"
        r"_process_hicache_local_restores|_start_hicache_prefetch|process_decode_queue|"
        r"run_batch|process_batch_result|recv_requests|synchronize|pop_preallocated|"
        r"pop_transferred|poll_and_all_reduce|cache_finished_req|cache_unfinished_req|match_prefix"))
    a = ap.parse_args()
    incl, leaf = collections.Counter(), collections.Counter()
    total = 0
    for path in a.raw:
        for line in open(path, errors="replace"):
            stack, _, n = line.rstrip("\n").rpartition(" ")
            if not stack.startswith("thread") or "MainThread" not in stack.split(";")[0]:
                continue
            n = int(n)
            fs = frames(stack)
            total += n
            for f in set(fs):
                incl[f] += n
            if fs:
                leaf[fs[-1]] += n
    if not total:
        print("no MainThread samples")
        return
    pct = lambda v: 100 * v / total
    print(f"MainThread samples: {total} over {len(a.raw)} files")
    focus = re.compile(a.focus)
    print("focus functions (inclusive %):")
    for f, v in sorted(((f, v) for f, v in incl.items() if focus.search(f)), key=lambda x: -x[1]):
        print(f"  {pct(v):6.2f}%  {f}")
    print(f"top {a.top} inclusive:")
    for f, v in incl.most_common(a.top):
        print(f"  {pct(v):6.2f}%  {f}")
    print(f"top {a.top} leaves (self):")
    for f, v in leaf.most_common(a.top):
        print(f"  {pct(v):6.2f}%  {f}")


if __name__ == "__main__":
    main()
