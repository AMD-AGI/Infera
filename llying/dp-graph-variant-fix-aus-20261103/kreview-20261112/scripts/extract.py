#!/usr/bin/env python3
"""Reduce torch-profiler decode traces (one .trace.json.gz per DP rank) to compact pickles with every GPU kernel
event kept (name id, absolute ts, dur) and tied to its hipGraphLaunch + enclosing CPU annotation.

Nothing is aggregated or dropped here so that later analysis never has to reopen the raw traces.
Usage: extract.py OUT_DIR TRACE_ROOT   (walks TRACE_ROOT for *.trace.json.gz; output mirrors relative paths)
"""
import gzip, json, os, pickle, re, sys
import multiprocessing as mp

KEEP_RT = {"hipGraphLaunch", "hipEventSynchronize", "hipStreamWaitEvent", "hipDeviceSynchronize",
           "hipMemcpyWithStream", "hipStreamSynchronize", "hipEventQuery"}


def one(job):
    src, dst = job
    try:
        d = json.load(gzip.open(src))
    except Exception as e:  # truncated / corrupt
        return src, f"ERR {e!r}"
    ev = d["traceEvents"]
    pname = next((e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"
                  and "scheduler" in str(e.get("args", {}).get("name", ""))), None)
    names, nid = [], {}

    def idx(n):
        i = nid.get(n)
        if i is None:
            i = nid[n] = len(names)
            names.append(n)
        return i

    ann = sorted(((e["name"], e["ts"], e["dur"]) for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                  and not e["name"].startswith("gloo")), key=lambda x: x[1])
    gloo = sorted(((e["name"], e["ts"], e["dur"]) for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                   and e["name"].startswith("gloo")), key=lambda x: x[1])
    gann = sorted(((e["name"], e["ts"], e["dur"]) for e in ev if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation"),
                  key=lambda x: x[1])
    rt = sorted(((e["name"], e["ts"], e["dur"], e.get("args", {}).get("correlation")) for e in ev
                 if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver") and e["name"] in KEEP_RT),
                key=lambda x: x[1])
    rt_counts = {}
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = rt_counts.setdefault(e["name"], [0, 0.0])
            c[0] += 1
            c[1] += e["dur"]
    graph_corr = {r[3] for r in rt if r[0] == "hipGraphLaunch"}
    kern = {}   # corr -> [(name_id, ts, dur, stream)]
    eager = []
    nk = 0
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        nk += 1
        a = e["args"]
        c = a.get("correlation")
        rec = (idx(e["name"]), e["ts"], e["dur"], a.get("stream"))
        if c in graph_corr:
            kern.setdefault(c, []).append(rec)
        else:
            eager.append(rec + (a.get("grid"), a.get("block")))
    for v in kern.values():
        v.sort(key=lambda x: x[1])
    eager.sort(key=lambda x: x[1])
    mem = sorted(((e["name"], e["ts"], e["dur"], e["args"].get("bytes"), e["args"].get("correlation"), e["args"].get("stream"))
                  for e in ev if e.get("ph") == "X" and e.get("cat") in ("gpu_memcpy", "gpu_memset")), key=lambda x: x[1])
    out = {"src": src, "pname": pname, "base_ns": d.get("baseTimeNanoseconds"), "n_kernel_events": nk,
           "names": names, "ann": ann, "gloo": gloo, "gann": gann, "rt": rt, "rt_counts": rt_counts,
           "graph_kernels": kern, "eager": eager, "mem": mem,
           "meta": {k: v for k, v in d.items() if k not in ("traceEvents", "deviceProperties")}}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with gzip.open(dst + ".tmp", "wb", compresslevel=4) as f:
        pickle.dump(out, f, protocol=4)
    os.replace(dst + ".tmp", dst)
    return src, f"ok kernels={nk} launches={len(kern)} eager={len(eager)}"


def main():
    out_dir, root = sys.argv[1], sys.argv[2]
    jobs = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".trace.json.gz"):
                src = os.path.join(dp, f)
                dst = os.path.join(out_dir, os.path.relpath(src, root)).replace(".trace.json.gz", ".pkl.gz")
                if not os.path.exists(dst):
                    jobs.append((src, dst))
    print("jobs", len(jobs), flush=True)
    with mp.Pool(int(os.environ.get("NPROC", "64"))) as p:
        for src, msg in p.imap_unordered(one, jobs):
            print(msg, os.path.relpath(src, root), flush=True)


if __name__ == "__main__":
    main()
