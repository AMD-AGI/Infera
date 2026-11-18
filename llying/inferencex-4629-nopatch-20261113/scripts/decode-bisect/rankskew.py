import gzip, json, glob, sys, collections, re
for d in sys.argv[1:]:
    print("==", d)
    for f in sorted(glob.glob(f"{d}/profile/*.json.gz")):
        ev = [e for e in json.load(gzip.open(f, "rt"))["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
        comm = sum(e["dur"] for e in ev if "nccl" in e["name"].lower())
        tk = [e["dur"] for e in ev if "topk_main" in e["name"] or "topk_mode" in e["name"]]
        tk.sort()
        sm = sum(e["dur"] for e in ev if "sparse_mla_fwd_split_dim" in e["name"])
        nsm = sum(1 for e in ev if "sparse_mla_fwd_split_dim" in e["name"])
        busy = sum(e["dur"] for e in ev)
        rank = re.search(r"TP-(\d)", f).group(1)
        print(f"  rank {rank}: busy {busy/1e3:7.1f} ms  comm {comm/1e3:6.1f} ms  non-comm {(busy-comm)/1e3:7.1f} ms  sparse_mla calls {nsm} ({sm/1e3:.1f} ms)  topk n={len(tk)} med={tk[len(tk)//2] if tk else 0:.1f} p90={tk[len(tk)*9//10] if tk else 0:.1f} max={tk[-1] if tk else 0:.1f} us")
