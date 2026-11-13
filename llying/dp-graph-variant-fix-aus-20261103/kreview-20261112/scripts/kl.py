"""Loader for the compact pickles written by extract.py: graph launches in time order, each tagged with the CPU
annotation that encloses the hipGraphLaunch call (verify bs / draft / draft_extend)."""
import bisect, glob, gzip, os, pickle, re

STEP = re.compile(r"step\[(\w+) bs=(\d+)\]")


def short(n, w=100):
    n = re.sub(r"^void ", "", n)
    n = re.sub(r"^_ZN5aiter(\d+)", "aiter::", n)
    return n if len(n) <= w else n[:w] + "~"


def load(path):
    d = pickle.load(gzip.open(path))
    names = d["names"]
    ann = [a for a in d["ann"] if a[0].startswith("step[") or a[0] in ("draft", "draft_extend")]
    starts = [a[1] for a in ann]
    launches = []
    for name, ts, dur, corr in d["rt"]:
        if name != "hipGraphLaunch":
            continue
        ks = d["graph_kernels"].get(corr)
        kind, bs, a_ts, a_dur = "unknown", None, None, None
        i = bisect.bisect_right(starts, ts) - 1
        # innermost enclosing annotation: scan back a few (annotations do not nest among these three kinds)
        while i >= 0 and i > bisect.bisect_right(starts, ts) - 6:
            a = ann[i]
            if a[1] <= ts and ts + dur <= a[1] + a[2] + 1:
                m = STEP.match(a[0])
                kind, bs = (m.group(1).lower(), int(m.group(2))) if m else (a[0], None)
                a_ts, a_dur = a[1], a[2]
                break
            i -= 1
        launches.append({"corr": corr, "cpu_ts": ts, "cpu_dur": dur, "kind": kind, "bs": bs, "ann_ts": a_ts,
                         "ann_dur": a_dur, "k": ks or []})
    launches.sort(key=lambda l: l["cpu_ts"])
    d["launches"] = launches
    m = re.search(r"DP(\d+)_TP(\d+)", d.get("pname") or "")
    d["dp"] = int(m.group(1)) if m else None
    return d


def rank_files(set_dir):
    """Compact files of one capture (one per DP rank), ordered by DP rank; nested duplicate dirs are ignored."""
    fs = sorted(glob.glob(os.path.join(set_dir, "*.pkl.gz")), key=lambda p: int(re.search(r"DP-(\d+)", p).group(1)))
    return fs


def captures(root):
    """All capture dirs under root that directly hold per-rank pickles, skipping a dir nested in a same-named dir."""
    out = []
    for dp, _, fs in os.walk(root):
        if any(f.endswith(".pkl.gz") for f in fs):
            if os.path.basename(os.path.dirname(dp)) == os.path.basename(dp):
                continue
            out.append(dp)
    return sorted(out)
