"""Maps the GPU time of a torch-profiler trace (one PP rank of a prefill) to model code: each kernel is joined to
its launch (correlation id), the launch to the Python frames and the operator (cpu_op, with input shapes) that
enclose it on the launching thread. NCCL kernels (PP send / recv, mostly waiting for the peer stage) and copies on
side streams are reported apart and left out of the compute time.

Prints per rank: steps (step[EXTEND ...] annotations), compute time per step, and the compute time by
  - module: a coarse class from the frames (MoE, DSA indexer, sparse MLA core, attention projections, ...),
  - model frame: the innermost frame in sglang/srt/models/ (file:line function),
  - leaf frame: the innermost frame under sglang/ or aiter/,
  - kernel name,
  - operator + input shapes for GEMM-like operators (aten::mm / linear, aiter gemm, fused_moe, ...).
  python3 prefill_trace_map.py TRACE.json.gz ... [--top 30] [--json OUT.json]"""

import argparse
import bisect
import collections
import gzip
import json
import re

MODULES = [
    ("moe", r"layers/moe/|fused_moe|moe_|_moe|experts|topk_softmax|grouped_topk|moe_sorting|mxfp4_moe"),
    ("dsa_indexer", r"nsa_indexer|indexer|fp8_mqa|mqa_logits|topk_transform|topk_packed"),
    ("sparse_mla_core", r"nsa_backend|nsa/|sparse_mla|forward_extend|attn_backend|radix_attention|forward_core"),
    ("rope_kv_cache", r"rope|rotary|set_kv_buffer|cache_mla|kv_cache"),
    ("norm", r"layernorm|rmsnorm|rms_norm|add_rmsnorm"),
    ("linear", r"layers/linear\.py|quantization/|gemm|linear"),
    ("logits_sampling", r"logits_processor|sampler|lm_head"),
    ("hicache_kv_io", r"hicache|memory_pool_host|cache_controller|move_indices"),
    ("scheduler_io", r"managers/|model_executor/forward_batch|resolve_forward_inputs|prepare_|_prefix_kv_indices"),
]
MODULE_RX = [(n, re.compile(rx, re.I)) for n, rx in MODULES]
GEMM_OP = re.compile(r"aten::(mm|addmm|bmm|linear|matmul)$|gemm|fused_moe|moe|a4w4|a8w8|a16w16", re.I)


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)["traceEvents"]


def frame_file(name):
    m = re.match(r"(.*?)\((\d+)\): (.*)", name)
    return (m.group(1), m.group(2), m.group(3)) if m else (name, "", "")


def short(path):
    for key in ("sglang/srt/", "aiter/", "sglang/"):
        i = path.find(key)
        if i >= 0:
            return path[i:]
    return path


class Stacks:
    """Enclosing intervals (well nested per thread) of sorted X events, queried at increasing times."""

    def __init__(self, events):
        self.by_tid = collections.defaultdict(list)
        for e in events:
            self.by_tid[e["tid"]].append(e)
        for v in self.by_tid.values():
            v.sort(key=lambda e: (e["ts"], -e["dur"]))

    def query_all(self, queries):
        """queries: list of (tid, ts, key); returns {key: [enclosing events, outer -> inner]}."""
        out = {}
        per_tid = collections.defaultdict(list)
        for tid, ts, key in queries:
            per_tid[tid].append((ts, key))
        for tid, qs in per_tid.items():
            evs, i, stack = self.by_tid.get(tid, []), 0, []
            for ts, key in sorted(qs):
                while i < len(evs) and evs[i]["ts"] <= ts:
                    e = evs[i]
                    while stack and stack[-1]["ts"] + stack[-1]["dur"] < e["ts"]:
                        stack.pop()
                    stack.append(e)
                    i += 1
                while stack and stack[-1]["ts"] + stack[-1]["dur"] < ts:
                    stack.pop()
                out[key] = [e for e in stack if e["ts"] + e["dur"] >= ts]
        return out


def module_of(chain, kernel):
    text = " ".join(chain) + " " + kernel
    for name, rx in MODULE_RX:
        if rx.search(" ".join(chain[-3:]) + " " + kernel):
            return name
    for name, rx in MODULE_RX:
        if rx.search(text):
            return name
    return "other"


def analyze(path, top):
    ev = load(path)
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    launches = {e["args"]["correlation"]: e for e in ev
                if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    py = Stacks(e for e in ev if e.get("ph") == "X" and e.get("cat") == "python_function")
    ops = Stacks(e for e in ev if e.get("ph") == "X" and e.get("cat") == "cpu_op")
    steps = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")]
    queries = []
    for k in gpu:
        l = launches.get(k["args"].get("correlation"))
        if l is not None:
            queries.append((l["tid"], l["ts"], id(k)))
    py_st, op_st = py.query_all(queries), ops.query_all(queries)

    stream_time = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"])
    main_stream = stream_time.most_common(1)[0][0]
    agg = {n: collections.Counter() for n in ("module", "model_frame", "leaf_frame", "kernel", "gemm_shape", "kernel_site", "side_site")}
    cnt = collections.Counter()
    durs = collections.defaultdict(list)
    side, comm, compute = 0.0, 0.0, 0.0
    steps.sort(key=lambda s: s["ts"])
    step_ts = [s["ts"] for s in steps]
    per_step = [collections.Counter() for _ in steps]
    step_compute = [0.0] * len(steps)
    step_side = [collections.Counter() for _ in steps]
    for k in gpu:
        d = k["dur"]
        l = launches.get(k["args"].get("correlation"))
        si = bisect.bisect_right(step_ts, l["ts"]) - 1 if l is not None else -1
        if si >= 0 and l["ts"] > steps[si]["ts"] + steps[si]["dur"]:
            si = -1
        if "nccl" in k["name"].lower() or "rccl" in k["name"].lower():
            comm += d
            if si >= 0:
                step_side[si]["nccl"] += d
            continue
        if k["args"].get("stream") != main_stream:
            side += d
            if si >= 0:
                step_side[si][f"s{k['args'].get('stream')} {k['name'][:24]}"] += d
            fr = [frame_file(e["name"]) for e in py_st.get(id(k), [])]
            ch = [f"{short(f)}:{ln} {fn}" for f, ln, fn in fr if "sglang/" in f or "aiter/" in f]
            ol = op_st.get(id(k), [])
            inner = ol[-1] if ol else {}
            site = (f"s{k['args'].get('stream')} {k['name'][:40]} <- {inner.get('name', '')} "
                    f"{str(inner.get('args', {}).get('Input Dims', ''))[:50]} @ {' > '.join(ch[-2:])}")
            agg["side_site"][site] += d
            cnt[site] += 1
            continue
        compute += d
        frames = [frame_file(e["name"]) for e in py_st.get(id(k), [])]
        chain = [f"{short(f)}:{ln} {fn}" for f, ln, fn in frames if "sglang/" in f or "aiter/" in f]
        model = [c for c in chain if "/models/" in c]
        oplist = op_st.get(id(k), [])
        mod = module_of(chain + [o["name"] for o in oplist], k["name"])
        agg["module"][mod] += d
        agg["model_frame"][model[-1] if model else "(none)"] += d
        agg["leaf_frame"][chain[-1] if chain else "(none)"] += d
        agg["kernel"][k["name"][:120]] += d
        cnt[k["name"][:120]] += 1
        durs[k["name"][:120]].append(d)
        inner = oplist[-1] if oplist else {}
        site = (f"{k['name'][:60]} <- {inner.get('name', '')} {str(inner.get('args', {}).get('Input Dims', ''))[:70]}"
                f" @ {chain[-1] if chain else ''} | {model[-1] if model else ''}")
        agg["kernel_site"][site] += d
        cnt[site] += 1
        if si >= 0:
            step_compute[si] += d
            per_step[si][site] += d
        gop = next((o for o in oplist if GEMM_OP.search(o["name"])), None)
        if gop is not None:
            dims = gop.get("args", {}).get("Input Dims", [])
            agg["gemm_shape"][f"{gop['name']} {dims[:3]} @ {model[-1] if model else ''}"] += d
    n_steps = max(len(steps), 1)
    print(f"\n== {path}")
    print(f"steps {len(steps)}, compute (main stream {main_stream}) {compute / 1e3:.0f} ms = {compute / 1e3 / n_steps:.1f} ms/step; "
          f"NCCL {comm / 1e3:.0f} ms, side streams {side / 1e3:.0f} ms (both excluded)")
    toks = [int(m.group(1)) for s in steps if (m := re.search(r"toks=(\d+)", s["name"]))]
    if toks:
        print(f"tokens per step: mean {sum(toks) / len(toks):.0f}, min {min(toks)}, max {max(toks)}")
    res = {"path": path, "steps": len(steps), "compute_ms": compute / 1e3, "nccl_ms": comm / 1e3, "side_ms": side / 1e3}
    if steps:
        # Kernels launched inside a step annotation; a step far above the median is a stall (the extra time sits in
        # kernels that normally take microseconds), so the median is the steady-state cost.
        sc = sorted(step_compute)
        med = sc[len(sc) // 2]
        print(f"main-stream compute per step (by launch inside the step): median {med / 1e3:.1f} ms, "
              f"mean {sum(sc) / len(sc) / 1e3:.1f} ms, min {sc[0] / 1e3:.1f}, max {sc[-1] / 1e3:.1f}")
        res["step_compute_ms"] = [c / 1e3 for c in step_compute]
        res["step_toks"] = [int(m.group(1)) if (m := re.search(r"toks=(\d+)", s["name"])) else None for s in steps]
        for i, c in enumerate(step_compute):
            sd = "; ".join(f"{v / 1e3:.0f} ms {key}" for key, v in step_side[i].most_common(3))
            mark = "STALL" if c > 1.3 * med else "     "
            print(f"  {mark} step {i:2d} {steps[i]['name'][5:38]:33s} wall {steps[i]['dur'] / 1e3:6.0f} ms  main {c / 1e3:6.0f} ms"
                  f"  side/nccl: {sd}")
            if c > 1.3 * med:
                top3 = "; ".join(f"{v / 1e3:.0f} ms {key[:90]}" for key, v in per_step[i].most_common(3))
                print(f"        top: {top3}")
    # What else ran on the GPU while a normally-short main-stream kernel took > 50 ms.
    others = sorted((k for k in gpu if k["args"].get("stream") != main_stream), key=lambda k: k["ts"])
    o_ts = [k["ts"] for k in others]
    long_k = [k for k in gpu if k["args"].get("stream") == main_stream and k["dur"] > 50e3]
    for k in long_k[:12]:
        t0, t1 = k["ts"], k["ts"] + k["dur"]
        ov = collections.Counter()
        j = max(0, bisect.bisect_left(o_ts, t0 - 5e6))
        while j < len(others) and others[j]["ts"] < t1:
            o = others[j]
            x = min(t1, o["ts"] + o["dur"]) - max(t0, o["ts"])
            if x > 0:
                ov[f"s{o['args'].get('stream')} {o['name'][:40]}"] += x
            j += 1
        desc = "; ".join(f"{v / 1e3:.0f} ms {key}" for key, v in ov.most_common(4))
        print(f"  long main kernel {k['dur'] / 1e3:.0f} ms {k['name'][:50]}; overlapping: {desc or 'nothing'}")
    for name, c in agg.items():
        print(f"  -- by {name}")
        for key, d in c.most_common(top if name != "module" else None):
            extra = f"  x{cnt[key]}" if name in ("kernel", "kernel_site", "side_site") else ""
            if name == "kernel":
                ds = sorted(durs[key])
                extra += f"  median {ds[len(ds) // 2] / 1e3:.3f} ms x{len(ds)} = {ds[len(ds) // 2] * len(ds) / 1e3:.0f} ms"
            print(f"    {d / 1e3:9.1f} ms {100 * d / compute:5.1f}%  {key}{extra}")
        res[name] = {k: v / 1e3 for k, v in c.most_common(200)}
    return res


ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+")
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--json")
a = ap.parse_args()
out = [analyze(p, a.top) for p in a.traces]
if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
