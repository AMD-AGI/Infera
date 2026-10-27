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
    agg = {n: collections.Counter() for n in ("module", "model_frame", "leaf_frame", "kernel", "gemm_shape")}
    cnt = collections.Counter()
    side, comm, compute = 0.0, 0.0, 0.0
    for k in gpu:
        d = k["dur"]
        if "nccl" in k["name"].lower() or "rccl" in k["name"].lower():
            comm += d
            continue
        if k["args"].get("stream") != main_stream:
            side += d
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
    for name, c in agg.items():
        print(f"  -- by {name}")
        for key, d in c.most_common(top if name != "module" else None):
            extra = f"  x{cnt[key]}" if name == "kernel" else ""
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
