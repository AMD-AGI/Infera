#!/usr/bin/env python3
"""Per-definition time breakdown of one sweep point (stdlib only, low memory).

Inputs (one run dir): profile/*.trace.json.gz (graph-ON window), profile/eager/*.trace.json.gz
(eager window, gives launching op + input dims), profile/device_timer_rank_0_yihou.json,
steps_rank_0_yihou.jsonl, config_yihou.json.

Method: every kernel name seen in the eager window is attributed to a workset definition from
its launching op and dims; graph-ON kernel time per (phase, kernel name) is then assigned to
definitions with the eager split. Draft-decode kernels are invisible in graph-ON traces, so that
phase uses eager kernel time scaled to the DeviceTimer graph-ON median.

Usage: analyze_sweep.yihou.py RUN_DIR [--json OUT.json]
"""
import argparse
import gzip
import importlib.util
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ks", HERE.parents[1] / "scripts" / "kernel_shapes.yihou.py")
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)

PHASES = {"TARGET_VERIFY (step[VERIFY])": "target_verify",
          "DRAFT_EXTEND_V2 (draft_extend)": "draft_extend",
          "draft DECODE x4 (draft > step[DRAFT])": "draft_decode",
          "draft glue (draft, outside steps)": "draft_glue",
          "<none>": "other"}
GEMM_NK = {(6144, 16384): "gemm_bf16_n6144_k16384 (o_proj)",
           (16384, 2048): "gemm_bf16_n16384_k2048 (q_b_proj)",
           (2624, 6144): "gemm_bf16_n2624_k6144 (qkv_a)",
           (256, 6144): "gemm_bf16_n256_k6144 (router)",
           (38720, 6144): "gemm_bf16_n38720_k6144 (draft lm_head)",
           (32, 6144): "indexer projections (wq_b/wk/weights gemm; not exported)",
           (4096, 2048): "indexer projections (wq_b/wk/weights gemm; not exported)",
           (128, 6144): "indexer projections (wq_b/wk/weights gemm; not exported)",
           (6144, 12288): "draft eh_proj gemm (not exported)",
           (6144, 6144): "dense MLP layers 0-2 (not exported)",
           (6144, 3072): "dense MLP layers 0-2 (not exported)"}
DIMS = re.compile(r"\[([0-9, ]+)\]")


def classify(kname, opname, dims):
    k, o = kname, opname or ""
    if k.startswith("ncclDevKernel") or "nccl" in o or "all_gather" in o or "reduce_scatter" in o or "all_reduce" in o:
        return "communication (RCCL, not exported)"
    if "fused_moe" in o or "a4w4" in k or "kernel_moe_gemm" in k or "moe_sort" in k or "moe_sorting" in k:
        return ("moe_bf16_routed_topk9_e257_h6144_i512" if "BFloat16[257, 1024, 6144]" in dims or "kernel_moe_gemm" in k
                else "moe_mxfp4_routed_topk9_e257_h6144_i512")
    if "grouped_topk" in k or "biased_grouped_topk" in o or "_fused_append_shared_experts" in k:
        return "moe_routing_sigmoid_bias_topk8_e256_shared1"
    if "_sparse_mla" in k:
        return "dsa_sparse_mla_fp8_h64_ckv512_kpe64_topk2048_ps64"
    if "paged_mqa_logits" in k or "dsa_topk" in k or "fast_topk_transform" in o:
        return "dsa_topk_indexer_fp8_h32_d128_topk2048_ps64"
    if "_batched_gemm_a8w8" in k:
        return "mla_absorb_bmm_fp8w_h64 (W_UK + W_UV)"
    if "add_rmsnorm" in o or "rmsnorm2d_fwd_with_add" in o:
        return "fused_add_rmsnorm_h6144"
    if o.endswith(("aten::mm", "aiter::gemm_a16w16", "aiter::gemm_a16w16_")) or "gemm_a16w16" in o or o == "aten::mm":
        shapes = [tuple(int(x) for x in m.split(",")) for m in DIMS.findall(dims)]
        if len(shapes) >= 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2:
            kdim = shapes[0][1]
            b = shapes[1]
            n = b[1] if b[0] == kdim else b[0]
            return GEMM_NK.get((n, kdim), f"other gemm N{n} K{kdim}")
    if "act_quant" in k or "indexer_k_quant" in k or "hadamard" in k or "Layernorm2d" in k or "kn_entry_2c" in k:
        return "indexer prologue (q/k quant, hadamard, k_norm, rope; not exported)"
    if "fused_qk_rope_cat_and_cache" in k or "fused_qk_rmsnorm" in k or "rmsnorm" in k:
        return "rope/qk-norm/kv-write (not exported)"
    return "glue: elementwise/cat/copy/argmax/sampling (not exported)"


def load(path):
    with gzip.open(path) as f:
        return json.load(f)["traceEvents"]


def kernel_table(events):
    """{(phase, kernel): (us, n, {definition: us})} with definitions from the launching op."""
    ctx = ks.attribute_context(events)
    out = defaultdict(lambda: [0.0, 0, defaultdict(float)])
    for e in events:
        if e.get("cat") != "kernel":
            continue
        phase, op, outer = ctx.get(e["args"].get("correlation"), ("<none>", None, None))
        name = op["name"] if op else ""
        if outer is not None and outer is not op:
            name = f"{outer['name']} > {name}"
        d = classify(e["name"], name, ks.dims(op) if op else "")
        row = out[(PHASES.get(phase, phase), e["name"])]
        row[0] += e.get("dur", 0)
        row[1] += 1
        row[2][d] += e.get("dur", 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    run = Path(a.run_dir)
    cfg = json.loads((run / "config_yihou.json").read_text())["benchmark"]
    pw = (cfg["profile_start_step"], cfg["profile_start_step"] + cfg["profile_num_steps"])
    ew = tuple(cfg.get("eager_window") or (10**9, 0))
    on = kernel_table(load(next((run / "profile").glob("*.trace.json.gz"))))
    off = kernel_table(load(next((run / "profile" / "eager").glob("*.trace.json.gz"))))
    n_on, n_off = cfg["profile_num_steps"], ew[1]

    # Steps: graph-ON, outside both windows -> clean iteration wall time.
    steps = [json.loads(l) for l in open(run / "steps_rank_0_yihou.jsonl")]
    clean = [s["seconds"] for i, s in enumerate(steps)
             if s.get("target_graph") and not (pw[0] <= i < pw[1]) and not (ew[0] <= i < ew[0] + ew[1])]
    dt = json.loads((run / "profile" / "device_timer_rank_0_yihou.json").read_text())["by_category"]
    med = {k: statistics.median(v["samples"]) * 1e3 for k, v in dt.items() if v.get("samples")}

    per_def = defaultdict(lambda: defaultdict(float))   # phase -> def -> us/iter
    for (phase, kname), (us, n, _) in on.items():
        split = off.get((phase, kname))
        if split and split[0] > 0:
            for d, dus in split[2].items():
                per_def[phase][d] += us / n_on * dus / split[0]
        else:
            per_def[phase][classify(kname, "", "")] += us / n_on
    # Draft decode: invisible in graph-ON traces; use eager split scaled to the DeviceTimer median.
    eager_dd = defaultdict(float)
    for (phase, kname), (us, n, split) in off.items():
        if phase == "draft_decode":
            for d, dus in split.items():
                eager_dd[d] += dus / max(1, n_off)
    tot_dd = sum(eager_dd.values())
    if tot_dd and "eagle_draft" in med:
        scale = med["eagle_draft"] * 1e3 / tot_dd
        per_def["draft_decode"] = {d: v * scale for d, v in eager_dd.items()}
    result = {"run": run.name, "local_bs": cfg["batch_size"] // 4, "isl": cfg.get("input_len"),
              "iteration_ms_median_clean": statistics.median(clean) * 1e3 if clean else None,
              "clean_iterations": len(clean), "device_timer_median_ms": med,
              "per_phase_def_us": {p: dict(sorted(d.items(), key=lambda kv: -kv[1])) for p, d in per_def.items()}}
    text = json.dumps(result, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
