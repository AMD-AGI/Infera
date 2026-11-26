# After tuning aiter's a8w8 bpreshuffle table for GLM-5.2's two untuned online-FP8 (PTPC) dense shapes, with our table
# mounted under aiter/configs/model_configs/ (and the bpreshuffle JIT modules rebuilt from the merged table):
#   (a) the kernel v0.34.0 now selects for o_proj 6144x16384 and dense MLP gate_up 24576x6144 (Fp8PtpcOnlineLinearMethod
#       -> init_fp8_linear_kernel, the ROCm list in kernels/linear/__init__.py), expected
#       AiterPreshuffledPerTokenFp8ScaledMMLinearKernel;
#   (b) median us/call of that path vs the path production runs today, i.e. the same method built with vLLM's
#       "(N, K) is tuned" check forced False (-> RowWiseTorchFP8ScaledMMLinearKernel = torch._scaled_mm / hipBLASLt),
#       per M, through apply() (per-token activation quant included), plus rel. error of both vs fp32;
#   (c) which tuned row aiter's runtime lookup hits per M (exact or padded M) and the tuner's own us for it.
# Setup (VllmConfig context, 1-process TP1 world, FP8 layer construction, timing with rotated weights behind a GPU
# sleep) is scripts/bench_dense_fp8.py's.
# usage: python3 verify_fp8_bpreshuffle.py <out_dir> [--ms 1,2,...] [--step-m 56]
# Exit 0 = both shapes select CK bpreshuffle and its error is within the hipBLASLt path's; 1 otherwise.
import argparse, json, math, os, statistics, sys, time, traceback
from types import SimpleNamespace

import torch

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--ms", default="1,2,4,8,16,24,32,40,48,52,56,64,72,80,96,104,128,160,192,256,2048,4096,8192")
ap.add_argument("--step-m", type=int, default=56)
ap.add_argument("--iters", type=int, default=40); ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--rot-mb", type=int, default=768)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
MS = [int(x) for x in a.ms.split(",")]
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.manual_seed(0)
SHAPES = [("o_proj", 6144, 16384, 78), ("mlp.gate_up", 24576, 6144, 3)]   # name, N, K, layers per decode step (TP1)

import vllm.envs as envs
from vllm.platforms import current_platform
import vllm.model_executor.layers.quantization.online.fp8 as ofp8
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm._aiter_ops import rocm_aiter_ops

ofp8.get_current_vllm_config = lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
torch.set_default_dtype(torch.bfloat16)
try:
    from vllm.config import VllmConfig, set_current_vllm_config
    _vctx = set_current_vllm_config(VllmConfig()); _vctx.__enter__()
except Exception as e:
    print("running without a VllmConfig context:", type(e).__name__, e, flush=True)
try:
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    for _i, _backend in enumerate(("nccl", "gloo")):
        try:
            init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"file:///tmp/vpd_fp8_verify_init{_i}",
                                         local_rank=0, backend=_backend)
            break
        except Exception as e:
            print(f"{_backend} init failed: {type(e).__name__}: {e}", flush=True)
            if _i == 1:
                raise
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    print("distributed world 1 / TP1 initialised", flush=True)
except Exception as e:
    print("distributed init failed:", type(e).__name__, e, flush=True)
    traceback.print_exc()
FP8 = current_platform.fp8_dtype()

from aiter.jit.core import AITER_CONFIGS
from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
merged = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE   # the merged table both vLLM's check and aiter read
ours_in_merged = 0
with open(merged) as f:
    hdr = f.readline().strip().split(",")
    iN, iK = hdr.index("N"), hdr.index("K")
    for line in f:
        c = line.strip().split(",")
        ours_in_merged += (c[iN], c[iK]) in {(str(n), str(k)) for _, n, k, _ in SHAPES}
env_info = {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("VLLM_ROCM", "AITER", "HIP_VIS"))}
env_info.update(fp8_dtype=str(FP8), gpu=torch.cuda.get_device_name(0), merged_table=merged, our_rows_in_merged=ours_in_merged,
                linear_fp8_enabled=bool(rocm_aiter_ops.is_linear_fp8_enabled()),
                hipbmm_enabled=bool(rocm_aiter_ops.is_linear_hipbmm_enabled()))
print("env", json.dumps(env_info), flush=True)


def _fp8_layer_shell(N, K):
    layer = torch.nn.Module()
    layer.tp_size, layer.input_size, layer.output_size = 1, K, N
    layer.input_size_per_partition, layer.output_size_per_partition = K, N
    m = ofp8.Fp8PtpcOnlineLinearMethod()
    m.create_weights(layer, K, [N], K, N, torch.bfloat16, weight_loader=default_weight_loader)
    return layer, m


def make_fp8(w):
    N, K = w.shape
    try:
        layer, m = _fp8_layer_shell(N, K)
        with torch.device(dev):
            layer.weight.weight_loader(layer.weight, w)
        if layer.weight.dtype != FP8:
            m.process_weights_after_loading(layer)
        assert layer.weight.dtype == FP8 and layer.weight.device.type == "cuda", (layer.weight.dtype, layer.weight.device)
        return layer, m
    except Exception:
        pass
    layer, m = _fp8_layer_shell(N, K)
    del layer.weight
    layer.register_parameter("weight", torch.nn.Parameter(w.clone(), requires_grad=False))
    m.process_weights_after_loading(layer)
    assert layer.weight.dtype == FP8
    return layer, m


_ORIG = rocm_aiter_ops.__dict__["is_shuffled_per_token_w8a8_gemm_tuned"]


def make_fp8_untuned(w):
    """Production today: the same method with vLLM's tuned-shape check answering False for this (N, K)."""
    rocm_aiter_ops.is_shuffled_per_token_w8a8_gemm_tuned = staticmethod(lambda N, K, q: False)
    try:
        return make_fp8(w)
    finally:
        rocm_aiter_ops.is_shuffled_per_token_w8a8_gemm_tuned = _ORIG


def _sleep_cycles_per_us():
    if not hasattr(torch.cuda, "_sleep"):
        return None
    try:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1000); torch.cuda.synchronize()
        s.record(); torch.cuda._sleep(2_000_000); e.record(); torch.cuda.synchronize()
        return 2_000_000 / (s.elapsed_time(e) * 1000)
    except Exception:
        return None


CPU_US = _sleep_cycles_per_us()


def timeit(fn, xs, n_rot):
    for i in range(3 * n_rot):
        fn(i, xs[i % len(xs)])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(a.iters):
        fn(i, xs[i % len(xs)])
    torch.cuda.synchronize()
    host_us = (time.perf_counter() - t0) * 1e6 / a.iters
    samples = []
    for _ in range(a.reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if CPU_US:
            torch.cuda._sleep(int(CPU_US * (host_us * a.iters * 1.5 + 200)))
        s.record()
        for i in range(a.iters):
            fn(i, xs[i % len(xs)])
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000 / a.iters)
    return statistics.median(samples)


def rel_err(y, ref):
    return ((y.float() - ref).norm() / ref.norm()).item()


def row_hit(M, N, K):
    c = get_GEMM_config_with_quant_type(M, N, K, FP8, merged)
    if c is None:
        return "default (no row)", None
    return f"{c.get('libtype')}:{str(c.get('kernelName'))[:60]} splitK={c.get('splitK')}", float(c.get("us", "nan"))


rows, kernels, ok = [], {}, True
for name, N, K, cnt in SHAPES:
    n_rot = max(2, min(16, math.ceil(a.rot_mb * 2**20 / (N * K))))
    tuned = bool(rocm_aiter_ops.is_shuffled_per_token_w8a8_gemm_tuned(N, K, FP8))
    print(f"== {name} N={N} K={K} x{cnt}: vLLM tuned-shape check {tuned}; {n_rot} weight copies", flush=True)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(n_rot)]
    ck = [make_fp8(w) for w in ws]
    hb = [make_fp8_untuned(w) for w in ws]
    kc, kh = type(ck[0][1].fp8_linear).__name__, type(hb[0][1].fp8_linear).__name__
    kernels[f"{N}x{K}"] = {"name": name, "tuned_check": tuned, "selected": kc, "baseline": kh}
    print(f"   selected now: {kc} | production baseline: {kh}", flush=True)
    sel_ok = "Preshuffled" in kc
    ok = ok and sel_ok and tuned
    w0 = ws[0].float()
    for M in MS:
        r = dict(name=name, N=N, K=K, count=cnt, M=M, selected=kc, baseline=kh)
        try:
            r["row"], r["tuner_us"] = row_hit(M, N, K)
            n_x = max(1, min(n_rot, (64 << 20) // (M * K * 2)))
            xs = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) for _ in range(n_x)]
            ref = xs[0].float() @ w0.t()
            (lc, mc), (lh, mh) = ck[0], hb[0]
            r["err_sel"], r["err_base"] = rel_err(mc.apply(lc, xs[0]), ref), rel_err(mh.apply(lh, xs[0]), ref)
            r["us_sel"] = timeit(lambda i, x: ck[i % n_rot][1].apply(ck[i % n_rot][0], x), xs, n_rot)
            r["us_base"] = timeit(lambda i, x: hb[i % n_rot][1].apply(hb[i % n_rot][0], x), xs, n_rot)
            if r["err_sel"] > 1.1 * r["err_base"] + 0.005:
                ok = False; r["err_flag"] = "CK error above hipBLASLt path"
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:300]; ok = False
            traceback.print_exc()
        rows.append(r)
        print("  ", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    del ws, ck, hb
    torch.cuda.empty_cache()

f = lambda v, p=1: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
lines = ["# online FP8 PTPC, tuned CK bpreshuffle (selected now) vs hipBLASLt (production today), us/call median, TP1, eager, rotated weights",
         f"merged table {merged}: {ours_in_merged} rows for these two (N, K)", "",
         "| shape | N x K | M | selected us | hipBLASLt us | sel/hipb | tuner us | row hit | err sel | err hipb |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "error" in r:
        lines.append(f"| {r['name']} | {r['N']}x{r['K']} | {r['M']} | ERROR {r['error'][:100]} |"); continue
    lines.append(f"| {r['name']} | {r['N']}x{r['K']} | {r['M']} | {f(r['us_sel'])} | {f(r['us_base'])} | "
                 f"{r['us_sel'] / r['us_base']:.2f} | {f(r['tuner_us'])} | {r['row']} | {r['err_sel']:.1e} | {r['err_base']:.1e} |")
lines += ["", "## kernel per (N, K)"]
for k, v in kernels.items():
    lines.append(f"- {v['name']} {k}: tuned check {v['tuned_check']}; selected {v['selected']}; production baseline {v['baseline']}")
sel = base = 0.0; full = True
for name, N, K, cnt in SHAPES:
    r = next((r for r in rows if r["name"] == name and r["M"] == a.step_m and "error" not in r), None)
    if r is None:
        full = False; continue
    sel += cnt * r["us_sel"]; base += cnt * r["us_base"]
summary = (f"{'PASS' if ok else 'FAIL'}: M={a.step_m} per decode step (o_proj x78 + gate_up x3): CK {sel / 1000:.3f} ms vs hipBLASLt "
           f"{base / 1000:.3f} ms (delta {(sel - base) / 1000:+.3f} ms){'' if full else ' [incomplete]'}")
lines += ["", "**" + summary + "**", "",
          "Notes: eager calls; production decode is a FULL CUDA graph (launch overhead hidden there). 'row hit' is aiter's",
          "runtime lookup (exact M, then M padded to 16/32/64/128, then next power of two); 'default' = untuned CK kernel."]
open(os.path.join(a.out_dir, "verify.md"), "w").write("\n".join(lines) + "\n")
json.dump({"env": env_info, "kernels": kernels, "rows": rows}, open(os.path.join(a.out_dir, "verify.json"), "w"), indent=1)
print("\n".join(lines))
print("SUMMARY", summary, flush=True)
sys.exit(0 if ok else 1)
