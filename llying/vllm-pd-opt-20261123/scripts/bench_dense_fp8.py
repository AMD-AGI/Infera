# Item 9b (dense online FP8): 1-GPU GEMM time / error of GLM-5.2's TP1 dense target shapes, BF16 as decode runs it today
# vs v0.34.0's online fp8_per_channel (PTPC: per-output-channel FP8 weight, dynamic per-token FP8 activation).
#
#   BF16: vLLM's UnquantizedLinearMethod (process_weights_after_loading = ROCm cache-aliasing pad, apply =
#         torch.ops.vllm.rocm_unquantized_gemm -> wvSplitKrc / aiter triton / wvSplitK / aiter tgemm, the latter
#         reading our tuned table when the job mounts it over model_configs/glm52_bf16_tuned_gemm.csv).
#   FP8:  vLLM's Fp8PtpcOnlineLinearMethod (the class `targets: fp8_per_channel` resolves to): create_weights ->
#         init_fp8_linear_kernel (the ROCm list in kernels/linear/__init__.py) -> BF16 weight loaded through the
#         online loader -> process_weights_after_loading; timed through apply(), i.e. per-token activation quant
#         included. The selected kernel class is reported per (N, K).
#   CK:   (headroom, --ck) AiterPreshuffledPerTokenFp8ScaledMMLinearKernel built directly (skipping its
#         "tuned shape" can_implement), same FP8 weights: aiter gemm_a8w8_bpreshuffle with its default config for
#         shapes missing from the tuned CSV. Not a tuned number; tuning can only do better.
#
# Timing: eager (no CUDA graph), torch.cuda events. Each sample = one chunk of ITERS back-to-back calls queued behind a
# GPU sleep so host launch overhead does not leak into GPU time; weights rotate over enough copies (>= ROT_MB total)
# that the 256 MB MALL does not hold them (decode reads every weight once per step). Median over REPS chunks, us/call.
# Error: ||y - y_ref|| / ||y_ref|| with y_ref = fp32 x @ fp32 W (the BF16 weights), on N(0,1) activations and on an
# "outlier" variant (1% of input channels x30), weights N(0, 0.02).
#
# usage: python3 bench_dense_fp8.py <model_dir> <out_dir> [--ck] [--ms 1,4,...]
import argparse, json, math, os, statistics, sys, time, traceback
from types import SimpleNamespace

import torch

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out_dir")
ap.add_argument("--ms", default="1,4,8,16,32,56,64,96,128,192,256,2048,4096")
ap.add_argument("--step-m", type=int, default=56)
ap.add_argument("--iters", type=int, default=40); ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--rot-mb", type=int, default=768)
ap.add_argument("--ck", action="store_true", help="also time aiter CK bpreshuffle directly (headroom)")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
MS = [int(x) for x in a.ms.split(",")]
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.manual_seed(0)

# ---- shapes from config.json (TP1, target layers 0..L-1 only; MTP layers stay BF16 and are not benchmarked) ----
cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
nh, qlr, kvlr, rope = cfg["num_attention_heads"], cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
qk_hd = cfg.get("qk_head_dim") or cfg["qk_nope_head_dim"] + rope
v_hd = cfg["v_head_dim"]
kd = cfg.get("first_k_dense_replace", 0)
n_full = sum(1 for t in cfg.get("indexer_types", ["full"] * L)[:L] if t == "full")
SHAPES = [  # name, N (out), K (in), layer count
    ("fused_qkv_a", qlr + kvlr + rope, H, L),
    ("q_b", nh * qk_hd, qlr, L),
    ("o_proj", H, nh * v_hd, L),
    ("indexer.wq_b", cfg["index_n_heads"] * cfg["index_head_dim"], qlr, n_full),
    ("mlp.gate_up", 2 * cfg["intermediate_size"], H, kd),
    ("mlp.down", H, cfg["intermediate_size"], kd),
]

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
import vllm.model_executor.layers.utils as lu
import vllm.model_executor.layers.quantization.online.fp8 as ofp8
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

# OnlineLinearBase.__init__ reads get_current_vllm_config().model_config.dtype; no engine here, so give it the
# model dtype (BF16) directly. Kernel selection itself runs without a config (linear_backend "auto").
_real_gcvc = ofp8.get_current_vllm_config
ofp8.get_current_vllm_config = lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
torch.set_default_dtype(torch.bfloat16)
try:   # default engine config (linear_backend "auto", default compilation config for CustomOp dispatch)
    from vllm.config import VllmConfig, set_current_vllm_config
    _vctx = set_current_vllm_config(VllmConfig()); _vctx.__enter__()
except Exception as e:
    print("running without a VllmConfig context:", type(e).__name__, e, flush=True)
FP8 = current_platform.fp8_dtype()

env_info = {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("VLLM_ROCM", "AITER", "HIP_VIS"))}
env_info.update(VLLM_ROCM_USE_SKINNY_GEMM=envs.VLLM_ROCM_USE_SKINNY_GEMM, fp8_dtype=str(FP8),
                gpu=torch.cuda.get_device_name(0), gcn=getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?"))
try:
    import vllm, aiter  # noqa
    env_info.update(vllm=vllm.__version__, aiter=getattr(aiter, "__version__", "?"))
except Exception:
    pass
from vllm._aiter_ops import rocm_aiter_ops
env_info.update(tgemm_enabled=bool(rocm_aiter_ops.is_tgemm_enabled()),
                linear_fp8_enabled=bool(rocm_aiter_ops.is_linear_fp8_enabled()))
print("env", json.dumps(env_info), flush=True)


def bf16_branch(n_rows, N, K):
    """Which rocm_unquantized_gemm_impl branch a [n_rows, K] x [N, K] call takes (labels only, mirrors v0.34.0)."""
    from vllm.platforms.rocm import on_gfx950
    cu = lu.num_compute_units()
    try:
        _, fits = lu.wvsplitkrc_dispatch(n_rows, K, N, cu)
    except Exception:
        fits = False
    if envs.VLLM_ROCM_USE_SKINNY_GEMM and on_gfx950() and 10 <= n_rows <= 128 and K % 8 == 0 and K > 512 \
            and N % 16 == 0 and fits:
        return "wvSplitKrc"
    if lu.use_aiter_triton_gemm(n_rows, N, K, torch.bfloat16):
        return "aiter_triton_a16w16"
    if envs.VLLM_ROCM_USE_SKINNY_GEMM and K % 8 == 0 and 0 < n_rows <= 5 and (N == 1 or N > 8):
        return "wvSplitK"
    return "aiter_tgemm" if rocm_aiter_ops.is_tgemm_enabled() else "torch_linear"


def make_bf16(w):
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(w.clone(), requires_grad=False))
    m = UnquantizedLinearMethod()
    m.process_weights_after_loading(layer)
    return layer, m


def _fp8_layer_shell(N, K):
    layer = torch.nn.Module()
    layer.tp_size, layer.input_size, layer.output_size = 1, K, N
    layer.input_size_per_partition, layer.output_size_per_partition = K, N
    m = ofp8.Fp8PtpcOnlineLinearMethod()
    m.create_weights(layer, K, [N], K, N, torch.bfloat16, weight_loader=default_weight_loader)
    return layer, m


def make_fp8(w):
    """Returns (layer, method, path): path 'loader' = BF16 weight pushed through the online loader as at model load."""
    N, K = w.shape
    try:
        layer, m = _fp8_layer_shell(N, K)
        with torch.device(dev):
            layer.weight.weight_loader(layer.weight, w)
        if layer.weight.dtype != FP8:      # loader buffered but did not process: finish as the model loader would
            m.process_weights_after_loading(layer)
        assert layer.weight.dtype == FP8 and layer.weight.device.type == "cuda", (layer.weight.dtype, layer.weight.device)
        return layer, m, "loader"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    layer, m = _fp8_layer_shell(N, K)      # fallback: real weight in place of the meta one, then the same processing
    del layer.weight
    layer.register_parameter("weight", torch.nn.Parameter(w.clone(), requires_grad=False))
    m.process_weights_after_loading(layer)
    assert layer.weight.dtype == FP8
    return layer, m, "direct (loader: " + err[:120] + ")"


def make_ck(fp8_layer, N, K):
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterPreshuffledPerTokenFp8ScaledMMLinearKernel as CK
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
    ok, why = CK.is_supported()
    if not ok:
        raise RuntimeError(f"CK bpreshuffle unsupported: {why}")
    k = CK(FP8ScaledMMLinearLayerConfig(weight_quant_key=kFp8StaticChannelSym, activation_quant_key=kFp8DynamicTokenSym,
                                        input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(N, K)),
           layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"])
    layer = torch.nn.Module()
    w = fp8_layer.weight
    w_nk = w if w.shape == (N, K) else w.t()           # PTPC stores [K, N] views of an [N, K] tensor
    layer.register_parameter("weight", torch.nn.Parameter(w_nk.t(), requires_grad=False))   # kernel expects [K, N]
    layer.register_parameter("weight_scale", torch.nn.Parameter(fp8_layer.weight_scale.data.clone(), requires_grad=False))
    layer.input_scale = None
    k.process_weights_after_loading(layer)
    return layer, k


# ---- timing ----
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
print("sleep cycles/us", CPU_US, flush=True)


def timeit(fn, xs, n_rot):
    """fn(i, x) for i in range(iters); returns median us/call over reps."""
    for i in range(3 * n_rot):
        fn(i, xs[i % len(xs)])
    torch.cuda.synchronize()
    # host enqueue time per call, to size the sleep that hides it
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
    return statistics.median(samples), host_us


def rel_err(y, ref):
    return ((y.float() - ref).norm() / ref.norm()).item()


rows, kernels = [], {}
for name, N, K, cnt in SHAPES:
    if cnt == 0:
        continue
    wbytes = N * K * 2
    n_rot = max(2, min(16, math.ceil(a.rot_mb * 2**20 / (N * K))))    # copies so the FP8 set alone exceeds ROT_MB
    print(f"== {name} N={N} K={K} x{cnt}: {n_rot} weight copies", flush=True)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(n_rot)]
    bf = [make_bf16(w) for w in ws]
    fp = [make_fp8(w) for w in ws]
    kname = type(fp[0][1].fp8_linear).__name__
    kernels[f"{N}x{K}"] = {"name": name, "fp8_kernel": kname, "fp8_build": fp[0][2],
                           "weight_shape_after": list(fp[0][0].weight.shape), "weight_stride": list(fp[0][0].weight.stride())}
    print("   fp8 kernel:", kname, "| build:", fp[0][2], flush=True)
    ck = None
    if a.ck and "Preshuffled" in kname:
        kernels[f"{N}x{K}"]["ck_error"] = "selected kernel already is CK bpreshuffle (tuned)"
    elif a.ck:
        try:
            ck = [make_ck(l, N, K) for l, _, _ in fp]
        except Exception as e:
            kernels[f"{N}x{K}"]["ck_error"] = f"{type(e).__name__}: {e}"[:300]
            print("   ck:", kernels[f"{N}x{K}"]["ck_error"], flush=True)
    w0 = ws[0].float()
    for M in MS:
        r = dict(name=name, N=N, K=K, count=cnt, M=M, bf16_branch=bf16_branch(M, N, K), fp8_kernel=kname)
        try:
            n_x = max(1, min(n_rot, (64 << 20) // (M * K * 2)))
            xs = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) for _ in range(n_x)]
            xo = xs[0].clone(); xo[:, torch.randperm(K, device=dev)[: max(1, K // 100)]] *= 30
            ref, refo = xs[0].float() @ w0.t(), xo.float() @ w0.t()
            lb, mb = bf[0]; lf, mf, _ = fp[0]
            r["bf16_err"], r["bf16_err_outlier"] = rel_err(mb.apply(lb, xs[0]), ref), rel_err(mb.apply(lb, xo), refo)
            r["fp8_err"], r["fp8_err_outlier"] = rel_err(mf.apply(lf, xs[0]), ref), rel_err(mf.apply(lf, xo), refo)
            r["bf16_us"], r["bf16_host_us"] = timeit(lambda i, x: bf[i % n_rot][1].apply(bf[i % n_rot][0], x), xs, n_rot)
            r["fp8_us"], r["fp8_host_us"] = timeit(lambda i, x: fp[i % n_rot][1].apply(fp[i % n_rot][0], x), xs, n_rot)
            q = mf.fp8_linear.quant_fp8
            r["quant_us"], _ = timeit(lambda i, x: q(x), xs, n_rot)
            try:
                r["quant_native_us"], _ = timeit(lambda i, x: q.forward_native(x), xs, n_rot)
            except Exception:
                r["quant_native_us"] = None
            if ck:
                lc, kc = ck[0]
                r["ck_err"] = rel_err(kc.apply_weights(lc, xs[0]), ref)
                r["ck_us"], _ = timeit(lambda i, x: ck[i % n_rot][1].apply_weights(ck[i % n_rot][0], x), xs, n_rot)
            r["bf16_TBps"] = wbytes / r["bf16_us"] / 1e6
            r["fp8_TBps"] = (wbytes / 2) / r["fp8_us"] / 1e6
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()
        rows.append(r)
        print("  ", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    del ws, bf, fp, ck
    torch.cuda.empty_cache()

# ---- outputs ----
json.dump({"env": env_info, "kernels": kernels, "rows": rows, "args": vars(a)}, open(os.path.join(a.out_dir, "bench.json"), "w"), indent=1)
f = lambda v, p=1: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
lines = ["# dense BF16 (decode today) vs online fp8_per_channel (PTPC), us/call median, TP1, eager, rotated weights",
         "| shape | N x K | M | bf16 branch | bf16 us | fp8 us | fp8/bf16 | quant us | ck us | bf16 TB/s | fp8 TB/s | err bf16 | err fp8 | err fp8 outl |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "error" in r:
        lines.append(f"| {r['name']} | {r['N']}x{r['K']} | {r['M']} | ERROR {r['error'][:80]} |"); continue
    lines.append(f"| {r['name']} | {r['N']}x{r['K']} | {r['M']} | {r['bf16_branch']} | {f(r['bf16_us'])} | {f(r['fp8_us'])} | "
                 f"{r['fp8_us'] / r['bf16_us']:.2f} | {f(r['quant_us'])} | {f(r.get('ck_us'))} | {f(r['bf16_TBps'], 2)} | "
                 f"{f(r['fp8_TBps'], 2)} | {r['bf16_err']:.1e} | {r['fp8_err']:.1e} | {r['fp8_err_outlier']:.1e} |")
lines += ["", "## FP8 kernel selected by v0.34.0 per (N, K)"]
for k, v in kernels.items():
    lines.append(f"- {v['name']} {k}: {v['fp8_kernel']} (build {v['fp8_build']}){'; ck: ' + v['ck_error'] if 'ck_error' in v else ''}")
lines += ["", f"## one decode step, target dense GEMMs only, M = {a.step_m} (sum over layer counts; MTP layer not included)"]
tot = {"bf16": 0.0, "fp8": 0.0, "ck": 0.0, "quant": 0.0}; ok = True
for name, N, K, cnt in SHAPES:
    r = next((r for r in rows if r["name"] == name and r["M"] == a.step_m and "error" not in r), None)
    if r is None:
        if cnt:
            ok = False; lines.append(f"- {name}: no result")
        continue
    ck_us = r.get("ck_us") or r["fp8_us"]
    for k, v in (("bf16", r["bf16_us"]), ("fp8", r["fp8_us"]), ("ck", min(ck_us, r["fp8_us"])), ("quant", r["quant_us"])):
        tot[k] += cnt * v
    lines.append(f"- {name} x{cnt}: bf16 {cnt * r['bf16_us'] / 1000:.3f} ms, fp8 {cnt * r['fp8_us'] / 1000:.3f} ms "
                 f"(of which act quant {cnt * r['quant_us'] / 1000:.3f} ms)")
summary = (f"M={a.step_m}: bf16 {tot['bf16'] / 1000:.2f} ms, fp8 {tot['fp8'] / 1000:.2f} ms "
           f"(delta {(tot['fp8'] - tot['bf16']) / 1000:+.2f} ms; act quant {tot['quant'] / 1000:.2f} ms)"
           + (f", fp8 with CK where faster {tot['ck'] / 1000:.2f} ms" if a.ck else "") + ("" if ok else " [incomplete]"))
lines += ["", "**" + summary + "**",
          "", "Notes: eager calls (production decode is a FULL CUDA graph and may compile/fuse the activation quant into the",
          "preceding RMSNorm, so 'quant us' is an upper bound); random weights/activations (error on real activations with",
          "outliers is closer to the 'outl' column)."]
open(os.path.join(a.out_dir, "table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("SUMMARY", summary, flush=True)
