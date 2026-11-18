"""sglang PR #47276 (gfx950 fused indexer logits + histogram, top-k from the histogram) at vLLM's flattened MTP3 decode
shapes, built standalone (hipcc + tvm-ffi, the same wrapper sglang's load_jit generates) and compared with what vLLM
can run: #47897 gluon logits (overlay) + aiter top-k (what vLLM v0.34 uses for decode top-k).

The sglang kernels read the same cache page as vLLM's 64-token preshuffled indexer cache (64 x 128 fp8 K, then 64 fp32
scales, 8448 bytes a page); their top-k maps positions through a page table, so an identity table gives vLLM's token
positions. The sibling-group logits kernel (variant 2) is compiled for HOST_GR rows per request (sglang: 6, MTP5);
here it is rebuilt for 4 (MTP3) and serves at most 64 rows.

Candidates per row count (4 rows per request, lengths L-3..L):
  vllm      gluon_fp8_paged_mqa_logits (4-row LPT groups) + aiter top_k_per_row_decode
  sgl       logits_hist (variant from sglang's logits_launch rules, group 4) + topk_transform (identity page table)
Timing: CUDA graph replay of ITERS calls (warm cache, no host launch cost). Correctness: logits on visible columns
vs gluon, top-k exact against the row's k-th largest logit.
Usage: python bench_sgl47276.py [rows ...]   env CTX_MIN/CTX_MAX (65536/69632), ITERS (20), WIDTH (logits row width,
       default 1048576 as vLLM's max_model_len)
"""

import os
import subprocess
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
JIT = os.path.join(here, "sgl47276", "jit")
BUILD = os.environ.get("BUILD_DIR", "/tmp/sgl47276-build")
HEADS, DIM, PAGE, NEXT_N, TOPK = 32, 128, 64, 4, 2048
CTX_MIN = int(os.environ.get("CTX_MIN", 65536))
CTX_MAX = int(os.environ.get("CTX_MAX", 69632))
ITERS = int(os.environ.get("ITERS", 20))
WIDTH = int(os.environ.get("WIDTH", 1048576))
HIST_BITS, GROUP_ROWS, GROUP_MIN_ROWS, BPR, SHORT_BPR, SHORT_MAX_ROWS, TOPK_G = 12, 4, 30, 48, 32, 24, 48
dev = torch.device("cuda")

FFI = """#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
"""


def build(name, src, wrappers, flags):
    import tvm_ffi
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path, find_libtvm_ffi

    os.makedirs(BUILD, exist_ok=True)
    cu = os.path.join(BUILD, f"{name}.cu")
    so = os.path.join(BUILD, f"{name}.so")
    with open(cu, "w") as f:
        f.write(FFI + f'#include "{src}"\nnamespace sglang {{\n')
        for export, kernel in wrappers:
            f.write(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export}, ({kernel}));\n")
        f.write("}\n")
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    lib = find_libtvm_ffi()
    arch = os.environ.get("ARCH") or torch.cuda.get_device_properties(0).gcnArchName
    cmd = [f"{rocm}/bin/hipcc", "-DUSE_ROCM", "-DHIP_FP8_TYPE_E4M3=1", "-std=c++20", "-O3", "-fPIC",
           "-D__HIP_PLATFORM_AMD__=1", "-fno-gpu-rdc",
           f"--offload-arch={arch}", f"-I{JIT}/include", f"-I{find_include_path()}",
           f"-I{find_dlpack_include_path()}", f"-I{rocm}/include", *flags, cu, "-shared", "-o", so,
           f"-L{os.path.dirname(lib)}", "-l" + os.path.basename(lib)[3:].rsplit(".so", 1)[0],
           f"-L{rocm}/lib", "-lamdhip64"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(" ".join(cmd))
        print(r.stderr[-4000:])
        raise RuntimeError(f"build {name} failed")
    return tvm_ffi.load_module(so)


def modules():
    src = os.path.join(JIT, "csrc", "dsa_gfx950", "paged_mqa_logits.cuh")
    patched = os.path.join(BUILD, "paged_mqa_logits_gr.cuh")
    os.makedirs(BUILD, exist_ok=True)
    text = open(src).read()
    assert "constexpr int HOST_GR = 6;" in text
    open(patched, "w").write(text.replace("constexpr int HOST_GR = 6;", f"constexpr int HOST_GR = {GROUP_ROWS};"))
    logits = build("logits", patched, [("logits_hist", "PagedMqaLogitsKernel::run")],
                   ["-fno-honor-nans", "-mllvm", "-amdgpu-mfma-vgpr-form", "-mllvm", "-amdgpu-early-inline-all=true",
                    "-mllvm", "-amdgpu-function-calls=false", "-Wno-unused-result"])
    topk = build("topk", os.path.join(JIT, "csrc", "dsa_gfx950", "topk_transform.cuh"),
                 [("topk_transform", "TopKTransformKernel::run"), ("hist_stride", "TopKHistStride::run")],
                 [f"-DDSA_TOPK_HIST_BITS={HIST_BITS}"])
    return logits, topk


def logits_launch(rows, table_pages, rows_per_request):
    if rows_per_request == GROUP_ROWS and rows % GROUP_ROWS == 0 and GROUP_MIN_ROWS <= rows <= 64:
        return 0, 2
    if rows <= SHORT_MAX_ROWS:
        need = -(-table_pages // (8 * 64))
        return max(SHORT_BPR, -(-need // 8) * 8), 1
    return BPR, 0


def make_inputs(rows, seed):
    from bench_paged_logits import make_inputs as mk  # same cache / table construction as the logits bench
    import bench_paged_logits as bp

    bp.MAX_MODEL_LEN = WIDTH
    bp.CTX_MIN, bp.CTX_MAX = CTX_MIN, CTX_MAX
    inp = mk(rows // NEXT_N, seed)
    ctx = inp["ctx"]
    lens = (ctx[:, None] - (NEXT_N - 1) + torch.arange(NEXT_N, device=dev, dtype=torch.int32)[None]).reshape(-1)
    table = inp["table"].repeat_interleave(NEXT_N, dim=0).contiguous()
    q = inp["q"].reshape(rows, HEADS, DIM).contiguous()
    return dict(q=q, kv=inp["kv"], w=inp["w"].contiguous(), lens=lens.contiguous(), table=table)


def graph_time(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(ITERS):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (5 * ITERS)


def topk_bad(logits, lens, out):
    bad = 0
    for r in range(out.shape[0]):
        n = int(lens[r])
        k = min(n, TOPK)
        sel = out[r, :TOPK]
        sel = sel[sel >= 0]
        if sel.numel() != k or sel.unique().numel() != k or (sel >= n).any():
            bad += 1
            continue
        kth = torch.topk(logits[r, :n], k).values[-1]
        if (logits[r, sel.long()] < kth).any():
            bad += 1
    return bad


def main():
    if os.environ.get("BUILD_ONLY") == "1":
        modules()
        print("built", flush=True)
        return
    from aiter.ops.topk import top_k_per_row_decode as aiter_topk

    from vllm.v1.attention.ops import rocm_gluon_paged_mqa_logits as gm  # overlay file, mounted by run_kf.sh

    lmod, tmod = modules()
    ghs = int(tmod.hist_stride())
    rows_list = [int(a) for a in sys.argv[1:]] or [16, 32, 48, 64, 104, 128]
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} {props.gcnArchName}; ctx {CTX_MIN}-{CTX_MAX}, logits width {WIDTH}, hist stride {ghs}, "
          f"graph-timed, iters {ITERS}", flush=True)
    print("rows | vllm logits us | vllm topk us | vllm total | sgl variant | sgl logits us | sgl topk us | sgl total | "
          "sgl/vllm | logits max|diff| | topk bad rows (vllm, sgl)", flush=True)
    for rows in rows_list:
        inp = make_inputs(rows, 11 + rows)
        q, kv, w, lens, table = inp["q"], inp["kv"], inp["w"], inp["lens"], inp["table"]
        pages = table.shape[1]
        out_v = torch.empty(rows, WIDTH, device=dev)
        idx_v = torch.empty(rows, TOPK, dtype=torch.int32, device=dev)
        kv8 = kv.view(kv.shape[0], -1).view(torch.float8_e4m3fn)
        out_s = torch.empty(rows, pages * PAGE, device=dev)
        ghist = torch.zeros(rows, ghs, dtype=torch.int32, device=dev)
        ident = torch.arange(pages, dtype=torch.int32, device=dev)[None].expand(rows, pages).contiguous()
        ctr = torch.zeros(rows, 32, dtype=torch.int32, device=dev)
        cap = max(TOPK, pages * PAGE)
        cand_i = torch.empty(rows, cap, dtype=torch.int32, device=dev)
        cand_v = torch.empty(rows, cap, dtype=torch.float32, device=dev)
        idx_s = torch.empty(rows, TOPK, dtype=torch.int32, device=dev)
        bpr, variant = logits_launch(rows, pages, NEXT_N)

        def v_logits():
            gm.gluon_fp8_paged_mqa_logits(q.view(rows, 1, HEADS, DIM), kv, w, out_v, lens, table, WIDTH,
                                          Preshuffle=True, KVBlockSize=PAGE, ChunkK=256, LptGroupRows=NEXT_N)

        def v_topk():
            aiter_topk(out_v, 1, lens, idx_v, rows, out_v.stride(0), out_v.stride(1), k=TOPK)

        def s_logits():
            lmod.logits_hist(q, kv8, w, lens, table, out_s, ghist, bpr, variant)

        def s_topk():
            tmod.topk_transform(out_s, lens, ident, idx_s, ghist, ctr, cand_i, cand_v, TOPK_G, PAGE)

        def s_both():
            s_logits()
            s_topk()

        try:
            v_logits(); v_topk(); torch.cuda.synchronize()
            s_logits(); torch.cuda.synchronize()
            vis = torch.arange(pages * PAGE, device=dev)[None] < lens[:, None]
            diff = float((out_v[:, : pages * PAGE][vis] - out_s[vis]).abs().max())
            s_topk(); torch.cuda.synchronize()
            bad_v = topk_bad(out_v, lens, idx_v)
            bad_s = topk_bad(out_s, lens, idx_s)
            tvl, tvt = graph_time(v_logits), graph_time(v_topk)
            tsl = graph_time(s_logits)
            ghist.zero_()  # logits alone leaves its histogram behind; top-k expects it zeroed on entry
            torch.cuda.synchronize()
            tsb = graph_time(s_both)  # top-k consumes (zeroes) the histogram the logits kernel writes
            tst = tsb - tsl
            print(f"{rows} | {tvl:.1f} | {tvt:.1f} | {tvl + tvt:.1f} | {variant} | {tsl:.1f} | {tst:.1f} | {tsb:.1f} | "
                  f"{tsb / (tvl + tvt):.2f} | {diff:.3g} | {bad_v}, {bad_s}", flush=True)
        except Exception as e:  # report and keep going
            print(f"{rows} | error: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
