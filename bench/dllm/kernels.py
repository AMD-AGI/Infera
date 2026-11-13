# ruff: noqa: B023, F821 - each closure is timed in the iteration that builds it
"""Kernel microbenchmarks for dLLM-Sim on the local GPU vs the simulator's prediction."""

import json
import os
import sys

import torch
import torch.nn.functional as F

from infera.projection.core.projection.inference_projection.dllm import DLLMHardware
from infera.projection.core.projection.inference_projection.dllm.ops import (
    ATTENTION,
    GEMM,
    MEMORY,
    Op,
)

ARCH = os.getenv("ARCH", "mi325x")
OUT = os.getenv("OUT", os.path.expanduser("~/dllm/results/kernels.json"))
dev = torch.device("cuda")
hw = DLLMHardware.from_arch(ARCH)
roof = DLLMHardware.from_arch(ARCH, backend="roofline")
rows: list[dict] = []


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def record(group, name, measured, op, **extra):
    pred = op.time_ms(hw, launch=False)
    rl = op.time_ms(roof, launch=False)
    rows.append(
        {
            "group": group,
            "name": name,
            "measured_ms": measured,
            "origami_ms": pred,
            "roofline_ms": rl,
            **extra,
        }
    )
    print(
        f"{group:>9} {name:<44} meas {measured:9.4f}  sim {pred:9.4f} ({pred / measured:5.2f}x)  roof {rl:9.4f}",
        flush=True,
    )


# -- device ceilings -----------------------------------------------------------
x = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
y = torch.empty_like(x)
t = timed(lambda: y.copy_(x))
rows.append(
    {"group": "ceiling", "name": "hbm_copy_1GiB", "measured_ms": t, "gbps": 2 * x.numel() / t / 1e6}
)
print(
    f"HBM copy: {2 * x.numel() / t / 1e6:.0f} GB/s (model peak {hw.hbm_gbps:.0f}, achievable {hw.hbm_gbps * hw.mem_eff:.0f})"
)
del x, y
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
t = timed(lambda: a @ a)
print(
    f"8192^3 bf16 GEMM: {2 * 8192**3 / t / 1e9:.0f} TFLOP/s (model peak {hw.peak_tflops_bf16:.0f})"
)
rows.append(
    {"group": "ceiling", "name": "gemm_8192", "measured_ms": t, "tflops": 2 * 8192**3 / t / 1e9}
)
del a

# -- GEMMs at LLaDA-8B / Dream-7B shapes -------------------------------------------
shapes = {
    "llada.qkv": (3 * 4096, 4096),
    "llada.o": (4096, 4096),
    "llada.up_gate": (2 * 12288, 4096),
    "llada.down": (4096, 12288),
    "llada.lm_head": (126464, 4096),
    "dream.qkv": ((28 + 8) * 128, 3584),
    "dream.up_gate": (2 * 18944, 3584),
    "dream.down": (3584, 18944),
}
for m in (32, 256, 1024, 2048, 4096, 8192):
    for name, (n, k) in shapes.items():
        if name.endswith("lm_head") and m > 4096:
            continue
        w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
        xx = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        t = timed(lambda: F.linear(xx, w))
        op = Op(
            name,
            GEMM,
            flops=2.0 * m * n * k,
            bytes_read=(m * k + n * k) * 2,
            bytes_written=m * n * 2,
            gemm=(m, n, k, 1),
        )
        record("gemm", f"{name} M={m}", t, op, m=m, n=n, k=k)
        del w, xx
torch.cuda.empty_cache()


# -- attention ---------------------------------------------------------------------
def attn_case(b, h, hkv, q, kv, d, causal):
    qq = torch.randn(b, h, q, d, dtype=torch.bfloat16, device=dev)
    kk = torch.randn(b, hkv, kv, d, dtype=torch.bfloat16, device=dev)
    vv = torch.randn(b, hkv, kv, d, dtype=torch.bfloat16, device=dev)
    gqa = h != hkv
    t = timed(lambda: F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, enable_gqa=gqa))
    flops = 4.0 * b * h * q * kv * d * (0.5 if causal else 1.0)
    op = Op(
        "attn",
        ATTENTION,
        flops=flops,
        bytes_read=(b * h * q * d + 2 * b * hkv * kv * d) * 2,
        bytes_written=b * h * q * d * 2,
        attn=(b, h, q, kv, d, hkv, causal),
        kv_bytes_per_token=hkv * d * 2 * 2,
    )
    tag = f"b{b} h{h}/{hkv} q{q} kv{kv} {'causal' if causal else 'bidir'}"
    record(
        "attention",
        tag,
        t,
        op,
        b=b,
        h=h,
        hkv=hkv,
        q=q,
        kv=kv,
        d=d,
        causal=causal,
        tflops=flops / t / 1e9,
    )


for b in (1, 8):
    for s in (256, 512, 1024, 2048, 4096):
        for causal in (False, True):
            attn_case(b, 32, 32, s, s, 128, causal)
for b in (1, 8):
    for q, kv in ((32, 544), (32, 1056), (32, 2080), (32, 4128), (512, 1024), (1024, 2048)):
        attn_case(b, 32, 32, q, kv, 128, False)
for s in (512, 1024, 2048):
    attn_case(8, 28, 4, s, s, 128, False)
torch.cuda.empty_cache()

# -- head sampler: fp32 softmax confidence + argmax + top-k ------------------------
for tokens in (256, 1024, 4096, 8192):
    v = 126464
    logits = torch.randn(tokens, v, dtype=torch.bfloat16, device=dev)

    def sampler():
        conf, pred = torch.softmax(logits.float(), dim=-1).max(dim=-1)
        return conf.topk(max(1, tokens // 8)).indices, pred

    t = timed(sampler, iters=10)
    op = Op("unmask_select", MEMORY, bytes_read=2 * tokens * v * 2, bytes_written=tokens * 2)
    record("sampler", f"T={tokens} V={v}", t, op, tokens=tokens, vocab=v)
    del logits
torch.cuda.empty_cache()

# -- FP8 pack/unpack of an activation, one round -----------------------------------
fp8 = (
    torch.float8_e4m3fnuz
    if "fnuz" in str(getattr(torch, "float8_e4m3fnuz", ""))
    else torch.float8_e4m3fn
)
for tokens in (256, 2048, 8192, 32768):
    h = 4096
    act = torch.randn(tokens, h, dtype=torch.bfloat16, device=dev)

    def pack():
        s = act.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) / 240.0
        return (act.float() / s).to(fp8), s

    q8, sc = pack()
    t_pack = timed(pack)
    t_unpack = timed(lambda: (q8.float() * sc).to(torch.bfloat16))
    elements = tokens * h
    model_ms = elements * (2 + 1) * 2 / hw.mem_bytes_per_s() * 1e3 + 2 * hw.kernel_launch_us / 1e3
    rows.append(
        {
            "group": "pack",
            "name": f"fp8 T={tokens} H={h}",
            "measured_ms": t_pack + t_unpack,
            "pack_ms": t_pack,
            "unpack_ms": t_unpack,
            "model_ms": model_ms,
            "tokens": tokens,
        }
    )
    print(
        f"     pack fp8 T={tokens:<6} pack {t_pack:.4f} unpack {t_unpack:.4f}  model {model_ms:.4f} ms ({model_ms / (t_pack + t_unpack):.2f}x)",
        flush=True,
    )
    del act, q8, sc

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(
        {
            "arch": ARCH,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "rows": rows,
        },
        f,
        indent=1,
    )
print("wrote", OUT, file=sys.stderr)
