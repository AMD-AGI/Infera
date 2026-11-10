#!/usr/bin/env python3
"""KF MoE against aiter on real GLM-5.2-MXFP4 weights (layer 3, the TP4 rank-0 slice of every expert's intermediate
dim, shared expert in slot 256) and real router top-k, to settle whether the 4 dB KF / aiter gap on random weights
(kf_eager_check.py) is the kernel or the synthetic inputs."""
import importlib.util
import json
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import moe_bench as mb  # noqa: E402

M_DIR = "/perf_apps/data/models/GLM-5.2-MXFP4"
LAYER, TP, RANK = 3, int(sys.argv[1]) if len(sys.argv) > 1 else 4, 0


def tensors(names):
    wm = json.load(open(f"{M_DIR}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for f in sorted({wm[n] for n in names}):
        with safe_open(f"{M_DIR}/{f}", framework="pt", device="cpu") as h:
            for n in names:
                if wm[n] == f:
                    out[n] = h.get_tensor(n)
    return out


def main():
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    inter = 2048 // TP
    r0, r1 = RANK * inter, (RANK + 1) * inter
    p = f"model.layers.{LAYER}.mlp"
    experts = [f"{p}.experts.{e}" for e in range(256)] + [f"{p}.shared_experts"]
    names = [f"{x}.{w}_proj.{k}" for x in experts for w in ("gate", "up", "down") for k in ("weight", "weight_scale")]
    names += [f"{p}.gate.weight", f"{p}.gate.e_score_correction_bias"]
    t = tensors(names)
    w13 = torch.stack([torch.cat([t[f"{x}.gate_proj.weight"][r0:r1], t[f"{x}.up_proj.weight"][r0:r1]]) for x in experts])
    s13 = torch.stack([torch.cat([t[f"{x}.gate_proj.weight_scale"][r0:r1], t[f"{x}.up_proj.weight_scale"][r0:r1]])
                       for x in experts])
    w2 = torch.stack([t[f"{x}.down_proj.weight"][:, r0 // 2:r1 // 2] for x in experts])
    s2 = torch.stack([t[f"{x}.down_proj.weight_scale"][:, r0 // 32:r1 // 32] for x in experts])
    print("shapes", tuple(w13.shape), w13.dtype, tuple(s13.shape), tuple(w2.shape), tuple(s2.shape), flush=True)
    E = w13.shape[0]
    w13, s13, w2, s2 = (x.contiguous().cuda() for x in (w13, s13, w2, s2))
    s13 = e8m0_shuffle(s13.reshape(E * 2 * inter, -1)).view(E, 2 * inter, -1)
    s2 = e8m0_shuffle(s2.reshape(E * 6144, -1)).view(E, 6144, -1)
    w13 = shuffle_weight(w13, (16, 16)).view(torch.float4_e2m1fn_x2)
    w2 = shuffle_weight(w2, (16, 16)).view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = w2.is_shuffled = True
    ws = (w13, w2, s13, s2)
    gate = t[f"{p}.gate.weight"].cuda().to(torch.bfloat16)
    bias = t[f"{p}.gate.e_score_correction_bias"].cuda().float()
    kf = None
    if TP == 4:
        spec = importlib.util.spec_from_file_location("kf_moe", "/yihou_overlay/yihou_kf/kf_moe.py")
        kf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kf)
    for M in ((160, 288, 448) if TP == 4 else (384, 704)):
        g = torch.Generator(device="cuda").manual_seed(M)
        x = torch.randn(M, 6144, device="cuda", dtype=torch.bfloat16, generator=g)
        s = torch.sigmoid((x @ gate.t()).float())
        ids = (s + bias).topk(8, dim=1).indices.to(torch.int32)
        w = s.gather(1, ids.long())
        w = w / w.sum(1, keepdim=True) * 2.5
        ids = torch.cat([ids, torch.full((M, 1), 256, dtype=torch.int32, device="cuda")], 1).contiguous()
        tw = torch.cat([w, torch.ones(M, 1, device="cuda")], 1).contiguous()
        a = mb.aiter_call(x, ws, ids, tw).float().clone()
        a2 = mb.aiter_call(x, ws, ids, tw).float().clone()
        msg = f"TP{TP} M {M}: aiter run-to-run {mb.snr(a, a2):.1f} dB"
        if kf is not None:
            k = kf.glm52_moe_mxfp4(x, *ws, tw, ids).float().clone()
            msg += f", KF vs aiter {mb.snr(a, k):.1f} dB"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
