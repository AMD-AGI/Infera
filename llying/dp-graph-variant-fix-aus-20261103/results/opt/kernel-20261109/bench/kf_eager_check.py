import importlib.util, sys
import torch
sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import moe_bench as mb
spec = importlib.util.spec_from_file_location("kf_moe", "/yihou_overlay/yihou_kf/kf_moe.py")
kf = importlib.util.module_from_spec(spec); spec.loader.exec_module(kf)
for inter, M in ((512, 288), (512, 448), (256, 384)):
    ws = [mb.make_weights(inter, 7 + i) for i in range(2)]
    x, ids, tw = mb.make_inputs(M, 1234 + M)
    a0 = mb.aiter_call(x, ws[0], ids, tw).clone()
    k0 = kf.glm52_moe_mxfp4(x, *ws[0], tw, ids)
    k0c = k0.clone()
    k1 = kf.glm52_moe_mxfp4(x, *ws[1], tw, ids)
    print(f"inter {inter} M {M}: snr(kf0 vs aiter0) {mb.snr(a0, k0c):.1f} dB; kf0 buffer reused by next call: "
          f"{k0.data_ptr() == k1.data_ptr()}; snr(kf0 after next call vs aiter0) {mb.snr(a0, k0):.1f} dB", flush=True)
