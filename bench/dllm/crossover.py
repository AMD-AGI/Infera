"""When does a dLLM beat AR? Simulated break-even tokens per step.

AR = block diffusion with one-position blocks over an exact cache, so one step
model prices both. For each cache mode and batch, the smallest tokens/step
(gen_len / steps) at which the dLLM's latency for gen_len tokens drops below
AR's, and the speedup at 2 and 4 tokens/step.
"""

import json
import sys

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.simulator import simulate_static
from infera.projection.core.projection.inference_projection.dllm.workload import DLLMWorkload

P, G, BLOCK = 512, 256, 32
spec = load_model("dream_7B")
out = []
for arch, clock in (("mi325x", 1510), ("mi355x", None)):
    hw = DLLMHardware.from_arch(arch, gpu_clock_mhz=clock)
    for runtime in ("engine", "hf_eager"):
        m = DLLMStepModel(spec, Parallel(), hw, runtime=runtime)
        print(f"== {arch} {runtime}  (P={P}, G={G}, block={BLOCK})")
        for batch in (1, 8, 32, 128):
            ar = simulate_static(
                m, DLLMWorkload(P, G, block_len=1, steps=G, cache="block"), batch
            ).latency_ms
            line = f"  b{batch:4d}  AR {ar:8.0f} ms |"
            for cache in ("none", "prefix", "dual", "block"):
                be = None
                sp = {}
                for steps in (256, 128, 64, 32, 16, 8):
                    lat = simulate_static(
                        m, DLLMWorkload(P, G, block_len=BLOCK, steps=steps, cache=cache), batch
                    ).latency_ms
                    tps = G / steps
                    sp[tps] = ar / lat
                    if be is None and lat <= ar:
                        be = tps
                out.append(
                    {
                        "arch": arch,
                        "runtime": runtime,
                        "batch": batch,
                        "cache": cache,
                        "ar_ms": ar,
                        "speedup": sp,
                        "breakeven_tok_per_step": be,
                    }
                )
                line += f" {cache}: be {be if be else '>32':>4} x{sp[4.0]:.2f}@4 |"
            print(line, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
