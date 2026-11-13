"""AR baseline: HF generate on Qwen2.5-7B-Instruct, measured and simulated.

AR decoding is block diffusion with one-position blocks and an exact cache:
prefill the prompt, then one q=1 step per token.
"""

import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.simulator import simulate_static
from infera.projection.core.projection.inference_projection.dllm.workload import DLLMWorkload

out_path = sys.argv[1]
P, G = int(os.getenv("P", "256")), int(os.getenv("G", "256"))
model = (
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
    .cuda()
    .eval()
)
sim = DLLMStepModel(
    load_model("dream_7B"),
    Parallel(),
    DLLMHardware.from_arch("mi325x", gpu_clock_mhz=1510),
    runtime="hf_eager",
)
wl = DLLMWorkload(prompt_len=P, gen_len=G, block_len=1, steps=G, cache="block")
rows = []
for b in (1, 8, 32):
    ids = torch.randint(100, 150000, (b, P), device="cuda")
    kw = dict(max_new_tokens=G, min_new_tokens=G, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        model.generate(ids, **{**kw, "max_new_tokens": 8, "min_new_tokens": 8})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(ids, **kw)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    s = simulate_static(sim, wl, b)
    row = {
        "batch": b,
        "meas_ms": ms,
        "sim_ms": s.latency_ms,
        "ratio": s.latency_ms / ms,
        "meas_tok_s": b * G / ms * 1e3,
        "sim_tok_s": s.tokens_per_s,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump({"P": P, "G": G, "rows": rows}, open(out_path, "w"), indent=1)
