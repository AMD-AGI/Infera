"""Simulated vs measured (host.py) small-q steps against a cache."""

import json
import sys

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    SeqShape,
    load_model,
)

hw = DLLMHardware.from_arch("mi325x", gpu_clock_mhz=1510)
for path, preset in zip(sys.argv[1::2], sys.argv[2::2]):
    rows = json.load(open(path))["rows"]
    m = DLLMStepModel(load_model(preset), Parallel(), hw, runtime="hf_eager")
    print("==", preset)
    for r in rows:
        b = m.step([SeqShape(r["q"], r["kv"], r["q"])] * r["batch"])
        d = b.as_dict()
        cats = {
            k: round(v, 2)
            for k, v in d.items()
            if k.endswith("_ms") and isinstance(v, float) and v > 0.05
        }
        print(
            f"  b{r['batch']} q{r['q']:4d}: busy {r['busy_ms']:6.2f} sim {b.total_ms:6.2f} ({b.total_ms / r['busy_ms']:.2f}x)  kernels {r['kernels']}  {cats}"
        )
