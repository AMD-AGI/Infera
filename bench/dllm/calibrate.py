"""Fit the hf_eager element-wise chain on one model's kernel census.

measured_elementwise(shape) = kernels(shape) * floor_us + m * modeled_chain_ms(shape)
"""

import json
import sys

import numpy as np

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    Parallel,
    SeqShape,
    build_stage_graph,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.runtime import _CHAIN

prof_path, preset = sys.argv[1], sys.argv[2]
rows = json.load(open(prof_path))
spec = load_model(preset)
hw = DLLMHardware.from_arch("mi325x", backend="roofline")
X, y = [], []
for r in rows:
    g = build_stage_graph(
        spec,
        [SeqShape(r["seq"], r["seq"], r["seq"])] * r["batch"],
        Parallel(),
        first=True,
        last=True,
    )
    chain_ms = sum(o.memory_ms(hw) * o.count for o in g.ops if o.name in _CHAIN)
    n = r["by_cat_n"]["elementwise"]
    X.append([n / 1e3, chain_ms])
    y.append(r["by_cat_ms"]["elementwise"])
X, y = np.array(X), np.array(y)
(floor_us, m), *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ np.array([floor_us, m])
print(
    f"{preset}: floor {floor_us:.2f} us/kernel, traffic x{m:.2f}, kernels/layer {X[0, 0] * 1e3 / spec.num_layers:.1f}"
)
for r, p, t in zip(rows, pred, y):
    print(f"  b{r['batch']} s{r['seq']}: elementwise meas {t:7.2f} fit {p:7.2f}")
