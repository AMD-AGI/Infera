"""FP8 collectives at refinement-step payloads on the MI325X mesh: per-layer ms."""

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    Parallel,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.comm import (
    CommConfig,
    DLLMCommModel,
)

spec = load_model("llada_8B")
hw = DLLMHardware.from_arch("mi325x", backend="roofline")
print("tokens = batch x canvas rows per all-reduce; dense layer = 2 all-reduces")
for tp in (2, 4, 8):
    for pack in ("fused", "eager"):
        cm = DLLMCommModel(spec, Parallel(tp=tp), hw, CommConfig("fp8", "all", pack=pack))
        cells = []
        for tok in (32, 512, 4096, 16384, 65536):
            r = cm.tradeoff("dense", tok, "fp8")
            cells.append(f"{tok:>6}: {r['plain_ms']:6.3f}->{r['quantized_ms']:6.3f}")
        print(f"TP{tp} {pack:5s} | " + " | ".join(cells))
