#!/usr/bin/env python3
"""Check source patch markers and imports without requiring a GPU at build time."""
from pathlib import Path
import subprocess
import infera
import sglang

root = Path(sglang.__file__).parent
checks = {
    "srt/models/glm4_moe.py": 'fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"',
    "kernels/jit/csrc/kvcacheio/hicache.cuh": "pick_group_bytes",
    "kernels/ops/kvcache/hicache.py": "_tiles_across_lanes",
    "srt/mem_cache/pool_host/mha.py": "can_use_jit = (_is_cuda or _is_hip)",
}
for relative, marker in checks.items():
    if marker not in (root / relative).read_text():
        raise SystemExit(f"missing patch marker: {relative}: {marker}")
subprocess.run(["/usr/local/bin/infera-router", "--help"], check=True, stdout=subprocess.DEVNULL)
print(f"markers=ok sglang={sglang.__version__} infera={infera.__file__}")
print("Router affinity is not part of this overlay; use independent P/D ranks.")
