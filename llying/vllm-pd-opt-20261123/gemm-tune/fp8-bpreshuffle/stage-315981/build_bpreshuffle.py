import os, torch
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_ck, gemm_a8w8_bpreshuffle_cktile
from aiter.jit.core import AITER_CONFIGS
print("merged table", AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE, flush=True)
M, N, K = 64, 6144, 16384
x = torch.randn(M, K, device="cuda").to(dtypes.fp8)
w = shuffle_weight((torch.randn(N, K, device="cuda") * 0.1).to(dtypes.fp8), layout=(16, 16))
xs, ws = torch.ones(M, 1, device="cuda"), torch.ones(N, 1, device="cuda")
y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
for f in (gemm_a8w8_bpreshuffle_ck, gemm_a8w8_bpreshuffle_cktile):   # first call JIT-builds the module
    try:
        f(x, w, xs, ws, y, 0); torch.cuda.synchronize(); print(f.__name__, "ran", flush=True)
    except Exception as e:
        print(f.__name__, "call failed after build:", type(e).__name__, str(e)[:200], flush=True)
