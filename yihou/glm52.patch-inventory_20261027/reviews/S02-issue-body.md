### Description

Current SGLang `8a455f1b9005a05091ad1825f0f3493bfd74d27d` still generates an absorb-BMM hand configuration that aborts the gfx950 Triton/LLVM compiler at M=256. Reproduced today on an allocated MI355X; the same operation with `config=None` completes with finite output.

This is a bug report, not a PR disabling all absorb-BMM tuning. The archived workaround returned None unconditionally; a scoped configuration/compiler fix is needed instead.

### Environment

- MI355X (gfx950), single-GPU reproducer, October 27.
- SGLang source: `8a455f1` (current-main port checkout; the gather PR on that checkout does not touch this operation).
- PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015`, Triton `3.7.0`.
- Aiter image commit `e7d2453f25e5aaeea72a3a40b284ba453e026624`.
- Compared with Aiter main `e270b8427f089e57cdfc67f2404b366724ed4abb`: this Triton kernel file is byte-identical. The current wrapper adds gfx1250 Gluon dispatch; gfx950 with an explicit config retains the Triton path. A full latest-Aiter/latest-Triton installation has not been retested.

### Reproduction

Run each case in a separate process, with a writable Triton cache. `default` passes; `hand` aborts. Disable core dumps if desired with `ulimit -c 0`.

```bash
PYTHONPATH=python python repro.py default
PYTHONPATH=python python repro.py hand
```

```python
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm as module

print('torch', torch.__version__, 'triton', triton.__version__, 'GPU', torch.cuda.get_device_name(), flush=True)
# Retain the model's strided Q view: non-RoPE width 192 inside a wider per-head row.
q = torch.randn((256, 64, 256), device='cuda', dtype=torch.bfloat16)[:, :, :192]
w = torch.randn((64, 192, 512), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
attn = SimpleNamespace(w_kc=w, w_scale=torch.ones((64,), device='cuda', dtype=torch.float32))
print('mode', sys.argv[1], 'config', module._absorb_bmm_config(64, 256, 512), flush=True)
if sys.argv[1] == 'default':
    with patch.object(module, '_absorb_bmm_config', return_value=None):
        result = module.rocm_absorb_q_bmm(attn, q, is_capture_mode=False)
else:
    result = module.rocm_absorb_q_bmm(attn, q, is_capture_mode=False)
torch.cuda.synchronize()
print('PASS', tuple(result.shape), bool(result.isfinite().all()), flush=True)

```

The generated configuration is:

```python
{'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 1,
 'num_warps': 4, 'num_stages': 2, 'waves_per_eu': 4,
 'matrix_instr_nonkdim': 16, 'cache_modifier': '.cg'}
```

Observed default result: `PASS (64, 256, 512) True`. Hand-config failure:

```text
llvm/ADT/Sequence.h:275: llvm::iota_range<unsigned int>::iota_range(...):
Assertion `Begin <= End && "Begin must be less or equal to End."' failed.
```

The individual compiler-option trigger has not been isolated; `waves_per_eu=4` is a hypothesis, not a proven root cause. The current SGLang config is the independently varied input in the reproducer.

### Historical evidence / limits

GLM-5.2 PD run `c44b`, ledger R003, first hit this compiler abort during warmup. The archived sweep reproduced it at M=256 while M=2/16/64/128 passed; `c44c` bypassed the configuration and completed warmup. Today's one-case reproduction confirms that the failure persists in the stated software combination. No simulated-acceptance throughput or stack performance is used as evidence.
