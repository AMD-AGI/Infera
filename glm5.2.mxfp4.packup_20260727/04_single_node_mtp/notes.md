# Notes — 04 single-node MTP (the 3-bug debugging story)

Enabling GLM-5.2 MTP on the **rc6** image hit three bugs in sequence. Only the last two are real
rc6 bugs; the first was a false start from reusing the reference program's patch.

## Bug 1 (false start) — reference's patch is incompatible with rc6

The reference program (`/mnt/vast/jiejing/crusoe_glm_52/`, image **v0.1.1**) ships a patched
`deepseek_nextn.py`. Mounting it on **rc6** → `ValueError: 'DeepseekV3ForCausalLMNextN' is not a
registered model`. **Why:** her file is structurally divergent from rc6's (423 vs 365 lines,
different imports/API) → it fails to import cleanly → the `EntryClass` never registers.
**Lesson:** do NOT reuse a patch across image versions. Patch the version you actually run.

## Bug 2 (REAL) — nextn eh_proj shape crash: `3072 vs 6144`

With rc6 stock (no patch): `RuntimeError: The size of tensor a (3072) must match tensor b (6144)`
in `deepseek_nextn.py` load_weights (6144 = hidden_size).

**Root cause (evidence-backed):** GLM-5.2's quark `exclude` list contains the **submodule-level**
entry `model.layers.78.eh_proj` and does NOT contain the bare `model.layers.78`. rc6 stock checks
the **bare** prefix:

    ckpt_prefix = f"model.layers.{config.num_hidden_layers}"     # = "model.layers.78"

`should_ignore_layer("model.layers.78", exclude)` → **False** → `nextn_quant_config` stays non-None
→ `eh_proj` is built MXFP4-packed (packed dim 3072) while the checkpoint weight is bf16 (6144) →
shape mismatch.

**Fix (1 line, `patches/deepseek_nextn.rc6.diff`):**

    - ckpt_prefix = f"model.layers.{config.num_hidden_layers}"
    + ckpt_prefix = f"model.layers.{config.num_hidden_layers}.eh_proj"

Now the submodule exclude matches → `nextn_quant_config=None` → `eh_proj` built bf16 → loads clean
(`Load weight end ... type=DeepseekV3ForCausalLMNextN quant=quark`). This is a genuine rc6 sglang
bug for any quark checkpoint whose MTP layer is fully bf16 (same idea as the reference's fix, but
applied surgically to rc6 stock rather than replacing the whole file).

## Bug 3 (REAL) — decode hangs on a gfx950-incompatible CUDA JIT kernel

After Bug 2 fixed: draft head loads, server "ready", a single short completion works — but the
correctness probe (repeated calls) **hangs**. The decode-time MTP verification path calls
`sglang.jit_kernel.fused_metadata_copy`, whose source `#include <cuda_runtime.h>` — it will not
JIT on gfx950 (`fatal error: 'cuda_runtime.h' file not found`), so sglang retries the failed
compile on every request → hang.

**Source (`dsa_backend.py`):** the fused-copy block is gated by
`if envs.SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA.get():` (default **True**). Its `>3`-steps branch
uses `fused_metadata_copy_multi_cuda`, its `<=3` branch uses `fused_metadata_copy_cuda` — **both are
CUDA kernels**, so lowering `--speculative-num-steps` does NOT help. The env's `else` branch runs the
plain per-backend replay (no CUDA JIT).

**Fix (env):** `SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0`. After this: probe 4/4, no hang,
accept len 3.5–4.8, ~219 tok/s.

## Why MTP is worth it here

Single-stream decode: ~80 tok/s (no MTP) → ~219 tok/s (EAGLE 5-draft) ≈ **2.7×**. accept len ~3.5–4.8
of 6 drafts means the small nextn head correctly predicts ~4 tokens per verify step, and the big
model accepts them in one parallel pass.

## Reproducibility gap (honest)

The single-node MTP run logged to `docker logs` (container removed at teardown), so no persistent
log file is included. `results/single_stream.txt` has the captured accept-len + throughput. Re-run
`scripts/launch.sh` to regenerate.
