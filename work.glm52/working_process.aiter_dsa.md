# aiter as DSA main backend — v0.5.16 debug (chi2835, 2026-07-29)

Context: backend.md §已验证 (A1–A5) concluded "the aiter DSA attention kernel itself
GPU-faults on gfx950/GLM-5.2; tilelang is the only correct backend". Re-tested on
`lmsysorg/sglang:v0.5.16-rocm720-mi35x`. Node chi2835 (10.2.122.78), 8x MI355X gfx950.

## STATUS: root cause NOT yet found. Fault reproduced and localized; not explained.

## What is now firmly established

### 1. aiter DOES support DSA — the backend.md claim is wrong
- `DSA_CHOICES` includes `aiter`; sglang has full `_forward_aiter` /
  `_forward_aiter_extend` / persistent-metadata plumbing.
- aiter ships gfx950 sparse-MLA asm kernels. `hsa/gfx950/mla/mla_asm.csv` (45 rows) has
  bf16q/fp8kv gqa16 ps, fp8/fp8 gqa 8/16/32/64, and `mla_pfl_qh192_vh128_*`.
- The GLM-5.2 shape RESOLVES a real kernel; the log shows
  `LoadKernel: _ZN5aiter41mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_psE`.
- The current tilelang recipe ALREADY runs aiter for the DSA indexer (paged MQA logits,
  `indexer_k_quant_and_cache`, fused QK-norm/RoPE/quant) — backend.md §一 says so itself.
- sglang PR #18526 publishes MI355X numbers where aiter DSA beats tilelang
  (DSv3.2-Exp 8192/1024, conc=64: 5283 vs 3759 tok/s; TPOT 92.4 vs 125.5 ms).

### 2. Environment (probed in-image)
sglang 0.5.16 / torch 2.9.1+rocm7.2.0 / triton 3.6.0 / tilelang 0.1.7.post3;
gfx950, cu_num=256; `is_experimental_enabled()==False` (so the `hk_mla_*` family is OFF);
v0.5.16 auto-sets `SGLANG_OPT_USE_TOPK_V2=False` for the DSA family on HIP (PR #30506).
ROCm DSA default = tilelang, applied ONLY when neither prefill nor decode is user-set
(`not user_set_prefill and not user_set_decode and is_hip()`) -> always set both flags.

### 3. The fault, precisely localized (Round B1, in-place trace patch)
GLM-5.2 + `--dsa-*-backend aiter` + exp01 recipe faults ~120 s in, during decode
CUDA-graph capture. `__DSA_TRACE__` instrumentation of `_forward_aiter` gives the exact
last call before the fault:

    prep bs=48 maxq=1 cap=64 qo=(49,) kvindptr=(65,) nhead_pad=16 topk=2048
    dec  bs=48 q=(48,16,576) o=(48,16,512) kv=(3552256,1,576) head_dim=576 v=512
         cu_q=(49,) kvindptr=(65,) kvidx=(131072,) maxq=1 pt=(48,2048)

Capture order is 64 -> 56 -> 48; **64 and 56 succeed, 48 faults**, on all 8 GPUs
("Write access to a read-only page" / "Unknown").

### 4. Hypotheses RAISED AND REFUTED (all by direct evidence)
| # | hypothesis | how it died |
|---|---|---|
| H1 | FlyDSL bf16 GEMM configs | v0.5.16 log shows every GEMM `using torch solution:0`; FlyDSL never fires, still faults |
| H2 | `layer.head_dim`(192) vs pool dim(576) mismatch in `kv_cache.view` | `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` -> absorbed MLA layer, trace confirms `head_dim=576`, matches pool. View is correct |
| H3 | uint8-packed KV layout vs raw | `calculate_mla_kv_cache_dim` explicitly returns raw 512+64 for HIP+aiter/tilelang |
| H4 | metadata buffer sized at max_bs but called with smaller bs | strict orthogonal scan (fresh process per pair): meta_bs=64 x call_bs {56,48,40,32,16,8,1} -> **ALL OK** |
| H5 | fix `capacity >= batch_size` -> `== batch_size` | applied to the real server (Round B2): **still faults at the same bs=48** |
| H6 | stale `kv_indices` tail | zeroing it each round changed nothing |
| H7 | stale `kv_indptr` tail | zeroing it each round changed nothing |
| H8 | oversized `cu_seqlens_q` making aiter infer the wrong bs | source shows it is correctly sliced `[:bs+1]`; trace confirms `cu_q=(49,)` for bs=48 |

### 5. Micro-repro status: reproduces nothing (important negative)
`repro5_exact.py` replays the traced call EXACTLY — same shapes, oversized
`kv_indptr`(65) / `kv_indices`(131072) persistent buffers, same 1.91 GiB KV pool, same
capture sequence 64->56->48 in one process — and **passes**. Single-GPU, no server.
So the fault needs something the isolated kernel path does not have: 8-GPU/TP8 context,
real KV contents, the CUDA-graph capture stream, or the surrounding allocator state.

NOTE: an earlier repro (`repro2_aiter_dsa.py`) DID fault at bs=40 and led me to H4. That
repro had its own out-of-bounds bug (kv_indices buffer vs indptr tail); its faults were
an artifact, not the sglang path. H4's "REBUILD fixes it" evidence came from that broken
script and must be discarded. Round B2 independently disproved H4 on the real server.

## Next steps (for whoever picks this up)
1. The fault is inside CUDA-graph capture. Try `--disable-cuda-graph` on v0.5.16:
   A5 said eager also faults, but that was the OLD image — worth one clean re-test,
   because it splits "capture-specific" from "always".
2. Vary only `--cuda-graph-max-bs` (e.g. 56, 48, 32). If the fault always lands on the
   3rd captured bs rather than on the value 48, it is sequence/state dependent, not
   shape dependent.
3. TP4 (16 q-heads/GPU -> `need_pad_heads=False`, native gqa16 path, no
   `repeat_interleave`) as a single-variable test: it removes the padding path entirely.
4. Only after that, consider filing upstream — with the `__DSA_TRACE__` shapes, which is
   the reusable artifact from this session.

## Reusable artifacts (work.glm52/)
- `probe2_v0516.sh` — versions, gfx950 kernel table, env resolution (zero GPU alloc)
- `dsa_patch_trace.py` — in-place patch printing exact aiter call args; **this is what
  finally localized the fault**. Prefer it over sitecustomize/runpy hooks.
- `launch_v0516.sh` / `watch_v0516.sh` — launch + verdict poller (HEALTHY/GPU_FAULT/TIMEOUT)
- `repro3_matrix.py` (one pair per process), `repro5_exact.py` (traced shapes)
- `fix_dsa_exact_bs.py` — H5 patch; **does not fix it**, kept as a negative result

## Traps hit
- `get_valid_kv_indices` moved to `sglang.kernels.ops.attention.dsa.triton_kernel` in v0.5.16.
- `import aiter` needs `--device /dev/kfd --device /dev/dri` (it shells out to rocminfo).
- A `sitecustomize.py` that imports aiter **hangs the server at startup** (conflicts with
  sglang's own init); a `runpy`-based wrapper breaks multiprocessing spawn. Patch the
  source file in place instead.
- /apps/xinyi/sglang is an old tree with no GLM-DSA; read source from the image.
