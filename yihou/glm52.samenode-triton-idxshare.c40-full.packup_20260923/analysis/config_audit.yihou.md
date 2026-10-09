# Config Audit — same-node 1P1D P4D4

| item | argv | server_info | verdict |
|---|---|---|---|
| 1. DSA prefill/decode backend = triton on both legs | prefill: `--dsa-prefill-backend triton --dsa-decode-backend triton`; decode: same | prefill `server_args.dsa_prefill_backend=triton`, `server_args.dsa_decode_backend=triton`; decode same | pass |
| 2. `--dsa-topk-backend` absent / default `sgl-kernel` | flag absent in both argv | `dsa_topk_backend=sgl-kernel` on both legs | pass |
| 3. `index_share_for_mtp_iteration=true` in the running engine | argv carries `--json-model-override-args {"index_share_for_mtp_iteration":true}`; server info echoes the same on both legs | resolved value is `True` in the running engine; the checkpoint `config.json` already carries `True`, so both the override-honoured and override-ignored branches land on the same value. Fresh-container `AutoConfig.from_pretrained(...)` also returns `GlmMoeDsaConfig` / `True`, but that does **not** discriminate the override path. Discriminating test (not run): apply the same override machinery with `false` and expect the resolved attribute to become `False`. | pass |
| 4. HiCache on prefill only | prefill has `--enable-hierarchical-cache --hicache-ratio 1.5 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first`; decode lacks `--enable-hierarchical-cache` | prefill `enable_hierarchical_cache=True`, `hicache_ratio=1.5`, `hicache_write_policy=write_through`, `hicache_io_backend=kernel`, `hicache_mem_layout=page_first`; decode `enable_hierarchical_cache=False`; prefill log shows `Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB host memory` **per rank** (4 ranks), matching the 20260920 cross-node reference exactly | pass |
| 5. Decode-only custom all-reduce disable | decode argv has `--disable-custom-all-reduce`; prefill lacks it | decode `disable_custom_all_reduce=True`; prefill `False` | pass |
| 6. MTP on decode | decode argv has `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`; env has `SGLANG_SIMULATE_ACC_LEN=3.61` | decode `speculative_algorithm=EAGLE`, `speculative_num_steps=5`, `speculative_eagle_topk=1`, `speculative_num_draft_tokens=6` | pass |
| 7. GPU split / TP4 DP4 / DPA | prefill env `HIP_VISIBLE_DEVICES=0,1,2,3`; decode env `HIP_VISIBLE_DEVICES=4,5,6,7` | both legs report `tp_size=4`, `dp_size=4`, `enable_dp_attention=True` | pass |
| 8. Engine ports 29001 / 29257 | prefill `--port 29001`; decode `--port 29257` | `server_info.port` matches on both legs | pass |
| 9. Per-role AITER JIT cache roots differ | prefill `/tmp/aiter-jit-yihou-sn-prefill/...`; decode `/tmp/aiter-jit-yihou-sn-decode/...` | `AITER_JIT_DIR=/aiter-jit` in both containers; host bind roots differ | pass |
| 10. Triton actually in use | live decode process has `/opt/venv/lib/python3.10/site-packages/triton/_C/libtriton.so` mapped, an sglang triton cache under `/root/.cache/sglang/triton/<hash>/`, 175 compiled `.hsaco` kernels under `/root/.cache/sglang`, and `TRITON_CACHE_DIR` unset with `HOME=/root` (container-private cache); resolved DSA backends are triton/triton on both legs | running-engine evidence: triton is loaded and compiling kernels in the live decode process; inference: this strongly supports triton activity for DSA, but the process-level observables do not separately distinguish DSA-triton from other triton users (`mamba_backend`, `linear_attn_backend`) | pass with limit |

## Gaps / notes

- No running-engine log observable was found for item 3; the value is consumed by `disaggregation/utils.py:70-78` and passed into PD metadata (`output_dsa_topk_indices_dim`) without a direct log line.
- This file only records the live state as verified while the benchmark was running; decode was restarted after an early failure, so any round-001 decode log is stale.
- **Prefill allocates TWO distinct host-memory regions per rank**, which are easy to
  confuse and were confused once in an earlier draft of this table:
  `Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB` (the **HiCache**
  pool — the evidence for item 4) and `Allocating 35.25 GB host memory for DSA indexer
  (layout=page_first)` (a **separate** DSA indexer allocation, not HiCache). Four of
  each, one per DP rank ⇒ ~189 GB/rank, ~756 GB total, consistent with the ~757 GB RSS
  observed on the prefill container.
- Item 10's evidence was also corrected once: an earlier draft cited
  `[aiter] import [module_aiter_core]`, which is **AITER, not triton** — AITER is
  enabled in every run in this project including the `tilelang` ones — and read it from
  the **round-001** decode log, which belongs to the attempt that died in the RCCL race
  at 08:13 and never served a request. The live decode leg is the round-002 restart.
