# [ROCm][Perf] DSA indexer: gfx950 gluon paged MQA logits for decode / MTP-verify rows

Draft for vLLM upstream (candidate U12). Status: function-level test passes; aggregated gate and decode A/B pending
(`../vllm-kf-port-20261117/`).

## Summary

On gfx950, `rocm_fp8_paged_mqa_logits` (the DSA indexer's decode scoring) calls aiter's
`deepgemm_fp8_paged_mqa_logits` on `[B, next_n]` queries. This PR adds an opt-in path
(`VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1`) that runs a re-tuned gluon kernel instead, ported from
sglang PR #47897 (itself derived from aiter's `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`):

- KV split decoupled from the occupancy hint, program ids swizzled so the pieces of one row share an XCD (L2);
- the MTP verify rows of one request form a longest-first dispatch group;
- sibling rows with identical page tables are scored by one program (checked on device, exact for any input);
- gfx950 full-rate scaled fp8 MFMA.

The kernel scores one query per row, so the wrapper gives every verify row its own context length (taken from a
`[B, next_n]` `context_lens` when the caller provides it, otherwise derived from the per-request length) and its own
block-table row (`repeat_interleave`, no host sync, CUDA-graph capturable). It serves 1–128 rows on the 64-token
preshuffled indexer cache with 2 ≤ next_n ≤ 8; every other case keeps aiter.

## Performance (MI355X, GLM-5.2 indexer: 32 heads × 128, fp8, 64-token pages, next_n 4, contexts 40K–200K)

| requests (rows) | aiter µs | this PR µs | speed-up |
|---|---|---|---|
| 4 (16) | 28.7 | 15.3 | 1.88× |
| 8 (32) | 83.5 | 40.5 | 2.06× |
| 16 (64) | 140.1 | 71.2 | 1.97× |
| 26 (104) | 251.2 | 110.3 | 2.28× |
| 32 (128) | 272.0 | 139.3 | 1.95× |

Building the per-row inputs costs ~28 µs per call. End-to-end (DP4, MTP3): TBD.

## Correctness

- Logits on every row's visible columns match aiter to 7.6e-6 absolute (logits magnitude 50–70), for 1-D and 2-D
  `context_lens`, 1–32 requests; 40 requests (160 rows) fall back to aiter; CUDA graph capture + replay with new
  contents matches.
- GSM8K / long-context gate with the path on: TBD.

## Notes

- Kernel file keeps its MIT header (AMD); attribution to sglang #47897 in the module docstring.
- Next candidates on the same path: single-pass Triton top-k for long decode rows (sglang #47872), fused indexer
  logits + top-k for verify rows (sglang #47276).
