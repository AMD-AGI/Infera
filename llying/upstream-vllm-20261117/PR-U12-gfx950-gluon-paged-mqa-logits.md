# [ROCm][Perf] DSA indexer: gfx950 gluon paged MQA logits for decode / MTP-verify rows

Draft for vLLM upstream (candidate U12). Status: function-level test passes; aggregated gate and decode A/B pending
(`../vllm-kf-port-20261117/`).

## Summary

On gfx950, `rocm_fp8_paged_mqa_logits` (the DSA indexer's decode scoring) calls aiter's
`deepgemm_fp8_paged_mqa_logits`. With MTP, ROCm decode is flattened (`indexer._use_flattening`: one query per
verify row, per-row lengths and block-table rows). This PR adds an opt-in path
(`VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1`) that runs a re-tuned gluon kernel instead, ported from
sglang PR #47897 (itself derived from aiter's `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`):

- KV split decoupled from the occupancy hint, program ids swizzled so the pieces of one row share an XCD (L2);
- the MTP verify rows of one request form a longest-first dispatch group;
- sibling rows with identical page tables are scored by one program (checked on device, exact for any input);
- gfx950 full-rate scaled fp8 MFMA.

The kernel scores one query per row, which is what flattened decode already passes; the LPT group is the request's
verify rows (1 + num_speculative_tokens). For a `[B, next_n]` caller the wrapper expands per-row lengths and
block-table rows (`repeat_interleave`, no host sync, CUDA-graph capturable). It serves 1–128 rows on the 64-token
preshuffled indexer cache with an even group of 2–8 rows; every other case keeps aiter.

## Performance (MI355X, GLM-5.2 indexer: 32 heads × 128, fp8, 64-token pages, next_n 4, contexts 40K–200K)

| requests (rows) | aiter, flattened (current vLLM) µs | this PR µs | speed-up |
|---|---|---|---|
| 4 (16) | 39.0 | 15.3 | 2.54× |
| 8 (32) | 96.1 | 39.9 | 2.41× |
| 16 (64) | 163.4 | 69.3 | 2.36× |
| 26 (104) | 247.9 | 110.8 | 2.24× |
| 32 (128) | 324.9 | 139.1 | 2.34× |

End-to-end (DP4, MTP3): TBD.

## Correctness

- Logits on every row's visible columns match aiter to 7.6e-6 absolute (logits magnitude 50–70), for flattened rows
  (1-D and `[rows, 1]` lengths) and `[B, next_n]` callers (1-D and 2-D lengths), 1–32 requests; 40 requests (160 rows) fall back to aiter; CUDA graph capture + replay with new
  contents matches.
- GSM8K / long-context gate with the path on: TBD.

## Notes

- Kernel file keeps its MIT header (AMD); attribution to sglang #47897 in the module docstring.
- Next candidates on the same path: single-pass Triton top-k for long decode rows (sglang #47872), fused indexer
  logits + top-k for verify rows (sglang #47276).
