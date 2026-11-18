# [ROCm][Perf] DSA indexer: gfx950 gluon paged MQA logits for decode / MTP-verify rows

Draft for vLLM upstream (candidate U12), updated from `../upstream-vllm-20261117/PR-U12-gfx950-gluon-paged-mqa-logits.md`
with the end-to-end data of track A (`README.zh-CN.md` §2) and the metadata-driven group size (overlay v3).

## Summary

On gfx950, `rocm_fp8_paged_mqa_logits` (the DSA indexer's decode scoring) calls aiter's
`deepgemm_fp8_paged_mqa_logits`. With MTP, ROCm decode is flattened (`indexer._use_flattening`: one query per
verify row, per-row lengths and block-table rows). This PR adds an opt-in path
(`VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1`, registered in `vllm/envs.py`) that runs a re-tuned gluon kernel instead,
ported from sglang PR #47897 (itself derived from aiter's `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`):

- KV split decoupled from the occupancy hint, program ids swizzled so the pieces of one row share an XCD (L2);
- the MTP verify rows of one request form a longest-first dispatch group;
- sibling rows with identical page tables are scored by one program (checked on device, exact for any input);
- gfx950 full-rate scaled fp8 MFMA.

The kernel scores one query per row, which is what flattened decode already passes. The dispatch group is a
request's verify rows: `rocm_aiter_sparse_attn_indexer` passes `flat_group_rows` from the indexer decode metadata
(`write_max_decode_len` when `decode_is_uniform`, else 0), so the size is known where the batch is built and needs no
global config lookup at forward time. For a `[B, next_n]` caller the wrapper expands per-row lengths and block-table
rows (`repeat_interleave`, no host sync, CUDA-graph capturable). It serves 1–128 rows on the 64-token preshuffled
indexer cache with an even group of 2–8 rows; every other case (including non-uniform steps) keeps aiter.

## Performance

Kernel (MI355X, GLM-5.2 indexer: 32 heads × 128, fp8, 64-token pages, MTP3 flattened rows, contexts 40K–200K):

| requests (rows) | aiter, flattened (current vLLM) µs | this PR µs | speed-up |
|---|---|---|---|
| 4 (16) | 39.0 | 15.3 | 2.54× |
| 8 (32) | 96.1 | 39.9 | 2.41× |
| 16 (64) | 163.4 | 69.3 | 2.36× |
| 26 (104) | 247.9 | 110.8 | 2.24× |
| 32 (128) | 324.9 | 139.1 | 2.34× |

End-to-end (GLM-5.2 MXFP4, DP4 TP1, MTP3, fp8 KV, one MI355X node, 104 requests × 64K input × 4K output, decode-only
window after all prefills, single run):

| | decode step p50 | mean | p99 | indexer logits per call (engine trace) | per step per rank |
|---|---|---|---|---|---|
| aiter (current) | 52.93 ms | 52.99 | 54.50 | 138–146 µs | 3.04–3.21 ms |
| this PR | **50.89 ms (−3.9%)** | 50.90 | 52.61 | 68.3 µs | 1.50 ms |

The gain grows when DP ranks have unequal context lengths (the rank with the longest contexts sets every step).

## Correctness

- Logits on every row's visible columns match aiter to 7.6e-6 absolute (logits magnitude 50–70), for flattened rows
  (1-D and `[rows, 1]` lengths) and `[B, next_n]` callers (1-D and 2-D lengths), 1–32 requests; 40 requests (160 rows)
  fall back to aiter; CUDA graph capture + replay with new contents matches, for both the `[B, next_n]` and the
  flattened form.
- GSM8K 200 / 16K long-context gate with the path on (DP4 MTP3): 196/200, 32/32 + 31/32, MTP acceptance length 3.56;
  stock run side by side: 197/200, 32/32 + 32/32, 3.44 (stock A/A runs: 195–199).

## Notes

- Kernel file keeps its MIT header (AMD); attribution to sglang #47897 in the module docstring.
- Before this change the group size came from `get_current_vllm_config()` at forward time, where workers have no
  config context: it raised, the fallback returned 0 and `functools.cache` kept it, so the path silently never ran.
  Passing the size from the metadata removes that failure mode.
- Next candidates on the same path: single-pass Triton top-k for long decode rows (sglang #47872; at 64K contexts
  aiter's radix one-block top-k is already 32 µs per call), fused indexer logits + top-k for verify rows (sglang #47276).
