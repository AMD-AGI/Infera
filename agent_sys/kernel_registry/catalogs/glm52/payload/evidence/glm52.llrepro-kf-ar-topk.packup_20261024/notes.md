> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Notes (what / why / how)

## 1. Decode-only hooks added to contributor's kit (worker.sh / common.sh, both in `kit/`)
- `DECODE_MOUNT_EXTRA` / `DECODE_ENV_EXTRA` (space separated K=V): appended to the decode `docker run` AFTER the kit's
  default env list, so a key given here overrides the kit default (`SGLANG_OPT_USE_TOPK_V2=true` in llr7). Verified with
  `docker exec ... env`.
- `DECODE_CUSTOM_AR=1`: removes `--disable-custom-all-reduce` from the decode args only. contributor's kit disables custom AR
  on both legs; the prefill is PP4 x TP1 and has no TP all-reduce. Verified from the server-args line
  (`disable_custom_all_reduce': False` in llr6/llr7, `True` in llr5).

## 2. KF kernel engagement (decode `[infera-kernels]` lines, `results/llr/llr*/decode.keylines.txt`)
- `sitecustomize.py` imports `infera_kernels` when any `INFERA_KERNEL_*=1`; it patches sparse MLA, MoE, indexer logits
  (`dsa_indexer.aiter_paged_mqa_logits`) and indexer top-k (`DSATopKBackend.topk_transform`). Each first call per shape
  logs `rows=N -> KF|orig`; a CHECK compares against the original once (sparse MLA identical, MoE SNR ~51 dB).
- Indexer top-k takes the KF path only if `not should_use_topk_v2()` (code read, `infera_kernels/__init__.py`
  `_patch_indexer_topk`). The kit sets `SGLANG_OPT_USE_TOPK_V2=true`, so in llr5/llr6 every top-k shape logged
  `v2=True -> orig`; llr7 sets it false on the decode and top-k then logs `-> KF` at the same shapes as the logits.
- Indexer rows are TARGET_VERIFY rows = 6 x running requests. KF engaged at rows 60..120; rows > 128 stay original
  (`INDEXER_MAX_ROWS = 128`). Rows 1..48 also stayed original for both logits and top-k - cause NOT investigated
  (one of the `ok` conditions in `_patch_indexer_logits` / `_patch_indexer_topk` fails there; not yet read per shape).
  Rows 60..120 logged both `KF` and `orig` once each; which call took which path is not resolved.
- The share of decode steps per row bucket was not measured; running per rank p50 10, p90 15-17 (rows 60, 90-102).

## 3. Reading the results
- Closed loop: the AgentX mix changes between runs (ISL mean 116.4k -> 120.6k here). Judge by qps, TTFT/TPOT and the
  queue statistics together with the ISL change, not by the score alone.
- Custom AR gave -4.8 % TPOT but only +0.4 % score: running requests per rank fell and prealloc waits rose
  (frac>0 0.15 -> 0.26), i.e. the queue moved upstream. That is an observation; the mechanism is not established.
- Correctness is NOT checked: sim acceptance 3.61; custom AR and KF were enabled performance-first by the user's order.
  A real-acceptance run + output comparison is still owed before any of this is called correct.

## 4. Open / next
- llr8 = llr7 + decode mem 0.95 (running at pack-up; llr2 showed 0.95 is not a bad point on this base).
- KF indexer at rows <= 48 (cause), rows > 128 (share of steps first; then ask the kernel session for a larger RMAX).
- Decode CPU thread at 100 % on REDACTED_NODE: py-spy dump + GPU idle gaps from a trace before acting.
- Prefill / transfer: per-stage breakdown of the ~2 s KV latency; rail 5 carries ~55 % of rail 4 (120 s average).
- Router R2 (decode demand) on this kit: our ppda3 R2 cut TTFT p90 23 -> 10 s; the contributor router's support is unchecked.
