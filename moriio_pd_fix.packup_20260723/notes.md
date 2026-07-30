# Notes — debug narrative, wrong turns, gotchas

Distilled from `working_process_raw.md` (the raw iteration log). Read this before
re-attacking anything MoRIIO — it will save you from re-walking dead hypotheses.
Format: what / why / how / context.

## The single most important gotcha

**`vllm bench serve` NEVER verifies output correctness.** A "CLEAN N/N, X tok/s"
result can be pure gibberish — it sends random token IDs and only counts
completions. ALWAYS run a temp=0 France→Paris probe through the router before
believing any PD run works. Every "correctness" claim in this kit is a temp=0 probe.

## The debug loop that cracked it (method)

iterative-debug-loop + **对拍 (differential) vs Mooncake**, which is correct on the
SAME nodes/model/config — only the connector differs. Change ONE thing per pass;
temp=0 check every run; instrument, don't guess. The winning move was a `DBGSPEC`
dump comparing, per layer, the connector's `geom.block_len` vs the spec's
`page_size_bytes` — a single table showed block_len SHORT for every cache.

## Hypotheses tried and RULED OUT (in order)

1. **Router wire-shape drift** (infera's forged request_id vs v0.25.1 regex).
   RULED OUT: `_PREFILL/_DECODE_ZMQ_RE` match infera's forge; `add_new_req` falls
   back to request_id parse exactly when infera omits remote_host/ports; `meta.tp_size`
   is consumed only as REMOTE tp on handshake (=4, correct). infera wire is correct
   (more correct than the upstream toy_proxy, which omits tp_size).
2. **WRITE-path async race** (seal/finalize/notify ordering). RULED OUT: READ mode
   fails IDENTICALLY, and READ has no async write-worker. So the bug is common to
   both modes → the shared transfer geometry.
3. **fp8 KV scale LOSS** (scales in a separate tensor not transferred). RULED OUT:
   `fp8_ds_mla` packs the UE8M0 scale INLINE in the 576-aligned page; a byte-copy
   carries it. (But the SHORT copy dropped the tail that HOLDS it — see root cause.)
   Also: bf16 is impossible on DSv4 (`fp8_ds_mla layout only supports fp8 kv-cache`).
4. **Missing layer coverage / indexer skipped** (the old GLM `dsa_write` bug).
   RULED OUT: `DBGWCOUNT` = 243 on all ranks → every layer written. v0.25.1's
   `wait_for_save` natively loops all `kv_caches`.
5. **Registration under/over-coverage, P/D stride mismatch, block-count.** ALL
   RULED OUT by measurement: `DBGREGION` ratio=1.000; P and D register identical
   strides; `DBGMETA` block_size correct, nblk=1 for ≤1-block prompts.

## Root cause (what / why)

`get_layer_transfer_geometry` MLA (3-dim) branch computed the per-block transfer
SIZE and STRIDE from the tensor `.shape`/`.stride()`:
```
block_len = block_size * latent_dim * element_size   # shape-derived, WRONG
block_stride = stride[0]
```
The authoritative page is `spec.page_size_bytes`. Two independent mismatches:
- **DSv4** `fp8_ds_mla` is 576-byte-aligned → page > shape bytes; dropped tail =
  UE8M0 per-block scale → decode dequantizes with stale scale → facts garble,
  structure survives ("France"→"a good idea").
- **GLM** cache is per kernel block of size 1 (shape[1]=1, ~1.25M blocks) while the
  scheduler pages at spec.block_size=16 → page = 16×slot but shape bytes = 1×slot →
  16× short → 1/16 of the KV moved → total garbage ("is is is").

Same defect. Mooncake avoids it: it transfers `stride(0)*es` and uses
`layer_spec.page_size_bytes` for MLA.

## The fix (how)

`patches/patch_moriio_pagelen.py`: MLA branch `block_len = spec.page_size_bytes`,
`block_stride = page_size_bytes // element_size`. No-op for contiguous matched-block
fp16/bf16 K/V. See `patches/patch_moriio_pagelen.md` for the full rationale +
provenance (an earlier `stride[0]*es` draft fixed DSv4 but was 16× short for GLM;
must use page_size_bytes).

## Corrections to prior beliefs (this session overturned them)

- **DSv4-Pro is NOT single-MLA.** It registers 243 layers / 6 heterogeneous cache
  shapes (swa 256-block, MLA latent 2-block, compressor 256-block, DSA indexer
  64-block), all fp8_ds_mla. It IS a dual-cache DSA hybrid like GLM (has the
  Lightning Indexer, index_head_dim=128). Any note saying "DSv4 single MLA" is WRONG.
- The garbage is **NOT non-deterministic** — it's a deterministic wrong-scale/short-
  transfer corruption; earlier "non-determinism at temp=0" was just sampling on
  already-garbage logits.
- The 3 dead GLM MoRIIO patches (dsa_write/hetero/blocksize) are genuinely no-op AND
  their concerns are handled natively in v0.25.1 — but v0.25.1 introduced THIS new
  page-len bug none of them covered.

## Operating discipline (RDMA / distributed — keep doing)

- **Teardown ritual every round:** kill only your engine procs (`killworkers.sh` —
  matches vllm.entrypoints/EngineCore/Worker_TP, spares infera.server + other users'
  containers), confirm VRAM ~298 MB/card AND ports 36100-36203 free, THEN relaunch.
  Skipping it → `Address already in use` + OOM phantom failures.
- **Restart BOTH engines together**, never one side alone (engine_id re-handshake).
- **Routers run from the HOST** (they `docker exec`); **engines run INSIDE** the
  container (`docker exec glm_pd bash -lc 'bash ...'`). Don't nest.
- **`infera.server` rejects `--trust-remote-code`** (removed from router scripts here).
- **Nested-curl JSON quoting is a trap** — use the urllib probe .py files, not
  inline `curl -d '{...}'` through `docker exec bash -c "..."` (mangles the body →
  HTTP 500 / 000 that looks like a hang but isn't).
- **CG capture is slow** (DSv4 ~15 min, GLM ~10 min with torch.compile) — don't kill
  a "stuck" server mid-capture.
- On chi2866 (the jump host): never fill disk, don't touch foreign containers
  (titan training on card0-3); this run used card4-7 for decode.
