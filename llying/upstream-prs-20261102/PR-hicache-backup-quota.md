# [DRAFT, not submitted] HiCache: make the workgroup count of the all-layer D2H write-back copy configurable (`SGLANG_HICACHE_BACKUP_BLOCK_QUOTA`)

Diff: `hicache-backup-quota.diff` (`pool_host/mla.py`, `pool_host/dsa.py`; against v0.5.22; env var to be moved into `sglang.srt.environ`).

## Problem
`transfer_kv_all_layer_mla_lf_pf` takes a `block_quota` argument (workgroups launched), default 2. For a write_back eviction of ~500K tokens (KV +
DSA indexer, 75 nodes in the production shape) that default caps the copy at ~22 GB/s on MI355X, and the scheduler waits for it.

## Change
Pass `block_quota=_BACKUP_BLOCK_QUOTA` from both call sites (KV in `mla.py`, indexer in `dsa.py`); `_BACKUP_BLOCK_QUOTA` comes from
`SGLANG_HICACHE_BACKUP_BLOCK_QUOTA`, default 2 = current behaviour.

## Evidence
- Microbenchmark (production shape: KV 20 layers x 576 B, indexer 5 layers x 8,448 B, one eviction = 75 nodes), quota 1..64: output byte-identical to the
  default in every setting, no writes outside the target rows (host memory elsewhere kept its sentinel). 6.5 GB copy 278 ms -> 125 ms at 16; ~22 -> ~49 GB/s.
- End to end (same setup as the evict-headroom draft, with that knob also on): 49,719 (mean of two) -> 51,374 tok/s/GPU (+3.3%); per-eviction time summed over
  the four PP ranks 1.45 s -> 1.04 s.
- Correctness: capped-pool gate as in the other draft: probe 0.985, garbled 0, long-context needle 64/64, GSM8K 62/63.

## Known limits
- Only measured on MI355X (gfx950). Whether 16 is right on other GPUs is unknown, hence default 2 and an env var, not a new default.
- The kernel's grid size only changes how items are split across warps (each warp owns a disjoint item range), so results are unchanged; this was checked by
  reading the launcher and by the byte-exact benchmark, not by a separate kernel unit test.
