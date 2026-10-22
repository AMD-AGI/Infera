Verdict: our mounted files do not get in the way of `prefill_complete`. Every mount is the image source with lines added and nothing removed or changed. But the planned run as written fails `_validate_prefill_complete` at startup, because prefill HiCache runs with `write_through` and this policy requires `write_back`. A few runtime conditions can only be confirmed from a live log.

Paths below use the brief's abbreviations: S = the image's `sglang` package root (`.../image-9d755dd6/sglang`), H = the harness root (`/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`).

**1. `mooncake/conn.py`: image vs `conn.yihou-regbase.py`** (3283 vs 3336 lines)
- There are three hunks and all are additions (`diff` shows only `a` hunks):
  - `82a83,133` adds the diagnostic function `_yihou_log_regbase`, wrapped in try/except and log-only.
  - `382a434` adds one call `_yihou_log_regbase(..., "kv/aux/state")`.
  - `507a560` adds one call `_yihou_log_regbase([ptr],[size],"staging")`.
- The only `defer_decode_allocation` use in the image file is `S/srt/disaggregation/mooncake/conn.py:3075`, inside `poll()`. It is present unchanged in the mount at `conn.yihou-regbase.py:3128`, shifted by the 53 added lines.
- All the other `prefill_complete` logic lives in files we do not mount, so the image version runs:
  - `disaggregation/common/conn.py:371-378`, `788`, `1101`, `1377`, `1460`, `1847-1852`, `2022-2068`, `2379-2393`, `2554-2560`
  - `prefill.py:522`, `1050-1053`, `1396`, `1723`
  - `decode.py:1025`, `1746`
- Same base: yes. Taking away the added lines leaves the image file exactly.

**2. The other mounts** (`H/mounts.yihou.v2-regbase.tsv`)
- `kv_events_publisher.yihou-noblock.py`: adds `import logging`, a logger and a `metrics_drop_warned` field, and makes the metrics `sock_send` non-blocking with `zmq.Again` handling (image line 90 becomes 95-103). Its only disaggregation reference is the existing `from sglang.srt.disaggregation.kv_events import` (line 17). It has no allocation logic.
- `forward_mla_rocm.yihou-noabsorbcfg.py`: one hunk, `158a159-162`, an early `return None` in `_absorb_bmm_config`. It has no disaggregation logic.
- Both are the image file plus additions only, so same base.

**3. How the harness passes the flags today**
- `allocation_policy`, `host_receive_threshold` and `num_reserved_decode_tokens` appear nowhere in `H` outside `mounts-v0521/src` (recursive grep over `*.sh`, `*.py`, `*.tsv` returned nothing). `engine.sh` does not emit them on either leg.
- So today's run uses the sglang defaults: `early` (`arg_groups/fields/disagg.py:59-64`), threshold `0.0` (`disagg.py:179-182`) and 512 reserved tokens (`disagg.py:106-109`).
- There is no knob for the policy. The way in is `PREFILL_EXTRA_ARGS_ADD` / `DECODE_EXTRA_ARGS_ADD`, appended at `config.yihou.v0521-base.sh:243-244` and word-split into the command at `engine.sh:320-323`. It has to go on both legs. If the two legs differ, every request fails with "policy mismatch" (`common/conn.py:2057-2063`).
- There is nothing to duplicate, so no conflict. For the record, the parser is plain `argparse.ArgumentParser` (`server_args.py:655`, `672`) with no duplicate check; under standard argparse the last value would win.
- Blockers for the planned run, all from `_validate_prefill_complete` (`arg_groups/pd_disaggregation_hook.py:204-248`):
  - **HiCache write policy.** Prefill HiCache must be L2 `write_back` (`:231-239`). The base config sets `PREFILL_HICACHE_WRITE_POLICY:=write_through` (`config.yihou.v0521-base.sh:98`), and `engine.sh:305` emits it, so prefill would refuse to start. The overlay must set `PREFILL_HICACHE_WRITE_POLICY=write_back`. That is a second variable changed alongside the allocation policy, and it is forced by the code.
  - **Host receive threshold.** It must be 0 (`:224`). It is at its default of 0.
  - **Decode radix cache / decode HiCache.** These must be off (`:226-229`). `DECODE_HICACHE:=0` (`base.sh:92`) and radix defaults to False (`disagg.py:85-88`).
  - **Attention TP.** `tp_size` must equal `attn_dp_size` (`:213-215`): 4 = 4 with DPA, so this passes. The peer also needs attention TP1 at runtime (`common/conn.py:2065-2068`).
  - **Staging buffers.** `SGLANG_DISAGG_STAGING_BUFFER` must be off (`:240`). It defaults to False (`environ.py:803`).
- Optimistic prefill: prefill turns it on automatically (`pd_disaggregation_hook.py:49-55`). It is turned off if HiCache is not L2 `write_back`/`write_through` (`serving_hook.py:476-495`), and then the run fails at `serving_hook.py:497-505`. With `write_back` and no storage backend it stays on.
- Runtime risk I could not settle from source: `common/conn.py:375-378` raises `ValueError` if any KV state type is not `StateType.DSA`. `DSA_TAIL` gets added when `kpool_use_compress` is true (`disaggregation/utils.py:1345-1355`, `1566-1576`; `memory_pool.py:5219`). Whether GLM-5.2 turns that on is not settled. A prior run's `YIHOU-REGBASE` / `state_types` log, or the startup log of the new run, would answer it.
- Audit and env checks:
  - `H/tools/audit_live.v3.yihou.py:28-91` does not check the allocation policy. It does compare `hicache_write_policy` (`:52`), `hicache_size` (`:50`) and `mem_fraction_static` (`:33`), but against the config environment. Setting the knobs (not raw extra args) keeps them consistent.
  - `H/bench-harness/tools/agentx_env.py:135-224` checks `hicache_size`, chunk, `max_running` and similar. It has no policy or write-policy check, so nothing there fails because of the policy change.

**4. HiCache size, mem fraction, register chunk**
- `PREFILL_HICACHE_SIZE` becomes `--hicache-size` on prefill only, when it is greater than 0 (`base.sh:97`, `238`). A size above 0 overrides the ratio (`mem_cache/pool_host/base.py:211-216`).
- `DECODE_MEM_FRACTION` is read in `engine.sh:56` (base default 0.95, `base.sh:88`). Set it to 0.96.
- The chunk default is `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB = EnvInt(256)` (`environ.py:770`). The register loop at `mem_cache/pool_host/common.py:146-176` makes one call for the whole buffer when no granularity is passed. When granularity is passed (it is for MLA page_first at `pool_host/mla.py:398-401` and for the DSA indexer at `pool_host/dsa.py:292`), it splits into chunks of at most the limit.
- With 370 GB, the main MLA buffer is probably above 256 GB [inferred: 370 GB is split between the KV and indexer buffers, and the KV share is larger]. At the default it would then be registered in 2 calls; with 512 it is 1 call.
- The E2-target value of 512 is labelled a candidate fix, "H1" (`config.yihou.E2-target.sh:7-12`). Nothing in the code shows that 2 calls are wrong, so 512 is the known-tested setting rather than a proven requirement.
- Base leaves the chunk unset (`base.sh:103`). If set, it goes into the prefill container environment only (`base.sh:248-249`).

Overlay to add:
- `PREFILL_HICACHE_SIZE=370`
- `HICACHE_HOST_REGISTER_CHUNK_GB=512`
- `PREFILL_HICACHE_WRITE_POLICY=write_back`
- `DECODE_MEM_FRACTION=0.96`
- `PREFILL_MEM_FRACTION=0.88`
- `CONC=48`
- `DECODE_MAX_RUNNING=96`
- `PREFILL_EXTRA_ARGS_ADD="--disaggregation-decode-allocation-policy prefill_complete"`
- `DECODE_EXTRA_ARGS_ADD="--disaggregation-decode-allocation-policy prefill_complete"`

The base config defaults to `YIHOU_MOUNTS=mounts.yihou.v2.tsv`. The regbase TSV is only selected by the `mc4340*` configs (`config.yihou.mc4340.sh:36`), so the overlay should source one of those or set `YIHOU_MOUNTS` itself.

=====VERIFY=====
**Verdict:** The report holds up on everything that decides the run, with two corrections. First, the TSV overlay has to set `YIHOU_MOUNTS` after it sources the base config, because `base.sh:80` assigns it unconditionally. Second, the claim that the mounts change nothing in the image is slightly off: the kv_events mount has one changed hunk, which is harmless to this policy.

Paths are as in the report: S = `.../image-9d755dd6/sglang/srt`, H = `.../v0521dbg_20261010/harness`.

**Decision-relevant claims**

1. **CONFIRMED: the startup blocker is the prefill HiCache write policy.** `pd_disaggregation_hook.py:231-239` adds the failure when prefill has `enable_hierarchical_cache` and either `hicache_storage_backend is not None` or `hicache_write_policy != "write_back"`. The error is raised at `:244-248`.
   - `base.sh:98` sets `PREFILL_HICACHE_WRITE_POLICY:=write_through`.
   - `engine.sh:305` emits `--hicache-write-policy "$hicache_policy"`.
   - So the overlay must set `PREFILL_HICACHE_WRITE_POLICY=write_back`.
   - Missing check: `:242` (`SGLANG_RUST_SERVER`) also blocks, but nothing indicates it is set.
2. **CONFIRMED: the other checks pass.**
   - Host receive threshold: checked at `:224`, default 0.0 (`fields/disagg.py:179-182`).
   - Decode radix cache / decode HiCache: checked at `:226-229`. Radix defaults to False (`disagg.py:85-88`) and `DECODE_HICACHE:=0` (`base.sh:92`).
   - Attention TP: `tp_size == attn_dp_size` at `:213-215`.
   - Staging buffers: `SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)` (`environ.py:803`).
3. **CONFIRMED: both legs need the flag, and a mismatch fails per request, not at startup.** `common/conn.py:2056-2063` sets reason "policy mismatch" and the room goes Failed. The peer's attention TP1/CP1/PP1 is checked at `:2065-2068`. The default is `early` (`disagg.py:59-64`).
4. **CONFIRMED: optimistic prefill.**
   - It is auto-enabled to 1 at `pd_disaggregation_hook.py:48-55`.
   - It is turned off for unsupported HiCache configs at `serving_hook.py:476-495`, and the run then hard-fails at `:497-505`.
   - With `write_back` and no storage backend it stays on.
   - The only other turn-off sites are pp>1 (`:469-474`); repo-wide grep finds no other `optimistic_prefill_attempts=0`.
5. **The `DSA_TAIL` runtime risk is now resolved from config (it was UNVERIFIED in the report).**
   - `common/conn.py:375-378` raises if any state type is not `DSA`.
   - `DSA_TAIL` only appears when `kpool_use_compress = index_kpool > 1 and index_kpool_compress` (`memory_pool.py:5219`). `get_compress_tail_buf_infos` returns empty lists otherwise (`memory_pool.py:5272-5274`), and `utils.py:1564-1576` appends a tail only if `tail_ptrs` is non-empty.
   - `index_kpool_compress` defaults to False when it is absent from the model config (`configs/model_config.py:446-447`).
   - `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4/config.json` has no `index_kpool*` keys (`model_type glm_moe_dsa`; the only `index_*` keys are at lines 18-24).
   - The harness JSON override sets only `index_share_for_mtp_iteration` (`base.sh:228`).
   - So no `DSA_TAIL` [inferred from config plus code; a startup log would confirm]. `MHATokenToKVPoolMXFP8` / `BLOCK_SCALE` (`utils.py:1358-1361`) is the MHA pool; whether GLM's MLA pool can land there is UNVERIFIED (`KV_CACHE_DTYPE=fp8_e4m3`, `base.sh:118`).
6. **Mounted `conn.py` compatibility: CONFIRMED.** `diff` shows only `82a83,133`, `382a434` and `507a560`. `defer_decode_allocation` is at image `:3075` and mount `:3128`.
7. **The verdict that every mount is "added lines only" is WRONG for `kv_events_publisher`.** The diff has a change hunk, `90c95,103`: the `sock_send(...)` line is replaced by a non-blocking try/except with `zmq.Again`. The report's own detail section described this correctly. It has no allocation logic, so the conclusion still stands. `forward_mla_rocm` is `158a159,162` only (CONFIRMED).
8. **Harness knob: CONFIRMED.**
   - A grep for `allocation.policy|allocation_policy` over H (`*.sh`, `*.py`, `*.tsv`), excluding `mounts-v0521/src`, returns nothing.
   - `PREFILL_/DECODE_EXTRA_ARGS_ADD` are appended at `base.sh:243-244` and word-split at `engine.sh:321-323`.
   - `DECODE_MEM_FRACTION` is used at `engine.sh:56` and emitted at `:261`.
   - `--hicache-size` is emitted when it is above 0 (`base.sh:238`).
   - The register chunk goes into the prefill environment only (`base.sh:248-249`).
9. **Register chunk: CONFIRMED.** `environ.py:770` is `EnvInt(256)`. `pool_host/common.py:146-176` makes one call without a granularity and splits into chunks with one. `mla.py:396-401` passes a granularity for `page_first`, and `dsa.py:290-292` passes the indexer granularity. The claim that 370 GB means 2 calls remains [inferred].

**Missing facts that matter**

- **A. `YIHOU_MOUNTS` is assigned unconditionally at `base.sh:80`.** The overlay must source the base chain first and then set `YIHOU_MOUNTS=.../mounts.yihou.v2-regbase.tsv`, as `config.yihou.mc4340.sh:33-36` does (its comment at `:35` says so). Setting it before sourcing gets overwritten.
- **B. Prefill holds KV under `prefill_complete`.**
  - On the last chunk, prefill calls `mark_prefill_complete()` (`prefill.py:1049-1053`) and sends only if `not req.pending_bootstrap` (`:1054-1055`).
  - The mid-chunk "yield to waiting requests" release is skipped under this policy (`:1393-1404`).
  - The attempt-exhausted fallback is also skipped under it (`:1724-1727`).
  - Bootstrapping requests are admitted to prefill regardless of attempt count (`:517-523`).
  - So finished prefill KV stays in prefill GPU memory until decode admits it [inferred: no release path other than the test hook `:969-973` and the mid-chunk pending path `:1080-1081`]. `PREFILL_MEM_FRACTION` 0.88 (base default 0.85, `base.sh:87`) and decode admission at CONC 48 determine prefill pool pressure. The startup log and token-usage metrics would show it.
- **C. `--hicache-size` is GB per process.** `pool_host/base.py:211-214` converts `host_size*1e9 // size_per_token` per process, and `sync_fixed_hicache_size` (`base.py:112`) only syncs across PP. Per-DP-rank host totals of about 4×370 GB on prefill are [inferred]. The base default was already 450 (`base.sh:97`), so 370 is lower than prior runs.
- **D. Base default `CONC:=44`** (`base.sh:197`) needs to be overridden to 48.
- **UNVERIFIED (not re-read):** the argparse "last value wins" claim (`server_args.py:655`, `672`), `audit_live.v3` and `agentx_env.py` line refs, and `decode.py:1746` beyond the early-only rebootstrap recompute (`decode.py:1744-1746` CONFIRMED). The 370 GB KV/indexer split is also unverified.

**Corrected overlay**
1. Source `config.yihou.v0521-base.sh`.
2. Set `YIHOU_MOUNTS=$WS/harness/mounts.yihou.v2-regbase.tsv`.
3. Set the knobs:
   - `PREFILL_HICACHE_SIZE=370`
   - `HICACHE_HOST_REGISTER_CHUNK_GB=512`
   - `PREFILL_HICACHE_WRITE_POLICY=write_back`
   - `PREFILL_MEM_FRACTION=0.88`
   - `DECODE_MEM_FRACTION=0.96`
   - `CONC=48`
   - `DECODE_MAX_RUNNING=96`
   - `PREFILL_EXTRA_ARGS_ADD` and `DECODE_EXTRA_ARGS_ADD` both set to `--disaggregation-decode-allocation-policy prefill_complete`

   The `:=`-defaulted knobs can be exported before sourcing instead. `CONC` and `DECODE_MAX_RUNNING` were not checked for `:=` form, so setting all knobs after sourcing is the safe order.