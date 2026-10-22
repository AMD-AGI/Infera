# Triage: PD (22k DPA prefill, mode N) -> TP4/EP4 + DCP4 decode, ROCm, Mooncake HIP IPC

Branch `906b0915a9` (python/sglang = `ae529c96bf`). Everything here comes from reading code; none of it
has been measured.

## Pre-emptive answers

### R4: Mooncake registration of the DCP pack StagingBuffer (low risk)

Code on faae8dd4 (fetched via gh to `/var/tmp/yihou/analyst_ws/mooncake/`) plus the #4340 port in the image.

- **The pack buffer is only ever a transfer source.** It lives on the prefill, and the decode receives
  straight into its KV pool.
- **HipTransport never looks the source up.**
  - `startAsyncTransfer` uses `request.source` as a raw device pointer in `hipMemcpyAsync`.
  - Only the destination is relocated, through the target segment's published buffers
    (`relocateSharedMemoryAddress`, `hip_transport.cpp:455-510`).
  - Transport choice also keys on the target segment (`multi_transport.cpp:468-547`).
  - So a late, interior, or even failed registration of the pack buffer does not change a HIP transfer.
- **Late registration is not an issue for the decode either.**
  - KV, aux and state buffers are registered once at init (`mooncake/conn.py:389-425`, zero-length
    elided index-K layers skipped).
  - The pack buffer is registered lazily on the prefill (`_register_staging_memory`, `conn.py:549-550`)
    when the first DCP decode peer registers. The decode never needs to learn about it.
- **Caveat:** `batch_register` failures are logged at debug level and the return value is ignored
  (`distributed/device_communicators/mooncake_transfer_engine.py:166-181`). If an RDMA protocol were
  selected instead of HIP, an unregistered source would fail at transfer time, not at registration.
- **Interior destination registrations** (R018) on the decode are covered by the mc4340 image (#4340
  port). Use only `infera-sglang:*-mc4340` images.

### Router: decode DP1, prefill DP4 (`PD_DP_RANK_AFFINITY=0`) — resolved by reading the router: FORCE_QUERY is not needed

Router source: Infera at `68f06e29` (from the pack-up bundle, fetched into the scratch clone
`/var/tmp/yihou/analyst_ws/infera_clone`), file `rust/router/src/disagg.rs`.

- **The prefill DP rank** is chosen by the router policy (kv-aware `sessions.pick`) among the ranked
  prefill targets (`disagg.rs:61-68`).
- **Affinity only changes how the decode is picked** (`disagg.rs:69-98`). With it off, the decode is
  picked by the normal policy; with decode DP1 there is a single decode target.
- **Bootstrap room:** `room = dp::align_room_to_prefill_rank(random, &p)` makes
  `room % prefill_dp_size == prefill dp_rank` (`dp.rs:16-23`). `annotate_sglang` writes the same
  `bootstrap_host/port/room` into **both** bodies (`disagg.rs:122-127`, `protocol.rs:57-83`).
- **Decode body:** gets `disagg_prefill_dp_rank = p.dp_rank` whenever the prefill target is ranked
  (`disagg.rs:146-149`). This does not depend on affinity.
- **Prefill leg:** sent with header `X-Data-Parallel-Rank: p.dp_rank` (`leg_payload`, `disagg.rs:473-494`;
  `post_leg`). sglang maps that header to `routed_dp_rank` (`entrypoints/openai/serving_base.py:270-299`),
  and the DP controller dispatches it there (`managers/data_parallel_controller.py:840-841`).
- **sglang side, both checks pass:**
  - decode `_resolve_prefill_dp_rank` returns `req.disagg_prefill_dp_rank`;
  - the prefill sender's `follow_bootstrap_room` check sees `attn_dp_rank == room % 4` because the router
    aligned the room.
- **Verdict:** `SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK` is **not needed**. The prefill-side
  mechanism is the same one the 22k run used; affinity only changed the decode pick.
- **Preconditions:**
  - the image's router must be built from `68f06e29` (or a commit with the same `disagg.rs` / `dp.rs`);
  - the prefill must register with `dp_size 4` so that `p.dp_rank` and `worker.dp_size` are set (as in 22k).
  - If decode logs ever show `follow_bootstrap_room conflict` or a DP mismatch, the router build differs.

## Symptom -> likely cause -> where to look

| Symptom | Likely cause | Check |
|---|---|---|
| Decode launch: `DSA DCP with PD disaggregation is supported only as a DCP1 prefill feeding a ROCm DCP decode` | `--dcp-size>1` on the prefill, or not ROCm | prefill server_args `dcp_size` |
| Decode launch: `DSA DCP does not support HiSparse or HiCache` | `--enable-hierarchical-cache` on the decode | decode args |
| Decode launch: host_pool / prefill_complete rejection | `--disaggregation-decode-host-receive-threshold<1` or `--disaggregation-decode-allocation-policy prefill_complete` | decode args |
| Decode graph capture: `'lse' referenced before assignment` / `RotaryEmbedding ... cos_sin_cache` | pre-#44322 code is running (overlay not mounted) or DSA backend not triton | decode log `sglang.__file__` path (cd /tmp), server_args `dsa_*_backend='triton'` |
| Prefill: `Unsupported PD DCP topology` | the prefill has DCP>1 | prefill args |
| Prefill: `PD DCP source/destination KV geometry differs at entry i` | page size or KV dtype differ between legs; draft layout (mode S only) | both server_args `page_size=64`, `kv_cache_dtype='fp8_e4m3'`; entry i ≥ num_target means draft |
| Prefill: `PD DCP draft head slicing is unsupported for pure MLA` | mode S with old code (no sharded-draft commit) | sha mounted on the prefill |
| Prefill: `Rejecting state transfers to decode session ...` / `dsa item length mismatch` | index-K geometry differs: `--json-model-override-args` (index_share / skip layers) or kpool differ between legs | diff both legs' `json_model_override_args`; the rejection text names the entry |
| Prefill: `follow_bootstrap_room conflict` | the router does not align the room or the prefill is unranked (not expected with 68f06e29) | router build commit; the prefill worker registers dp_size 4; fallback: FORCE_QUERY env on both legs |
| Requests hang about 1800 s, then `KVTransferError` / waiting timeout; decode never leaves Bootstrapping or WaitingForInput | decode pulls from the wrong prefill DP rank, or a Mooncake session failed | decode `Fetched bootstrap info ... DP x` vs the prefill DP rank that ran the room; prefill `Session ... failed`, `Failed to send kv chunk` |
| Decode 503 from the router | `PD_DP_RANK_AFFINITY=1` with decode DP1 | router env |
| Prefill OOM at the first request (not at startup) | lazy pack buffers: 4 x 351 MB per prefill rank allocated after KV sizing | prefill log `PD DCP pack buffers allocated`; lower `--mem-fraction-static` by 0.01-0.02 |
| TTFT high, prefill log `PD DCP pack buffer too small ... falling back to per-token RDMA` | the transfer exceeds pack capacity (long prefill-side prefix hit) | count of that warning; `get_max_transfer_tokens` should bound sends |
| Mooncake `device memory not registered` / transfer rc != 0 | decode destination not in a registered region | decode registration (REGBASE-style log), image is mc4340 |
| Output correct for short prompts, wrong or garbled for long ones (passkey fails) | latent striping or index-K pairing wrong; decode prefix not aligned to 256 | decode radix must be off; compare with mixed DCP4 on the same prompt; unit tests T2/T3/fan-out |
| Garbled from the first tokens, accept rate ~0 | MTP + custom all-reduce (22k R-fix), or KV dtype mismatch | decode has `--disable-custom-all-reduce`; both fp8 |
| GPU memory access fault on the decode | draft pool layout mismatch (replicated vs sharded) or OOB slot | decode log draft pool size (sharded ≈ 2.4 GB/rank), `dcp_size` on the draft backend |
| Decode `max_total_num_tokens` far below ~9.2M | different mem fraction; replicated draft | decode startup pool log |
| Eager decode very slow or OOM at large batch | recipe `dsa_dp_sync` full-width page table when `seq_lens_cpu` is absent | keep eager at low concurrency |
| Decode steps mostly non-graph | running batch > `--cuda-graph-max-bs-decode` | decode log graph/eager counters; set max-bs ≥ max-running |
| Mode S only: accept length about 1 | PD MTP seed mishandled under DCP | A/B with `SGLANG_DSA_DCP_PD_DROP_MTP_SEED=1` |
| Mode N: low real accept length on long prompts | expected: the draft KV for the prompt is never sent (runbook "Mode N") | n/a |

Do not enable `--disaggregation-enable-kv-checksum` on this stack. Whether its whole-page checksum is
valid when the latent KV is striped is unverified.
