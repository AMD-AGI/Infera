# Environment — P8D8 AgentX sweep (2026-09-19 → 2026-09-20)

Hardware and software of record for the five-point AgentX sweep. Every figure
below is followed by the file it was read from, under this packup directory. Two
provenance classes are kept distinct throughout:

- **LIVE** — read from the running container's `server_args` dump captured in
  `logs/prefill-0.excerpt.txt.gz`, or from the orchestrator's own startup print
  in `logs/launch-summary.txt`. These are what the engine actually ran with.
- **CONFIG** — read from a config or launch script (`scripts/config.yihou.p8d8.sh`,
  `scripts/bench-harness/config.full.sh`, `scripts/bench-harness/engine.sh`).
  These state intent; they are authoritative only where a LIVE value does not
  contradict them.

One asymmetry matters and is called out where it bites: **the prefill leg's live
argv IS captured** (`logs/prefill-0.excerpt.txt.gz` contains the full
`server_args={...}` line). **The decode leg's live argv is NOT** — its excerpt
holds only per-batch decode lines, no `server_args` dump. So every prefill fact
below can be LIVE; several decode facts can only be CONFIG plus the one-line
orchestrator print in `logs/launch-summary.txt`.

## 1. Nodes and roles

| node | role | GPUs | engine ip:port | source |
|---|---|---|---|---|
| crsuse2-m2m-137 | prefill + router + etcd | 0–7 | 10.245.153.247:29001 | LIVE (`logs/launch-summary.txt`) |
| crsuse2-m2m-136 | decode | 0–7 | 10.245.154.168:29002 | LIVE (`logs/launch-summary.txt`) |

`logs/launch-summary.txt` also records etcd `10.245.153.247:22379` and router
`http://10.245.153.247:28000`, both on 137. Scope of the hardware prep was 137
and 136 only; 135/138 belonged to other sessions and were not touched
(`analysis/hardware_prep.yihou.md` §1).

## 2. GPU model, count, VRAM

- **MI355X, 8 per node**, indices 0–7, on both 137 and 136
  (`analysis/hardware_prep.yihou.md`; `scripts/rdma_map.yihou.md`).
- Per-GPU VRAM **309.22 GB** (`analysis/hardware_prep.yihou.md` §2: idle GPUs at
  ~284 MiB "of 309.22 GB used"). After teardown all 16 GPUs were confirmed idle
  at ~284 MiB each with no sglang/python compute PIDs (same section).
- LIVE prefill engine reported `available_gpu_mem=41.44 GB` and
  `max_total_num_tokens=3143680` per DP rank after pool init
  (`logs/prefill-0.excerpt.txt.gz`), with `mem_fraction_static=0.85`.

## 3. Host RAM and why it matters

The prefill leg runs HiCache with the host as the KV backing pool, so host RAM,
not VRAM, is the ceiling on the prefill radix cache.

- During the deployment, host memory on the prefill node reached **2,047 GB** in
  use; teardown returned it to 50 GB **"as the 1.7 TB HiCache pool released"**
  (`spec/poll_log.md:222`). So the prefill HiCache host pool is **~1.7 TB
  aggregate across the 8 DP ranks**; the node carries ~2 TB of RAM.
- Per-DP-rank share is therefore **≈ 212 GB** by arithmetic (1.7 TB / 8). The
  precise value **~211.86 GB per rank** stated in the task brief is **NOT
  independently locatable in any file in this kit** — only the 1.7 TB aggregate
  and the 2,047 GB / 50 GB host-memory readings are on disk. Treat 211.86 as
  UNKNOWN-to-this-kit; the load-bearing, file-backed facts are 1.7 TB aggregate
  and ~2 TB node RAM.
- The pool was genuinely saturated, not merely allocated: at the CONC 80 point
  the prefill leg showed **CPU KV usage 100 %**, GPU KV usage 9 %, prefix-cache
  hit 93.9 % (`spec/poll_log.md:502`) — PR #37152's host-pool path was under real
  load.

## 4. RDMA fabric

Re-measured first-hand 2026-09-18 from sysfs / rocm-smi on both nodes; full table
in `scripts/rdma_map.yihou.md`, summary in `analysis/hardware_prep.yihou.md` §3.

- **8 rails per node, `ionic_0..7`**, all ACTIVE with a live netdev and non-zero
  GID on both 137 and 136. **No defective rail** (contrast 135's dead `ionic_7`).
- **`GPU_n ↔ ionic_n` is NUMA-local for every n**: GPUs 0–3 + ionic_0–3 on PCI
  domain 0002 / NUMA0; GPUs 4–7 + ionic_4–7 on 0003 / NUMA1.
- **Rail id per index is identical across 137 and 136** (ionic_0→08, _1→07,
  _2→05, _3→06, _4→04, _5→03, _6→01, _7→02), so `GPU_n ↔ ionic_n` on both legs
  yields an automatic same-rail KV path — the property Mooncake
  `MC_ENABLE_DEST_DEVICE_AFFINITY` needs. Per-node GID index-1 values are tabled
  in `scripts/rdma_map.yihou.md`.
- Fabric result across the whole 13 h 33 m deployment: **rails 0/0 on every one
  of the five points** (`results/sweep_results.yihou.md`; `results/c*/rails-*.txt`).

Two settings carry the same-rail path, both LIVE-confirmed on the prefill leg via
`disaggregation_ib_device` in `logs/prefill-0.excerpt.txt.gz`:

- `RDMA_DEVICE` is the **per-GPU JSON map**
  `{"0":"ionic_0",...,"7":"ionic_7"}` — the engine received it verbatim as
  `disaggregation_ib_device` (LIVE). CONFIG source and rationale for the JSON
  (vs a shared list, which lets Mooncake auto-pick mismatched NICs on physically
  isolated rails) is `scripts/config.yihou.p8d8.sh:187-189`.
- `MC_TE_FILTERS=ionic_0,...,ionic_7` (comma list), `MC_GID_INDEX=1`,
  `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, `PD_DP_RANK_AFFINITY=1` — CONFIG
  (`scripts/config.yihou.p8d8.sh:176,194`; `scripts/bench-harness/config.full.sh:94-96,107`).
  `MC_GID_INDEX=1` is the live rail per `scripts/rdma_map.yihou.md`.

## 5. Image

- **Tag** `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, **id
  `sha256:fd7220a57b7d…`** (short `fd7220a57b7d`, 66.3 GB) — id verified identical
  on the source (135) and both targets (137, 136) in
  `analysis/image_transfer.yihou.md`; the same id appears in the AITER JIT cache
  path in `logs/launch-summary.txt` (LIVE).
- **In-image sglang `0.5.19.dev20260917+ga9fb1c3238`** — verified in-container on
  both nodes (`analysis/image_transfer.yihou.md`).
- **Base image**: `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917` per the
  mission spec. This exact base tag is **not printed in any kit log**; the only
  base tag on disk is `config.full.sh:14`'s default
  `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916`, which is a build-time
  default that does **not** apply here — the run used a pre-built, transferred
  image, so `SGLANG_BASE_IMAGE` was never consulted at runtime. The `-20260917`
  base is therefore CONFIG/spec-asserted, not log-verified in this kit.
- Contents verified by marker greps in-container (`analysis/image_transfer.yihou.md`):
  the **NextN shared-experts fusion fix** (`fused_shared_experts_architecture =
  "GlmMoeDsaForCausalLMNextN"`) and **sglang PR #37152** (ROCm HiCache JIT copy
  rounds + K-only host pool: `pick_group_bytes`, `_tiles_across_lanes`,
  `can_use_jit`). PR #37152 is **OPEN upstream, never merged, its AMD ROCm CI RED**
  (`spec/mission.md:87-91`); it is present only because the user asked for it.

## 6. Bench harness provenance

- Private worktree **`/home/yihou/dev/git/infera.yihou.glm52.p8p4` @ commit
  `ad3b85d3`**, seeded with patched `bench/glm5p2_pd` scripts from
  `/home/yihou/dev/git/infera.glm52.pd`
  (`scripts/config.yihou.p8d8.sh:227` points `BENCH_DIR` at
  `…/infera.yihou.glm52.p8p4/bench/glm5p2_pd`; worktree/commit per task brief).
- Vendored copies live in `scripts/bench-harness/` with SHA256 in
  `scripts/bench-harness/SHA256SUMS.txt`:

  | file | sha256 |
  |---|---|
  | `agentx_bench.sh`  | `a183a8f6e57b35931067655064da18602f4d61c8e196c1032a5927c6854fb945` |
  | `collect_agentx.py`| `320344d729a27f49481ba8db9fe186992748a5350b4da33e63ca70174cc99057` |
  | `config.full.sh`   | `b03500585023ae631c4c16d805b301c7994ff695644ecfe299a75d38d6114d48` |
  | `config.sh`        | `d65516d0884fe8627149784b841a8ad09160c350304afcdc4be2844259710712` |
  | `engine.sh`        | `a4d299e16bc42cf68530b5a72f0227f2915fd4fedafc5efd02afca0d52ee71c3` |
  | `launch.sh`        | `995b62de3dcfdc28b7b813c626e4b1b6005615cc97c467f6b49ddd21735bdb7f` |
  | `stop.sh`          | `5e09784712abb772923fe461755ad412cf1ede9f597b2c4c1271acf506c80efa` |

  The shape/sweep configs (`config.yihou.p8d8.sh` etc.) are in `scripts/` and are
  not covered by that SHA file.

## 7. Absolute paths

- **Weights** `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` — LIVE
  (`model_path` and `tokenizer_path` in `logs/prefill-0.excerpt.txt.gz`);
  served model name `glm5.2-mxfp4`, `kv_cache_dtype=fp8_e4m3` (LIVE).
- **Host RDMA lib** `/lib/x86_64-linux-gnu/libionic.so`, bind-mounted into the
  container at `/host-libionic/libionic.so` — CONFIG
  (`scripts/bench-harness/config.full.sh:112-113`).
- **Node-local scratch** `/mnt/m2m_nobackup` (docker root and launch tree). Server
  logs and inspection files live at
  `/mnt/m2m_nobackup/yihou_p8p4/launch/sweep-20260919T113947Z/`
  (LIVE, `logs/launch-summary.txt`). Raw server logs (110 MB prefill / 90 MB
  decode) stay on the nodes at `…/server-logs/`; only gzipped grep excerpts are
  packed here.

## 8. Engine configuration — prefill vs decode

Shape is symmetric **P8D8: TP8 / DP8 on both legs, DP attention on**. LIVE prefill
`server_args` confirm `tp_size=8`, `dp_size=8`, `enable_dp_attention=True`,
`max_running_requests=256`, `cuda_graph_max_bs_prefill=256`, `mem_fraction_static=0.85`,
`page_size=64`. The pool-init log line in the same excerpt additionally reports
`context_len=1048576` (`server_args` itself carries `context_length=None`).

| aspect | prefill (LIVE unless noted) | decode (CONFIG + launch-summary; **no live argv captured**) |
|---|---|---|
| node | 137 | 136 |
| TP / DP / DPA | 8 / 8 / 1 (LIVE) | 8 / 8 / 1 (`logs/launch-summary.txt`; CONFIG `config.full.sh:64-67`) |
| max_running_requests | 256 (LIVE) | 256 (CONFIG `config.yihou.p8d8.sh:59`) |
| cuda-graph max bs | 256 (LIVE `cuda_graph_max_bs_prefill`) | 256 (CONFIG `config.yihou.p8d8.sh:61`) |
| **HSA_NO_SCRATCH_RECLAIM** | **0** (CONFIG `config.full.sh:56`, forwarded `engine.sh:49/121`) | **1** (CONFIG `config.full.sh:72`, forwarded `engine.sh:62/121`) |
| **HiCache** | **ON** — `enable_hierarchical_cache=True`, `hicache_host_memory_mode='cache'`, `hicache_ratio=1.5`, `write_through`, `io_backend=kernel`, `mem_layout=page_first` (all LIVE) | **OFF** — `HiCache=0` (`logs/launch-summary.txt`). Forced off: `engine.sh:76-79` hard-rejects decode HiCache + MTP, so PR #37152 can only help the prefill leg |
| MTP / EAGLE | none — `speculative_algorithm=None` (LIVE) | EAGLE, `--speculative-num-steps 5`, `--speculative-num-draft-tokens 6`, `--speculative-eagle-topk 1` (CONFIG `config.full.sh:78-81`, `engine.sh:200-206`) |
| simulated acceptance | n/a | `DECODE_SIMULATE_ACC_LEN=3.61` (CONFIG `config.full.sh:82`). **Waives correctness** — forces the accept *count*, not which tokens are right; the deployment emits garbled text and the ~3.6 acceptance gauge carries no correctness signal |
| custom all-reduce | ON — `disable_custom_all_reduce=False` (LIVE) | ON — `DECODE_NO_CUSTOM_AR` unset, no `--disable-custom-all-reduce` (CONFIG `config.yihou.p8d8.sh:105-108`). The "IndexShare=false + custom-AR ON" cell is untested for correctness (`config.yihou.p8d8.sh:152-157`) |
| DSA backends | prefill `tilelang`, top-k `sgl-kernel` (LIVE `dsa_prefill_backend`/`dsa_topk_backend`) | decode `tilelang` (CONFIG `config.yihou.p8d8.sh:206-207`; top-k backend cleared so sglang defaults to `sgl-kernel`, `config.yihou.p8d8.sh:230-232`) |
| chunked prefill | `chunked_prefill_size=4096`, `max_prefill_tokens=16384` (LIVE — differs from CONFIG `config.full.sh:52`'s 32768; the live value governs) | not captured |

Shared engine env, forwarded to both legs by `engine.sh` (CONFIG
`config.full.sh` + `engine.sh`, values not in the captured argv unless noted):
`SGLANG_OPT_USE_TOPK_V2=false` (`config.full.sh:101`),
`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`, `SGLANG_USE_AITER=1`,
`SGLANG_ENABLE_FAILED_SESSION_PROBE=1` (`config.full.sh:110`),
`--enable-aiter-allreduce-fusion` (LIVE `enable_aiter_allreduce_fusion=True` on
prefill), `disaggregation_transfer_backend=mooncake` (LIVE).

**`index_share_for_mtp_iteration=false`** — the issue.md 3.3 mitigation that keeps
the fused DSA indexer while removing the cross-step MTP IndexShare seed. Delivered
as `--json-model-override-args '{"index_share_for_mtp_iteration":false}'` and
**LIVE-confirmed on the prefill leg** (`json_model_override_args` in
`logs/prefill-0.excerpt.txt.gz`). It must be assigned **after** sourcing
`config.full.sh`, because `config.full.sh:90` is a plain assignment that clears it
(`config.yihou.p8d8.sh:162-165,238-240`).

Two items from the same defect, recorded so they are not re-derived
(`config.yihou.p8d8.sh:130-135`): `SGLANG_DSA_FUSE_TOPK=0` was a **withdrawn**
workaround — it stopped the memory-access-fault crash but the deployment then
emitted garbled text (`1!au!au!au!…`) while the forced acceptance gauge still read
3.5–3.8, so it was replaced by the IndexShare mitigation. The written-in-advance
prediction that `SGLANG_DSA_FUSE_TOPK=0` *caused* the garbling was tested and
**disproved**.

## 9. Deployment lifetime

One deployment served all five points: up 2026-09-19 12:14:08, driver finished
2026-09-20 01:47:15 — **13 h 33 m, zero memory access faults, zero rail faults**
(`results/sweep_results.yihou.md`; `logs/launch-summary.txt`;
`results/c*/rails-*.txt`). The mission expected an out-of-memory end state at the
top of the sweep; none was reached — the limit was queueing collapse, not OOM
(`spec/mission.md`; `results/sweep_results.yihou.md`).
