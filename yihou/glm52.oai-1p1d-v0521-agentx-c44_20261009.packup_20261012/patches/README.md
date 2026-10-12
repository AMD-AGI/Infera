# Patches — what / why / how / context

Ledger IDs (R0xx) point into `../analysis/working_process.md`. **The 22,114 run (e2g A2) used exactly layers 1 + 2 + 3
below**; layer 4 is experimental and was NOT active in the headline run.

## 1. Image recipe — `image-recipe/` (baked into `infera-sglang:v0521-yihou-1010-mc4340`, built at commit `68f06e29`)

Apply with `git am` on `infera.glm52.pd` @ `28121cb40ef0b03bf7b8481fde214562d634a9a0`
(`git@github.com:AMD-AGI/Infera.git`, branch `dev/pd_opt/glm_5.2_agentx`). The same commits as a git bundle:
`infera-src.yihou.28121cb4..702af10f.bundle` (`git fetch <bundle> yihou.agentx44.20261009`). Full files in `image-recipe/files/`.

| patch | commit | what | why / context |
|---|---|---|---|
| `0001-router-…` | `e3e46f32` | resolve conflict markers in `rust/router/src/{policy,disagg}.rs`; port PD DP-rank affinity onto upstream `pick_targets`/sessions API | the user's rebase left the router unbuildable (phase 1) |
| `0002-docker-v0.5.21-…` | `fb8ae9c9` | `Dockerfile.sglang.v0521.yihou`: base pinned to v0.5.21 nightly; GLM-5.2 DSA patches re-cut for v0521 (`dsa_page_table_rows.diff`, NextN fused-shared-experts arch fix, prefill non-final-chunk wait-event); drops patches now upstream; builds Mooncake faae8dd4 | v0519 recipe diffs do not apply to v0521 |
| `0003-docker-v0521-Mooncake-4340-…` | `cdba2446` | `Dockerfile.sglang.v0521-mc4340.yihou` + `build_mooncake_sglang.mc4340.yihou.sh` + `mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff` | R018/R019: HIP IPC handle at allocation base → interior registrations mis-relocated; #4340 fixes it (safe transport) |
| `0004-docker-mc4340-avoid-pipefail-…` | `68f06e29` | build-script artifact check: no pipefail SIGPIPE false failure | r1 build failed on its own check |
| `0005-docker-mc4340prio-…` | `702af10f` | **optional, not used for 22k**: `Dockerfile.sglang.v0521-mc4340prio.yihou` + `mooncake-hip-stream-priority…diff` (`MC_HIP_STREAM_PRIORITY=high` → high-priority Mooncake HIP copy streams) | R039/R043: peer copies stall behind long kernels via shared HW queues; high-priority streams removed the stall in the microbench (200 → 11 ms). Not production-tested |

## 2. Runtime mounts — `runtime-mounts/` (bind-mounted `:ro` by `engine.sh` on both legs; table `scripts/harness/mounts.yihou.v2-regbase.tsv`)

Each `.py` is the full mounted file; each `.diff` is against the file inside the image.

| mounted file → target (`/sgl-workspace/sglang/python/sglang/srt/…`) | what | why |
|---|---|---|
| `kv_events_publisher.yihou-noblock.py` → `managers/scheduler_components/kv_events_publisher.py` | KV-metrics ZMQ PUSH sends NOBLOCK, drop on EAGAIN | R001: blocking send to an unbound endpoint wedged the prefill scheduler |
| `forward_mla_rocm.yihou-noabsorbcfg.py` → `models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` | `_absorb_bmm_config` → None | R003: v0521 hand config at M=256 aborts the triton LLVM backend |
| `conn.yihou-regbase.py` → `disaggregation/mooncake/conn.py` | logs `YIHOU-REGBASE … interior=` per registered region | diagnostic for R018 only (all runs `interior=0`); behaviour-neutral |

## 3. Harness — `harness/` (the runnable harness is `../scripts/harness/`, verbatim)

- `bench-harness.vs-infera.glm52.pd@28121cb4.yihou.diff` — whole `bench-harness/` vs `bench/glm5p2_pd` @ 28121cb4.
- `engine.sh.yihou.diff`, `agentx_bench.sh.yihou.diff`, `agentx_env.py.yihou.diff` — phase-2 changes (copied from the
  phase-2 pack-up): `DECODE_PID_OFFSET` (R022), `ENGINE_PID_HOST`/`IPC_MODE_LEGACY`/`DEBUG_AGENT` knobs, `NODE_RW_ROOT`,
  agentx_env validation of the new knobs.
- `bench-harness.since-packup_20261010.yihou.diff` — this phase: per-leg `{PREFILL,DECODE}_CPUSET_{CPUS,MEMS}` +
  `ENGINE_SET_CPU_AFFINITY` (R045) and role-only `{PREFILL,DECODE}_MOUNTS_ADD` (R050). **All default empty → the
  e2g command line is byte-identical to before** (verified by re-dryrun).
- Warm protocol driver: `scripts/harness/a2_e2g.yihou.sh` (flush decode → flush prefill → BENCH_TAG=A2 bench).

## 4. Experimental — NOT in the 22k config

`runtime-mounts/experimental/` (full file + diff vs image) and `mooncake-experimental/`:

| file | gate | what | result |
|---|---|---|---|
| `conn.yihou-regbase-xferlog-par.py` | `YIHOU_XFER_LOG=1`, `YIHOU_XFER_PAR=k` | per-minute KV-transfer stats; optional k-way layer-group submission | e2e (R037); log-only at PAR=1 |
| `conn.yihou-xferfix.py` + `prefill.yihou-nonfinal-event.py` | `YIHOU_XFER_SHARD=room`, `YIHOU_NONFINAL_EVENT=launch` | room-sharded transfer queues; record the non-final-chunk event at launch (removes the over-wait of R036) | e2f 19,864 (fresh); TTFT best, score not improved (decode-bound, R040-R042) |
| `eagle_disaggregation/eagle_worker_v2.yihou-seedguard.py` | `YIHOU_DSA_SEED_GUARD=count` | count/mask out-of-pool DSA seed entries | e2d: 0 bad entries in ~1.24M rows (R031) |
| `graph_variants/eagle_utils.yihou-hostvariant.py` | `YIHOU_DSA_HOST_VARIANT=1` (decode) | pick the SPARSE DSA graph variant from a host lower bound, skipping `seq_lens.max().item()` | e2j: verified sync removed (R054); warm R2 21,681 — no measurable gain (R057) |
| `mooncake-hip-batch-kernel…diff` | `MC_HIP_BATCH_KERNEL=1` | one batched peer-copy kernel per batch (option B') | not built; moot after R035 (blocks are large) |
