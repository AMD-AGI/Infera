# Patches — what / why / how / context

Four layers. Ledger IDs (R0xx) point into `../analysis/working_process.md`.

## 1. Image recipe — `image-recipe/` (baked into `infera-sglang:v0521-yihou-1010-mc4340`)

Apply in order on `infera.glm52.pd` @ `28121cb4` (or use the clone, see REPRODUCE §1).
Full Dockerfiles also copied to `image-recipe/files/`.

| file | what | why | context |
|---|---|---|---|
| `01-e3e46f32-router-stash-conflicts.patch` | resolve conflict markers left in `rust/router/src/{policy.rs,disagg.rs}` by the user's rebase/stash; PD DP-rank affinity ported onto upstream's `pick_targets`/sessions API | router did not build | phase 1, mission "find and fix errors from the rebase" |
| `02-fb8ae9c9-v0521-engine-recipe.patch` | `Dockerfile.sglang.v0521.yihou`: base pinned to the v0.5.21 nightly; GLM-5.2 DSA patches re-cut for v0521 (`dsa_page_table_rows.diff`, NextN fused-shared-experts arch fix, prefill non-final-chunk wait-event); drops patches now upstream (#32196, #41395, #41756, #37152); builds mooncake faae8dd4 | v0519 recipe's context diffs do not apply to v0521 | phase 1 |
| `03-cdba2446-v0521-mc4340-recipe.patch` | `Dockerfile.sglang.v0521-mc4340.yihou` + `build_mooncake_sglang.mc4340.yihou.sh` + `mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff` (Mooncake PR #4340 HIP-IPC sub-allocation relocation, ported to faae8dd4; transfer_engine_impl hunks hand-rebased) | **R018/R019**: HIP opens an IPC handle at the allocation base; faae8dd4 relocated with the registered pointer → KV written at (ptr − base) offset for interior registrations, silently | proven end-to-end by GPU te_pair (interior region correct only with #4340). In the passing rounds REGBASE showed `interior=0`, so the bug was not triggered by this config — the image is kept because it is the safe transport |
| `04-68f06e29-mc4340-pipefail.patch` | build-script artifact check: avoid pipefail SIGPIPE false failure | r1 build failed on its own check | build r2 succeeded |

## 2. Runtime mounts — `runtime-mounts/` (bind-mounted `:ro` over the image's sglang files by `engine.sh`, both legs)

Each `.py` is the full file mounted; each `.diff` is against the file inside the image.

| mounted file → target | what | why | context |
|---|---|---|---|
| `kv_events_publisher.yihou-noblock.py` → `sglang/srt/managers/scheduler_components/kv_events_publisher.py` | auxiliary KV-metrics ZMQ PUSH sends with NOBLOCK and drops on EAGAIN | **R001**: v0521 PUSH connects to an endpoint nothing binds; a blocking send wedged the prefill scheduler event loop | c44a hung |
| `forward_mla_rocm.yihou-noabsorbcfg.py` → `sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` | `_absorb_bmm_config` returns None (aiter picks its own config, as v0519 did) | **R003**: v0521's hand-built config at M=256 aborts the triton LLVM backend | c44b crashed |
| `conn.yihou-regbase.py` → `sglang/srt/disaggregation/mooncake/conn.py` | logs `YIHOU-REGBASE … interior=` per registered region (hipMemGetAddressRange) | diagnostic for R018: are production registrations interior? | **diagnostic only, removable**; all passing rounds: `interior=0`, `extends past its allocation`=0 |

## 3. Harness — `harness/`

The runnable harness is `../scripts/harness/` (copied verbatim). Diffs:

- `bench-harness.vs-infera.glm52.pd@28121cb4.yihou.diff` — the whole `bench-harness/` vs
  `infera.glm52.pd/bench/glm5p2_pd` @ 28121cb4 (lineage: 0922/0923 same-node pack-ups → oai pack-up →
  this). Files: agentx_bench.sh, config.sh, config.full.sh, engine.sh, launch.sh, stop.sh,
  tools/{agentx_env,topology,wait_healthy}.py.
- `engine.sh.yihou.diff`, `agentx_bench.sh.yihou.diff`, `agentx_env.py.yihou.diff` — this phase's
  changes only (vs the phase-1 copies). Key items:
  - **`DECODE_PID_OFFSET=K`** (R022): decode command becomes
    `bash -c 'i=0; while [ $i -lt K ]; do ( : ); i=$((i+1)); done; exec "$@"' _ <argv>`.
    Rank schedulers forked later get PIDs > K, disjoint from prefill → avoids the CLR same-PID
    IPC guard (R020, 201) and the ROCr `@hsa<pid>` socket collision (R012). `exec` keeps the
    wrapper's PID, so only later forks are offset (fine: the IPC users are the forked ranks).
  - `ENGINE_PID_HOST` / `IPC_MODE_LEGACY` / `DEBUG_AGENT` knobs (off in the passing rounds);
    `ENGINE_PID_HOST=1` refused on spur nodes (spur-authz).
  - `NODE_RW_ROOT` for nodes where /shared_nfs is read-only (136; unused on 102) — engine.sh and
    agentx_bench.sh.
  - agentx_env validates the new knobs against the live container (fused-indexer flag + writer env,
    `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB`, PidMode, legacy IPC). The `attn_dp_size` comparison
    (v0521 reports dp_size=1 with attn_dp_size=4, R002) predates this phase and is in the lineage diff.
  - `INFERENCEX_DIR=$(readlink -f …)` in `config.yihou.v0521-base.sh:181` so agentx_bench gives it its
    own `:ro` mount (R023: the symlinked cache dangled inside the client container).

## 4. Optional, NOT used by the passing rounds — `optional-pr42553/`

sglang PR #42553 @ 8c27ef48 (open): HiCache host pools via one exact-size `hipHostMalloc` (GTT)
when `HSA_USERPTR_FOR_PAGED_MEM=0`. GPU-validated (unit test 15/15, 150 GB pool D2H/H2D OK).
**Cannot be used with `--hicache-size 450`**: GTT is node-wide MemTotal/2 ≈ 1477 GB, target needs
≈ 2.21 TB. Overlay `scripts/harness/config.yihou.gtt42553.sh` has a guard. See R021.
