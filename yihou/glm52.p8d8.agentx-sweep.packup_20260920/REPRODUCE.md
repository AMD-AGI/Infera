# REPRODUCE — P8D8 AgentX concurrency sweep, GLM-5.2 MXFP4

This reproduces the five-point AgentX sweep recorded in
`results/sweep_results.yihou.md`: CONC = 80, 112, 144, 192, 256, each a full
3,600 s profiling window, all run against **one** 1P1D P8D8 deployment. The
result of record is the FINAL table in that file — five complete points, zero
memory-access faults, zero RDMA rail faults, deployment up 13 h 33 m.

Read `spec/mission.md`, `results/sweep_results.yihou.md`, and
`analysis/config_design.yihou.md` before starting. This file is the ordered
runbook; those explain *why*.

Every command below uses `crsuse2-m2m-137` as the prefill node and
`crsuse2-m2m-136` as the decode node, with the data-plane IPs recorded in
`scripts/topology.yihou.tsv` (`137` = `10.245.153.247`, `136` = `10.245.154.168`).
These are the exact nodes and IPs of the run; on a different allocation you must
substitute your own and re-survey the rails (step 4b).

Notation: `$KIT` = `/home/yihou/dev/git/infera.glm52.view/glm52.p8d8.agentx-sweep.packup_20260920`
(this directory).

---

## 0. What this run is, and is NOT — read before spending cluster time

- **Correctness is waived by construction.** The decode leg runs with
  `DECODE_SIMULATE_ACC_LEN=3.61` (simulated MTP acceptance). Simulation forces
  the *count* of accepted draft tokens, not *which* tokens are right, so the
  deployment **emits garbled text** (`results/sweep_results.yihou.md`, the
  correctness-caveat sections). These numbers are timing under forced acceptance,
  on the same footing the reference c32/c40 kit discloses about its own numbers —
  they are NOT a correctness run. The acceptance gauge reading ~3.6 carries no
  correctness information.
- **The `spec_accept_length` metric is not a health signal here** — it reports the
  value it was told to report. Do not use it to judge the run.
- These are the things the record forbids softening; keep them attached to any
  number you quote downstream.

---


## 0a. External dependencies you must supply — read before starting

This kit is **not** fully self-contained. Four things live outside it:

| what | where to get it | why not in the kit |
|---|---|---|
| **Docker image** `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, id `sha256:fd7220a57b7d…` | transferred node-to-node with `docker save \| ssh \| docker load`; base is `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917` on Docker Hub | 66.3 GB |
| **Bench harness** `bench/glm5p2_pd` from worktree `/home/yihou/dev/git/infera.yihou.glm52.p8p4` @ `ad3b85d3`, seeded from `/home/yihou/dev/git/infera.glm52.pd` | that checkout, or reconstruct from `scripts/bench-harness/` | it is a repo, not a file set |
| **InferenceX / aiperf checkout** — `agentx_bench.sh` clones it into `$BENCH/.cache/InferenceX` at the ref pinned by `INFERENCEX_REPOSITORY` / `INFERENCEX_REF` | fetched automatically by `tools/ensure_inferencex.py` | fetched at run time |
| **Model weights** `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | shared NFS | ~ hundreds of GB |

**What IS vendored here**, so the harness side can be audited or rebuilt:
`scripts/bench-harness/` holds `config.sh`, `config.full.sh`, `engine.sh`, `launch.sh`,
`stop.sh`, `agentx_bench.sh` and `tools/{topology,wait_healthy,agentx_env,ensure_inferencex,plot_agentx,collect_agentx}.py`,
all with `SHA256SUMS.txt`. Every `file:line` citation in this kit refers to those copies.
`patches/` holds both patches that are baked into the image, with verification commands.

## 1. Prerequisites and secrets

Access and mounts (all first-hand from the run):

1. **SSH** (key-based, `BatchMode=yes`) from your orchestration host to
   `crsuse2-m2m-135` (image source), `-137` (prefill), `-136` (decode). Every
   script uses `ssh -o BatchMode=yes`; no passwords.
2. **Docker** usable on 135/137/136, with docker root on the node-local disk
   `/mnt/m2m_nobackup/docker` (per `analysis/image_transfer.yihou.md`: 137 had
   8.7 TB free, 136 had 7.7 TB free; threshold ~120 GB).
3. **Model on NFS**: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, readable
   on 137 and 136. This path is `MODEL` in `scripts/bench-harness/config.full.sh:15`
   and the `--tokenizer`/router tokenizer source.
4. **Host RDMA lib**: `/lib/x86_64-linux-gnu/libionic.so` present on both nodes
   (ionic NICs).
5. **Node-local scratch**: `/mnt/m2m_nobackup` writable on both nodes; the run
   used `/mnt/m2m_nobackup/yihou_p8p4/**` for launch artifacts, server logs, and
   per-point results. **Keep all writes off `$HOME` / NFS** (shared, near-full).
6. **The bench worktree** at `$BENCH` (see step 3), present on whichever node
   sources the config — `launch.sh` runs `engine.sh` on the *remote* node with an
   absolute `CONFIG=` path, so both the config file and its `BENCH_DIR` target must
   exist there.

Secrets — **names and sources only, never values**:

- **Hugging Face token** — env `HF_TOKEN` (standard HF cache/`HF_HOME` under
  `$AGENTX_CACHE_DIR/hf`). The AgentX client pulls the trace dataset
  `semianalysisai/cc-traces-weka-062126` (`results/c080/agentx_conc80.json`
  `dataset` block). Source: your team's HF account.
- **InferenceX checkout credentials** — `agentx_bench.sh` calls
  `tools/ensure_inferencex.py --repository "$INFERENCEX_REPOSITORY" --ref
  "$INFERENCEX_REF"` (values come from `config.full.sh`). If that repo is private,
  its git/HTTPS token is needed. Source: the bench repo's own README / your team.

No secret VALUES appear anywhere in this kit; supply them from your own store.

---

## 2. Obtain the image — transferred node-to-node, NOT built

The image is `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, id
`sha256:fd7220a57b7d…` (66.3 GB), base
`lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`, in-image sglang
`0.5.19.dev20260917+ga9fb1c3238`. It was **transferred** from `-135` to `-137`
and `-136`, never rebuilt — this removes "did the patches apply" as a variable.
Full record: `analysis/image_transfer.yihou.md`.

**2a. Disk pre-check** (both targets):

```bash
ssh crsuse2-m2m-137 "df -h /mnt/m2m_nobackup/docker"
ssh crsuse2-m2m-136 "df -h /mnt/m2m_nobackup/docker"
```

**2b. Confirm the source (read-only, do not disturb 135/138):**

```bash
ssh crsuse2-m2m-135 "docker images infera-sglang:v0519-yihou-0917-nextnfix-hicache \
  --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'"
# expect: infera-sglang:v0519-yihou-0917-nextnfix-hicache fd7220a57b7d 66.3GB
```

**2c. Stream `docker save | docker load`, one target at a time** (sequential is
safer than parallel; no 66 GB tar ever lands on NFS):

```bash
set -o pipefail
ssh crsuse2-m2m-135 "docker save infera-sglang:v0519-yihou-0917-nextnfix-hicache" \
  | ssh crsuse2-m2m-137 "docker load"    # -> "Loaded image: ...", exit 0
ssh crsuse2-m2m-135 "docker save infera-sglang:v0519-yihou-0917-nextnfix-hicache" \
  | ssh crsuse2-m2m-136 "docker load"    # -> "Loaded image: ...", exit 0
```

**2d. Confirm the id on both targets:**

```bash
ssh crsuse2-m2m-137 "docker images infera-sglang:v0519-yihou-0917-nextnfix-hicache --format '{{.ID}} {{.Size}}'"
ssh crsuse2-m2m-136 "docker images infera-sglang:v0519-yihou-0917-nextnfix-hicache --format '{{.ID}} {{.Size}}'"
# both -> fd7220a57b7d 66.3GB
```

**2e. Verify the FIVE in-container markers — run on 137 AND on 136, each on its
own node** (uses a `--rm` CPU-only container, touches no GPU):

```bash
ssh crsuse2-m2m-<NODE> 'docker run --rm --entrypoint bash \
  infera-sglang:v0519-yihou-0917-nextnfix-hicache -lc "
S=/sgl-workspace/sglang/python/sglang
python3 -c \"import sglang;print(sglang.__version__)\"
grep -c '\''fused_shared_experts_architecture = \"GlmMoeDsaForCausalLMNextN\"'\'' \$S/srt/models/glm4_moe.py
grep -c pick_group_bytes      \$S/kernels/jit/csrc/kvcacheio/hicache.cuh
grep -c _tiles_across_lanes   \$S/kernels/ops/kvcache/hicache.py
grep -c '\''can_use_jit = (_is_cuda or _is_hip)'\'' \$S/srt/mem_cache/pool_host/mha.py
"'
```

Expected on both nodes: `0.5.19.dev20260917+ga9fb1c3238` / `1` / `2` / `2` / `2`.
Marker 2 is the **NextN shared-experts fusion fix**; markers 3–5 are **sglang
PR #37152** (OPEN upstream, never merged, its AMD ROCm CI RED — applied because
the user asked for it). A miss on any line means the wrong image; stop.

---

## 3. Checkout and where the configs go

Two trees are involved:

1. **Bench harness** — a private worktree of the infera repo, seeded with the
   patched `bench/glm5p2_pd` scripts. The run used
   `/home/yihou/dev/git/infera.yihou.glm52.p8p4` @ `ad3b85d3`, so
   `$BENCH = /home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd`. The
   exact vendored copies of the harness (`launch.sh`, `engine.sh`, `agentx_bench.sh`,
   `stop.sh`, `config.sh`, `config.full.sh`, `collect_agentx.py`) are in
   `$KIT/scripts/bench-harness/` with `SHA256SUMS.txt`. Place them at
   `$BENCH` (together with the repo's `tools/` — `topology.py`, `wait_healthy.py`,
   `agentx_env.py`, `ensure_inferencex.py` — which the worktree already carries).
   The rust router patch is baked INTO the image, so the rust sources are not
   needed.
2. **This kit's driver + config** — `$KIT/scripts/`. The sweep configs
   (`config.yihou.p8d8.sh`, `topology.yihou.tsv`) and the orchestration scripts
   (`run_full_sweep.yihou.sh`, `sweep_driver.yihou.sh`, `wait_gpus_free.yihou.sh`)
   live here.

The orchestration scripts read `$WS/scripts/...`; point `WS` at this kit:

```bash
export KIT=/home/yihou/dev/git/infera.glm52.view/glm52.p8d8.agentx-sweep.packup_20260920
export WS="$KIT"
export BENCH=/home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd
```

`config.yihou.p8d8.sh:227` hard-defaults `BENCH_DIR` to that same worktree path
and `source`s `$BENCH_DIR/config.full.sh`; override `BENCH_DIR` if you placed the
harness elsewhere.

**What the config fixes (all in `scripts/config.yihou.p8d8.sh`, rationale in
`analysis/config_design.yihou.md`):**

- Shape: `PREFILL/DECODE_GPU_DEVICES=0..7`, `TP=8`, `DP=8` both legs
  (`:28-33`); DP attention on.
- `*_MAX_RUNNING=256` and `*_GRAPH_MAX_BS=256` (`:58-61`) — **raised from
  config.full.sh's 128 on purpose**. `--max-running-requests` is a *global* cap
  divided across DP ranks; at 128 the 144/192/256 points would be silently
  queue-limited. See `analysis/max_running_cap.yihou.md`.
- Simulated acceptance left UNSET so `config.full.sh:82`'s no-colon
  `${DECODE_SIMULATE_ACC_LEN-3.61}` yields 3.61 (simulation ON).
- DSA compute backends `tilelang` (`:206-207`); `DSA_TOPK_BACKEND` cleared
  **after** the source (`:232`) so `aiter` is not silently restored.
- Same-rail KV path: `PD_DP_RANK_AFFINITY=1` (`:176`, translated `1→true` for the
  router's clap in `launch.sh:128`) **plus** the 8-entry per-GPU `RDMA_DEVICE`
  JSON map and `MC_TE_FILTERS` comma list (`:187-194`, `GPU_n↔ionic_n`).
- **`index_share_for_mtp_iteration=false`** — the issue.md §3.3 memory-fault
  mitigation. It is assigned at `:238-240` **AFTER** `source "$BENCH_DIR/config.full.sh"`,
  because `config.full.sh:90` is a plain unconditional `JSON_MODEL_OVERRIDE_ARGS=""`
  that would otherwise clear it. This ordering is load-bearing; do not move it
  above the source.

Prefill HiCache is on (ratio 1.5); decode HiCache is off — `engine.sh:76`
hard-rejects decode HiCache with MTP, so PR #37152 acts on the prefill leg only.

---

## 4. Pre-launch GPU-VRAM gate — why it exists, and it is not optional

`run_full_sweep.yihou.sh` step 1 blocks on `wait_gpus_free.yihou.sh` until every
GPU on both nodes reports ≤1 % VRAM (the idle floor is ~284 MiB of driver
reserve, never literally 0).

**Why (from the script header, a real 2026-09-18 incident):** after `stop.sh` the
containers vanish and *host* memory returns fast, so teardown *looks* finished —
but GPU VRAM keeps draining for many minutes, one device at a time. A deployment
relaunched two minutes into that drain could not get its allocation on several
ranks, startup jammed with three GPUs still near-empty, and the wreckage was
misdiagnosed as a node fault needing an admin reset. The node was fine and
recovered on its own ~20 min later. Checking `docker ps` and `free` is not
checking the thing that matters. An unreachable node or unparseable `rocm-smi`
counts as NOT free.

Run it standalone first if you want to watch it:

```bash
TIMEOUT_S=3600 INTERVAL_S=30 bash "$WS/scripts/wait_gpus_free.yihou.sh" \
    crsuse2-m2m-137 crsuse2-m2m-136
# -> "ALL GPUS FREE on: crsuse2-m2m-137 crsuse2-m2m-136", exit 0
```

**4b. (only if you changed nodes)** Re-survey rails/GID/NUMA and rebuild the
`RDMA_DEVICE` map and `MC_TE_FILTERS` list. On 137/136 the record verified
`GPU_n↔ionic_n` NUMA-local, all rails ACTIVE, same rail-id per index across both
nodes (`analysis/hardware_prep.yihou.md`, `scripts/rdma_map.yihou.md`). On other
nodes these values may differ and a wrong map breaks the same-rail KV path.

---

## 5. Launch — one deployment for all five points

`run_full_sweep.yihou.sh` does gate → remove stale `p8d8` containers (all names
carry `yihou`) → launch → wait for health → pre-flight checks → run every point.
The whole sweep is one call:

```bash
cd "$WS"
LAUNCH_OUT=/mnt/m2m_nobackup/yihou_p8p4/launch/sweep-$(date -u +%Y%m%dT%H%M%SZ) \
  bash scripts/run_full_sweep.yihou.sh 80 112 144 192 256
```

Internally the launch is (do not run by hand unless debugging):

```bash
cd "$BENCH" && ./launch.sh \
  CONFIG="$WS/scripts/config.yihou.p8d8.sh" \
  TOPOLOGY="$WS/scripts/topology.yihou.tsv" \
  OUT_DIR="$LAUNCH_OUT"
```

The launch summary of the run of record is `logs/launch-summary.txt`:
prefill `GPUs=0..7 TP=8 EP=1 DP=8 DPA=1 HiCache=1`, decode
`GPUs=0..7 TP=8 EP=1 DP=8 DPA=1 HiCache=0`, then `router ready`.
Server logs go to `$LAUNCH_OUT/server-logs/{prefill-0,decode-0}.log`.

Health is polled (up to ~40 min; a cold start on this model has taken ~22 min)
on all three endpoints — these must all return `200`:

```
prefill  http://10.245.153.247:29001/health
decode   http://10.245.154.168:29002/health
router   http://10.245.153.247:28000/health
```

---

## 6. Checks that MUST pass before you spend an hour

The orchestrator runs these automatically and aborts on failure; verify them
yourself if you launched by hand.

**6a. The engine must actually carry `index_share_for_mtp_iteration`.** A setting
written into a config is not a setting the engine received.

```bash
ssh -o BatchMode=yes crsuse2-m2m-136 \
  'docker inspect glm52-pd-yihou-p8d8-decode-0 \
   --format "{{join .Config.Cmd \" \"}}" | grep -c index_share_for_mtp_iteration'
# expect 1 — else ABORT (mitigation missing)
```

**6b. `SGLANG_DSA_FUSE_TOPK` must be ABSENT.** `SGLANG_DSA_FUSE_TOPK=0` was the
earlier §3.3 workaround; it stopped the crash, that run was garbled, and it was
**withdrawn** on 2026-09-19. Do NOT read the withdrawal as blaming it for the
garbling: `SGLANG_DSA_FUSE_TOPK=0` was predicted in writing to be the cause, and
that prediction was **falsified** — restoring the fused path did not fix the
garbling (`results/sweep_results.yihou.md`, "Hypothesis falsified"). Garbling is
attributable to simulated acceptance (§0), not to this flag. It must still be
absent because it is the superseded config; if it is set, you are not on the run
of record.

```bash
ssh -o BatchMode=yes crsuse2-m2m-136 \
  'docker inspect glm52-pd-yihou-p8d8-decode-0 \
   --format "{{join .Config.Env \"\n\"}}" | grep -c "SGLANG_DSA_FUSE_TOPK=0"'
# expect 0 — else ABORT (withdrawn workaround still set)
```

**6c. Rail health is zero, before you start.** A benchmark over a dead rail is not
a measurement:

```bash
ssh -o BatchMode=yes crsuse2-m2m-137 \
  "strings $LAUNCH_OUT/server-logs/prefill-0.log | \
   grep -cE 'transport retry counter exceeded|wqe is not posted'"
# expect 0
```

**6d. Text-coherence gate — a note, not a pass/promise.** The orchestrator sends
three "first ten primes" probes and by default aborts if they are garbled. **This
configuration is expected to garble** (§0), so the driver only proceeds past this
gate with `ALLOW_GARBLED=1` set — which is the deliberate acceptance that
correctness is waived. `run_full_sweep.yihou.sh` will otherwise stop here. Set
`ALLOW_GARBLED=1` in the environment for the sweep, exactly as the record did, and
keep the §0 caveat attached to the numbers.

`SGLANG_OPT_USE_TOPK_V2=false` should also be confirmed present in the live decode
env (it is the `config.full.sh:101` default, forwarded by `engine.sh`):

```bash
ssh -o BatchMode=yes crsuse2-m2m-136 \
  'docker inspect glm52-pd-yihou-p8d8-decode-0 \
   --format "{{join .Config.Env \"\n\"}}" | grep SGLANG_OPT_USE_TOPK_V2'
# expect SGLANG_OPT_USE_TOPK_V2=false
```

---

## 7. Run the five points

`run_full_sweep.yihou.sh` (step 5) hands off to `sweep_driver.yihou.sh`, which
runs each point against the ONE live deployment — it never launches or stops
engines. Per point it: health-gates all three endpoints, counts rails before,
runs `agentx_bench.sh CONC=<n>` (DURATION inherited 3,600 s), counts rails after,
and stops the driver on any non-zero exit. Each point invokes:

```bash
cd "$BENCH" && ./agentx_bench.sh CONC=<n> \
  CONFIG="$WS/scripts/config.yihou.p8d8.sh" \
  TOPOLOGY="$WS/scripts/topology.yihou.tsv" \
  OUT_DIR=/mnt/m2m_nobackup/yihou_p8p4/sweep/c<NNN> \
  AGENTX_CACHE_DIR=/mnt/m2m_nobackup/yihou_p8p4/agentx-cache
```

Artifacts land in `$SWEEP/c<NNN>/` (`agentx_conc<n>.json`,
`profile_export_aiperf.csv`) plus `$SWEEP/rails-c<NNN>.txt` and the running
`$SWEEP/driver.log`. `DURATION=3600 ≥ 900`, so `submission_valid` is preserved.

To resume or run a subset (a point whose `$SWEEP/c<NNN>/` already exists is
skipped):

```bash
SRVLOG=$LAUNCH_OUT/server-logs/prefill-0.log \
  bash "$WS/scripts/sweep_driver.yihou.sh" 144 192 256
```

---

## 8. Expected wall-clock (from `logs/driver.log`, run of record)

Profiling is 3,600 s (1 h) per point. Warmup is 10 requests/lane; past saturation
a share of requests each burn the full 1,800 s `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`
before failing and releasing, so warmup lengthens sharply with concurrency
(`results/sweep_results.yihou.md`, final section).

The last column is a single derived quantity — `wall − 3,600 s`, the non-profiling
overhead (warmup + drain + per-point setup) — computed the same way for every row
from the `logs/driver.log` START→END stamps. It is NOT the record's warmup figure;
`results/sweep_results.yihou.md` reports warmup only for c080/c192/c256 (~27 min /
~2 h 11 m / ~3 h 10 m), which is a smaller quantity than this total overhead.

| point | driver START→END (wall) | overhead = wall − 3,600 s (derived) |
|---|---|---|
| c080 | 12:23:01 → 13:52:00  (~1 h 29 m) | ~29 min |
| c112 | 13:52:03 → 15:24:20  (~1 h 32 m) | ~32 min |
| c144 | 15:24:23 → 17:49:02  (~2 h 25 m) | ~1 h 25 m |
| c192 | 17:49:04 → 21:15:53  (~3 h 27 m) | ~2 h 27 m |
| c256 | 21:15:56 → 01:47:15  (~4 h 31 m) | ~3 h 31 m |

Whole sweep: deployment up 2026-09-19 12:14:08, driver finished 2026-09-20
01:47:15 = **13 h 33 m** (`results/sweep_results.yihou.md`). Budget a full day.
Do NOT read a long c144+ warmup as a hang — it is the timeout-drain crawl
(~8 req/min), documented in the "slow-warmup mechanism" section.

**Teardown** costs GPU-memory release, not instant: ~33 min measured
(`run_full_sweep.yihou.sh` header; the record frames it as 33–40 min). Use
`stop.sh`, then re-run `wait_gpus_free.yihou.sh` before any relaunch:

```bash
cd "$BENCH" && ./stop.sh \
  CONFIG="$WS/scripts/config.yihou.p8d8.sh" \
  TOPOLOGY="$WS/scripts/topology.yihou.tsv"
```

CUDA-graph capture at bring-up is also slow (`GRAPH_MAX_BS=256`); watch the build
dir, do not assume a hang.

---

## 9. How to read the results

Primary per-point file: `results/c<NNN>/agentx_conc<n>.json`. The headline
figures come from the `request_metrics` block:

- **tok/s/chip** = `.request_metrics.throughput.per_gpu.total_tput_tps`
  (c080 = 19513.2).
- **TTFT p50/p90 (s)** = `.request_metrics.latency.ttft.p50` / `.p90`.
- **ITL p50 (s)** = `.request_metrics.latency.itl.p50` (JSON is seconds; the
  table reports ms).
- **interactivity p50** = `.request_metrics.latency.intvty.p50`.
- **profiled / errors** = `.request_accounting.records_profiled` /
  `.records_error_dropped` (all errors were `InvalidInferenceResultError`).
- **rails** = `results/c<NNN>/rails-c<NNN>.txt` (before/after
  `transport retry counter exceeded|wqe is not posted`); every point recorded
  `0/0`.

Server logs are NOT in this kit (110 MB prefill / 90 MB decode); they remain on
the nodes at
`/mnt/m2m_nobackup/yihou_p8p4/launch/sweep-20260919T113947Z/server-logs/`. Only
gzipped grep excerpts are packed under `logs/`.

**Reading discipline (do not invert):**

- **Peak throughput is at or below CONC 80**, the sweep's lowest point — the
  maximum is OUTSIDE the measured range. 80→112 loses only 2.6 %; 112→144 falls
  34 %; the knee is between 112 and 144.
- **ITL and interactivity IMPROVE monotonically as the system collapses** — their
  best readings belong to the WORST point (c256). In PD the queue forms on the
  prefill side and decode starves; these metrics measure the survivors, not the
  system. Use tok/s/chip, TTFT p90, and completed-request count as the honest
  signals; ITL actively misleads.

---

## 10. Success criteria vs. the actual result

Mission target (`spec/mission.md`): P8D8, simulation ON at 3.61, `full` mode
(`DURATION=3600`), five points CONC = 80/112/144/192/256, one deployment; an
out-of-memory failure at the top point was an accepted end state; then pack up.

Actual result of record (`results/sweep_results.yihou.md`, FINAL table):

- All **five points completed** their full 3,600 s window, each
  `submission_valid`-eligible (`DURATION ≥ 900`).
- **The anticipated OOM never happened.** The real limit is **queueing collapse**:
  at CONC 256 TTFT p90 is 510 s (~8.5 min), errors reach 487, profiled falls to
  3,659 — 38 % of CONC 112's completions. No out-of-memory end state was reached.
- **13 h 33 m fault-free**: zero `Memory access fault`, zero `Fatal Python error`,
  zero RDMA rail faults across all five points, with
  `index_share_for_mtp_iteration=false` and custom all-reduce ON. (The same shape
  with IndexShare ON faulted at 1 h 26 m — do not run it that way.)
- **Correctness waived by construction** (§0) — garbled text; timings only,
  comparable to the reference c32/c40 kit, not to a correctness run. Measured cost
  of the `index_share=false` mitigation at CONC 80: 0.3 %, inside noise.

You have reproduced the run when: the five `agentx_conc<n>.json` files exist with
per-GPU throughput and TTFT tracking the FINAL table
(19,513 / 19,014 / 12,555 / 7,409 / 5,329 tok/s/chip), all five `rails-c<NNN>.txt`
read `0/0`, and the server logs show no memory-access or Python-fatal faults for
the deployment's lifetime.
