# Task: finish validating and land three upstream sglang GLM-5.2 PRs

Continue the hardware validation that stalled on n06-33 (another cluster, now gone).
Review, validate on gfx950, and flip from draft to ready:
[#33968](https://github.com/sgl-project/sglang/pull/33968),
[#33970](https://github.com/sgl-project/sglang/pull/33970),
[#33973](https://github.com/sgl-project/sglang/pull/33973).

Hand-off from the previous session: `pr-verify.experiment/` (README -> context.md ->
plan.md -> working_process.md). Read it; it holds nine traps that each cost real time.
This session's workspace: `pr-verify.experiment/rounds/`.

## The three PRs

| PR | Fix | Upstream file(s) | Local patch |
|----|-----|------------------|-------------|
| #33968 | HiCache ROCm host-pool allocator | `mem_cache/pool_host/common.py` | `sglang_rocm/patch_hicache_rocm_host_alloc.py` |
| #33970 | mooncake KV transfer waits on prefill forward | `disaggregation/{common/utils,mooncake/conn,prefill}.py` | `sglang_disagg/patch_mooncake_early_send_wait_event.py` |
| #33973 | DSA decode DP-divergent D2H syncs | `layers/attention/dsa_backend.py` | `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` (2a half) |

Branches live on `dorado269/sglang`. Local checkout: `/home/yihou/dev/git/sglang`
(`origin`=sgl-project only — **the `fork` remote must be re-added**).

## Corrections to the hand-off, established first-hand this session (2026-08-24)

1. **The previous session's rebase was never pushed.** PR head SHAs are
   `7ea3ee8e` / `d044c7a3` / `8b374fed`, not the `33f0ea6cd3` / `780fbb3018` /
   `2b3c9ea7a3` recorded in `context.md`; `updatedAt` is still 2026-08-07 on all
   three. **The rebase has to be redone**, and the three defects re-confirmed
   present on whatever `main` is now.
2. **This cluster runs the same amdgpu 6.14.14** as n06-33 — the one uncontrolled
   variable `plan.md` step 3 names for #33968 has *not* changed. Expect the probe
   to measure `same=True` again. Run it to confirm, then stop; do not plan work
   behind it.

## Environment — this cluster (crsuse2-m2m, spur), first-hand

- Login node `crs-m2m-cpu-spur-012`. Scheduler is **spur**, not stock Slurm —
  see the `spur-cluster-usage` and `spur-interactive-debug` skills.
- Held nodes: **`crsuse2-m2m-237` (job 58799)** and **`crsuse2-m2m-106` (job 58800)**,
  12 h limit, started 2026-08-24 02:49 UTC. Reach them with `spur exec <jobid> <cmd>`
  (runs as the real user on the host; `ssh` to compute nodes is blocked).
- Both: 8x MI355X gfx950, **0 KFD procs**, amdgpu 6.14.14, ROCm 7.0.1 host-side.
- Storage: `/mnt/m2m_nobackup` 28T (~25T free), **node-local**, docker root lives there.
  `/` is only 123G — keep everything off it.
- **Model: `/mnt/m2m_nobackup/models/mlx-community__GLM-5.2-mxfp4`**, 368 G, 76
  shards, present on **both** nodes. `config.json` says `GlmMoeDsaForCausalLM`, so
  it is the DSA architecture #33973 needs. `zai-org__GLM-5.2-FP8` (708 G) also present.
- Docker: daemon per node, registry reachable, pinned base
  `lmsysorg/sglang:v0.5.17-rocm720-mi35x` manifest-inspects OK. No infera image
  cached — **it has to be rebuilt here**; the n06-33 one is gone with that cluster.

### Network — the big difference from n06-33

- **The rail NICs carry no IP.** `ionic_0..7` are all `PORT_ACTIVE` and mapped to
  `enP2p0s9..12` / `enP3p0s9..12`, but none of those interfaces has an address.
  The n06-33 `192.168.N.x/31` rail scheme does not exist here.
- The only routed interface is **`ens3` (10.245.x/20)**, and **`mlx5_0` is bound to
  it** with `gid[3] = ::ffff:0af5:9abf`, RoCE v2. 237 = 10.245.154.191,
  106 = 10.245.159.121, 0.56 ms apart, ping OK.
- So the cross-node path n06-33 could not build **may be available here over
  mlx5_0/ens3**. Probe it before choosing; do not assume either way.
- **Enumerate the GID table on this fabric** rather than copying `MC_GID_INDEX=1`
  from n06-33 — trap 1 in `context.md`. For `mlx5_0` the RoCE v2 IPv4 entry is
  index **3**; for `ionic_0` it is 0 or 1. And remember a passing `ib_write_bw -x N`
  does *not* validate mooncake's GID index.

## Decisions taken with the user this session

- **#33970**: probe cross-node RDMA over mlx5_0/ens3 first. If it works, run the
  real 2-node 1P1D — the evidence n06-33 explicitly could not produce. If not,
  fall back to the single-node TP4+TP4 loopback plan, whose four premises r01
  already proved.
- **Priority**: #33970 and #33973 in parallel across the two nodes. #33968 gets
  only the `probe_host_devptr` run; if it measures `same=True` it is a negative
  control here too and stays draft.

## Core principles

1. **The positive control comes first.** Stock must reproduce the defect before a
   clean result on the patched tree means anything. `validate_{A,B,C}.py` all PASS
   without the defect ever reproducing — a PASS is *not* evidence a fix works.
2. **Suspend, don't conclude.** State what was measured. Where a mechanism is
   unknown, say so and name the measurement that would settle it.
3. **Rank evidence.** Code read / experiment run = high confidence. Anything from
   `pr-verify.experiment/*.md` is second-hand until re-verified here — two of its
   claims have already turned out stale.
4. **Scope.** Three PRs. Do not widen to the four patches that correctly get no PR
   (`work.todo.md`). The `#30350` fallout is a TODO the user owns, not work.
5. **DCO.** Every commit `-s`, signed off as the actual author. Never a bot or
   assistant identity. Upstream also rejects assistant `Co-Authored-By` trailers.
6. Work in English; report to the user in Chinese.

## Other notable details

- `gh` is at `/home/yihou/miniconda3/bin/gh`, authenticated as `dorado269`.
- Build wrapper: `.github/scripts/build_test_push.sh build sglang`, tags
  `${IMAGE}:sglang-${ID}`. One n06-33 build hit a transient
  `Could not resolve host: index.crates.io`; that was probed and is **not** a
  Dockerfile bug — retry it.
- All the hand-off scripts are present under `pr-verify.experiment/scripts/`:
  `validate_{A,B,C}.py`, `probe_host_devptr{,_sizes}.py`, `mvp_mooncake_loopback.py`.
- Keep temporary artifacts under `pr-verify.experiment/rounds/`, not scattered in
  the repo tree.
