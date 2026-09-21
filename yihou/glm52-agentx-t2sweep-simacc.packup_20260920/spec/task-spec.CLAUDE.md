# Task: GLM-5.2 1P1D AgentX with the corrected MTP config + HiCache PR #37152

Two benchmarks on the now-known-good MTP configuration, with two extra
optimisations layered in.

| # | mode | concurrency | simulated acceptance | deliverable |
|---|---|---|---|---|
| **T1** | fast (1200 s, warmup 1/lane) | 40 | **OFF** — real | report `spec_accept_length`; pack up |
| **T2** | full (3600 s, warmup 10/lane) | sweep 40, 56, 72, 96, 128 | **ON** (3.61) | pack up; 128 may OOM — stopping there is an acceptable end |

`accept_length` and `accept_rate` are **different metrics**. T1 reports length.

## What "the corrected MTP config" means

`--disable-custom-all-reduce` on the **decode** leg. Established 2026-09-18 by a
five-round single-variable A/B; without it this P4D4 shape emits garbled text
(first token correct, the rest degenerate) at `accept len 1.25`. With it, accept
length lands at 2.55–3.05. Full evidence and reproduction:
`yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/`.

Delivered through `DECODE_EXTRA_ARGS`, an escape hatch added to `engine.sh`
(uncommitted; also captured in that packup's `patches/`).

## The two added optimisations, and their real status

1. **`SGLANG_OPT_USE_TOPK_V2=false` — already the default.** `config.sh:93` and
   `config.full.sh:101` both read `${SGLANG_OPT_USE_TOPK_V2:-false}`, committed.
   `engine.sh:123` passes it into the container. **Nothing to change.** Assert it
   in the runtime snapshot rather than setting it again.
2. **Upstream sglang PR #37152 — HiCache ROCm copy-round widening.** Three
   source hunks: `kernels/jit/csrc/kvcacheio/hicache.cuh`,
   `kernels/ops/kvcache/hicache.py`, `srt/mem_cache/pool_host/mha.py`. All three
   are plain source files under `python/sglang/`, and the `.cuh` is JIT-compiled
   at runtime against a content-addressed cache — so a **thin docker layer is
   sufficient**; no 40–60 min full rebuild.
   - **Open upstream, never merged; upstream's own AMD ROCm CI on it is RED.**
     Applied because the user asked for it.
   - It only does anything when HiCache is on. **T1 runs `config.sh`, where
     `PREFILL_HICACHE=0`, so in T1 the patch is inert.** T2 runs
     `config.full.sh` with `PREFILL_HICACHE=1` — that is where it matters.
   - Source: `/home/yihou/dev/git/infera.glm52.view/agentx.c72.oai176219_20260918.packup_20260918/patches/`.

## Workspace — everything lives here

```
bench/glm5p2_pd/results/yihou-agentx-hicache/
```

Gitignored via `.gitignore:45`. Nothing outside it. Every container and artifact
name carries `yihou`. Override with `CONFIG=` / `TOPOLOGY=` / `OUT_DIR=`; never
mutate the repo's `topology.tsv` or `config.sh` in place — the tracked
`topology.tsv` points at 136/137/140, **not** our nodes.

## Fixed parameters

| | |
|---|---|
| topology | 1P1D — 135 prefill (`10.245.148.209`), 138 decode (`10.245.157.237`) |
| control / builder | crsuse2-m2m-135 |
| base image | `infera-sglang:v0519-yihou-0917-nextnfix` (on both nodes) |
| run image | `infera-sglang:v0519-yihou-0917-nextnfix-hicache` (thin layer, +PR37152) |
| GPUs | devices **2,3,4,5** on each node |
| shape | P4+DPA / D4+DPA, decode MTP, mem_fraction 0.85 |
| same-rail | `PD_DP_RANK_AFFINITY=1` **and** per-GPU `RDMA_DEVICE` JSON map |
| decode | `DECODE_EXTRA_ARGS=--disable-custom-all-reduce` in **both** tests |

The `nextnfix` base carries the GLM NextN shared-experts-fusion fix. R06 proved
it is not required for correct output, but R05 (with it) measured the higher
acceptance, and it is a genuine upstream defect — so it stays in.

## fast vs full — verified, not assumed

`AIPERF_EXPERIMENTAL_FAST=1` in InferenceX `benchmarks/benchmark_lib.sh:1979`
does exactly two things: `duration=1200` and `warmup_requests_per_lane=1`.
`config.sh` already sets `AGENTX_DURATION=1200` and
`AGENTX_WARMUP_REQUESTS_PER_LANE=1`, so **running `config.sh` IS fast mode** —
the env var is not plumbed through `agentx_env.py` and does not need to be.
`config.full.sh` sets 3600 / 10, i.e. full mode.

## config.full.sh needs two substitutions (already established)

`DSA_PREFILL_BACKEND`/`DSA_DECODE_BACKEND=flydsl` and `DSA_TOPK_BACKEND=aiter`
are **not accepted by the pinned nightly** — verified first-hand against the
image's own argparse choices in the previous task. Substitute `tilelang` and the
default `sgl-kernel`. This is forced, not a preference.

`config.full.sh` also sets `JSON_MODEL_OVERRIDE_ARGS=""`, dropping the
`index_share_for_mtp_iteration=false` workaround. Keep that as written — T2 runs
under simulated acceptance, where correctness is not the measurement — but
record it, because the previous C40 run in that shape dropped 3
`InvalidInferenceResultError`.

## Core principles

1. **Verify first-hand.** Code read, command run, or official source.
2. **Suspend, don't conclude.**
3. **One variable at a time.**
4. **Docker, not host.**
5. **Deletion rule.** Never delete a file whose path lacks `yihou`.

## Settle the GPUs between teardown and the next launch

After every `stop.sh`, before the next `launch.sh`: wait for the GPUs to go
quiet and **verify** it rather than assuming. `rocm-smi` must show the target
GPUs back at the idle baseline (~298 MB) with no KFD processes and no other
load; then wait longer still before launching.

Evidence, such as it is: the T2 run killed by a GPU memory access fault on
decode GPU[3] was launched **1 minute** after a decode leg that had run under
sustained load for an hour. The byte-identical re-run, launched onto GPUs that
had been idle ~23 minutes, cleared the same point. **N=1 correlation, not a
demonstrated cause** — the check is adopted because it is cheap, not because the
mechanism is established. Note that "HiCache did not release" does **not** fit
this failure: `DECODE_HICACHE=0`, so the decode leg never had a host pool, and
the 153.81 GB pool that did exist was on the *prefill* node, a different machine
from the one that faulted.

## Landmines (all verified)

- **HiCache release takes time.** T2 enables prefill HiCache with ratio 1.5;
  host-pool teardown and eviction lag the request stream. Slow VRAM/host-memory
  return between sweep points is **expected** — do not call it a leak or abort
  the sweep on it.
- **One launch serves the whole sweep.** `agentx_bench.sh` runs against the live
  router, so T2's five points need one bring-up, not five.
- **Verify the image layer by importing, not grepping.** `__pycache__` is keyed
  on mtime. For the `.cuh` there is no import — assert the `kCopyGroupThreads`
  marker inside a **running container**, and check `_tiles_across_lanes` by
  importing the module.
- **No GPU during `docker build`.** Importing `sglang.srt.models.*` in a build
  `RUN` fails with `No HIP GPUs are available` (`fp8_kernel.py:50` →
  `torch.cuda.get_device_properties(0)`). Verify post-build with
  `docker run --device /dev/kfd --device /dev/dri`.
- **The idle-gauge trap.** A rank that served no decode tokens reports
  `spec_accept_length 0.0`. Generate real traffic first; trust only ranks whose
  counters moved.
- 135 GPU[1] is a root-owned k8s vLLM pod (`pod5d84e491`, Qwen3-32B). **Do not
  touch it.** `check_nodes.sh` flags it; that is expected.
- 135's `ionic_7` is defective (no netdev, GID index 1 all-zero).
- `launch.sh` / `preflight.sh` / `agentx_bench.sh` refuse a pre-existing `OUT_DIR`.
- `CONTROL_NODE` must appear in `topology.tsv`.
- `config.sh` uses `${DECODE_SIMULATE_ACC_LEN-3.61}` (single dash), so an
  **empty but set** value survives and disables simulation. `:-` would not.
- Workspace configs must sit exactly two levels under `bench/glm5p2_pd/` —
  `config.yihou.base.sh` reaches the repo `config.sh` via `../..`.

## Timing expectations

CUDA graph capture ~30 min per bring-up. T1 ≈ 1.2 h. T2 ≈ 6 h (5 × 3600 s plus
per-point dataset setup, which routinely takes 4–5 min and can stretch to 14).

## Repository conventions

### DCO sign-off is required on every commit

CI blocks any PR containing a commit without a `Signed-off-by:` trailer. Commit
with `-s`, always, and sign off **as yourself** — built from your own
`user.name` / `user.email`. Never a bot or assistant identity, never a
colleague's line.

```bash
git log --format='%h %s | %(trailers:key=Signed-off-by,valueonly)' origin/main..HEAD
git rebase --signoff <last-already-signed-commit>
```

Cherry-picks do not inherit it — use `git cherry-pick -s`.

## Language

Work in English (code, comments, commits, notes). Report to the user in Chinese.
