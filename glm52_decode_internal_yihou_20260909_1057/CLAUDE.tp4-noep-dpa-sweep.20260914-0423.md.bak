# Task: TP4 / no-EP / DPA-on internal decode sweep, concurrency 4->24

## Goal
Use the internal scheduler-free method (`bench/profile_decode.py`, no server launch) to measure
TP4, **EP1 (no expert parallel)**, **DP attention enabled**, global concurrency 4/8/16/20/24,
ISL 70000, OSL 10000, simulated acceptance 3.61, EAGLE steps=5 / draft=6 / topk=1.
Deliver a report pairing the five points, plus the delta against the existing TP4/**EP4**/DPA-on
sweep so the EP4-vs-noEP difference is visible.

Start from parameters only: drop `--ep-size 4` (default 1) and always pass `--enable-dp-attention`.

**User authorization (2026-09-11):** the source is under git, so if EP1+DPA does not run, debug and
fix it in the source until it does — "跑通为止". Rules for such fixes:
- Drive it with the `iterative-debug-loop` discipline: explicit goal, one hypothesis per iteration,
  first-hand evidence (logs, source, LSP) over conjecture, TP4/EP4 DPA-on as the known-good
  differential reference.
- Every source change must be recorded (diff + rationale) in the workspace working_process.md, and
  must not silently change the semantics of the already-published EP4 sweep. If a fix would alter
  EP4 behaviour, gate it or state the impact explicitly.
- Never weaken a validation, relax a completeness check, or tune numbers to make a run "pass".

## Resource
- Allocation: job **133750**, node **crsuse2-m2m-217**, 8h from 2026-09-11T07:54Z (ends 15:54Z).
- Spur allocations are exclusive; a leftover foreign container `s1b_cap45` held all 8 GPUs and was
  stopped under explicit user instruction at ~08:0xZ. All GPUs verified at 0% VRAM afterwards.
- Docker root on 217 is `/var/lib/docker` on `/` (123G total, ~90G free) — watch free space while
  the pinned image loads (~24GB compressed archive).
- Use GPUs 0-3. Container name must start with `yihou-`. No allocation requests or cancellations.
- Never delete any file whose name lacks the `yihou` substring.

## Workspace
Everything for this task lives in
`sweeps/tp4_noep_dpa_yihou_20260911-0800/` (`iterations/`, `results/`, `scripts/`,
`working_process.md`). Previous CLAUDE.md saved as
`CLAUDE.reverse-fake-server-sweep.20260911-0800.md.bak`. Do not modify old sweeps, comparisons or
packups. Preserve the existing dirty working tree; no commit/push.

## Method (unchanged tooling)
- `scripts/create_container_yihou.sh` creates the pinned-image container (image
  `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`).
- `scripts/run_decode.sh` runs one point; it verifies allocation state, container image/owner,
  snapshots code + hashes, and writes `console.log`, `runtime.log`, `launch_status.json`.
- New sweep driver: `sweeps/tp4_noep_dpa_yihou_20260911-0800/scripts/run_tp4_noep_dpa_sweep_yihou.sh`.
- Per-point args: `--tp-size 4 --ep-size 1 --enable-dp-attention --batch-size C --input-len 70000
  --output-len 10000 --accept-length 3.61 --warmup-steps 10 --enable-aiter-allreduce-fusion
  --enable-fused-qk-norm-rope --mem-fraction-static 0.85`.

## Key semantics (verified in source)
- `bench/topology.py`: `--batch-size` is GLOBAL. With DPA, `dp_size = tp_size = 4`, so
  `local_batch_size = C/4` and C must be divisible by 4 (4/8/16/20/24 all are).
- `ep_size` must divide `tp_size`; `ep_size=1` gives `moe_ep_rank=0, moe_ep_size=1` on all ranks
  (TP MoE). `--moe-a2a-backend none` is already forced by `server_cli`.
- CUDA graph bs is set to `local_batch_size` (1/2/4/5/6), not the global C.
- Model `/shared_nfs/models/GLM-5.2-MXFP4` mounted read-only.

## Baselines for comparison
- TP4/EP4 DPA on/off sweep: `sweeps/tp4_ep4_dpa_yihou_20260910-0452/summary.csv` and packup
  `/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/packups/glm52_tp4_ep4_dpa_sweep_yihou.packup_20260910-062400`.

## Discipline
- Research -> plan -> workspace -> CLAUDE.md -> work (already done in that order).
- Agent team: one execution teammate; leader polls every 20 min, records a first-time issue and only
  intervenes if it persists at the next poll. Mission book re-injected every 10 min.
- CUDA graph capture can take ~30 min per point; do not treat a long silence as a hang before that.
- Keep outliers; never tune numbers into agreement. Report unresolved items as open.
- Work files in English; user-facing report in Chinese.
