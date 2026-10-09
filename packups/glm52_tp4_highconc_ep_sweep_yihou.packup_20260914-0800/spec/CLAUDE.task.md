# Task: TP4 / DPA-on high-concurrency points, EP on vs EP off, C = 128 and 160

## Goal
Extend the existing TP4 / DP-attention internal decode sweep to high global concurrency with
**both arms**: EP off (`--ep-size 1`) and EP on (`--ep-size 4`).
Requested points: C = 128, 160, 192, 224, 256, 288. Same workload as the prior sweep —
ISL 70000, OSL 10000, simulated acceptance 3.61, EAGLE steps=5 / draft=6 / topk=1.
Method is unchanged: the internal **scheduler-free** harness `bench/profile_decode.py`
(no server, no scheduler, no PD fake-input path).

Reference for everything procedural (scripts, flags, pitfalls, reproduction):
`packups/glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/`.

## The binding constraint: TP4 cannot reach C >= 192 at ISL 70000

Measured per rank from the prior sweep's own `runtime.log` (first-hand):

| quantity | value |
|---|---|
| total device memory | 287.33 GiB |
| target weights | 119.84 GiB |
| draft weights | 6.92 GiB |
| free after weights | **151.66 GiB** |
| KV cost per token | 46.58 KiB |
| reserved tokens per request | 80064 (= ceil((70000+10000+reserve)/64)*64) |
| **KV cost per request** | **3.556 GiB** |

With DP attention each DP rank carries `C/4` requests, so required KV = `(C/4) * 3.556 GiB`:

| C | local batch | KV needed | verdict |
|---:|---:|---:|---|
| 128 | 32 | 113.8 GiB | fits; needs `--mem-fraction-static` ~0.88 (0.85 gives only 109.87 GiB) |
| 160 | 40 | 142.3 GiB | ~7 GiB margin; needs ~0.97; **attempt, may OOM** |
| 192 | 48 | 170.7 GiB | **exceeds the 151.66 GiB hard ceiling — impossible** |
| 224 / 256 / 288 | 56 / 64 / 72 | 199 / 228 / 256 GiB | **impossible** |

`--mem-fraction-static` cannot fix 192+: the ceiling is total device memory minus weights, not the
static fraction. The absolute TP4 ceiling is C ~= 168 with zero headroom.

**User decision (2026-09-14): keep TP4 and run only the feasible points.** C >= 192 is reported as
infeasible with this arithmetic, not attempted. Do not silently switch to TP8 — that would be a
different, non-comparable curve.

Useful prior observation: `max_memory_allocated_bytes` was **identical (237.17 GiB) for every local
batch from 1 to 12**, i.e. peak allocation is dominated by weights + pool and barely grows with
batch. That is why C=160 is worth attempting despite the thin margin — but it is a prediction, not
a result.

## Resources
- Node `crsuse2-m2m-254`, job **136670**; node `crsuse2-m2m-267`, job **136669**. Both held, not
  requested. 8h limits from 2026-09-14T04:00Z-ish.
- `crsuse2-m2m-267` carried a foreign 8-GPU Ray/verl container `dsv4`. **User confirmed it was
  residual and authorized cleanup (2026-09-14).** It was `docker stop`ped, never removed; the image
  and the stopped container remain on disk. Verified afterwards: `docker ps` empty, 0 KFD processes.
- `zihaoan2` holds concurrent Slurm allocations on both nodes (136568 on 254, 136569 on 267).
  Treat both nodes as potentially shared: re-check `docker ps` and KFD processes before each arm.
- Forbidden nodes remain 234 / 036 / 249 (enforced by `scripts/run_decode.sh`).

## Workspace
Everything for this task lives in
`sweeps/tp4_highconc_yihou_20260914-0423/`:
- `scripts/run_highconc_yihou.sh` — one arm per invocation, points given as `C:MEMFRAC`
- `scripts/collect_highconc_yihou.py` — the gate (not a formatter)
- `iterations_254/`, `iterations_267/` — per-node results
- `results/`, `working_process.md`

Nothing outside this directory is written except the container and the image store on the nodes.

## Design: both nodes end up with the full matrix, run in two phases
Target dataset: `{node 254, node 267} x {ep1, ep4} x {C=128, C=160}` = 8 points. Each node must end
with **both arms**, because the prior sweep established that a cross-node EP comparison is
confounded and only a same-node control removes that confound. The second node is then an
independent **replication**, which also yields the variance estimate the prior packup listed as an
open question.

Execution order (chosen to overlap the ~14 min cold AITER JIT cost on both nodes at once):
- Phase 1, in parallel: 254 runs `ep1`, 267 runs `ep4`.
- Phase 2, chained automatically off each phase-1 driver pid
  (`scripts/chain_opposite_arm_yihou.sh`): 254 runs `ep4`, 267 runs `ep1`, both against a now-warm
  JIT cache.

Phase 1 alone is a cross-node comparison and is **not** sufficient. The reportable EP delta comes
from phase 1 + phase 2 together, per node.

`--mem-fraction-static` per point: **0.88 at C=128** (0.85 yields only 2,436,864 pool tokens, below
the 2,562,048 required) and **0.97 at C=160** (requires 3,202,560). Derived from the measured slope
of ~6.47M pool tokens per 1.0 of mem-fraction on a 287.98 GiB device. If a point reports
`KV capacity insufficient`, raise the fraction rather than shrinking the workload.

## Core principles (carried over, still binding)
- `tok/s = C * 1000 / TPOT` is an **identity** in this harness. Reporting "latency down and
  throughput up" as two findings is double-counting one measurement.
- Never relax a validation or tune a number to make a point pass. A point that does not fit is a
  reported outcome.
- `--batch-size` is **global**; with DP attention `dp_size = tp_size = 4` and the per-rank batch is
  `C/4`. C must divide by 4.
- Run drivers detached (`setsid nohup ... > driver.log 2>&1 &`). **Never pipe `runtime.log` through
  `tail`** — it holds multi-megabyte single-line tqdm bars and previously killed the driver shell
  via host memory pressure. Use `grep -c` or file size.
- CUDA-graph capture is slow. Do not kill a quiet point before ~30 minutes. Cold first point on a
  node costs ~14 min in `load_pool_capture` alone (AITER JIT cache is per-node).
- Delete nothing whose name lacks `yihou`.

## Success criteria
Per point: `complete: true`, `useful_output_tokens == C * 10000`, resolved topology
`(ep, tp, dp, dpa) == (EP_SIZE, 4, 4, True)`, and driver `exit_code 0`.
`collect_highconc_yihou.py` asserts all four and exits non-zero otherwise.

## Deliverable
A report giving, per node: EP-on vs EP-off at each feasible C, the same-node delta, the
node-to-node replication spread, the `--mem-fraction-static` actually used at each point, and an
explicit, arithmetic-backed statement of why C >= 192 was not run.
