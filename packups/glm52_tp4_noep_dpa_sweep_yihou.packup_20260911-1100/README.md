# GLM-5.2 TP4 / EP1 (no expert parallel) / DP-attention internal decode sweep

**Ran:** 2026-09-11, 08:00Z – 10:56Z (single day)
**Author:** yihou
**Status:** **PASS** — all 8 sweep points and all 6 control points completed and verified.

## Goal
Measure decode performance of GLM-5.2-MXFP4 on 4× MI355X with **expert parallelism disabled**
(`ep_size=1`) and **DP attention enabled**, sweeping global concurrency 4 → 48, using the internal
**scheduler-free** harness (`bench/profile_decode.py`, which intercepts SGLang's model forward via
`TpModelWorker` + `EAGLEWorkerV2` — **no server, no scheduler, no PD fake-input path**).
Second objective: quantify the EP1-vs-EP4 difference on the **same node** so the comparison is
single-variable.

**Spec:** `spec/CLAUDE.task.md` (task instructions) and `spec/mission.md` (the standing mission book).
**Success criteria** (from the spec): each point must report `complete: true`, exactly
`useful_output_tokens = C × 10000`, the requested topology (`tp=4, ep=1, dp=4, dpa=true`), and a
driver `exit_code 0`. Numbers must never be tuned or validations relaxed to make a point pass.

## Result
All criteria met, verified mechanically by `scripts/collect_noep_sweep_yihou.py` (exit 0).

| Config | Criterion | Target | Actual | Verdict |
|---|---|---|---|---|
| 8 sweep points (C=4…48) | `complete` | true | true, all 8 | ✅ |
| 8 sweep points | `useful_output_tokens` | C × 10000 | exact, all 8 | ✅ |
| 8 sweep points | topology | tp4/ep1/dp4/dpa | exact, all 8 | ✅ |
| 8 sweep points | driver exit code | 0 | 0, all 8 | ✅ |
| 6 control points (EP4) | same four checks with ep=4 | — | all pass | ✅ |
| whole sweep | no parameter relaxed to force a pass | — | mem-fraction never changed | ✅ |

### Headline numbers — TP4 / EP1 / DPA-on, ISL 70000, OSL 10000, accept 3.61

| C | local batch | TPOT ms | output tok/s |
|---:|---:|---:|---:|
| 4 | 1 | 6.245638818759471 | 640.4468967986998 |
| 8 | 2 | 7.814737337362021 | 1023.7068316745905 |
| 16 | 4 | 9.821627188567073 | 1629.0579649189801 |
| 20 | 5 | 10.714399154949934 | 1866.6469029913098 |
| 24 | 6 | 11.361972574423998 | 2112.309270489213 |
| 32 | 8 | 12.757144562341272 | 2508.3983209270114 |
| 40 | 10 | 14.3613898829557 | 2785.24574055834 |
| 48 | 12 | 15.429530137684196 | 3110.9178031784368 |

### EP1 vs EP4, same node / container / allocation (single variable)

| C | EP1 TPOT ms | EP4 TPOT ms | TPOT delta |
|---:|---:|---:|---:|
| 4 | 6.245639 | 7.205778 | **−13.32%** |
| 16 | 9.821627 | 10.968559 | **−10.46%** |
| 24 | 11.361973 | 12.321298 | **−7.79%** |
| 32 | 12.757145 | 13.397674 | **−4.78%** |
| 40 | 14.361390 | 15.248736 | **−5.82%** |
| 48 | 15.429530 | 16.430828 | **−6.09%** |

Dropping EP is faster at every measured concurrency. The advantage shrinks steeply to C=32 and then
**stops shrinking and widens slightly at C≥40 — unexplained by this sweep** (single runs, no
per-stage attribution). See `notes.md` "Open questions".

**Do not read TPOT and throughput as two independent results:** the harness defines
`tok/s = C × 1000 / TPOT` identically. They are one measurement in two units.

## How to reproduce
See `REPRODUCE.md`. TL;DR: hold a 1-node spur allocation, load the pinned image, create the container
with `scripts/create_container_yihou.sh`, then run `scripts/run_tp4_noep_dpa_sweep_yihou.sh 4 8 16 20
24 32 40 48` and collect with `scripts/collect_noep_sweep_yihou.py`. ~35 min for the whole sweep once
the AITER JIT cache is warm; the first point alone takes ~16 min cold.

## Folder map
- `REPRODUCE.md` — ordered, copy-pasteable reproduction steps
- `environment.md` — exact hardware/software the numbers came from, and what could not be re-probed
- `notes.md` — gotchas, incidents, wrong turns, open questions
- `scripts/` — every script that ran, verbatim
- `results/` — `summary_yihou.csv`, `ep1_vs_ep4_same_node_yihou.csv`, full `report.md`
- `evidence/points/` — per-point `result_yihou.json`, per-rank reports, `config_yihou.json`,
  `command.txt`, `launch_status.json`, gzipped `runtime.log`
- `evidence/code_snapshot/` — the exact `bench/*.py` that ran, plus hashes, git HEAD and the
  working-tree diff at run time
- `spec/` — the task instructions and mission book this work was driven by
- `notes_src/working_process.md` — the raw, timestamped iteration log (unedited)
- `env/` — raw `collect_env.sh` output from the node
- `logs/` — the four sweep-driver logs, gzipped
- `MANIFEST.sha256` — hash of every file in this packup

**No `patches/` folder:** this experiment required **no code change**. See `notes.md`.

## What was deliberately left out (originals are intact)
Nothing was deleted. The original experiment workspace is preserved in full at
`glm52_decode_internal_yihou_20260909_1057/sweeps/tp4_noep_dpa_yihou_20260911-0800/` (52 MB).
Excluded from this packup to keep it committable (671 KB):
- **Per-step JSONL traces** (`steps_yihou.jsonl`, `steps_rank_{0..3}_yihou.jsonl`, ~4 MB per point,
  ~45 MB total). Needed only for per-iteration forensics, not for reproducing or checking the result.
  Originals: `<workspace>/iterations/<point>/` and `<workspace>/control_iterations/<point>/`.
- **`console.log`** — byte-for-byte identical to `runtime.log` (verified with `cmp`); only the
  gzipped `runtime.log` is shipped.
- **`bench_snapshot/` per point** — all 14 snapshots hash-identical (verified), so one copy is
  shipped in `evidence/code_snapshot/` along with `code_hashes.sha256` to prove the identity.
- **`__pycache__/`** — build artifacts.

The `noep_dpa_on_c24_orphan_driver_yihou` point is included in `evidence/points/` for completeness:
it is the driver-killed C=24 run that has a valid result but **no** `launch_status.json`, and is
therefore excluded from every table. See `notes.md`.
