# GLM-5.2 TP4 / DP-attention high-concurrency decode: EP on vs EP off

**Ran:** 2026-09-14, 04:10Z – 07:45Z (single day)
**Author:** yihou
**Status:** **PASS with a scoped negative result** — 12/12 measured points verified; the requested
C >= 160 points are proven infeasible at TP4 rather than left untried.

## Goal
Extend the existing TP4 / DP-attention internal decode sweep (which stopped at C=48) to high global
concurrency, measuring **both** arms — expert parallelism **off** (`--ep-size 1`) and **on**
(`--ep-size 4`) — using the internal **scheduler-free** harness `bench/profile_decode.py`, which
intercepts SGLang's model forward via `TpModelWorker` + `EAGLEWorkerV2`. **No server, no scheduler,
no PD fake-input path.** Workload: ISL 70000, OSL 10000, simulated acceptance 3.61,
EAGLE steps=5 / draft=6 / topk=1.

**Spec:** `spec/CLAUDE.task.md` (task file) and `spec/mission.md` (standing mission book).
**Method reference:** `packups/glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/`.

## Success criteria and what actually happened

Criteria from the spec: every point must report `complete: true`, exactly
`useful_output_tokens = C x 10000`, the requested topology, and driver `exit_code 0`; numbers must
never be tuned nor validations relaxed to make a point pass.

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| `complete` | true | true, 12/12 | ✅ |
| `useful_output_tokens` | C x 10000 | exact, 12/12 | ✅ |
| topology `(ep, tp, dp, dpa)` | as requested / (4,4,True) | exact, 12/12 | ✅ |
| driver `exit_code` | 0 | 0, 12/12 | ✅ |
| `verify_iterations` | 2768 (deterministic) | 2768, 12/12 | ✅ |
| `realized_accept_length` | 3.6134393063583814 | exact, 12/12 | ✅ |
| no validation relaxed to force a pass | — | none; the C=160 failures are reported as failures | ✅ |
| requested C=128 | measured | measured, both arms, both nodes | ✅ |
| requested C=160 | measured | **infeasible** — 12 attempts, evidence in `notes.md` | ⚠️ negative result |
| requested C=192/224/256/288 | measured | **infeasible by arithmetic**, not attempted | ⚠️ negative result |

C = **64** and **96** were added with the user's agreement to fill the gap between the prior sweep's
ceiling (C=48) and C=128.

## Headline numbers

TP4 / DP-attention on / ISL 70000 / OSL 10000 / accept 3.61, per node:

| C | local batch | node | EP off TPOT ms | EP on TPOT ms | EP off tok/s | EP on tok/s |
|---:|---:|---|---:|---:|---:|---:|
| 64 | 16 | 254 | 17.5528 | 17.9472 | 3646.1 | 3566.0 |
| 64 | 16 | 267 | 17.5190 | 17.9769 | 3653.2 | 3560.1 |
| 96 | 24 | 254 | 21.9358 | 22.1619 | 4376.4 | 4331.8 |
| 96 | 24 | 267 | 21.9994 | 22.3749 | 4363.7 | 4290.5 |
| 128 | 32 | 254 | 25.5301 | 25.7811 | 5013.7 | 4964.9 |
| 128 | 32 | 267 | 25.7918 | 25.9187 | 4962.8 | 4938.5 |

**`tok/s = C x 1000 / TPOT` is an identity in this harness.** The throughput columns are the latency
columns in different units — not independent confirmation.

### The EP-off advantage decays and is nearly gone by C=128

Same node, same container, same allocation (single variable), negative = EP off faster:

| C | node 254 | node 267 |
|---:|---:|---:|
| 64 | -2.198 % | -2.547 % |
| 96 | -1.020 % | -1.678 % |
| 128 | -0.973 % | -0.490 % |

Joined to the prior sweep (same method): -13.32 % at C=4, -6.09 % at C=48, then the above.

**Claimable:** EP off was faster in **6 of 6** same-node pairs here and in every point of the prior
sweep — the sign is consistent and the magnitude falls steeply with concurrency.
**Not claimable:** the C=128 magnitude. It is the same size as the node-to-node spread (below), and
these are single runs.

### Noise floor measured by this sweep
Node-to-node spread on identical configurations: 0.165 % – 1.025 %.
Within-node near-repeat (the allocator control, identical but for one PyTorch env var):
0.205 % and 0.265 %, with opposite signs.

This **resolves an open question from the previous packup**, which flagged as "unexplained" that the
EP advantage widened between C=32 (-4.78 %) and C=48 (-6.09 %). That 1.3-point excursion is within
what unreplicated single runs produce here. Downgraded from "unexplained" to "consistent with
single-run variation" — not proven.

## How to reproduce
See `REPRODUCE.md`. TL;DR: hold a 1-node allocation, load the pinned image, create the container,
then `scripts/run_highconc_yihou.sh 64:0.85 96:0.85 128:0.88` once per arm, and collect with
`scripts/collect_highconc_yihou.py`. **You must pass `--max-running-requests`** — the driver does it
for you, and `notes.md` explains why nothing above C=48 runs without it.

## Folder map
- `REPRODUCE.md` — ordered, copy-pasteable reproduction steps
- `environment.md` — hardware/software, captured first-hand on **both** nodes
- `notes.md` — gotchas, the two hidden ceilings, wrong turns, open questions
- `scripts/` — every script that ran, verbatim
- `patches/` — the one script variant created for this sweep, with rationale
- `results/` — `report.md`, `summary_yihou.csv`, `ep_delta_same_node_yihou.csv`,
  `node_replication_yihou.csv`
- `evidence/points/` — all **41** attempt directories (14 pass + 27 failures), each with
  `result_yihou.json`, per-rank reports, `config_yihou.json`, `command.txt`, `launch_status.json`,
  gzipped `runtime.log`
- `evidence/code_snapshot/` — the exact `bench/*.py` that ran, plus hashes, git HEAD, working-tree diff
- `env/` — raw environment probes for nodes 254 and 267
- `logs/` — 27 gzipped driver logs + the two image-load logs
- `spec/`, `notes_src/working_process.md` — the task file, mission book, and the raw timestamped log
- `MANIFEST.sha256` — hash of every file here

## What was deliberately left out (originals are intact)
Nothing was deleted. The full workspace (105 MB) remains at
`glm52_decode_internal_yihou_20260909_1057/sweeps/tp4_highconc_yihou_20260914-0423/`.
Excluded to keep this packup committable (1.3 MB, largest file 25 KB):
- **Per-step JSONL traces** (`steps_yihou.jsonl`, `steps_rank_{0..3}_yihou.jsonl`, ~1.5 MB each,
  ~100 MB total) — per-iteration forensics, not needed to reproduce or check the result.
- **`console.log`** — verified byte-for-byte identical to `runtime.log` in **all 41** directories
  (`cmp`); only the gzipped `runtime.log` ships.
- **`bench_snapshot/` per point** — `code_hashes.sha256` is identical across all 41 points
  (1 unique hash), so one copy ships in `evidence/code_snapshot/` with the hash file as proof.

**Decision made without the user:** the skill asks the author to confirm before including log files.
The user was asked twice whether to package this experiment and did not reply, so a conservative
default was taken — logs are included **gzipped**, no file exceeds 4 MB, and every exclusion is
listed above. Say the word and any of it can be changed.
