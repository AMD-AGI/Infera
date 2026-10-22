# GLM-5.2-MXFP4 v0.5.21 same-node P4D4 — OffloaderV2 expert-weight offload on the prefill leg (2026-10-14 … 15)

**Status at packing (2026-10-15 ~03:10 UTC): PARTIAL.** S0 (1/8 offload) is complete (correctness + scored C48).
The ratio sweep (1/4, 1/3, 1/2) has finished its correctness rounds. Its perf rounds were **in flight or aborted** at packing time; see "Not in this kit".

## Question

Earlier rounds found that the prefill leg's KV capacity is the throughput lever (packup `glm52.v0521-prefill-kv-bottleneck.packup_20261014`).
The question here: does moving routed-expert weights (w13/w2) of some MoE layers to pinned host memory buy prefill HBM for KV?
The mechanism is sglang OffloaderV2 (`--offload-group-size/--offload-num-in-group/--offload-prefetch-step --offload-mode cpu`).
Each forward streams the weights back H2D. The score must hold while the KV pool grows.

User requests, verbatim:
1. 2026-10-14: "分析调研另一个问题：…OffloaderV2…你开一个调研workflow, 调研下sglang目前的可用程度和开发量，如果评估一下，如果可行，我们就采用着各种方式开始开发实验和debug" → research (`analysis/research/`). Then "开始" → S0.
2. 2026-10-15 01:35: "继续offload的实验呀，提高offload率观察性能的同时，堆高conc, 正确，因为有多个机器，可以并行的来" → ratio sweep ofg4/ofg3/ofg2.

## Result at a glance

Prefill per rank, from `logs/server/<round>/prefill-0.log.gz`:

| round | node | offload (G,K) | MoE layers offloaded / rank | weights after load (GB) | prefill KV tokens / rank | vs pf90 | avail mem after pool (GB) | correctness (23 tests) |
|---|---|---|---|---|---|---|---|---|
| pf90 / cx90 (ref) | 089 | none | 0 | 119.84 | 2,541,376 | — | 27.67 | cx90 22/23 (needle_128k_d10 hit max_tokens) |
| ofs0c / ofs0 (S0) | 089 | G8 K1 (1/8) | 9 | 110.77 | 2,717,696 | +6.9 % | 27.72 | **23/23** |
| ofg4c | 252 | G4 K1 (1/4) | 19 | 99.40 | 2,938,752 | +15.6 % | 27.72 | in flight |
| ofg3c | 266 | G3 K1 (1/3) | 26 | 91.50 | 3,092,288 | +21.7 % | 27.70 | **23/23** |
| ofg2c | 288 | G2 K1 (1/2) | 39 | 76.78 | 3,378,496 | +32.9 % | 27.79 | **23/23** |

AgentX C48 (score = `request_metrics.throughput.per_gpu.total_tput_tps`, simulated acceptance 3.61; full table in `results/results.csv`):

| run | node | score | qps | ISL | TTFT mean/p95 (s) | TPOT (ms) | GPU hit | CPU hit | err |
|---|---|---|---|---|---|---|---|---|---|
| pf90 A1 (ref, fresh) | 089 | 21,859 | 1.219 | 142,517 | 9.57/31.32 | 15.7 | 0.911 | 0.035 | 0 |
| **ofs0 A1** (fresh) | 089 | **22,380** | 1.228 | 144,840 | 9.19/27.12 | 15.2 | 0.931 | 0.024 | 2 |
| pf90 A2 (ref, scored) | 089 | 23,127 | 1.276 | 143,964 | 9.30/28.76 | 13.1 | 0.860 | 0.071 | 3 |
| **ofs0 A2** (scored) | 089 | **22,825** | 1.267 | 143,116 | 9.75/30.51 | 12.9 | 0.887 | 0.049 | 4 |

- **S0 against the gate** (assessment §E: "≤2 % loss → go to 25 %; >10 % → abandon"):
  - A2 is −1.3 % vs pf90 A2, and A1 is +2.4 % vs pf90 A1.
  - Both are single replicates on the same node, and the two deltas have opposite signs. They do not separate offload from run-to-run variation.
  - The gate (≤2 %) was met, so the sweep proceeded.
- **The 23-test correctness suite passed with offload on** at 1/8, 1/3 and 1/2. "Correctness" here means the suite only: smoke greedy+sampled, needle 8k-256k × 3 depths, 5-fact 100k. It is not a logit diff.
  - The `is_shuffled` patch was applied in every offload round.
  - No unpatched offload round was run, so this kit does not show what the patch changes. The assessment's argument for it is in `patches/README.md`.

## Not in this kit (in flight / aborted at packing)

- **ofg4 (252, G4K1)**:
  - The correctness round ofg4c was still running.
  - Its prefill PD warm-up took 20 min 45 s (02:21:24 → 02:42:09). ofg3c/ofg2c took 60 s / 85 s. The cause is unknown.
  - The log at the stuck point is preserved in `logs/server/ofg4c.hang1/` and `results/rounds/ofg4c.hang1/`.
  - Perf ofg4 (C48/C48/C56) runs after it.
- **ofg3 (266, G3K1) perf: ABORTED in bring-up**, ~02:50. The admins asked for node 266, so the user had it released. `results/rounds/ofg3/stop.release266.log`.
- **ofg2 (288, G2K1) perf**: bring-up in flight at packing (`results/rounds/ofg2/` snapshot).
- When they finish:
  - Re-run `python3 -B scripts/tools/results_csv_v15.yihou.py offload > results/results.csv`.
  - Copy `$R/ofg4/`, `$R/ofg2/`, `$N/runs/ofg4c/`.

## Navigate

- `REPRODUCE.md` — node prep, env files, launch, read-out.
- `environment.md` — nodes, image, versions, gaps.
- `notes.md` — gotchas, incidents, design choices (K=1, step=1).
- `patches/` — YIHOU-OFFLOAD-ATTRS (`create_device_tensor` carries `is_shuffled`), with orig/new/diff/builder.
- `analysis/research/` — the pre-experiment research: assessment, cost model, implementation reading, upstream status.
- `results/` — `results.csv`, `correctness/<round>/results.jsonl.gz`, `rounds/<round>/` (status, chain logs, audit), bench dirs, dryruns + tokdiffs, `h2d/h2d_bench.089.json`.
- `logs/server/<round>/` — prefill/decode server logs (gzip; live rounds are snapshots).
- `analysis/ledger/` — `poll.md`, `working_process.md` as of packing.
