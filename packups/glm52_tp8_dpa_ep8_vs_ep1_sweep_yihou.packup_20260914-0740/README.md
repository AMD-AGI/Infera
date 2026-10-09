# GLM-5.2 TP8 / DP-attention internal decode sweep — EP8 vs EP1, C = 48 … 320

**Ran:** 2026-09-14, 04:23Z – 07:29Z (single day, single host)
**Author:** yihou
**Status:** **PASS** — all 20 measured points completed and gate-verified.

## Goal
Measure decode performance of GLM-5.2-MXFP4 on **8× MI355X** using the internal **scheduler-free**
harness (`bench/profile_decode.py`, which drives SGLang's `TpModelWorker` + `EAGLEWorkerV2` directly
— **no HTTP server, no Scheduler, no PD fake-input path**), at TP8 with DP attention on, sweeping
global concurrency, once with expert parallelism **on** (`ep_size=8`) and once **off** (`ep_size=1`).

This is the first run of this method on **this** machine: `smci355-ccs-aus-n06-25`, a bare-metal host
with no Slurm/spur scheduler. The method and every reference number come from the spur-cluster
packups in `../glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/` (TP4) — see
"Deviations from the spur packups" in `notes.md` for everything that had to change here, and why.

**Spec:** `spec/mission.md` (the standing mission book) and `spec/CLAUDE.task.md` (the task file).

### Success criteria (from the spec, verbatim in intent)
Per point: `complete: true`; `useful_output_tokens == C × 10000` exactly; topology
`(tp=8, dp=8, dpa=true)` with `ep=8` (Phase 2) / `ep=1` (Phase 3); `local_batch_size == C/8`;
`moe_a2a_backend == none`; driver `exit_code == 0`. Across the sweep: the collector must exit 0.
**No parameter may be tuned and no validation relaxed to make a point pass.**

## Result — all criteria met

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| EP8 points measured | 10 (C=48…320) | 10 | ✅ |
| EP1 points measured | 10 (C=48…320) | 10 | ✅ |
| `complete` | true | true, all 20 | ✅ |
| `useful_output_tokens` | C × 10000 | exact, all 20 | ✅ |
| topology / `local_batch_size` / `moe_a2a_backend` | as specified | exact, all 20 | ✅ |
| driver `exit_code` | 0 | 0, all 20 | ✅ |
| `collect_sweep_yihou.py` (both sweeps) | exit 0 | exit 0 | ✅ |
| no parameter relaxed to force a pass | — | `--mem-fraction-static` never changed | ✅ |

Fixed for every point: ISL 70000, OSL 10000, accept length 3.61, warmup 10,
`--mem-fraction-static 0.85`. Verified invariants, **identical across all 20 points**:
`verify_iterations = 2768`, `realized_accept_length = 3.6134393063583814`.

### EP8 (`--ep-size 8`)

| C | local batch | TPOT ms | output tok/s | tok/s per GPU |
|---:|---:|---:|---:|---:|
| 48 | 6 | 12.083191042300314 | 3972.460572042904 | 496.563 |
| 64 | 8 | 13.142841791105457 | 4869.57090538156 | 608.696 |
| 96 | 12 | 16.237883286201395 | 5912.100629617084 | 739.013 |
| 128 | 16 | 17.847756853699686 | 7171.769598231988 | 896.471 |
| 160 | 20 | 20.842523304204224 | 7676.613702896799 | 959.577 |
| 192 | 24 | 22.725100385496624 | 8448.807562695574 | 1056.101 |
| 224 | 28 | 24.21575831020018 | 9250.174912162323 | 1156.272 |
| 256 | 32 | 26.251939422101714 | 9751.660472920015 | 1218.958 |
| 288 | 36 | 28.91812167810276 | 9959.15306000244 | 1244.894 |
| 320 | 40 | 30.771934552298625 | 10399.086201621223 | 1299.886 |

### EP1 (`--ep-size 1`, plain TP MoE) and the delta

Both sweeps ran **on the same host, in the same container, from the same image, back to back**, so
this is a single-variable comparison. `delta = (EP1 − EP8) / EP8`; negative TPOT delta = EP1 faster.

| C | EP8 TPOT ms | EP1 TPOT ms | TPOT delta | EP8 tok/s | EP1 tok/s |
|---:|---:|---:|---:|---:|---:|
| 48 | 12.083191 | 10.558471 | **−12.62 %** | 3972.46 | 4546.11 |
| 64 | 13.142842 | 11.974589 | −8.89 % | 4869.57 | 5344.65 |
| 96 | 16.237883 | 14.912221 | −8.16 % | 5912.10 | 6437.67 |
| 128 | 17.847757 | 17.323120 | −2.94 % | 7171.77 | 7388.97 |
| 160 | 20.842523 | 19.727784 | −5.35 % | 7676.61 | 8110.39 |
| 192 | 22.725100 | 21.995037 | −3.21 % | 8448.81 | 8729.24 |
| 224 | 24.215758 | 24.262255 | **+0.19 %** | 9250.17 | 9232.45 |
| 256 | 26.251939 | 25.674006 | −2.20 % | 9751.66 | 9971.17 |
| 288 | 28.918122 | 28.373166 | −1.88 % | 9959.15 | 10150.44 |
| 320 | 30.771935 | 30.410779 | −1.17 % | 10399.09 | 10522.58 |

EP1 is faster at nine of ten concurrencies; C=224 is the single point where it is not. **Every point
is a single run with no repeats**, and no per-stage attribution was done, so a few percent is within
unmeasured run-to-run variation and the C=224 sign flip is recorded as an **open question**, not as
an effect. See `notes.md` → "Open questions".

**Do not read TPOT and throughput as two independent results:** the harness defines
`tok/s = C × 1000 / TPOT` identically. They are one measurement in two units.

## How to reproduce
See `REPRODUCE.md`. TL;DR: on an idle 8× MI355X host with the `rocm-llm-bench:latest` image present,
run `scripts/create_container_yihou.sh`, then `scripts/run_tp8_ep8_dpa_yihou.sh 48 64 96 128 160 192
224 256 288 320` and `scripts/run_tp8_noep_dpa_yihou.sh <same list>`, then the collector. Measured
compute: 3107 s (0.86 h) for the ten EP8 points and 3055 s (0.85 h) for the ten EP1 points, summing
the per-point `launch_wall_seconds` — so ~1.7 h for both sweeps once warm, plus a slower first point
(cold AITER JIT + cold page cache).

## Folder map
- `REPRODUCE.md` — ordered, copy-pasteable reproduction steps
- `environment.md` — exact hardware/software the numbers came from, and the gaps
- `notes.md` — gotchas, the five environment failures and their fixes, open questions
- `scripts/` — every script that ran, verbatim
- `results/` — `sweep_summary_yihou.csv` (EP8), `sweep_summary_noep_yihou.csv` (EP1),
  `verify_c128_yihou.json`, and the full `REPORT.md`
- `evidence/points/` — per-point `result_yihou.json`, 8 per-rank reports, `config_yihou.json`,
  `command.txt`, `launch_status.json`, `container-inspect.json`, gzipped `runtime.log` (20 points)
- `evidence/aborted/` — the five failed attempts, each with its config, status and failure tail
- `evidence/code_snapshot/` — the exact `bench/*.py` that ran, plus hashes, git HEAD and the
  working-tree diff at run time
- `spec/` — the mission book and the task file this work was driven by
- `notes_src/working_process.md` — the raw, timestamped iteration log (unedited)
- `env/` — raw environment probes: the host and the container
- `MANIFEST.sha256` — hash of every file in this packup

**No `patches/` folder:** this experiment required **no source change**. `bench/*.py` is
byte-identical to the committed version across all 20 points (`evidence/code_snapshot/`). Everything
that had to be fixed was container/environment configuration, and all of it lives in
`scripts/create_container_yihou.sh`. See `notes.md`.

## What was deliberately left out (originals are intact)
Nothing was deleted. The original workspace is preserved in full at
`../../glm52_tp8ep8_dpa_c128_yihou_20260914-0423/` (212 MB). Excluded here to keep the packup
committable (1.6 MB):
- **Per-step JSONL traces** (`steps_yihou.jsonl` + `steps_rank_{0..7}_yihou.jsonl`, ~961 KB each,
  **207 MB over 180 files**). Needed only for per-iteration forensics, not to reproduce or check the
  result. Originals: `<workspace>/iterations/<point>/`.
- **`console.log`** — content-identical to `runtime.log` at all 20 points (verified by comparing the
  decompressed streams; the `.gz` files differ only in the gzip header's embedded name/mtime). Only
  the gzipped `runtime.log` is shipped.
- **`bench_snapshot/` per point** — all 20 snapshots hash-identical, so one copy is shipped in
  `evidence/code_snapshot/` with `code_hashes.sha256` to prove the identity.
- **`logs/driver_*.log`** — all zero bytes: `run_decode.sh` tees the child's output into the point's
  own `runtime.log`/`console.log`, so the driver's own stdout stayed empty. Nothing was lost.
