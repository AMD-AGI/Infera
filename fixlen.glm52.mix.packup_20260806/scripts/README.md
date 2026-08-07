# scripts/ — what each file is, and which ones this packup actually uses

Every file here is a **verbatim copy** of what ran on the cluster. Verified by
md5 against `/mnt/vast/c_huggingface/glm52_mix_20260806/scripts/` — byte
identical, not paraphrased, so flag-level details survive.

## Used by the fixlen sweep (this packup)

Call order is the order below.

| script | role |
|---|---|
| `mix_site.sh` | **The only file carrying site values.** Node IP, image, model path, and the deployment shape (TP/GMU/CHUNK/CTX/DPA/MTP/KVAWARE/KVD). Dispatches `up` / `smoke` / `down`. Edit this one, not the others. |
| `mix_common.sh` | Shared helpers: `start_container`, `start_etcd`, `start_kvd`, `start_router`, `wait_health`, `reap`. Sourced by everything. Carries the router self-kill fix (`notes.md` §2) and the VRAM-drain wait. |
| `mix_up.sh` | Orchestrates bring-up in five stages: container → etcd → kvd → worker → router. Starts the router only *after* the worker is serving, so the etcd registry is never empty at router start. |
| `mix_engine.sh` | Launches the one mix worker via `python3 -m infera.engine.sglang`. Carries the mandatory DSA-on-ROCm env block and the tuned flags. Derived from the validated 1P1D recipe with every `--disaggregation-*` flag removed. |
| `mix_smoke.sh` | **The feature gate.** Seven checks, each of which goes red if its feature is silently absent. Run this before measuring anything; read the blocks, not the exit code. |
| `mix_bench_fixlen.sh` | **The sweep itself.** 3 arms × 4 concurrencies via `sglang.bench_serving` against the router. |
| `summarize_fixlen.py` | Collapses the per-run jsonl into `summary.csv` + a markdown table. Takes the *last* json object per file (bench_serving appends). |
| `envsnap.sh` | Snapshots node + image + **resolved** engine cmdline + kvd counters. Must be run while the deployment is live — some of it is unrecoverable afterwards. |
| `accept_len.sh` | Reads the MTP acceptance-length distribution off an engine log. **Unscoped** — scope it by time window yourself if anything else has used the deployment (`REPRODUCE.md` §8, `notes.md` §4). |

## NOT used by the fixlen sweep

`run_agentic.sh` drives the Optimus-AgenticBench workloads for **Phase 2**
(agentic at conc 1) and **Phase 3** (agentic under load). It is carried here
only because it shares the same staging directory and the same live deployment.
It plays no part in reproducing the 12 fixlen points, and it depends on an
external driver checkout (`/mnt/vast/c_huggingface/bench_20260801/agbench`) that
this packup does not document. See the Phase 2 / Phase 3 packups for that.

The same applies to `../spec/mix_solo_p50.yaml`, `mix_solo_p90.yaml`,
`mix_solo_p99.yaml` and `mix_load.yaml` — those are the Phase 2 / Phase 3
workload definitions. The spec file that governs *this* packup is
`../spec/mission.mix.md`, task 1.
