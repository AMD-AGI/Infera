# Scripts — what each one is for in THIS packup

All copied **verbatim** from the cluster staging directory
`/mnt/vast/c_huggingface/glm52_mix_20260806/scripts/`, md5-verified byte-identical
to the copies that ran (table in `../environment.md`).

The staging directory is shared across all three phases of the bench, so a few of
these belong to Phase 1 and are not carried here. What is here:

## Used directly by this run

| script | role |
|---|---|
| `run_agentic.sh` | **the one that ran the benchmark.** `WORKLOAD=specs/mix_load.yaml TAG=load bash run_agentic.sh`. Passes **no load knobs on the CLI** by design — the YAML is the single source of truth for offered load, which matters more here than anywhere else in this bench because the load knobs *are* the experiment. Passes `--dashboard-mode`, which is mandatory. |
| `analyze_solo.py` | **the analysis.** Sustain-phase-only per-request TTFT / **E2E** / TPOT from `metrics.jsonl`. Despite the name it makes **no** concurrency-1 assumption and works unchanged on this loaded run — verified by running it against the shipped results. Drops `new_tpots == 0.0` (the SOLO_M1 "filtered" marker) rather than averaging it in as a zero-latency token. Note `new_e2es` is in **seconds**; this script multiplies by 1000. |

## Bring-up (needed for a full re-run, §3 of REPRODUCE.md)

| script | role |
|---|---|
| `mix_site.sh` | site entry point — `up` / `smoke` / `down`. Holds the two site-specific values (`MY_IP`, `MODEL`). |
| `mix_up.sh` | the five-stage bring-up: container → etcd → kvd → mix worker → router. |
| `mix_common.sh` | shared helpers: container start, etcd, kvd daemon, router start, `reap()`, `wait_health`. Carries the fix for the `pgrep -f` self-kill defect found in Phase 1 (`fixlen.glm52.mix.packup_20260806/notes.md` §2) — **if you refactor a process-reaping line here, carry its warning comment with it.** |
| `mix_engine.sh` | the mix worker launch: one `python3 -m infera.engine.sglang`, TP8, `disaggregation_mode` unset, plus the mandatory DSA-on-ROCm env block. |
| `mix_smoke.sh` | the feature gate — the seven-row table in REPRODUCE.md §3. Read the blocks, not the exit code. |

## Diagnostics

| script | role |
|---|---|
| `envsnap.sh` | on-node environment snapshot: GPUs, driver, image id + digest, container binds, versions, and the engine's **resolved** cmdline. Run it while the deployment is live — the resolved cmdline and kvd counters cannot be recovered afterwards. Output for this deployment is `../env/env_chi2835.txt`. |
| `scan_err.sh` | **time-scoped** engine-log error scan. Takes `(logfile, window-prefix)`. This exists precisely because the engine log is appended across all three phases — a whole-file grep is a guaranteed false positive. This is how "the engine rejected nothing / 1802 of 1802 HTTP 200" was established. |
| `accept_len.sh` | MTP accept-len distribution, **UNSCOPED** (whole file). Use only on a log you know belongs to one run. For this packup use the window-scoped form in REPRODUCE.md §7 instead, or the pre-sliced `../logs/engine_loadwindow.log.gz`. |

## Not carried

`mix_bench_fixlen.sh` and `summarize_fixlen.py` belong to Phase 1 only; see
`fixlen.glm52.mix.packup_20260806/scripts/`.
