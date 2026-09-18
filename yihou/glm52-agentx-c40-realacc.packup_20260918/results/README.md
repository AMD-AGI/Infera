# Results

## Start here

- **`accept/verdict.txt`** — the one number this run was for:
  `spec_accept_length` per decode rank under live load, how many ranks were
  active, and PASS/FAIL against the 2.0 bar.
- **`t1-osl-analysis.txt`** — the 40.95 % output shortfall, computed per request.
  Read `../notes.md` §2 before quoting it; the method matters more than the
  number.
- **`agentx/agentx_conc40.json`** — the AgentX aggregate.

## `accept/`

| file | what |
|---|---|
| `metrics-before.txt`, `accept-before.txt` | `/metrics` scraped **before** load. All ranks read `0.0`. This is **not** a measurement — an idle rank is indistinguishable from one that accepts nothing. Kept so the trap is visible. |
| `metrics-after.txt`, `accept-after.txt` | scraped **while the profiling phase was live**. This is the measurement. |
| `accept-from-log-after.txt` | the same quantity from the decode server log's own `accept len: …, accept rate: …` line — an independent second source, and the same source the failing baseline was read from |
| `verdict.txt` | per-rank values, active-rank count, min/mean, PASS/FAIL |

The numbers:

| dp_rank | `spec_accept_length` | `spec_accept_rate` |
|---|---|---|
| 0 | 2.1500 | 0.2300 |
| 1 | 2.3305 | 0.2661 |
| 2 | **3.0750** | 0.4150 |
| 3 | 2.7917 | 0.3583 |

4/4 ranks active. min **2.1500**, mean 2.5868 → **PASS** (bar 2.0).
Log series: `accept len: 2.33 / 4.50 / 2.46 / 3.29 / 2.77`.

`accept_length` and `accept_rate` are different metrics; the bar is on length.
The broken configuration read length **1.25** / rate **0.05**.

## `agentx/`

| file | what |
|---|---|
| `agentx_conc40.json` | the aggregate: throughput, TTFT/ITL/E2E percentiles, request accounting |
| `profile_export.jsonl.gz` | **per-request ground truth**, 1151 records. The only artifact from which the OSL shortfall can be recomputed. Each metric is a `{"value":…,"unit":…}` object, not a bare number |
| `profile_export_aiperf_timeslices.{csv,json}.gz` | per-timeslice detail; not used for any conclusion here, included for time-series work |
| `runtime.env`, `benchmark_command.txt` | exactly how AIPerf was invoked |
| `workload_distribution_summary.txt`, `*.png` | workload shape and metric plots |

`server_metrics_export.json` (584 MB) was **excluded**. Its content is covered by
the `/metrics` snapshots in `accept/`.

## `server-info/` and `launch-command-lines.txt` — two records, different scopes

**They are not interchangeable.** `server-info/{prefill,decode}-0.json` is each
engine's own `server_args` dump — CLI flags only. It does **not** carry the
container environment. `launch-command-lines.txt` is the launcher's emitted
`docker run` line, one token per line, and is the only place the env vars are
recorded. Check a claim against whichever one actually holds it.

Verified in `server-info/` (CLI flags):

| key | prefill | decode |
|---|---|---|
| `disable_custom_all_reduce` | `False` | **`True`** |
| `speculative_algorithm` / `num_steps` | — | `EAGLE` / `5` |
| `enable_aiter_allreduce_fusion` | `True` | `True` |
| `enable_hierarchical_cache` | `False` | `False` |
| `max_running_requests`, `tp_size`, `dp_size` | 64, 4, 4 | 64, 4, 4 |

Two things this table settles: the custom-all-reduce fix was applied to the
**decode leg only**, as intended; and HiCache is off on both legs, which is why
PR #37152 is inert in this run.

Verified in `launch-command-lines.txt` (container env and full CLI):

| token | occurrences | meaning |
|---|---|---|
| `SGLANG_OPT_USE_TOPK_V2=false` | 2 | both legs; already the committed default, asserted rather than set |
| `SGLANG_SIMULATE_ACC_LEN` | **0** | **real acceptance, proven by absence** |
| `--disable-custom-all-reduce` | 1 | decode only |
| `…-nextnfix-hicache` | 2 | both legs ran the same image tag |

## `workers.json`, `router-health.json`, `worker-health.json`

Router's view: two workers, correct roles, both healthy.
The launcher's `failures/` directory was **empty** — 0 Mooncake transfer
failures, 0 router affinity 503s.

## Reading the evidence

**The acceptance result stands on three independent legs**: the `/metrics` gauge
across four ranks, the decode log's own line, and the fact that the text is
coherent at all — a broken speculative path in this configuration produced
`1!!!!!!!` and an accept length of 1.25.

**The throughput result does not stand on its own.** 40.95 % of requested output
tokens were never produced, and AgentX still scored `0/1067` errors, because an
OSL mismatch is a warning rather than a failure. Any comparison against another
run must first check that run's own per-request deficit.
