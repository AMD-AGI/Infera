---
name: infera-bench
description: Measure one point (AgentX or a fixed ISL/OSL load) on a live deployment and write it as an infera.result/v1 file the ledger reads
---

# infera-bench

## Pick the load

| Load | How | Cost |
|---|---|---|
| InferenceX AgentX (the project's main metric) | `engine/agentx/agentx.sh` of the ATOM kit; client pinned (InferenceX `918524ff`, AIPerf 0.12.0, `constraints.txt`) | warm-up 13-25 min + `DURATION=3600` |
| AgentX fast (screening) | same, shorter: `DURATION=1200 WARMUP_PER_LANE=1` | warm-up + 20 min |
| fixed ISL/OSL | `python -m sglang.bench_serving --backend sglang-oai --dataset-name random --random-range-ratio 1.0 --num-prompts 10xC --request-rate inf`; vLLM: `bench/serve_standard/bench_vllm_serve.sh` | minutes |

Screen with the cheap one, confirm with the real one. The ATOM kit's `agentx.sh` hands the client only the
variables it writes into `runtime.env` (`DURATION`, `WARMUP_PER_LANE`, ...): other AIPerf variables set in your shell,
such as `AIPERF_EXPERIMENTAL_FAST`, do not reach it, so shorten the window with those two. Sweep concurrency from high
to low (C176 -> C128 -> C96 ..., about x0.7) so the most informative point lands first.

## Traps that produce plausible wrong numbers

- AIPerf exits 0 on an empty window: check the request count; count errors per request, not from the summary.
- Streaming must be on, or there is no TTFT/ITL. `--stream-interval > 1` makes ITL = interval x TPOT: read TPOT.
- A fresh stack measures ~10% below a warm one: compare like with like (both fresh, or warm-up then score).
- Reuse the same trace slice and seed for both arms; flush caches (`POST /flush_cache`, decode first, retry while
  prefill answers 400) only when nothing is in flight.
- The random dataset has no prefix reuse: it cannot show kv-aware routing or KV tiers.
- Performance runs may use forced MTP acceptance (`SGLANG_SIMULATE_ACC_LEN`, `MTP_AL`): their output is wrong by
  design; correctness is `infera-quality-gate`'s job on a stack with real acceptance.
- TP > 1: sum engine counters across ranks.

## Record the point

Per point keep: the run directory (`engine-config.env`, `workers.json`, `effective-config.json`), the client's
command (`benchmark_command.txt` / `replay-command.txt`), the raw client output (`agentx_conc<C>.json`,
`profile_export_aiperf.json`), `/metrics` before and after. Then write **one result file**, flat in the task's
`results/` (the ledger reads the files at its top; raw data goes in a subdirectory or `log/`):

```json
{"schema": "infera.result/v1", "label": "c64-spf", "time": "<ISO, end of the window>",
 "role": "baseline | candidate",
 "change": "one line: what differs from the baseline",
 "metrics": {"input_tput_per_gpu": 14120.5, "p90_intvty": 31.2, "p90_ttft": 2.1, "p90_itl": 0.032,
             "requests": 3120, "errors": 0},
 "workload": {"client": "agentx", "client_version": "InferenceX@918524ff/aiperf 0.12.0", "conc": 64,
              "duration_s": 3600, "seed": 42, "forced_acceptance": 3.61},
 "config": {"kit": "atom_4p4d_glm5.2", "image": "<tag@digest>", "run_dir": "<path>"},
 "evidence": {"run_dir": "<path>", "metrics_pre": "<path>", "metrics_post": "<path>"}}
```

The names and definitions are the InferenceX App's: `input_tput_per_gpu` = input tokens (cache hits included) /
duration / all P and D GPUs, `p90_intvty` = 1 / p90 ITL (tok/s/user), times in seconds. The project's
`ledger.primary` and `ledger.x` name the curve's Y and X. Check the file before you report:
`"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger check "$AGENT_SYS_WORKSPACE/results"`. A candidate's `verdict` comes
from `infera-experiment`.

**AgentX points: convert, do not copy numbers by hand.** The converter writes one result per `agentx_conc<C>.json`
with the metric names and definitions of the InferenceX App (`input_tput_per_gpu`, `tput_per_gpu`, `p90_intvty` =
1 / p90 full-response ITL, `p90_e2e_norm_intvty`, `p90_ttft`, ...), so our curves and the official ones plot on one
chart (the infera-perf template's ledger uses `input_tput_per_gpu` against `p90_intvty`):

```bash
"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger agentx <run dirs or agentx_conc*.json> --out "$AGENT_SYS_WORKSPACE/results" \
  --series v0 --round 1 --role baseline          # a version's curve; an experiment: --series r1-spf, role candidate
```

A result is never overwritten: a repeat of a series at the same concurrency needs its own `--label` (`--label
v0-rep1` writes `v0-rep1-c64.json`), else the converter reports the name it would have replaced.

Every point of a curve carries `round` (the project's round), `series` (the curve: `v0`, `r2-spf`, `ix:<entry>` for
a reference, `role: reference`) and `workload.conc`; the ledger draws each series as a Y–X curve with its frontier
and compares series at the project's working point. Add the verdict afterwards (`infera-experiment`).
