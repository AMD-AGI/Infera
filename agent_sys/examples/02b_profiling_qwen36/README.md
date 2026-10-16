# 02b · profile Qwen3.6 — profiler off (online) and on (offline)

Level 2 of the [demo ladder](../README.md), beside
[`02a_sglang_serve_qwen36`](../02a_sglang_serve_qwen36/). It runs 02a's
deployment and then profiles it:

```
main
├─ m1_deploy                      (e2e-flow stage 1)
│  └─ deploy_and_prove            AI agent   -> deploy_kit
└─ m2_profiling                   (e2e-flow stage 2)
   ├─ run_profiling_mode_off      program    -> profiling_mode_off.bench_result
   ├─ run_profiling_mode_on       program    -> profiling_mode_on.{bench_result,profile_result,kernel_table}
   └─ merge_profiling_evidence    program    -> profiling_evidence
```

- **Profiler off — the online numbers.** The stage brings the service up from
  the deploy kit, replays a fixed load with AIPerf, and records throughput, TTFT
  and inter-token latency as the user would see them.
- **Profiler on — the offline view.** The same load again with the torch
  profiler capturing; the trace is reduced to a kernel table (which kernels the
  GPU spent its time in, with input shapes), using Magpie.
- **Merged evidence.** Both benches, the trace and the kernel table in one
  handoff, with the profiler's overhead stated rather than hidden.

The two lines are program tasks: once the kit exists, stage 2 is deterministic.
Like 02a, this package writes only `main.yaml`, its root readme, this README and
`tools/make_trace.py`; the rest links into
[`../llm_e2e_performance_optimization/e2e-flow`](../llm_e2e_performance_optimization/e2e-flow/).

## Prerequisites

Everything 02a needs, plus:

- `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0` pulled (both lines replay with it).
- A Magpie checkout for the kernel table (`--var magpie_root=`).
- A load to replay. The lines replay a Mooncake trace at its recorded pacing;
  make a synthetic one:

```bash
python3 agent_sys/examples/02b_profiling_qwen36/tools/make_trace.py \
  --out /var/tmp/$USER/trace.jsonl          # 240 requests, 1024 in / 256 out, 2 req/s
```

## Run it on one host

02a's arguments, plus the load, the profiling settings and the Magpie path:

```bash
PYTHONPATH=$PWD/agent_sys python3 -m cli.main run \
  --package agent_sys/examples/02b_profiling_qwen36 \
  --demo-root /var/tmp/$USER/l2b \
  --var model_name=Qwen3.6-27B --var model_path=/data/models/Qwen3.6-27B \
  --var image=infera:sglang-gfx942-glm52 \
  --var tp=1 --var gpu_devices=7 --var context_length=32768 \
  --var "parser_args=--reasoning-parser qwen3" \
  --var container=${USER}_l2b \
  --var port_router=18201 --var port_worker=18202 --var port_etcd=18203 \
  --var validate_port_base=18240 \
  --var work_root=/var/tmp/$USER/l2b/work --var scratch_root=/var/tmp/$USER/l2b/scratch \
  --var transport=local --var node=$(hostname) --var node_ip=127.0.0.1 --var jobid=local \
  --var mock_stages=none \
  --var aiperf_trace=/var/tmp/$USER/trace.jsonl --var trace_end_ms=120000 \
  --var expect_ranks=1 --var stack_ranks=1 \
  --var magpie_root=/path/to/Magpie
```

| var | why |
|---|---|
| `aiperf_trace`, `trace_end_ms` | the load, and how much of it is the measured window |
| `expect_ranks`, `stack_ranks` | how many GPU ranks the trace must cover; equal to `tp` |
| `magpie_root` | the trace-to-kernel-table tool |

`validate_port_base` matters when 02a and 02b run at once: each run's validator
re-deploys the kit on its own port band, and two runs on the default band
collide.

## What a run looked like

Measured on one MI300X (`gpu_devices=7`, `tp=1`), the synthetic trace above
(240 requests, 1024 in / 256 out): the whole run took about 72 minutes, 44 of
them the stage-1 agent ($17.08 of model use); all 17 verdicts on the six handoffs passed.

| | profiler off | profiler on |
|---|---|---|
| output tok/s | 489.6 | 206.4 |
| requests/s | 1.91 | 0.81 |
| TTFT p50 | 188 ms | 2291 ms |
| inter-token latency p50 | 33.5 ms | 76.7 ms |

The "on" column is what the profiler costs, not what the service does; that is
why the online numbers come from the "off" line. The kernel table's top rows:
two hipBLASLt GEMMs (26.5 % and 22.2 % of GPU time) and
`fused_recurrent_gated_delta_rule_packed_decode_kernel` (9.5 %).

## What you get

```bash
R=$(ls -d /var/tmp/$USER/l2b/runs/* | tail -1)
ls $R/handoffs/                                                        # six handoffs
cat $R/handoffs/handoff.profiling_mode_off-bench_result.*/v*/content/README.md
head $R/handoffs/handoff.profiling_mode_on-kernel_table.*/v*/content/items/table.csv
cat $R/handoffs/handoff.profiling_evidence.*/v*/content/README.md      # the merged view
```

## Next

[`03_kernelforge_one_kernel`](../03_kernelforge_one_kernel/) takes one kernel
from a table like this one and optimises it.
