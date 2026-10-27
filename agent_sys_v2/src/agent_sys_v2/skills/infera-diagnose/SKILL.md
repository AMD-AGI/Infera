---
name: infera-diagnose
description: Find which layer limits a deployment (prefill, decode step, KV capacity, transfer, CPU path, JIT, correctness) from metrics, engine logs, GPU traces and CPU profiles, and turn it into ranked hypotheses
---

# infera-diagnose

Diagnose before choosing a lever. In recent Infera work most gains came from scheduling, CPU paths, memory/KV
recipes, transfer and routing; a single kernel was a small part (a real profile: collectives 52% of GPU time, the
top optimisable kernel 2.9%: by Amdahl ~1-2% end to end). Mark every finding `[M]` measured or `[I]` inferred.

## Observe (cheapest first)

1. **Little's law on the client**: throughput ~= qps x mean ISL / GPUs. Split a lane's cycle into TTFT, decode
   time and client gap; the biggest part is where to look.
2. **Router `/metrics`**: `infera_time_to_first_token_seconds`, `infera_inter_token_latency_seconds`,
   `infera_pd_dispatch_duration_seconds{leg}` (P leg vs D leg: the gap approximates KV transfer),
   `infera_engine_kv_cache_usage`, `infera_engine_prefix_cache_hit_rate`, `infera_engine_kv_transfer_queue_reqs`.
   ATOM workers are not federated: scrape `http://<worker>/metrics` (`atom:*`) directly. Sample every 5 s.
3. **Engine logs** (the most used "profiler"): SGLang `Decode batch` / `Prefill batch` lines (#running-req, token
   usage, accept len; one line per 40 steps per rank, so step time = interval / 40); queue triple
   `#prealloc-req` (KV short) / `#transfer-req` (transfer slow) / `#queue-req` (compute short); ATOM PP stage times.
   Decode-side gauges sometimes freeze: trust the log lines.
4. **GPU trace**: `curl -X POST http://<worker>/start_profile -d '{"output_dir": "...", "num_steps": 30,
   "activities": ["CPU","GPU"]}'` after warm-up, during load (Python router: `POST /v1/admin/profile/start`).
   Break it down into comm / moe / attention / gemm / idle; large idle gaps mean the CPU or the scheduler is the
   limit, not a kernel.
5. **CPU profile**: py-spy as a sidecar (the kit containers have `SYS_PTRACE`):
   `docker run --rm --pid container:<ctr> --cap-add SYS_PTRACE <img-with-py-spy> py-spy record --idle --threads
   --nonblocking -d 60 -o <out> --pid <engine pid>`, or `py-spy dump`. Look at the API server, scheduler,
   TokenizerManager and router; `top -H` for hot threads; gen-2 GC pauses via a gc callback.

## Classify, then pick levers

| Symptom | Bottleneck | Candidate levers |
|---|---|---|
| prefill queue builds, long full-chunk time, FCFS head-of-line blocking | prefill capacity | PP for prefill, `--schedule-policy shortest-prefill-first`, chunk size, more P instances, async batch depth |
| step time grows with the busiest DP rank's running count | decode imbalance | request weighting / decode-demand routing |
| step time high even when balanced | decode step | kernels, collectives (custom all-reduce, comm groups), CUDA graph sizes x MTP depth |
| any rank > 90% KV, `#prealloc-req` rising, a concurrency cliff | KV capacity | mem fraction, shared comm groups, CPU KV tier / HiCache write-back, decode radix, lower CONC |
| `#transfer-req` rising, NIC far below line rate | KV transfer | Mooncake QPs / send workers / rails, check TCP fallback |
| a process at 100% CPU, TTFT >> prefill time | CPU path | tokenizer workers, token-id reuse across legs, GC, Rust router, /metrics scraping rate |
| all ranks stall ~10 s on a new shape | online JIT | precompile / warm the buckets |
| HTTP 200 with wrong text | correctness | stop perf work; bisect (`infera-quality-gate`) |

Couplings to reason about together: CUDA graph sizes x MTP depth x GEMM tuning buckets; decode batch budget >=
(MTP_K + 1) x 2 x CONC; DP attention symmetric on both legs.

## Output

`report.md`: the bottleneck class with its evidence; a ranked list of hypotheses (lever, expected mechanism and
size, how to measure, cost); what the ledger already rules out. In a round of the infera-perf template this is the
profile task's job (`infera-profile`: the same list as `candidates.yaml`) and the plan drafts the experiments;
elsewhere draft one experiment task per top hypothesis (see the project's experiment skeleton), each with its
machines.
