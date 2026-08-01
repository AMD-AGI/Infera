# Optimus-AgenticBench Case A on infera + GLM-5.2-MXFP4 (spur)

**Ran:** 2026-07-31 (single day; measured window 17:10–18:17 UTC)
**Author:** yihou
**Status:** **PASS with caveats** — all four goal items delivered; item 3 is N/A by
instruction, and the success rate sits below the workload's SLA for a reason traced to
the load generator, not the deployment.

## Goal

Run Optimus-AgenticBench against a production-shaped infera deployment of
GLM-5.2-MXFP4 on the spur cluster and report correctness, classic serving metrics, MTP
acceptance, and session/turn/context distributions with percentiles. MTP must stay off
(the DPA+PD+MTP fix is not yet merged with kv-aware).

**Spec:** `spec/agentic.bench.md` — Goal items copied verbatim:

> 1. 简单的正确性验证。
> 2. 经典指标
> 3. mtp接受率
> 4. session并行数，session turns, context length (如果由p50 p90 p99等分位数据更好。)

Supporting spec: `spec/glm52_crxx_caseA.fix.yaml` (the Case A profile of record, left
untouched) and `spec/CASE_AB_GUIDE.md`.

**Deployment under test:** two-node PD over mooncake RDMA, DP-attention 8/8,
**kv-aware routing ON**, **kvd ON (prefill) / OFF (decode)**, **MTP OFF**,
`--context-length 262144`.

## Result

| # | Goal item | Result | Verdict |
|---|---|---|---|
| 1 | correctness | short factual **4/4**; needle-at-depth **5/5** at ~120K tokens | PASS |
| 2 | classic metrics | TTFT p50 **4,543 ms** / p90 8,821 / p99 13,070; TPOT p50 **31.3 ms** / p90 32.6 / p99 37.9; cache hit **0.8900** | PASS |
| 3 | MTP acceptance | **N/A — MTP off by instruction** (reason stated, no fabricated number) | N/A |
| 4 | sessions / turns / context | live sessions p50 **38** / p90 47 / p99 52; turns p50 **3** / p90 18 / p99 46; context p50 **73,862** / p90 151,526 / p99 226,854 | PASS (p99 turns truncated) |

Against the workload YAML's own SLA block:

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| `success_rate` | ≥ 0.97 | **0.953** | **FAIL** — all 96 errors are the client's hardcoded 240 s timeout; **0** server faults |
| `ttft_p90_ms` | ≤ 30,000 | **8,821** | PASS (3.4x margin) |
| `e2e_p50_ms` | ≤ 4,500 | not directly reported by the driver | N/A |

Deployment health across the full 67-minute window with kvd ON: **0 GPU faults,
0 scheduler exceptions** on both legs.

Sanity checks the guide asks for, all passing: cache hit **0.8900** vs configured
0.89 (efficiency 1.0002, **zero eviction**); live session population flat, not
climbing; `max_inflight` peaked at 46/48 and **never pinned**.

Full numbers, kvd counters, and the complete limitations list: **`RESULTS.md`**.

## How to reproduce

See `REPRODUCE.md`. TL;DR: build the kvaware+kvd image on both nodes, apply two runtime
patches, boot the two PD legs at ctx=262144, pass an 8-row gate, run a short probe to
calibrate the offered load, then run Case A for 67 minutes.

## Folder map

- `REPRODUCE.md` — ordered, copy-pasteable steps with each verification
- `environment.md` — exact HW/SW, pinned image digests, git SHAs, transport
- `spec/` — the originating task spec, the Case A profile of record, the guide
- `scripts/` — bring-up, leg launch, correctness, bench driver, result extraction
- `patches/` — the two runtime fixes, with rationale
- `workloads/` — `caseA_probe.yaml`, `caseA_full.yaml` (derived; originals untouched)
- `results/` — `summary.json`, `metrics.jsonl.gz`, kvd counters, human-readable reports
- `analysis/` — post-run analysis derived from `results/`:
  - `sli_percentiles.md` — full TTFT ladder (p1…p99.9) recomputed from the 2,781 raw
    per-request samples, TPOT, supporting distributions, and why E2E has no SLI
  - `yaml_vs_measured.md` — every YAML knob against its measured value, with the
    conversion rule; knobs with no measured counterpart carry their blast radius instead
- `notes.md` — gotchas, the aborted first attempt, what this does not establish
- `logs/` — correctness output

## Read this before trusting any number

* **The bench grades nothing.** It sends synthesized filler. Goal 1 is answered by
  `scripts/correctness.py`, never by the bench.
* **`--dashboard-mode` is mandatory** or no structured artifact is written at all. The
  probe run lost its `summary.json` to this before it was caught.
* **The first Case A attempt was aborted** for pinning `max_inflight`, and is described
  in `RESULTS.md` rather than quietly dropped.
* **No kvd-off A/B was run**, so no performance claim is made for kvd.

The companion kit `kvd.rocm.hostalloc.packup_20260731/` covers the ROCm hicache bug
that had to be fixed before kvd could be enabled here at all.
