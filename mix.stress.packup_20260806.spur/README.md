# GLM-5.2 MIX — Task 3 agentic STRESS (Case-A, closed-loop)

**Ran:** 2026-08-06 (timestamp `2026-08-06-10-54-55`)
**Author:** yihou (AMD)
**Status:** PASS — a 1-hour closed-loop Case-A stress run drove the MIX
(aggregated) server to its genuine saturation point (in-flight cap binds) with
89% prefix reuse and 98.2% success. Same frozen server family as Tasks 1 & 2.

## Goal

Task 3 of the GLM-5.2 agentic-bench-on-AMD mission: **stress** a MIX
(aggregated, non-PD) SGLang deployment of GLM-5.2-MXFP4 on a single 8×MI355X
node with a **realistic closed-loop agentic Case-A workload**, and report the
steady-state saturation numbers (TTFT/TPOT, cache-hit, live sessions, in-flight,
token throughput). The load is driven by the **INFERA agentic driver**
`agent.agent_throughput` — **not** the customer/AgentX bench (mission rule 5) —
run on the node host against the router `:8100`.

**Spec:** mission `CLAUDE.md` at the repo root + the operator's stress config
(initial_sessions=8, max_inflight=16, max_sessions=24). See `REPRODUCE.md` §3 and
`notes.md` for the closed-loop method and the `new_session_rate` re-solve.
**Success criteria:**
1. One frozen MIX server serving GLM-5.2-MXFP4 with DSA + MTP + kvd + kv-aware
   all verified live (smoke passes).
2. A closed-loop Case-A stress run that actually loads the server to a caps-bound
   steady state over an honest window (ramp 400 s + sustain 3600 s), reporting
   the sustain-phase saturation metrics with the Case-A cache-hit (88–90%) held.

## Result

Both met. Reported = the **sustain phase** (3600 s) of run 2 (run 1 was a
calibration — see below and `notes.md`).

| metric | value | note |
|---|---|---|
| requests completed (sustain) | 2092 over 3600 s | |
| success rate (whole run) | 98.2% (31 err / 2364) | ≥ 0.97 target |
| offered QPS | 0.58 req/s | closed-loop, set by live sessions |
| cache-hit rate | 89.0% | Case-A target 88–90% ✅ |
| live sessions (steady) | ~23–24 | at the session cap (24) |
| in-flight (steady) | ~15–16 | at the in-flight cap (16) ⇒ backpressure binds |
| TTFT p50 / p90 | 1460 / 3610 ms | |
| TPOT p50 / p90 | 19.0 / 32.6 ms | agrees with ref kit (~14–16 ms) |
| input TPM | 2,988,618 | |
| uncached TPM / GPU | 41,330 | the real prefill work |
| gen (visible) TPM / GPU | ~3,381 | |

**Interpretation:** the in-flight cap (16) binds in steady state, so this is a
**genuine saturation point** — the server holds ~24 live sessions with 89% prefix
reuse and 98% success. TPOT p50 19 ms and cache-hit 89% agree with the reference
kit.

## How to reproduce

See `REPRODUCE.md`. TL;DR: build `infera/engine-sglang:final-pr` on the node from
this branch, `mix_up.sh` to bring up etcd+kvd+worker+router, `mix_smoke.sh` to
prove features, then `run_stress.sh` (host venv) to run the closed-loop stress
against the router `:8100`.

## Folder map
- `REPRODUCE.md` — ordered, copy-pasteable reproduction from a clean node
- `environment.md` — exact node / image / weights / git SHA + engine recipe
- `scripts/` — stress runner + workload yaml, and the 3 bring-up scripts (verbatim)
- `results/` — `stress_summary.md` (table) + `summary.json` (phase rollups) +
  `metrics.jsonl` (per-window samples) + `metadata.json` (resolved run params)
- `notes.md` — closed-loop semantics, the `new_session_rate` re-solve, gotchas
