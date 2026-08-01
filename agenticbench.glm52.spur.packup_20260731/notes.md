# Notes — gotchas, wrong turns, and what this run does not establish

## The wrong turn that cost a 20-minute window

**What.** The first Case A launch used `new_session_rate = 0.145` and was aborted at
t=1195 s of 4000.

**Why it was wrong.** The probe held ~22 live sessions at rate 0.10, so I scaled
linearly to target 32: `0.10 x 32/22 = 0.145`. Linearity assumes service time is
independent of load. It is not. The model predicted ~26 in-flight; measurement gave
**44–48**, pinning `max_inflight` (25 of the last 120 ticks at the cap, mean 44.2)
while the live population climbed monotonically 40 → 57.

**How it showed.** Both abort criteria from the guide fired at once: in-flight pinned
at the cap, and sessions climbing rather than oscillating. When `max_inflight` binds,
backpressure — not the workload — sets the load, and the run is no longer the one you
configured. The server was healthy throughout (0 faults, 0 exceptions), so this was
purely my configuration error.

**Context.** ctx also doubled (131072 → 262144) between the probe and the full run,
raising per-request service time on its own. Two variables moved; the linear
extrapolation accounted for neither.

**The fix.** Anchor on measurement, not on a model. Two measured points existed:

    rate 0.100 -> in-flight 10-27 (mean ~18), sessions 19-26   STABLE
    rate 0.145 -> in-flight 44-48 (pinned),   sessions 40-57   SATURATED

Interpolating between an unsaturated and a saturated point is invalid, so the rerun
stepped modestly from the known-stable point to **0.110**, which held in-flight at
p50 28 / max 46 without pinning. Aborted run preserved at
`/shared_nfs/yihou_agentbench/bench/caseA_full_ABORTED_saturated/`.

## `--dashboard-mode` is mandatory, and nothing warns you

**What.** `summary.json`, `metrics.jsonl` and `metadata.json` are **all** written inside

    if dashboard_mode and benchmark_name and data_dir:     # agent_throughput.py:1674

**Why it matters.** Without the flag the run completes normally, prints a full report
to stdout, exits 0 — and persists nothing structured. Goal item 2's percentiles and
Goal item 4's `num_sessions_active` time series are simply gone.

**How it was caught.** The probe run finished cleanly and its output directory
contained only `run.log`. `run_bench.sh` now passes the flag unconditionally.

**Context.** The probe still served its purpose (calibration and stability), so it was
not re-run; the deliverable run has full artifacts.

## Why the context length had to be 262144

At `--context-length 131072` the probe showed prompt p90 **and** p99 both pinned at
exactly 131,072 — a visibly clipped distribution. Fitting a lognormal to the spec
triple (p50 74K / p90 155K / p99 235K) puts **16.1 %** of requests above a 131,072
clamp versus **1.4 %** above 262144.

Running the deliverable at 131072 would have misreported the spec's input distribution
while looking superficially fine. At 262144 the observed percentiles land within 2–4 %
of spec at every point (p50 73,862 / p90 151,526 / p99 226,854).

Cost: an engine restart and a re-gate, since the KV pool changes and the cold start is
not the same event.

## All 96 errors are one client-side timeout

`aiohttp.ClientTimeout(total=240)` is hardcoded at `agent_throughput.py:928`. With p99
input near 227K tokens plus a long generation tail, some requests legitimately exceed
240 s.

Verified exactly, not inferred: **96** lines matching `timed out`, **0** matching
`failed: HTTP`, **0** other exceptions — summing to the summary's `errors: 96`. Both
engine legs logged 0 GPU faults and 0 scheduler exceptions for the entire window.

Reported as measured (0.953) rather than adjusted upward. A future run wanting to meet
the 0.97 SLA should raise that timeout, not tune the workload.

## Deployment gotchas that will bite a cold reproducer

* **Never probe a PD leg directly** — `curl` to a leg's own port hangs. Go through the
  router.
* **Grep server logs with `strings`.** They contain binary bytes; plain `grep -c`
  returns 0, which reads exactly like "the bad thing never happened".
* **Check `Errno 98` *after* the ready line.** A `--kv-snapshot-port` collision lets a
  leg log `ready to roll` and *then* die during etcd registration — it looks healthy
  and simply never appears in `/v1/workers`.
* **`docker exec -d $CTR bash -lc '...'` does not persist.** The detached login shell
  exits and takes the child with it: no process, no log, no error. Stage a script file.
* **VRAM release is asynchronous.** After `kill -9`, processes go `Z` and `rocm-smi`
  still shows 90 % for a minute or two with no live holder. Poll to 0 % before
  rebooting or the next boot OOMs.
* **Cold start is ~5–8 min.** Eight live `sglang::scheduler_DP*` processes means it is
  working, not hung.
* **`--enable-cache-report` or Goal item 2 loses its cache column** — the bench reads
  `usage.prompt_tokens_details.cached_tokens`.
* **`acc_len` / `mtp_draft_tokens` must be 1.0 / 1** in the YAML. Left at the shipped
  1.56 / 5, the driver reports a fabricated MTP-adjusted TPS for a deployment with no
  speculative decoding.

## kvd wrote a lot and read almost nothing — expected, not a defect

Across Case A: **+131,162 sets, +18 gets**, long tier grown to 415 GB, +131,295
evictions.

Case A is one shared prefix at a 0.89 hit rate, so sglang's **in-GPU radix cache**
satisfies nearly every hit before L3 is ever consulted. The engine-reported hit rate of
0.890 is the GPU tier doing its job. This is why a latency win could never have
attributed anything to kvd, and why the read path was proven separately by
restart-and-replay (see the companion kit).

## What this run does NOT establish

* **No A/B against kvd-off.** kvd was ON (prefill) for the whole measured window. No
  claim is made that kvd improved any serving metric.
* **No MTP.** Goal item 3 is N/A, not measured-and-poor. The bench's
  `acceptance_length` is deliberately not quoted — without speculative decoding it
  degenerates to ~1 and would read as a measurement of a feature that is not running.
* **Case B not run** (needs a 520K-context engine).
* **One run, no repeat.** No confidence interval on any percentile.
* **p99 turns-per-session is truncated** by the 3600 s window, exactly as the guide
  predicts for a p99 session whose modelled lifetime is ~3557 s.
* **`max_sessions=128` was reached at t≈1089 s**, so new-session creation stopped for
  the remainder and the population plateaued partly by that ceiling rather than purely
  by service time. A larger `max_sessions` would let it find its own equilibrium.
* **The needle 5/5 was obtained with a warm L3 prefix cache.** The same suite scored
  3/5 and 4/5 on cold-prefill runs. That dependence is real and measured; its mechanism
  is **not established** and is not asserted. See the companion kit's `notes.md`.
