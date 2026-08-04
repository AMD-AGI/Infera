# Notes — traps, wrong turns, and what this run could not answer

Ordered by how much they would cost someone repeating this. The first two each
**produce no error** — a run that looks like it is working, or a failure line on
a run that actually succeeded.

---

## Trap 1 — a 1-turn session silently stalls the whole run (cost: 12 minutes)

**What.** I built a 1-session smoke corpus by picking the *smallest* session in
the trace. It had **1 turn**. aiperf then sat in warmup for **690 seconds** with:

```
Phase warmup progress | returned=0/2 | sent=0 | in_flight=0 | errors=0 | elapsed=690.1s
```

**`sent=0` and `errors=0` at the same time.** No exception, no timeout, no
diagnostic — just a benchmark that never starts.

**Why it happens.** `timing/trajectory_source.py:423` skips any trace with
`n <= 1` turns, because the agentic scenario needs at least one warmup turn
*and* one profiling turn. The skip is a `logger.warning`, and the surviving lane
count then floors to zero credits. The customer's README documents that
**conc=1** is unsupported; it does **not** document the ≥2-turn requirement,
which is the same failure with a different cause.

**How it was caught.** By reading the `TrajectorySource` lines in
`aiperf.log`, which print `root_next=0/1 (0% turns)` per lane — a lane sitting at
turn 0 of 1 is the tell. Re-picking a ≥3-turn session made warmup complete in
4 seconds.

**Context.** The real 200-session corpus is unaffected (its turn P50 is 3). This
only bites when you construct a reduced corpus for a smoke test — which is
exactly what a careful operator does first.

## Trap 2 — the run succeeds and the script reports FAILED (cost: nearly the whole run)

**What.** Both concurrency points completed cleanly — c8 `sent=233 completed=230
errors=0`, c16 `sent=329 completed=318` — and `summary.csv` read:

```
conc  out_tok_s  req_s  ttft_p50 ...
8     FAILED
16    FAILED
```

**Why.** `replay_caseA.sh`'s `run_aiperf()` mounts `$HERE`, `/models`, and
`/shared_nfs` — and nothing else. `--output-artifact-dir` is `$OUT/c$C/art`. I
set `OUT=/root/agentx_20260803/results`, **outside `$HERE`**, so aiperf wrote its
artifacts into the container's own filesystem. They were destroyed on container
exit, and the script's summary parser found no
`profile_export_aiperf.csv` — hence `FAILED` on a run that worked.

**How it was caught.** By checking `find results/c8 -type f` on the host (empty)
and then `docker exec <container> find ...` (populated). The evidence was
unambiguous before anything was changed.

**How it was resolved without modifying the customer's script.**
`scripts/rescue_artifacts.sh` polls for the running aiperf container and
`docker cp`s the artifact tree out every 20 s. The last copy before exit is the
complete one. Verified: 239 and 335 records recovered, matching the counts in
`aiperf.log`.

**The one-line alternative** is `OUT=$W/bench/results` (inside `$HERE`). We chose
the rescue loop so the defect would be documented rather than silently avoided —
it is worth reporting upstream.

## Trap 3 — 6,396 stack traces in the log, all from one harmless source

`grep -i error` on `aiperf.log` returns thousands of hits:

```
ValueError: Buckets are required for histogram time series
  .../aiperf/server_metrics/storage.py:525
```

**All of them are aiperf's optional server-metrics scraper** failing to parse a
Prometheus histogram from our endpoint. It does not touch request dispatch,
response parsing, or any reported metric — the run completed with 0 request
errors alongside 6,396 of these.

**Always exclude `server_metrics|Buckets are required` before counting faults
in this log.** Same discipline as the par8 kit's `server_args=` false positives:
read the matches before trusting a count.

---

## The measurement finding: the customer's README misstates its own cache metric

Their README:

> Reported cache-hit is the endpoint's realized server-side prefix hit

`aiperf/metrics/theoretical_prefix_cache.py:22-30` computes the reported
`Theoretical Prefix Cache Hit` **from the loader's walk of the trace file's
`hash_ids`** — an infinite-cache upper bound. It never queries the server.

The metric name is honest; the prose is not. Practical consequence: **that number
is invariant to the deployment under test.** A server with prefix caching
entirely disabled would report the same ~88 %.

The real server-side figure had to be computed here from
`usage_prompt_cache_read_tokens` in the raw records. It came out at
**p50 88.1 %** per request — which happens to agree, but agreeing by luck is not
the same as measuring.

**A subtlety in that number.** Token-weighted across all requests the server
cache rate reads **50.3 %**, not 88.1 %. Both are correct and they measure
different things: 54 of 231 requests at c8 are **first turns**, which have no
prefix to reuse and do not report the field at all (verified: 54/54 have
`turn_index == 0`). Weighting by tokens lets those large cold prompts dominate.
**Quote the per-request p50 when comparing against a per-request target like
"88 % cache hit"; quote the token-weighted figure when reasoning about total
prefill work saved.**

---

## What this run could not answer

### 1. Why c8 TTFT is 3.8× par8's at *lower* in-flight — NOT DETERMINED

c8 ran at mean in-flight **5.13** against par8's 12.1, and still showed TTFT p50
5,146 ms against par8's 1,365 ms. Queueing explains c16; it does not explain c8.

Four candidates, none confirmed, listed with the measurement that would settle
each in `analysis/vs_infera_bench.md`. The cheapest is candidate B:
**`cache_bust = FIRST_TURN_PREFIX`**, which the scenario forces — aiperf injects a
unique marker into every trajectory's first turn, deliberately preventing
cross-trajectory prefix sharing that par8's driver permits. Whether that costs
3.8× is **unmeasured**.

Do not attribute the gap to any one of these without running the experiment.

### 2. Where the concurrency cliff actually is — NOT LOCATED

The TTFT-vs-input-size curve is prefill-shaped at c8 (10.0× spread) and
queue-shaped at c16 (2.3× spread). The transition is somewhere between, and
**c16's TTFT p90 clears the 30 s bar by only 567 ms** — too close to call from
one sample. Bracketing it needs c12 and c24.

### 3. Why ITL's upper tail is fatter than par8's — NOT ESTABLISHED

ITL p99/p50 is 2.3 here vs par8's TPOT p99/p50 of 1.45, while the p50s agree
within 7 %. Both drivers compute the statistic differently and this run cannot
separate a real difference from a definitional one.

### 4. Whether the trace's think-times reach the server as authored

The trace carries `think_time` per turn (P50 3 s, P99 229 s) and the scenario
sets `use_end_to_start_delays`. Whether aiperf's lane model reproduces those gaps
as inter-arrival at the server, or absorbs them inside a saturated lane, is
**not verified**. The check is cheap: compare measured per-conversation
inter-arrival in `profile_export.jsonl` against the `think_time` field in the
corpus.

### 5. The customer's own deployment recipe — NOT RUN

`scripts/GLM-5.2-disagg/` (ATOM + atomesh + mooncake TCP,
**`--no-enable_prefix_caching`**, `--policy random`) was read from source only,
per the operator's decision to bench our existing deployment. Review and
recommendations in `analysis/customer_method_review.md`.

Notably, their recipe **disables prefix caching** on a workload whose defining
property is 88 % prefix reuse. On our stack that setting is worth **2.5× on
TTFT**. Whether it costs them the same on ATOM is unknown — different engine,
different KV dtype, different transport.

---

## Operational notes

- **`pkill -f rescue_artifacts.sh` kills your own ssh connection**, because the
  remote command line contains that string and `-f` matches the full cmdline.
  Use `pkill -f "rescue_[a]rtifacts"` — the bracket makes the pattern not match
  itself. Same family as the standing rule against bare `pkill -f infera.kvd`.
- **The deployment was never touched.** No restarts, no config changes, no
  `scancel`. Slurm holds on chi2835/chi2879 belong to `yeandy-debug`.
- **The load generator ran on the jump host**, so it did not contend for GPU with
  the deployment under test.
- **Both legs were still live at 12:42 UTC** when the environment snapshots were
  taken, so the recorded command lines are the ones that served the run — not a
  reconstruction.
