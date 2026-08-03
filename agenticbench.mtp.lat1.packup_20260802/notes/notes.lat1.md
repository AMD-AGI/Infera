# lat1 — configuration notes, defects found, and what they cost

## The workload in one paragraph

`lat1_full.yaml` is `caseA_full.yaml` with the population collapsed to one. Every
knob describing a REQUEST is byte-identical (input/output percentile triples,
`cache_hit_rate: 0.89`, `max_input_tokens`, `system_prompt_len`, tokenizer,
`acc_len`, `mtp_draft_tokens`, `mtp_overhead_factor`, `sla`, `gpus`, `window`).
Seven knobs differ. Six are the concurrency structure; the seventh is a seed that
had to change because of a real defect (below). **No server flag was touched** —
same image, same two legs, same `--mem-fraction-static 0.80` on prefill, same
`GLM52_P1V3` on decode.

| knob | Case A | lat1 | why |
|---|---|---|---|
| `turns_per_session` | 3/20/103 | **1/1/1** | one turn per session |
| `initial_sessions` | 32 | **1** | one session at t=0 |
| `max_sessions` | 128 | **1** | **this is what actually enforces it** |
| `new_session_rate` | 0.10 | **1.0** | respawn immediately after retirement |
| `max_inflight` | 48 | **1** | belt and braces |
| `ramp_duration` | 400 | **180** | see below |
| `sustain_duration` | 3600 | **1800** | 30 min, as specified |
| `random_seed` | 1337 | **20260802** | **defect, see below** |

### Why `max_sessions: 1` and not just `initial_sessions: 1`

They are different guarantees and only the first one holds. The control loop
spawns a replacement every tick it finds room (`agent_throughput.py:2509`):

```python
if new_session_rate > 0 and active_sessions < max_sessions:
    if random.random() < new_session_rate: <spawn>
```

`active_sessions` counts `is_available() or in_flight` (`:2416`, `:310`), so
while the current request is in flight the count is 1, the guard is false, and
nothing spawns. With `initial_sessions: 1` alone and the Case A `max_sessions:
128`, the population would grow by ~0.1/s and this would have become a slow ramp,
not a concurrency-1 measurement.

### Why `turns_per_session: 1/1/1` works

`PercentileSampler` requires `0 < p50 <= p90 <= p99` (`sampling.py:102`), so the
degenerate triple is legal; `_interp_lnv` returns `ln(1)` for every z and
`sample_int()` returns exactly 1. The session then **retires** rather than merely
breaking (`:2384-2391`) — which matters, because a session that only breaks still
reads as live, would hold the single `max_sessions` slot forever, and the run
would have issued exactly ONE request in 30 minutes.

### Why `ramp_duration` 400 → 180

The only knob changed for a reason other than concurrency. `ramp_duration` is a
warm-up **exclusion** window, and its job is different here. In Case A it covers
two effects: the shared prefix becoming resident, and a synchronized 32-session
cohort dispersing. At N=1 there is no cohort — only the prefix, which is resident
after one or two requests. At ~16 s/request, 400 s would have discarded ~25 of
124 samples (20 % of the dataset) to re-prove something the first two requests
settle. Measured after the fact: ramp captured 4 requests, sustain 119, drain 1.

---

## Defect 1 — cache contamination across runs (severe; invalidated the first attempt)

**Symptom.** The first lat1 attempt returned cache hit **~100 %** on its opening
requests against a configured 0.89, with only **14–62 uncached tokens** and TTFT
of 392–1,673 ms. Those requests were doing essentially no prefill, so their TTFT
is not a prefill service time. Had this gone unnoticed the headline "latency
floor" would have been roughly 3× too good.

**Cause, from source rather than inference.** Two seeds feed prompt construction
and **neither varies across runs**:

    :2054   profile_shared_base = make_filler_seeded(..., seed=987654321)
    :2187   fresh part          = make_filler_seeded(fresh, seed=request_id + 1)

The fresh-content seed is the **run-local `request_id`**, which restarts at 0
every run. So any two runs sharing `random_seed` draw the same input lengths →
the same `cached = round(0.89·L)` split → **byte-identical prompts in the same
order**. The second run replays the first run's prompts into whatever cache is
still warm.

**Verified, not assumed.** The aborted full run's first 17 prompt lengths
(61,473 / 53,668 / 164,146 / 183,017 / 130,129 / 35,342 / …) match the probe's
first 17 exactly.

**Which cache.** Not kvd: `gets_total` held at **15,210** across the contaminated
requests (snapshots preserved). It is the prefill leg's **in-GPU radix tree**,
still holding entries from the Case A run of 2026-08-01 — which used
`random_seed: 1337` and the same shared-base seed, and therefore has the same
opening request sequence. This also explains why the *probe* showed ~100 % for
its first 12 requests and then dropped: Case A's 32 interleaved sessions scramble
the draw order after roughly that point.

**Fix.** A distinct `random_seed` changes the sampled input lengths, which
changes the cached/fresh split, which breaks the shared-base prefix match before
the fresh remainder — so no stale entry can be hit. The probe carries its own
third seed (`2026080299`), because otherwise the probe would warm the full run's
cache and reintroduce the defect one level down.

**Verified after the fix**: cache hit 88.9–89.0 % from request #0, uncached
4.9K–16.5K tokens, TTFT monotone in input length. Final run: actual 0.8897 vs
ideal 0.8899, efficiency **0.9998**.

**Cost, stated plainly.** lat1 is no longer bit-comparable to Case A at the level
of individual prompts. It remains comparable at the level of the **distribution**,
which is what both configs specify and both reproduce. A shared seed was never
giving comparability here anyway — only contamination.

**Generalisation worth carrying**: on this harness, *any* two runs of the same
config against a warm engine will silently measure cache hits instead of compute.
This is not specific to lat1. Two runs of Case A back-to-back would have the same
problem.

## Defect 2 — numpy seed range (trivial, caught by the probe)

`random_seed: 20260802999` exceeds `2**32-1`; the driver seeds `np.random`
directly and numpy raises `ValueError: Seed must be between 0 and 2**32 - 1`,
killing the run at startup. Changed to `2026080299`. Costs nothing to state and
would have cost a 33-minute window to discover live — which is exactly what the
probe is for.

---

## Structural limit of this experiment: sample size

At concurrency 1 the request rate **is** the service time. 124 requests in
1,981 s = 16.0 s/request, which equals the mean composed E2E of 16.56 s. Reaching
n=1,000 would take **4.4 hours**.

Consequence, quantified with order-statistic CIs (n=120 sustain):

* **p50 (±11 %) and p90 (±12 %) are solid.**
* p75 (±22 %), p95 (±19 %) are weak.
* **p99 (±25 %) is not a percentile** — it is the 118.8th of 120 order
  statistics, i.e. essentially the third-largest observation. Reported as a range
  (5.8–9.3 s), never as a point.

This is inherent, not an oversight: the measurement requires concurrency 1 and
concurrency 1 caps the sample rate. The mitigation used instead is the
**TTFT-vs-input-length curve**, which at N=1 has R² = 0.956 over 124 points and
carries far more information than a percentile of a mixed distribution.

---

## What this run does NOT establish

1. **The cached-vs-uncached token cost split.** Cached and uncached tokens are
   collinear by construction here (ratio 7.98–8.09 across all 124 requests, i.e.
   effectively constant at the fixed 0.89 hit rate), so the regression cannot
   separate their coefficients. The comparison against the sweep's 0 %-cache
   conc=1 points bounds it — lat1's per-uncached-token rate is ~35 % worse, so
   cached tokens are **not** free — but the coefficient is not identified. A
   cache-hit sweep at fixed length would settle it.

2. **A controlled MTP ablation.** No MTP-off arm was run on this image. TPOT
   10.66 ms is the measured floor with MTP on; the counterfactual is not
   measured.

3. **Anything about multi-turn behaviour.** `turns_per_session: 1` removes it by
   design. The 0.89 cache hit here comes from the shared base prefix, not from
   conversational history.

4. **The knee location precisely.** The superlinear onset is visible between the
   160–200K bin (n=15) and the 200–240K bin (n=4), and the 240K+ bin has n=1. The
   knee is stated as "roughly 160K–200K"; pinning it would need targeted sampling
   at fixed lengths across that range.

---

## Known failure modes — what will bite a re-runner, and how to avoid it

These are not lat1's own defects (those are above). They are the ways *this
stack* fails, collected so a cold reproducer does not rediscover them. Each is
first-hand: every one was hit at least once in this workstream.

### Timing traps — things that look like hangs and are not

| symptom | reality | do this |
|---|---|---|
| leg silent for 5–6 min after launch | cold start: weights → tilelang JIT → DP cudagraph capture | **wait.** Killing it and retrying costs another 6 min and proves nothing |
| `wait_ready.sh` still polling at 300 s | normal; both legs reached ready at ~320 s here | budget 1,800 s, not 300 |
| probe seems stalled at ~10 requests | each lat1 request takes ~16 s, and a 260K prompt takes ~9 s of prefill alone | check `In-flight: 1` in the console — if it is 1, it is working |

### Restart discipline

**Restart BOTH legs together, never one.** Restarting one orphans the other's
c10d state and the survivor will hang on the next collective. And `boot.sh` waits
for *process* teardown but **not** for VRAM release — poll all 16 GPUs to 0 %
before relaunching, or the new leg allocates into a partially-freed pool and dies
with an OOM that looks nothing like the real cause.

### The binary-log grep trap

Both leg logs contain binary bytes. Plain `grep` prints `Binary file matches` and
**counts nothing** — so a fault scan reads as clean when it is not. Always pipe
through `strings`:

```bash
strings $W/logs/armB_decode.log | grep -cE 'HSA_STATUS_ERROR|...'   # correct
grep -cE 'HSA_STATUS_ERROR|...' $W/logs/armB_decode.log             # LIES
```

Related: `grep -c 'retracted'` on these logs returns thousands of matches that
are all `#retracted-req: 0`. Match on the *value*: `grep -cE '#retracted-req: [1-9]'`.

### Healthy-looking legs that never registered

Check `Errno 98` **after** the `ready to roll` line, not anywhere in the log. A
`--kv-snapshot-port` collision lets a leg print `ready to roll` and then die
during etcd registration. It looks up; it is not serving.

### Never probe a PD leg's own port

It hangs — the leg is waiting for its disaggregation peer. Always go through the
router (`:8190`).

### `--dashboard-mode` is not optional

`summary.json`, `metrics.jsonl` and `metadata.json` are **all** written inside
`if dashboard_mode and benchmark_name and data_dir:` (`agent_throughput.py:2067`).
Without the flag the run completes, prints a full report to stdout, exits 0 — and
persists nothing. This has already cost one run in this workstream.

### `temperature: 0` + MTP is indistinguishable from KV corruption

GLM-5.2's `generation_config.json` specifies temperature 1.0 / top_p 0.95. Greedy
decoding drives the model into repetition on a long prompt and EAGLE amplifies
it, producing output that looks exactly like a cache-coherency bug. Use the
official sampling in every probe. (Note this does **not** apply to the benchmark
requests themselves — the driver hardcodes `temperature: 0.0` at
`agent_throughput.py:2229`, and both Case A and lat1 ran that way. It matters for
`correctness.py` and any hand probe.)

### A needle miss is not automatically a bug

Run it twice. If the *failing depths move between runs* and every depth returns
its exact needle at least once, it is sampling variance, not corruption. That is
what happened here (3/5 then 4/5, disjoint failures). Do not sink time into it
beyond the second run.

### spur specifics

* **`ssh` to compute nodes is blocked and the error lies** — it says
  `Permission denied (publickey)`; the real cause is an `AllowUsers ubuntu root`
  whitelist. Your key is fine. Use `spur exec`.
* **Never background a long docker client inside `spur exec`** — the exec
  namespace teardown kills it. Stage a script and `docker exec -d` it.
* **`export DOCKER_CONFIG=/tmp/dockercfg` before every docker call** (docker 29
  buildx plugin discovery).
* **Artifacts >500 MB go to `/shared_nfs`, never `/home`.**

### Do not enable `SGLANG_DEBUG_DSA_ROWS=1` for a full run

It emitted 27k lines in one tail window and 144 MB over a full run on the sibling
cluster. Useful for a targeted 60-second reproduction, ruinous for 33 minutes.
