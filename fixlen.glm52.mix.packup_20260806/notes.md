# Notes — gotchas, defects, wrong turns, open questions

Written as **what / why / how / context**. The point is that a reader learns why
a step matters, not just that it exists.

---

## 1. Node selection: the cluster status snapshot's VRAM column can lag

**What.** The cluster-status snapshot reported `chi2878` idle at **0 GB VRAM on
every card**. A direct read said otherwise:

```bash
rocm-smi --csv --showmeminfo vram
# -> 90-136 GB per card, held by another user's live job (13 minutes in)
```

**Why it matters.** Launching on top of that would have OOM'd our engine, or
worse, degraded someone else's running job. The snapshot is a cache; the card is
the source of truth.

**How it was resolved.** Moved to `chi2835`. Its ~2,000 GB of VRAM was held by
**our own** stale `glm52_pd` container from the 2026-08-04 `verify_example` run.
Ownership was established *before* removing anything:

```bash
docker inspect glm52_pd --format '{{.Created}} {{json .HostConfig.Binds}}'
# created 2026-08-04 07:46, binds /mnt/vast + host libionic -> matches our own
# verify_example packup
```

Removed it and `glm52-etcd`; VRAM drained to 2 GB in 20 s.

**Context / rule.** Read per-node VRAM **directly** before committing to a node.
And prove a container is yours (`Created` + `Binds` + `Env`) before `docker rm`.
This is a shared cluster: never prune images, never prune `/tmp`, never
`scancel` someone's hold.

---

## 2. The one real defect — `mix_up.sh` exited rc 0 with no router

**What (the symptom).** `mix_up.sh` printed

```
[mix] === 5/5 router ===
```

and then **exited silently, rc 0**, with no router process and no
`/tmp/router.log`. Nothing in any log said why.

**Why (the cause — measured, not guessed).** Running the offending line alone:

```bash
docker exec glm52_mix bash -c "pgrep -f 'python3 -m infera.server' | xargs -r kill -9; true"
#   -> rc=137
```

`pgrep -f` matches against the **full command string**, and the `docker exec bash
-c "..."` command string *contains* the text `python3 -m infera.server`. So the
pattern matches the shell that is running the `pgrep` itself. It kills itself,
`docker exec` reports 137, and `set -e` in the caller aborts the script — before
`docker exec -d` ever runs. The trailing `; true` never executes, because the
shell is already dead.

rc 0 from `mix_up.sh` despite the abort is what made it silent: the failure
happened after the last thing that printed.

**How it was fixed.** In `scripts/mix_common.sh::start_router` — match on the
argv of a **real `python*` process** and exclude our own pid:

```bash
docker exec "$CTR" bash -c '
  self=$$
  ps -eo pid=,comm=,args= | awk -v me="$self" \
    "\$2 ~ /^python/ && \$0 ~ /-m infera\.server/ && \$1 != me {print \$1}" \
    | xargs -r kill -9 2>/dev/null
  true'
```

Two guards, both necessary: `comm` must be a python binary (a `bash -c` wrapper
is not), and the pid must not be `$$`.

**Context — this is a regression, not a new bug.** The 1P1D kit this script was
derived from carries a comment warning about exactly this hazard for `pkill`.
Rewriting it as `pgrep | xargs kill` dropped the comment and reintroduced the
bug. If you refactor a process-reaping line, carry its warning comment with it —
the comment *is* the reason the line looks the way it does.

The same class of bug is why `reap()` matches `infera.engine.sglang` /
`sglang.launch_server` / `multiprocessing.spawn` and then **waits for VRAM to
drain** instead of trusting the kill.

---

## 3. `--chunked-prefill-size` and `--max-running-requests` are GLOBAL budgets that DP-attention divides

**What.** Both are divided by `dp_size` to give the **per-rank** value. The
machine-wide budget is still what was requested — this is by design, not a flag
that failed to take:

| flag | requested (global) | resolved per rank | how it surfaced |
|---|---|---|---|
| `--chunked-prefill-size` | 65536 | **8192** | an explicit `WARNING` in the engine log |
| `--max-running-requests` | 256 | **32** | **no warning** — only the per-rank scheduler line shows it |

**Verified in the image's own source, not inferred from the arithmetic:**

- `srt/server_args.py:4902` — inside `if self._resolved().enable_dp_attention:`
  → `self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size`.
  A **division gated on DP-attention**, not a fixed clamp to 8192. The warning
  text ("adjusted to 8192 to avoid MoE kernel issues") reads like a clamp and is
  what makes this easy to misdiagnose.
- `srt/model_executor/pool_configurator.py:541-543` —
  `requested_max_running_requests_per_worker = server_args.max_running_requests // mr.dp_size`.

**Consequence for tuning.** Do **not** "fix" this by multiplying the request by
8: that would make the machine-wide budget 8× what was intended. The trap runs
the other way — hardcoding the per-rank number (8192) in a DPA-**off** arm cuts
the global budget 8×, silently, because the division only happens under DPA.

**Why it still matters to read the resolved value.** If you read the launch flags
and assume they are what the scheduler sees per rank, you will misattribute a
latency result. The `max_running_requests` case is worse because nothing warns
you: `server_args` still reports 256, and only

```
[... DP0 TP0 EP0] max_total_num_tokens=2812672, chunked_prefill_size=8192,
max_prefill_tokens=16384, max_running_requests=32, context_len=262144
```

reveals the 32.

**How to check.** Always read the engine's **resolved** state, not the request.
`env/resolved_server_args.txt` in this packup is exactly that, and
`scripts/envsnap.sh` captures it automatically.

**Settled by a source read, not left open.** At `dp_size=8` the numbers alone
cannot separate "divided by dp_size" from "clamped to 8192" — both give 8192. The
two source lines quoted above do separate them: it is a division, gated on
`enable_dp_attention`. Recorded here because the log message alone points the
wrong way.

---

## 4. Reading these logs: scope every grep by time window

**What.** The engine, router and kvd logs are **appended across the whole
session** — smoke traffic, the sweep, and (in our capture) Phase-2 traffic that
started afterwards.

**Why it matters.** A whole-file `grep` silently mixes them. Our shipped
`glm52_mix_base.log.gz` runs 07:02:13 → 10:02:04, and the sweep is only
07:13:48 → 09:45:14. An unscoped accept-len read over the shipped log gives
n=37,878 median 3.60; scoped to the sweep it gives **n=37,605 median 3.61** —
close here, but the closeness is luck, not a guarantee.

**How.** The lines carry timestamps. Filter on them:

```bash
strings /tmp/glm52_mix_base.log \
 | awk -F'[][]' '/accept len:/ {split($2,t," ");
     if (t[2]>="07:13:00" && t[2]<="09:46:00") print}'
```

Scoping to the pre-sweep window instead recovers the gate reading exactly:
`n=25 p10=2.48 MEDIAN=2.80 p90=3.08 at-4.00=0.0 %`. That the two windows give
clearly different distributions is itself the evidence that scoping matters.

**Also.** Use `strings` on these logs — they carry binary bytes, and a bare
`grep` then reports only "binary file matches". And never poll for readiness by
grepping a log for a ready line: it will match the *previous* run's line and
return early. `wait_health` polls the HTTP endpoint instead.

---

## 5. Why ISL is 7,400 / 15,500 / 23,500 and not 74K / 155K / 235K

**What.** The sweep sends the **10 % fresh remainder** of Case A's prompts, not
the whole prompt.

**Why.** Case A runs at an **89–90 % prefix-cache hit rate**: of a 74K-token
prompt, ~66K is served from cache and only ~7.4K is actually computed. A fixlen
arm that sent the full 74K would measure prefill work the real workload never
does.

**How it lines up.** `--dataset-name random` builds every prompt independently,
so there is **no shared prefix by construction** — the sent length *is* the
computed length. That equality is what makes the substitution valid.

**Context.** This was an **explicit user decision**, taken before the run, not
an inference made during analysis. Do not "correct" it.

**Corollary, and a trap.** The `--cache-report` column on this dataset is
therefore **not** a workload hit rate — the observed 9.95–68.98 % is emergent
overlap between independently generated random prompts, plus whatever the radix
tree retains between requests. It is reported to confirm the mechanism is wired
(and it does confirm `InferaKvdBackend` served tokens on two of the p50 points),
not to be read as the agentic workload's 89–90 %.

---

## 6. Sampling: never greedy-decode this model

**What.** `--temperature 1.0 --top-p 0.95` — the checkpoint's own
`generation_config`.

**Why.** At temperature 0 this reasoning model repeats on a long prompt. The MTP
draft model predicts the repetition **perfectly**, acceptance pins at 4.00, and
the whole run reads like KV corruption. A steady accept-len of 4.00 is a failure
signal, not a win. The healthy band is 2–3 at low load; we measured median 2.80
on the idle gate and 3.61 under the full sweep, with 9.0 % of batches at 4.00.

**Context.** This also means results are **not bit-reproducible**. Compare
distributions, not individual generations.

---

## 7. kvd is legal on a mix worker — this is by design, not an accident

**What.** kvd (8 adapters, one per DP rank) ran throughout, on a worker whose
`disaggregation_mode` is `null`.

**Why it's legal.** `kvd_wiring._skip_kvd_on_decode_leg` skips kvd only when
`disaggregation_mode == "decode"`. A mix worker keeps it. SGLang's storage
prefetch likewise runs on the aggregated branch of
`Scheduler._add_request_to_queue`.

**Context / trap.** `--max-bytes` and `--long-bytes` are **absolute**. The
ratio-based default sizes off `max_total_num_tokens` and can ask for hundreds of
GB per rank; L3 writes to a container-local path, so an oversized budget fills
the node's root filesystem. We pinned 64G / 64G.

Post-sweep counters: 70,676 entries, 84.8 GB host, 68.7 GB L3, gets 64,741,
sets 266,289, **hits 62,697 / misses 2,044**, evictions 151,525. Reported as
counters — no claim is made here about how much of the measured latency they
account for.

---

## 8. Open question, deliberately left open: TTFT non-monotonicity

**What was measured.** On the p90 and p99 arms, TTFT p50 *rises* from conc 1 to
conc 8, then *falls* at 16 and again at 24:

| arm | c1 | c8 | c16 | c24 |
|---|---|---|---|---|
| p90 | 1328 | 4523 | **3208** | **2691** |
| p99 | 1440 | 5368 | **4045** | **3522** |

The p50 arm shows a weaker version of the same shape (2012 → 1978 → 2276).

**What is NOT claimed.** No mechanism. Nothing was measured that would identify
one, and a fluent-sounding story here would be worse than silence — it would
discourage running the experiment that actually settles it.

**What would settle it.** Two measurements, both cheap:

1. **Per-request TTFT correlated with that request's `cached_tokens`.** Both
   arrays are already in the raw jsonl (`ttfts[]` and `cached_tokens[]`,
   index-aligned, same length as `completed`). This separates "waited longer in
   the queue" from "had less to prefill". Note the sweep-level hit rate is
   *itself* non-monotonic over the same points (p90: 52.5 / 20.9 / 39.3 /
   43.1 %), so the two must be looked at together.
2. **Scheduler queue depth from the engine log**, scoped to each run's window.
   The `Prefill batch` / `Decode batch` lines carry `#running-req` and token
   usage; a queue that is *shorter* at higher concurrency would point somewhere
   quite different from one that is longer.

Both are re-derivable from artefacts already in this packup — except that the
per-request arrays were trimmed from the shipped jsonl for size; see
`results/RESULTS.md` for where the full files live.

---

## 9. Smaller things worth knowing

- **Cold start is 390 s.** Weights (~400 GB) + aiter/tilelang JIT + CUDA-graph
  capture. Silence is not a hang.
- **Use the data-plane IP** (`10.2.x` on `enp193s0f1np1`), not the `45.76.x`
  management address.
- **Benchmark through the router** (`:8100`), not the engine (`:30000`) —
  otherwise kv-aware routing is bypassed and the deployment under test is not
  the one you configured.
- **`bench_serving` appends** to `--output-file`. A re-run of the same arm
  leaves two json objects in the file; `summarize_fixlen.py` deliberately takes
  the last.
- **`--ep-size` is independent of `--enable-dp-attention`.** It is emitted
  outside the DPA branch on purpose: expert parallelism and attention
  parallelism are different axes, and moving it inside would silently collapse
  the MoE in any DPA-off arm, making the comparison two-variable.
- **`--disable-custom-all-reduce` is set independently of MTP.** aiter's custom
  all-reduce deadlocks under EAGLE verify; letting the flag follow MTP would
  make any MTP A/B two-variable.
- **`--reasoning-parser glm45` means reasoning is billed against the same token
  budget as content.** In the smoke test, a small `max_tokens` returns an empty
  `content` on a perfectly healthy deployment. Do not read that as a failure.
- **The engine log is ~11 MB raw / 650 KB gzipped** and lives inside the
  container; `/tmp` is not on the shared mount.
