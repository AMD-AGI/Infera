# Notes — gotchas, method, and what was deliberately left open

The most re-read file. Ordered by how likely it is to cost someone time.

## 1. Proving "real acceptance" by absence, not by presence

`config.sh` reads `${DECODE_SIMULATE_ACC_LEN-3.61}` — a **single** dash. An
empty-but-set value therefore survives and disables simulation; `:-` would treat
empty as unset and silently restore `3.61`. So `config.yihou.fast.sh` sets
`: "${DECODE_SIMULATE_ACC_LEN=}"` and the assertion that this run used real
acceptance is that **`SGLANG_SIMULATE_ACC_LEN` appears nowhere in the launcher's
emitted `docker run` line**. Compare `spec/working_process.md`'s T1 section
against the T2 launch, where the variable *is* present with value `3.61`.

This distinction is the whole difference between a correctness measurement and a
timing measurement, and it hinges on one character in a parameter expansion.

## 2. The 40.95 % output shortfall — method first, because the method is the trap

**How it was computed.** Per request, from
`results/agentx/profile_export.jsonl.gz`, using each record's own
`metrics.osl_mismatch_diff_pct.value` to reconstruct what that request asked for:

```
requested = actual / (1 + diff_pct/100)      when diff_pct != 0
requested = actual                           when diff_pct == 0
```

Note the schema: every metric is a `{"value": …, "unit": …}` object, not a bare
number. A first pass that read them as floats silently matched nothing and
produced a division by zero — a loud failure, fortunately, rather than a wrong
answer.

**How it must NOT be computed.** From the aggregate JSON's
`request_metrics.tokens.output_expected` mean (1708.02 here) against
`output_actual` mean (509.11). Those are differently-defined fields; the C72
pack-up records a correction where exactly that substitution inflated the deficit
by more than 10×. The per-request ground truth is `osl_mismatch_diff_pct`, and it
is **exactly 0.0 for 82.8 % of requests**, including very short ones — a request
that produced 1 token has `diff_pct = 0.0` because 1 token is what the trace
asked for.

**The result.**

```
profiling records                      1067
diff_pct == 0.0 exactly                884  (82.8 %)
negative (short)                       183  (17.15 %)   worst -96.2 %
positive (over-long)                   0
sum(requested) 919,893   sum(actual) 543,225
deficit 376,668 tokens = 40.95 %
top-10 carry 30.0 %   top-50 carry 72.5 %
```

**What it invalidates.** `output 441.9 tok/s`, `per-GPU output 55.2 tok/s` and
`e2el` describe a run that produced well under half the requested output. They
are not comparable to a run that fulfilled the trace. Input throughput, TTFT and
prefix-cache figures are prefill-side and unaffected. AgentX does not catch this:
`--failed-request-threshold 0.10` counts *failed* requests, and an OSL mismatch
is a warning, not a failure — this run scored `0/1067` errors while dropping 41 %
of the output.

## 3. Why the shortfall is left undetermined

Three explanations were checked against the records and excluded:

| candidate | evidence against |
|---|---|
| client cancellation | `was_cancelled: False` on all 183 short requests |
| context overflow skip | `context_overflow_skip: False` on all 183 |
| a hard output cap | the exact-match group contains a 13,727-token response |

The short requests are the **long** ones — median actual OSL 744 versus 239 in
the exact-match group — and they skew to later turns (`turn_index` 6-9) while the
exact-match group skews to early turns (0-4).

Two explanations remain, and this pack-up cannot choose between them:

- **Benign.** This is the first run in the series where the model emits its own
  EOS rather than running under a forced acceptance length. GLM-5.2 may simply be
  more concise than whatever model produced the recorded trace, in which case the
  "deficit" is a workload-alignment artifact.
- **Not benign.** Something in the corrected MTP path terminates long generations
  early.

The controlled comparison that separates them is the same trace with
`DECODE_MTP=0`. It was **not run** — the user scoped it out after being shown the
finding. Recorded here so the next person does not have to rediscover the
question, and so nobody mistakes the absence of a conclusion for the absence of a
problem.

## 4. The idle-gauge trap, kept visible on purpose

`results/accept/metrics-before.txt` shows `spec_accept_length 0.0` and
`spec_accept_rate 0.0` on all four decode ranks. That is **not** a measurement.
An idle rank reports 0.0, which is indistinguishable from "accepts nothing" —
and 0.05 was the actual reading of the *broken* configuration, so the two are
easy to confuse. The measurement is `metrics-after.txt`, taken while the
profiling phase was live, and only ranks whose gauge moved are counted. Here all
four moved, so the sample is 4/4 rather than a subset.

## 5. Preflight was run, and ordering it was a mistake

Recorded because the reasoning is reusable, not because it produced a result.

Preflight was ordered out of habit. It should not have been: R05 had run at
~09:57 **the same morning** on these exact nodes, the same GPUs 2-5, and the same
pinned `RDMA_DEVICE` JSON, with 0 Mooncake failures and 0 router-affinity 503s.
The only change this round was a thin image layer over three source files. A
probe re-validation could not beat a 1.5-hour-old end-to-end result from the real
engine. It cost ~45 min and produced two "failures" that were both artifacts of
the probe's own invocation.

What it did produce, and what is worth keeping:

- **`preflight.sh` sources no config file.** It takes only `IMAGE=` plus node
  names, and mounts the host GPUDirect provider solely when the caller has
  exported `HOST_RDMA_LIB` (`preflight.sh:62-63`). `config.sh:104`'s default
  never reaches it, so the container keeps its stock `libionic` and every per-GPU
  transfer fails with an **empty** `"dev"` — a different signature from the
  cross-rail mismatch, where a device *is* chosen and is simply the wrong one.
- **Bare `preflight.sh` uses auto HCA selection only**, which on these two nodes
  reproducibly fails GPUs 4-7 in both directions while 0-3 pass ~45 GB/s. That is
  the isolated-rail artifact from the C32/C40 pack-up §1, not a fabric fault. A
  pinned probe on today's image passed all four run GPUs both directions:
  gpu2 29.9/28.9, gpu3 27.9/30.0, gpu4 40.4/41.3, gpu5 40.3/41.2 GB/s.
- On 135, `ionic_7`'s non-zero GID in `ibv_devinfo` is **stale**; sysfs
  (all-zero, no netdev) is ground truth. Sharper than the previous pack-up's
  wording, which relied on `ibv_devinfo`.

The leader's brief to the teammate was itself defective — it instructed passing
`CONFIG=` to `preflight.sh`, which has no such parameter. The teammate read the
script first and worked around it.

## 6. "fast mode" is a real flag, and `config.sh` already equals it

`AIPERF_EXPERIMENTAL_FAST=1` (InferenceX `benchmarks/benchmark_lib.sh:1979`) does
exactly two things: `duration=1200` and `warmup_requests_per_lane=1`.
`config.sh:121,123` already set both, and `agentx_env.py` writes them into
`runtime.env`. So running `config.sh` **is** fast mode. The variable is not
plumbed through our `agentx_bench.sh` and setting it would change nothing.
Checked rather than assumed, because "fast mode" reads like something that might
also swap the trace or the scenario — it does not.

## 7. `SGLANG_OPT_USE_TOPK_V2=false` was already the default

Requested as an added optimisation; it is the committed default in
`config.sh:93` and `config.full.sh:101`, passed through at `engine.sh:123`.
Nothing was changed.

**Where to check it, and a correction.** An earlier draft of this pack-up said
the value is asserted in `results/server-info/decode-0.json`. **It is not** —
`server-info` dumps `server_args`, i.e. CLI flags, and this is an environment
variable, so it does not appear there at all. The record that does hold it is
`results/launch-command-lines.txt`, the launcher's emitted `docker run` line,
where `SGLANG_OPT_USE_TOPK_V2=false` appears twice, once per leg. That file was
added to this pack-up because the cold-read pass caught the wrong citation.

The same file is where "real acceptance" is proven: `SGLANG_SIMULATE_ACC_LEN`
occurs **zero** times in it.

## 8. The image tag does not identify one blob

`infera-sglang:v0519-yihou-0917-nextnfix-hicache` is `fd7220a57b7d` on 135 and
`972d8fd952e9` on 138. Both are correct: each is a local layer built on that
node's own locally-built base, and neither was ever pushed, so there is no shared
digest. Reproduction rests on `patches/Dockerfile.yihou.hicache` plus the base
tag — not on the id. Anyone diffing image ids across the two nodes and concluding
the run was inconsistent would be wrong.

## 9. PR #37152 is in this image and does nothing here

Three source hunks — `hicache.cuh`, `kernels/ops/kvcache/hicache.py`,
`srt/mem_cache/pool_host/mha.py`. All plain source under `python/sglang/`, and
the `.cuh` is JIT-compiled at runtime against a content-addressed cache, which is
why a thin layer suffices instead of a 40-60 min full rebuild. But `config.sh`
has `PREFILL_HICACHE=0`, so no HiCache copy kernel runs in this benchmark. It is
carried only so that this run and the T2 sweep differ in configuration rather
than in binary. **Its effect is unmeasured here.** Also note the PR is open
upstream, never merged, and upstream's own AMD ROCm CI on it is red; it was
applied because the user asked for it.

## 10. `stop.sh` always says it failed

`cleanup completed with errors` is printed on clean teardowns too. Verify
instead: no `glm52-pd-yihou-*` containers on either node, and `rocm-smi`
reporting `No KFD PIDs currently running`. Both were checked here.
