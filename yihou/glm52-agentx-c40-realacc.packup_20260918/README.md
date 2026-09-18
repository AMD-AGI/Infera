# GLM-5.2 1P1D AgentX C40 on the corrected MTP config — real acceptance — 2026-09-18

One AgentX `CONC=40` fast-mode point against a 1-Prefill / 1-Decode GLM-5.2
MXFP4 deployment, run with **simulated MTP acceptance OFF**. This is the first
benchmark in this series whose decode output is actually correct, so the
acceptance number here is a measurement rather than a stand-in.

Run on **crsuse2-m2m-135** (prefill) and **crsuse2-m2m-138** (decode),
MI355X ×4 each (devices 2,3,4,5), 2026-09-18 11:04–11:57 UTC.

## Headline — the thing this run was for

| | value |
|---|---:|
| **`spec_accept_length`, min over 4 active ranks** | **2.1500** |
| `spec_accept_length` per rank (dp0/1/2/3) | 2.1500 / 2.3305 / 3.0750 / 2.7917 |
| mean over active ranks | 2.5868 |
| bar set by the user | ≥ 2.0 — **PASS** |
| decode-log second source | `accept len: 2.33 / 4.50 / 2.46 / 3.29 / 2.77` |

Before the fix, this same shape read `accept len 1.25 / accept rate 0.05` **and**
emitted garbled text. The fix is `--disable-custom-all-reduce` on the decode leg;
root cause and the five-round A/B behind it are in the sibling pack-up
`glm52-mtp-garbled-decode-rootcause.packup_20260918`.

`accept_length` and `accept_rate` are **different metrics**. Everything above is
length. The rates, for completeness, were 0.23 / 0.27 / 0.42 / 0.36.

## Correctness, verified before the benchmark

Both probes, `temperature=0`, real acceptance:

```
"What is 2+2? Answer with a single digit."
  -> '1.  **Analyze the Request:** … 2 + 2 = 4 …'      coherent
"Reply with exactly: BASELINE_OK_YIHOU111742"
  -> 'BASELINE_OK_YIHOU111742'                          exact
```

The NextN shared-experts-fusion fix is live in this image and confirmed from the
decode log: `Shared experts fusion optimization enabled` appears **2×** and
`Config does not support fused shared expert(s)` **0×**.

## AgentX C40 result

| | value |
|---|---:|
| requests total / profiled / warmup-dropped / **error-dropped** | 1151 / 1067 / 84 / **0** |
| aiperf request error rate | **0/1067 = 0.000 %** (gate < 10 %) |
| profiling window | 1229.3 s |
| throughput input / output / total | 96,817 / 441.9 / 97,259 tok/s |
| **per chip (8 GPU)** | **12,157 tok/s/chip** (output 55.2) |
| TTFT mean / p50 / p90 / p95 | 23.97 / **8.73** / 56.20 / 101.84 s |
| E2E mean / p50 / p90 | 31.25 / **16.45** / 74.04 s |
| ITL / TPOT mean (p95) | **16.4 ms** (23.3 ms) |
| interactivity p50 | 61.3 tok/s/user |
| prefix cache (theoretical / server) | 96.2 % / 90.0 % |
| router affinity 503s / Mooncake failures | **0 / 0** (`failures/` empty) |

## Read these limitations before quoting any number

**1. 40.95 % of the requested output tokens were never produced.** Computed
per request from `profile_export.jsonl`, not from aggregate means:

```
osl_mismatch_diff_pct == 0.0 exactly   884/1067  (82.8 %)
negative (short) requests              183       (17.15 %),  worst -96.2 %
sum(requested) 919,893  sum(actual) 543,225  ->  deficit 40.95 %
top-10 requests carry 30.0 % of it; top-50 carry 72.5 %
```

This is an order of magnitude worse than the C72 run's 3.11 % **and it is not
tail-driven**. So `output 441.9 tok/s`, `per-GPU output 55.2` and `e2el`
describe a run that produced well under half the output the trace asked for and
are **not comparable** to a run that fulfilled it. Input throughput and TTFT are
prefill-side and unaffected. Detail and method in `results/t1-osl-analysis.txt`
and `notes.md` §2.

**2. The cause of that shortfall is undetermined, deliberately.** Three
candidates were checked and excluded — client cancellation (`was_cancelled:
False` on all 183), context overflow (`context_overflow_skip: False` on all 183),
and a hard output cap (the exact-match group reaches 13,727 tokens). The short
requests are the *long* ones: median actual OSL 744 against 239 in the
exact-match group. The leading explanation is that this is the first run where
the model emits its own EOS, and GLM-5.2 is simply more concise than whatever
produced the recorded trace — but **the competing explanation, that something in
the corrected MTP path terminates long generations early, is not ruled out.**
The controlled comparison that would separate them (same trace,
`DECODE_MTP=0`) was **not run** — the user scoped it out. Do not resolve this
from the contents of this pack-up; it does not contain the evidence.

**3. HiCache PR #37152 is in the image but inert in this run.** `config.sh` sets
`PREFILL_HICACHE=0`, so no HiCache copy kernel executes. The patch is carried so
that this run and the T2 sweep differ in benchmark configuration and not in the
binary. Its effect is measured in the T2 pack-up, not here.

**4. `SGLANG_OPT_USE_TOPK_V2=false` was not a change.** It is already the
committed default in `config.sh:93`. It is asserted in
`results/server-info/`, not set by this run.

**5. P4D4 is not "half of P8D8."** TP4 and TP8 shard differently, so comparison
against the TP8/DP8 reference curve is indicative only.

## Navigation

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable reproduction |
| `notes.md` | the OSL analysis method, wrong turns, what was not established — **the most re-read file** |
| `environment.md` | hardware, fabric, image ids, git SHA |
| `scripts/` | all configs, topology, the acceptance collector, the sweep driver, env collector |
| `patches/` | the two image patches and the `engine.sh` escape hatch, each with what/why/how |
| `results/` | AgentX aggregate + raw JSONL, acceptance snapshots, OSL analysis, server-info |
| `spec/` | mission, full `working_process.md`, poll log, preflight and image-build reports |
| `env/` | raw `collect_env.sh` per node |
| `logs/` | both engine logs, gzipped |

## Provenance

- Repo `AMD-AGI/Infera`, branch `dev/pd_opt/glm_5.2_agentx`, HEAD
  `ad3b85d3838513010d7364309da3fcfebbf76518`, plus the uncommitted `engine.sh`
  change in `patches/`.
- Run image `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, built locally on
  each node: **135 `fd7220a57b7d`, 138 `972d8fd952e9`**. The ids differ by
  design — each is a local layer over that node's own locally-built base, never
  pushed. **The tag does not identify one blob across the two nodes**;
  reproduction rests on `patches/Dockerfile.yihou.hicache` plus the base tag.
- Base `infera-sglang:v0519-yihou-0917` (id `4190c3a99d0e` on both), itself from
  `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`, digest
  `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32`.
- In-image sglang `0.5.19.dev20260917+ga9fb1c3238`.
- Original working directory, untouched and still on disk:
  `bench/glm5p2_pd/results/yihou-agentx-hicache/` (gitignored).
