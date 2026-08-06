# GLM-5.2 1P1D deployment example — experiment pack-up

**Date:** 2026-08-04 (all work in a single day, 06:00–09:30 UTC)
**Repo:** `infera` · branch `yihou.dev.glm52.example` · HEAD `ff0f7ee`
**Deliverable produced:** `examples/sglang_1p1d_glm5.2/` (14 files, 3 commits)
**PR:** #84 (body in [`spec/pr84_body.md`](spec/pr84_body.md))

## What this experiment was

Turn the GLM-5.2-MXFP4 deployment experience that lived in six sibling experiment
packups into **one deliverable example** in the `infera` repo: SGLang **1P1D**
PD-disaggregation, KV over **Mooncake RDMA**, infera Rust router with **kv-aware**
policy, **DP-attention** on the decode leg, **MTP** (EAGLE) speculative decoding,
and the **kvd** cache tiers — packaged as `preflight → up → smoke` with a
`results/` folder carrying real conc=8 numbers.

The engineering problem was **not** "make it run" — that was already done across
six prior packups. It was **refactor six per-cluster scripts into one kit without
silently changing what they do**, then prove that. Hence the two things this
pack-up is mostly about:

1. a **dry-run equivalence harness** that stubs `docker` and `ssh`, records every
   engine argv and every remote command, and lets two kit revisions be diffed for
   behavioural identity;
2. a **real-machine validation run** of the finished kit — brought up by the kit's
   own wrapper with nothing edited outside it, then driven by the customer's
   unmodified benchmark.

## Result

**The kit works, as written, on real hardware.** Brought up by
`cluster/cluster.dmabuf.sh up` on a 2-node MI355X mode-B pair; all five features
show positive evidence in `smoke`; needle-in-a-haystack **12/12** to 238K tokens;
the customer's AgentX Case-A replay at c8 reproduced the previously recorded
numbers (TTFT p50 6,715 ms vs 6,698 ms recorded; ITL p50 14.0 vs 13.26 ms; cache
hit 88.1 % vs 88.1 %; **0 errors / 0 cancelled / 0 context overflows**).

**Eight defects were found and fixed** before shipping — two behavioural, one
information leak, five found only by running on a second cluster. All eight are
the kind that produce a *plausible result* rather than an error; see
[`notes.md`](notes.md), which is the most re-read file here.

**Not validated** (and flagged as such in the shipped README): the peer-mem
wrapper `cluster.peermem.sh`, `preflight_rdma.sh fabric`, `round-robin` routing,
and any concurrency other than 8.

## Navigation

| path | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **the reproduction kit** — ordered, copy-pasteable |
| [`notes.md`](notes.md) | the eight defects, each as what / why / how / context. Read this |
| [`environment.md`](environment.md) | hardware, fabric, image, model paths, versions |
| `spec/` | what was asked for: mission spec, plan, task CLAUDE.md, review rules, PR body |
| `kit/sglang_1p1d_glm5.2/` | the shipped deliverable, verbatim at `ff0f7ee` |
| `kit/revisions/` | the four intermediate kit snapshots + the diffs between them |
| `scripts/verify_harness/` | the dry-run equivalence harness (`dryrun.sh`, `cmp.sh`, `bin/`) |
| `scripts/collect_env.sh` | the env snapshotter that produced `logs/env_capture/` |
| `logs/dryrun_traces/` | T0–T3 traces — the equivalence evidence |
| `logs/real_run/` | the real-machine `up` / `smoke` / `needle` logs, **verbatim** |
| `logs/env_capture/` | per-leg host/GPU/driver/image/argv snapshots |
| `results/` | the three results documents + the raw aiperf export of the kit run |

> **`logs/` and `logs/env_capture/` are un-sanitised on purpose.** They carry real
> node names (`crsuse2-m2m-*`), internal IPs (`10.245.*`), `/shared_nfs` paths and
> the experiment image tag `infera/engine-sglang:final-pr`. This pack-up is an
> internal reproduction kit. The **shipped** deliverable under
> `kit/sglang_1p1d_glm5.2/` contains none of it — that separation was itself one of
> the review passes (T2→T3, see `notes.md` §D6).

## The four kit revisions

Every revision was dry-run against all six scenarios before moving on.

| rev | what changed | traces vs previous | maps to |
|---|---|---|---|
| **T0** | first refactor of six per-cluster scripts into the kit shape | — | — |
| **T1** | two behavioural bugs fixed (`PREFILL_MTP` forwarding, bench `N`) | **differ** — intended | — |
| **T2** | comment blocks compressed to the ≤3-line review rule; rationale moved to README | **byte-identical** | — |
| **T3** | local information stripped from two gotchas | **byte-identical** | `e2d462a` (0 diff) |
| **shipped** | five bugs found on a second cluster + the load-model correction | not re-run (needs the second cluster) | `ff0f7ee` |

That T1→T2 and T2→T3 leave the traces **byte-identical** is the point of the
harness: a 483-line comment rewrite and a documentation edit are proven to have
changed no behaviour, rather than asserted to have.
