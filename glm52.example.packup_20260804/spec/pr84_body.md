# Description

Adds a runnable deployment example for **GLM-5.2-MXFP4** under `examples/sglang_1p1d_glm5.2/`:
one prefill node and one decode node, KV moved between them over Mooncake RDMA, fronted by the
infera router, with DP-attention, MTP (EAGLE speculative decoding) and the kvd cache tiers on.

Everything site-specific lives in one of two files under `cluster/` — one per RDMA registration
mode, picked by the preflight probe rather than by autodetection at launch. `common.sh` and
`engine/*.sh` carry the tuned recipe and contain no addresses, paths, NIC names or GID indices,
so adapting to a cluster means editing a wrapper, never an engine script.

Directory shape mirrors `examples/deepseek_v4` (`common.sh` + an `engine/` subdirectory).

## Type of change

- [x] Documentation change (change only to the documentation, either a fix or a new content)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `cluster/` — two wrappers, the only files a user edits: `cluster.peermem.sh` (multi-rail
  with a peer-memory module) and `cluster.dmabuf.sh` (single ODP NIC, no peer-mem). No
  autodetection: `preflight_rdma.sh mode` tells you which one to use.
- `engine/leg.sh` — the launcher for one PD leg. Carries the tuned recipe and nothing
  site-specific.
- `engine/up.sh` / `down.sh` — bring up both nodes (containers → etcd + kvd → both legs →
  router) and tear down, waiting for VRAM to actually drain.
- `engine/smoke.sh` — service check **plus** positive evidence for each of the five features.
  A green `/health` proves the process is alive, not that PD paired or that speculation is
  doing anything; every check here goes red if its feature is silently absent.
- `engine/bench.sh` — reference throughput sweep using SGLang's own `bench_serving`.
- `preflight_rdma.sh` — registration-mode probe and cross-node fabric measurement.
- `results/` — concurrency-8 numbers from two independent agentic benchmarks on two clusters.

### Three couplings that fail quietly

Each of these produces a plausible result rather than an error when got wrong, so each is
commented at the point of use and written up in the README's "Notes & gotchas":

- **`--ep-size` is emitted unconditionally**, outside the DP-attention branch. Expert and
  attention parallelism are different axes; gating both on one condition collapses the MoE
  from ep8 to the TP default whenever DP-attention is off, so a run billed as "DPA off"
  differs in the expert-dispatch collective too and no latency delta is attributable.
- **`--chunked-prefill-size` is a global budget** that SGLang divides by `dp_size` only when
  DP-attention is on. One value serves both modes; hardcoding the per-rank number in a DPA-off
  branch cuts the global budget 8×.
- **Prefill activation OOM is fixed by *lowering* `--mem-fraction-static`** — the opposite of
  the decode-side retract fix. Low token usage at the moment of the abort is the tell: the KV
  pool is nearly empty, so it was never KV exhaustion.

### On `results/`

Numbers are recomputed from raw per-request records, not copied from summary lines. The
cross-cluster comparison is presented as **context, not a measurement**: more than one variable
moves between those clusters, four candidate causes are listed with the evidence behind each,
and none is confirmed. Where a benchmark's own summary metric is misleading it is called out —
the customer harness's `Theoretical Prefix Cache Hit` is computed from the trace file and never
asks the server, so it is invariant to the deployment under test.

The customer's benchmark is **referenced by URL, not vendored**; this kit ships no agentic bench
client, only the service self-check and the `bench_serving` reference sweep.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

**No tests.** This is a documentation/example change: shell scripts that drive a two-node GPU
deployment, with no unit-testable surface and no CI cluster to run them against. What was done
instead is recorded below.

## Validation status

Stated plainly rather than implied — the README carries the same table.

| what | status |
|---|---|
| the deployment shape (1P1D + mooncake + DPA + MTP + kvd + kv-aware) | **validated end-to-end on two clusters**, both fabric types |
| the tuned values (GMU, chunk, ctx, EAGLE settings, DSA env, router weights) | **validated** — each carried over from a run that completed cleanly |
| **these scripts as written** | **validated** — `preflight mode` → `up` → `smoke` → `bench` → `down` on a 2-node MI355X mode-B cluster, no edits outside the wrapper. Long context checked separately (needle, to 238K tokens) and under a real agentic workload at concurrency 8 |
| `preflight_rdma.sh fabric` | **not validated** — only `mode` was exercised |
| `cluster.peermem.sh`, `round-robin` routing | **not validated** — no peer-mem cluster was available; the shipped `kv-aware` default is what ran |

Static checks: `bash -n` on all 9 scripts; every infera flag grepped against `infera/**/args.py`
on `main`; the deliverable grepped for leaked local information (host names, absolute paths,
scheduler identifiers, credentials) before submission.

## Known placeholder

The image tag `inferaimage/infera-sglang:0.2.0` is a **placeholder**, flagged as such in the
README. It needs replacing with the released infera-sglang tag before this is useful to a
reader.
