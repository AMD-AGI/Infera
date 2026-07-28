# 03 — PD disaggregation over mooncake RDMA (GLM-5.2-MXFP4, sglang)

**Ran:** 2026-07-27 · **Status:** ✅ PASS (on the `pd-unified` image with PR #19's fix)

## Goal

Run GLM-5.2-MXFP4 PD-disaggregated across 2 nodes with KV transferred over **mooncake**,
coherent output, conc=64 pass.

**Success criteria:** correct output + conc=64 all-successful.

## Result

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Correctness (temp=0 via router) | coherent | 4/4 | ✅ |
| conc=64 (1k/1k, 256 prompts) | all succeed | 256/256, 0 transfer errors | ✅ |
| Transport | RDMA (not TCP/HIP) | mooncake RDMA (bare ibv_reg_mr+peermem) confirmed | ✅ |
| Total throughput | — | 5147 tok/s (≈ mori's 5167) | — |
| Median TTFT / TPOT | — | 543 ms / 20.9 ms | — |

Topology: prefill = chi2878 (10.2.122.3), decode = chi2879 (10.2.122.10), TP8 each, 8 ionic NICs.

## The fix that made mooncake RDMA work: the `pd-unified` image (PR #19)

A **prior attempt on the `rc6` image failed** — mooncake could only do correct output over TCP
(`MC_FORCE_TCP=1`), which collapsed at conc=64 (50/256, TTFT 51 s), because #2682's bundled mooncake
installs HIP transport unconditionally and prefers it, so a cross-node send picked intra-node HIP-IPC
(`hipIpcOpenMemHandle` fails cross-node). That failed attempt is preserved under
`tcp_fail_appendix/` for the record.

**PR #19** (`infera/engine-sglang:pd-unified`) rebuilds mooncake so HIP transport is **OFF by
default** (`MC_DISABLE_HIP_TRANSPORT=1`) → cross-node PD uses **RDMA**. dma-buf is compiled in but
OFF by default (ionic no-ODP 2× pin); the KV path is bare `ibv_reg_mr` + amdgpu peermem. On
chi2878/chi2879 this works cleanly. **No `MC_FORCE_TCP`.**

## How to reproduce

See `REPRODUCE.md`. TL;DR: use the `pd-unified` image, `pd_uni` container + libionic inject on both
nodes, launch both legs with `--disaggregation-transfer-backend mooncake` (dmabuf OFF, all-8-ionic),
`sglang_router` mini-LB, probe + bench.

## Folder map
- `REPRODUCE.md` — step-by-step (incl. how to get the image onto the 2nd node)
- `scripts/pd_leg.sh` — one PD leg launcher (mooncake, runtime dmabuf switch)
- `scripts/up.sh` — orchestrator (containers + libionic + both legs)
- `scripts/probe.py` — temp=0 correctness probe
- `results/bench_conc64.txt` — the numbers
- `logs/prefill.log` — the RDMA-confirmed prefill leg (+ `logs/README.txt` on the decode log caveat)
- `notes.md` — RDMA-vs-TCP story, the PR #19 mechanism, dmabuf boundary
- `tcp_fail_appendix/` — the earlier rc6-image TCP-only failure (50/256), kept for the record
