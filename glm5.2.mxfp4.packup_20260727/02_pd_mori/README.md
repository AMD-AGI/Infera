# 02 — PD disaggregation over MoRI RDMA (GLM-5.2-MXFP4, sglang)

**Ran:** 2026-07-27 · **Status:** ✅ PASS

## Goal

Run GLM-5.2-MXFP4 with prefill/decode **disaggregated across 2 nodes**, KV transferred over
**MoRI-IO RDMA** (ionic fabric), coherent output, conc=64 pass.

**Success criteria:** correct output through the PD pair + conc=64 all-successful.

## Result

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Correctness (temp=0 via router) | coherent | 4/4 | ✅ |
| conc=64 (1k/1k, 256 prompts) | all succeed | 256/256, 0 fail | ✅ |
| Total throughput | — | 5167 tok/s | — |
| Median TTFT / TPOT | — | 535 ms / 20.9 ms | — |

Topology: prefill = chi2878 (10.2.122.3), decode = chi2879 (10.2.122.10), TP8 each, 8 ionic NICs.
GLM-5.2 NSA cross-node KV transfer over MoRI RDMA works correctly on this node pair.

## How to reproduce

See `REPRODUCE.md`. TL;DR: NATS on both nodes + etcd + `infera.server` router (http transport,
kv-aware) on prefill; launch prefill leg (chi2878) + decode leg (chi2879) via `engine.sh` with
`--disaggregation-transfer-backend mori` and all 8 ionic NICs; probe + bench through the router.

## Folder map
- `REPRODUCE.md` — step-by-step
- `scripts/up.sh` — orchestrator (NATS/etcd/router + both legs)
- `scripts/engine.sh` — one PD leg (prefill|decode) launcher
- `scripts/pd_env.sh` — node IPs / topology config
- `scripts/probe.py` — temp=0 correctness probe
- `results/bench_conc64.txt` — the numbers
- `logs/prefill.log`, `logs/decode.log` — full engine logs (bootstrap, transfer, warmup)
- `notes.md` — the two router-flag bugs that blocked this + GID facts
