# 03 — PD disaggregation over mooncake (GLM-5.2-MXFP4, sglang)

**Ran:** 2026-07-27 · **Status:** ⚠️ PARTIAL — correct output over TCP, **conc=64 FAILS**.

## Goal

Run GLM-5.2-MXFP4 PD-disaggregated across 2 nodes with KV transferred over **mooncake**,
coherent output, conc=64 pass.

**Success criteria:** correct output + conc=64 all-successful.

## Result

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Correctness (temp=0, TCP transport) | coherent | 4/4 | ✅ |
| conc=64 (1k/1k, 256 prompts) | all succeed | **50/256** (206 dropped) | ❌ |
| Median TTFT | — | 50.7 s (!) | ❌ |
| Total throughput | — | 886 tok/s | — |

**Bottom line:** mooncake gives *correct output* but only over **TCP** (`MC_FORCE_TCP=1`), and TCP
cannot sustain conc=64 — the KV-transfer sessions drop under load. The intended **RDMA** path is a
known driver dead-end on this stack (see `notes.md`). This kit is kept as a **documented failure**
so the boundary is reproducible, not to claim a pass.

Topology attempted: prefill = chi2878 (10.2.122.3), decode = chi2879 (10.2.122.10), TP8 each.

## How to reproduce

See `REPRODUCE.md`. Same stack as 02_pd_mori but `--disaggregation-transfer-backend mooncake` +
`MC_FORCE_TCP=1` (and the other MC_* envs). The correctness probe passes; the conc=64 bench fails
with `KVTransferError: remote mooncake session ... is not alive`.

## Folder map
- `REPRODUCE.md` — step-by-step (both the TCP path that gives correct output, and how to see the failure)
- `scripts/engine.sh` — PD leg launcher (MODE=tcp|rdma)
- `scripts/up.sh`, `scripts/pd_env.sh`, `scripts/probe.py`
- `results/bench_conc64_FAIL.txt` — the failure numbers
- `logs/prefill.log`, `logs/decode.log` — full logs incl. the KVTransferError flood + TcpTransport lines
- `notes.md` — **the honest failure analysis**: TCP conc wall + the RDMA driver dead-end (and what we did/didn't test)
