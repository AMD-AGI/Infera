# 01 — Single-node mix (GLM-5.2-MXFP4, sglang)

**Ran:** 2026-07-27 · **Status:** ✅ PASS

## Goal

Run GLM-5.2-MXFP4 on one node (8× MI355X, TP8, colocated prefill+decode) on sglang, get
coherent output, and pass a conc=64 stress test.

**Success criteria:** correct (coherent) output + conc=64 completes with all requests successful.

## Result

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Correctness (temp=0 probe) | coherent | 4/4 (Paris/Beijing/4/Jupiter) | ✅ |
| conc=64 (1k/1k, 256 prompts) | all succeed | 256/256, 0 fail | ✅ |
| Total throughput | — | 4621 tok/s | — |
| Median TPOT / TTFT | — | 22.1 ms / 2.25 s | — |

No code fix needed — the base DSA-ROCm env recipe is sufficient.

## How to reproduce

See `REPRODUCE.md`. TL;DR: `docker run` the rc6 image with the DSA-ROCm envs + `sglang.launch_server`
on chi2879, wait ~5 min cold start, run `probe.py` then `bench.sh`.

## Folder map
- `REPRODUCE.md` — step-by-step
- `scripts/launch.sh` — the container + server launch (verbatim)
- `scripts/probe.py` — temp=0 correctness probe
- `scripts/bench.sh` — conc=64 1k/1k bench
- `results/bench_conc64.txt` — the numbers
- `notes.md` — gotchas
- (no logs/ — single-node ran to `docker logs`, not captured to file; see notes)
