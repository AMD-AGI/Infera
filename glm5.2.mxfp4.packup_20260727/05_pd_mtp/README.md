# 05 — PD disaggregated MTP (GLM-5.2-MXFP4, sglang, MoRI RDMA + decode spec-dec)

**Ran:** 2026-07-27 · **Status:** ✅ PASS

## Goal

Combine PD disaggregation (MoRI RDMA, from 02) with MTP speculative decoding (from 04), putting the
spec-dec **on the decode leg only**, and pass conc=64 — the highest-throughput config in this kit.

## Result

| Metric | Actual | vs 02 (PD, no MTP) | Verdict |
|--------|--------|--------------------|---------|
| Correctness (temp=0 via router) | 4/4 | — | ✅ |
| conc=64 (1k/1k, 256 prompts) | 256/256, 0 fail | same | ✅ |
| Total throughput | **7444 tok/s** | +44% (5167) | ✅ |
| Median TPOT | **12.1 ms** | 1.7× faster (20.9) | ✅ |
| Median TTFT | 412 ms | lower (535) | ✅ |
| Spec-dec on decode (accept len / rate) | 2.75–2.88 / 0.58–0.62 | — | ✅ |

Topology: prefill = chi2832 (10.2.122.79, **no MTP**), decode = chi2879 (10.2.122.10, **EAGLE 3-draft**),
TP8 each, MoRI RDMA over 8 ionic NICs.

## Design decisions

- **MTP on decode leg only.** Spec-dec accelerates token *generation*, which happens on decode.
  Adding it to prefill just loads a draft model + slows the prefill forward pass (throttles the
  pipeline). Prefill leg runs plain.
- **EAGLE 3-draft (not 5) + reserved-tokens 256 + decode mem-frac 0.80.** The reference program
  found MTP-on-PD-decode OOMs the decode KV pool at high concurrency with 5 draft steps; steps=3 +
  reserved 256 + mem-frac 0.80 is stable through conc=64 (no crash observed).
- Same two rc6 MTP fixes as 04 (nextn patch + precompute env) — they apply unchanged in PD.

## How to reproduce

See `REPRODUCE.md`. TL;DR: 02's mori PD stack, but the decode leg gets `MTP=1` (mounts the nextn
patch, sets the precompute env, adds EAGLE flags). Prefill leg unchanged.

## Folder map
- `REPRODUCE.md` — step-by-step
- `scripts/engine.sh` — PD leg launcher with optional decode-leg MTP (MTP=1)
- `scripts/up.sh` — orchestrator (prefill=chi2832 no-MTP, decode=chi2879 MTP)
- `scripts/probe.py`
- `patches/deepseek_nextn.rc6.diff` + `.rc6patch.py` — the 1-line MTP fix (same as 04)
- `results/bench_conc64.txt` — the numbers
- `logs/prefill.log`, `logs/decode.log` — full logs (decode log has the accept-len batch stats)
- `notes.md` — decode-leg MTP tuning + why prefill has no MTP + node-swap story
