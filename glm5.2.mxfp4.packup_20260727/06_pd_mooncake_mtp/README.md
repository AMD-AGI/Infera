# 06 — PD disaggregated MTP over mooncake RDMA (GLM-5.2-MXFP4, sglang)

**Ran:** 2026-07-27 · **Status:** ✅ PASS

## Goal

Combine mooncake RDMA PD (03) with MTP speculative decoding on the decode leg, on the `pd-unified`
image, and pass conc=64.

## Result

| Metric | Actual | vs 03 (mooncake, no MTP) | Verdict |
|--------|--------|--------------------------|---------|
| Correctness (temp=0 via router) | 4/4 | — | ✅ |
| conc=64 (1k/1k, 256 prompts) | 256/256, 0 fail | same | ✅ |
| Total throughput | 5302 tok/s | +3% (5147) | ✅ |
| Median TPOT | 19.0 ms | faster (20.9) | ✅ |
| Spec-dec on decode (accept len / rate) | 2.65–2.90 / 0.55–0.63 | — | ✅ |

Topology: prefill = chi2878 (10.2.122.3, **no MTP**), decode = chi2879 (10.2.122.10, **EAGLE 3-draft**),
mooncake RDMA, dmabuf OFF.

## The one MTP fix needed on pd-unified (and the one NOT needed)

- **NEEDED — 1-line nextn patch** (`patches/deepseek_nextn.unified.diff`): pd-unified's
  `deepseek_nextn.py` (423 lines) has the same bare-prefix `eh_proj` quark-exclude bug as rc6 (but at
  a different line, 363). Without the patch → `RuntimeError: size of tensor a (3072) vs b (6144)` at
  draft load. **This patch is pd-unified-specific — do NOT reuse the rc6 patch (04/05); the file
  structure differs.**
- **NOT needed — the `SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0` env** (which 04/05 needed on rc6).
  pd-unified's `dsa_backend.py` try/excepts the gfx950-incompatible CUDA `fused_metadata_copy`
  (steps>3) AND uses a plain loop for steps≤3, so the kernel is never a hard failure. With EAGLE
  steps=3 it isn't even reached. See `notes.md`.

## How to reproduce

See `REPRODUCE.md`. TL;DR: the 03 mooncake RDMA stack, but the decode container mounts the
pd-unified nextn patch and the decode leg adds `MTP=1` (EAGLE steps=3). Prefill leg unchanged.

## Folder map
- `REPRODUCE.md` — step-by-step
- `scripts/pd_leg_mtp.sh` — PD leg launcher with optional decode-leg MTP
- `scripts/up.sh` — orchestrator (from 03; the MTP decode leg is launched separately per REPRODUCE)
- `scripts/probe.py`
- `patches/deepseek_nextn.unified.diff` + `.unified_patch.py` — the pd-unified 1-line MTP fix
- `results/bench_conc64.txt` — the numbers
- `logs/prefill.log` (no MTP) + `logs/decode.log` (MTP, has accept-len batch stats)
- `notes.md` — pd-unified vs rc6 MTP differences, KV-pool tuning
