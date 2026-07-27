# 04 — Single-node MTP / speculative decoding (GLM-5.2-MXFP4, sglang)

**Ran:** 2026-07-27 · **Status:** ✅ PASS (needs 1 code patch + 1 env, both discovered here)

## Goal

Enable MTP (speculative decoding via GLM-5.2's own nextn head, EAGLE) on the single-node engine
and confirm it produces correct output AND actually accepts draft tokens (real speedup).

## Result

| Metric | Actual | Verdict |
|--------|--------|---------|
| Correctness (temp=0 probe) | 4/4 | ✅ |
| Spec-dec active (accept len / rate) | 3.52–4.83 of 6 / 0.51–0.77 | ✅ |
| Single-stream decode throughput | ~219 tok/s (vs ~80 no-MTP = **~2.7×**) | ✅ |

Config: EAGLE, `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`,
mem-fraction 0.80, on chi2879.

## The two fixes this experiment required (rc6-specific)

1. **1-line code patch** to `deepseek_nextn.py` — see `patches/deepseek_nextn.rc6.diff`.
   Without it: `RuntimeError: size of tensor a (3072) must match tensor b (6144)` at draft load.
2. **Env `SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0`** — without it, decode hangs on a
   gfx950-incompatible CUDA JIT kernel (`fused_metadata_copy`).

Full root-cause in `notes.md`.

## How to reproduce

See `REPRODUCE.md`. TL;DR: launch the rc6 image with the base DSA envs **+ the precompute env**,
mount the patched `deepseek_nextn.py`, add the EAGLE flags; probe + watch accept-len in decode log.

## Folder map
- `REPRODUCE.md` — step-by-step
- `scripts/launch.sh` — container + server launch (mounts the patch, sets the env, EAGLE flags)
- `scripts/probe.py` — temp=0 correctness probe
- `patches/deepseek_nextn.rc6.diff` — the 1-line fix (unified diff vs rc6 stock)
- `patches/deepseek_nextn.rc6patch.py` — the full patched file (what launch.sh mounts)
- `results/single_stream.txt` — accept-len + throughput
- `notes.md` — the 3-bug debugging story + why jiejing's patch does NOT work on rc6
