# The merged deliverable

One engine image carrying all three workstreams: **kvaware + kvd**, **PD +
DP-attention + EAGLE MTP** (the DSA patch set), and the **mooncake early-send KV
fix**. Validated end to end on 2 × 8×MI355X over mooncake RDMA — see
`../working_process.md` for the run log and `../notes/` for the findings.

## What is here

```
deploy/docker/Dockerfile.sglang                     the single deliverable image
deploy/docker/patches/sglang_dsa/                   3 DSA diffs + apply script  (PR58)
deploy/docker/patches/sglang_disagg/                mooncake early-send patch   (PR56)
deploy/docker/scripts/apply_sglang_dsa_patches.sh
infera_source_changes.diff                          4 infera files + 3 test files
```

The Dockerfile builds all of it. `infera_source_changes.diff` is **not** applied by
the build — those are source commits; the image picks them up through the existing
`COPY infera ./infera`. Apply it to the repo, don't bolt it onto the image.

## Image layer order, and why it is that order

| # | layer | why it sits there |
|---|---|---|
| 1 | mooncake unified rebuild | HIP-transport gate + dma-buf compiled in, chosen at runtime |
| 2 | `pip install .[sglang]` | brings in the infera source, including the 4 changed files |
| 3 | `patches/sglang/` loop | GLM-5.2 nextn `eh_proj` quark-exclude |
| 4 | `patches/sglang_disagg/` | mooncake early-send wait event |
| 5 | `patches/sglang_dsa/` | the 3 DSA diffs — **must follow 3**, which is their prerequisite |
| 6 | Rust router | |

Layer 5's apply script **asserts** the layer-3 edit rather than making it: that
script is idempotent, so a silent "skipped" would surface only at runtime as
GLM-5.2 dying at draft weight-load with `3072 vs 6144`.

Layers 3 and 4 differ deliberately in strictness. Layer 3 tolerates a failed
patch (`|| echo skipped`) because those no-op once the base carries the fix.
Layer 4 does **not** (`set -eu`): its script already reports "already present"
on its own, so a non-zero exit means the anchors drifted and the fix did *not*
go in — and an image that silently corrupts long prompts is worse than a failed
build.

## The infera source changes

| file | change | why |
|---|---|---|
| `router/kv_event/client.py` | `_flat_tokens` | under MTP, SGLang reports a block's tokens as bigram pairs `(t[i], t[i+1])`; hashing the pairs builds a view no query can match, and kv-aware routing silently degrades to round-robin |
| `router/kv_event/events.py` | widen `token_ids` | the msgspec schema must accept pairs or every event fails to decode |
| `engine/sglang/args.py` | gate the decode-radix append | SGLang forbids `--disaggregation-decode-enable-radix-cache` under speculative decoding; appending it kills an MTP decode leg at argument parsing |
| `engine/sglang/kvd_wiring.py` | skip kvd on a PD decode leg | kvd is **write-only** there — measured 180 sets / 0 gets. See `../notes/decode_leg_kvd_is_write_only.md` |

The last two are independent — different flags, different gates, different SGLang
checks. Neither subsumes the other: a decode leg with kvaware on and kvd off still
needs the `args.py` change.

**Tests.** Three new modules, 14 tests. Each fix was reverted in place and its
module re-run to confirm it fails without the fix (`../scripts/revert_check_tests.sh`
automates this; all 3 fail behaviourally, not as collection errors). Full unit
suite after the changes: **1161 passed, 1 skipped**.

## Building

```bash
docker build -f deploy/docker/Dockerfile.sglang -t infera/engine-sglang:merged .
```

`APPLY_SGLANG_DSA_PATCHES=0` builds a stock engine for A/B. The base tag stays
**pinned**: the DSA diffs apply at `--fuzz=0` against sglang
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, so a base bump fails the build at the
patch step rather than mis-applying silently. That is intended.

## Not in this deliverable

Deferred from PR56 as non-blocking on MI355X — see `../working_process.md`:

- `dsv4_gfx942` architecture-based detection — `apply_gfx942_dsv4` returns early
  on non-gfx942, so it is a **no-op here**. Needed for MI325X.
- `INFERA_SGLANG_READY_TIMEOUT` — convenience; 1800 s has sufficed.
- Rust router bigram decode — every run uses `--router-backend python` (the
  default). **A Rust-router deployment with MTP still has the original bug.**

Also open: whether `--disaggregation-decode-enable-offload-kvcache` reads back
from L3, which would need an exception carved into the decode-leg kvd skip. We
never enable it and did not test it. See the notes file.

## Validation status

| gate | criterion | result |
|---|---|---|
| G0 | patches do not break kvaware+kvd | 4/4, 32/32, kvd 102 gets / 102 hits after engine restart |
| G1 | + MTP on the decode leg | 4/4, 32/32, `accept len` 2.48–2.58, router view 51 blocks |
| G2 | + prompt spanning >1 prefill chunk | 5/5 needles, prompt really split 8192+8192+1728 |
| stress | conc=16, conc=128 | 64/64 and 256/256, 0 corrupt |

**Validated by in-container patching, not by a build of this Dockerfile.** The
image build has not been run. Every patch was verified present in the **bytecode**
inside both running containers, and the Dockerfile applies the same scripts in the
same order — but that is an argument, not a measurement. Build it and re-run G0–G2
before shipping.
