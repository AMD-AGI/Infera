# The merged branch: what is on it, and how to take it apart

`yihou.dev.glm52.merged.experiment`, branched from `main` @ `8692fb4`.

Three workstreams that had only ever been validated separately, merged into one
configuration and validated end to end on 2 × 8×MI355X over mooncake RDMA. The
run is packed up under `merge_kvaware_mtp_pd.packup_20260731/`.

The commits are grouped so that any group can be dropped with one
`git rebase --onto` once its upstream PR lands. **Nothing on this branch was
taken on faith** — every commit here is code the merge experiment actually ran.

## The four groups, oldest first

| # | group | commits | upstream |
|---|---|---|---|
| A | sglang DSA: PD + DP-attention + EAGLE MTP on gfx950 | `53ccea0..87460ae` (9) | PR #58 |
| B | kvaware + kvd | `826619b..7f2dac8` (7) | PR #59 |
| C | mooncake early-send + bigram kv-events | `2181136`, `6e6fdb7` (2) | **PR #56** — temporary, see below |
| D | this merge's own fixes | `0cc9593..c0450a4` (3) | new |
| E | the rest of PR #56 that is not gfx942 | `d3c0d6f..eef9bfc` (3) | **PR #56** — temporary, see below |

A and B are our own PRs, cherry-picked unmodified — the tree after A is
byte-identical to PR #58's branch, and B's files land identically too.

## Groups C and E are temporary

Both hold liyingli's PR #56. C is the part the merge experiment ran and could
not be validated without; E is the remainder of PR #56 that is **not gfx942**,
backfilled afterwards and validated on its own terms. Neither group is owned by
this branch.

**When PR #56 merges, drop both.** They are not adjacent, so it is two rebases —
innermost last:

```bash
# drop E (3 commits at the tip)
git rebase --onto c0450a4 eef9bfc yihou.dev.glm52.merged.experiment
# then drop C
git rebase --onto 7f2dac8 6e6fdb7 yihou.dev.glm52.merged.experiment
```

Dropping C replays group D straight onto the end of group B: D does not depend
on C's code — it touches `args.py`, `kvd_wiring.py`, and three test modules,
none of which C changes — so the replay is clean. E is at the tip, so dropping
it replays nothing.

E touches only files C does not (`kv_event.rs`, `worker.py`, `net.py`), so the
two can also be dropped in either order; the order above just avoids rewriting
E's SHAs before removing them.

### Group E — the rest of PR #56, minus gfx942

| commit | PR #56 | what, and how it was validated |
|---|---|---|
| `d3c0d6f` | `01b0534` (Rust half) | The bigram decode in `rust/router/src/kv_event.rs`. C took only the Python half. Validated by **control**: revert `as_u32_any`, and the new ZMQ integration test fails `left: 0, right: 2`; restore it and it passes. |
| `fd3540d` | `0bb23c7` | `INFERA_SGLANG_READY_TIMEOUT`. Liying's commit ships no test; one was added. |
| `eef9bfc` | `d63e48b` | The NodePort-range skip, **hand-merged** with B's `826619b` — both rewrite the same loop. Liying's separate test file was folded into ours for the same reason. |

Verified in the built image on chi2879: **cargo 70 passed / 0 failed**, and the
touched Python suites 22 passed. Locally, where `pytest-asyncio` is installed,
the whole unit tree is **1168 passed / 1 skipped** (was 1162; +6 NodePort tests).

> The four `test_kv_event_e2e.py` async tests that fail *in the image* fail
> there before these commits too: that image has no `pytest-asyncio`, so
> `@pytest.mark.asyncio` is an unknown mark and the coroutine is never awaited.
> They pass locally. New tests here avoid the marker for exactly this reason.

### What is still deliberately left out

Only gfx942 work remains, none of which this cluster can exercise:

| left out | from | why not here |
|---|---|---|
| `deploy/docker/Dockerfile.sglang.gfx942` early-send layer | `6121189` | the MI325X image. Never built or run here. |
| `infera/engine/dsv4_gfx942.py` arch detection | `1ebdc7e` | `apply_gfx942_dsv4` returns early on non-gfx942, so it is a **no-op on MI355X**. Matters on MI325X. |
| `Dockerfile.sglang.gfx942` v0.5.16 base | `b2150c3` | same image, same reason. |

The rule is unchanged: **only code an experiment here exercised enters this
branch.** Adding any of the above means adding it, then running the gate that
would catch it failing — on gfx942 hardware.

## Group D — what this merge itself produced

| commit | what |
|---|---|
| `0cc9593` | `args.py`: don't append `--disaggregation-decode-enable-radix-cache` under speculative decoding — SGLang rejects the combination, killing an MTP decode leg at argument parsing |
| `e948285` | `kvd_wiring.py`: skip kvd on a PD decode leg — it is **write-only** there in every configuration (measured 180 sets / 0 gets) |
| `c0450a4` | behavioural tests for the bigram decode, at the seam rather than through the publish path |

Neither fix is a conflict between the merged workstreams — no line of either
changed. Both are **pre-existing infera code meeting a configuration nobody had
run**: the kvaware/kvd validation never enabled MTP, and the MTP validation drove
`sglang.launch_server` directly, bypassing the wrapper that appends the flags.

The first two are independent — different flags, different gates, different
SGLang checks. A decode leg with kvaware on and kvd off still needs `0cc9593`.

## The image

`deploy/docker/Dockerfile.sglang` carries everything. Layer order and the reason
for it:

| # | layer | note |
|---|---|---|
| 1 | mooncake unified rebuild | HIP-transport gate + dma-buf, runtime-decided |
| 2 | `pip install .[sglang]` | brings in the infera source, including groups C and D |
| 3 | `patches/sglang/` | GLM-5.2 nextn `eh_proj` quark-exclude |
| 4 | `patches/sglang_disagg/` | mooncake early-send wait event (group C) |
| 5 | `patches/sglang_dsa/` | the 3 DSA diffs (group A) — **layer 3 is their prerequisite**, which the apply script asserts rather than applies |
| 6 | Rust router | |

Layers 3 and 4 differ in strictness on purpose. Layer 3 tolerates a failed patch
(`|| echo skipped`) because those no-op once the base carries the fix. Layer 4
does not (`set -eu`): its script already reports "already present" on its own, so
a non-zero exit means the anchors drifted and the fix did *not* go in — and an
image that silently corrupts long prompts is worse than a failed build.

```bash
docker build -f deploy/docker/Dockerfile.sglang -t infera/engine-sglang:merged .
```

Build **on each node** rather than building once and shipping a 28 GB tarball —
the claim being tested is that the Dockerfile reproduces the run. ~15 min per node
with the base pulled. The two nodes' image ids will **differ** (independent
builds, so Rust objects and layer timestamps differ); check content equivalence
with `verify_built_image.sh`, not digests.

The base tag stays **pinned**: the DSA diffs apply at `--fuzz=0` against sglang
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, so a base bump fails the build at the
patch step rather than mis-applying silently. That is intended.

## Validation

Four gates, each loading one more thing onto the last so a failure localises.
**Run twice**: once against a base image patched in place, and again against the
image built from this branch's `Dockerfile.sglang` on both nodes, with no
in-container patching at all. Full detail, including the three probe defects that
each produced a wrong verdict before being caught, is in
`merge_kvaware_mtp_pd.packup_20260731/`.

| gate | criterion | patched image | **built image** |
|---|---|---|---|
| G0 | the patches do not break kvaware+kvd | 4/4, 32/32 | **4/4, 32/32** |
| G0 | kvd restart-replay | 102 gets / 102 hits / sets unchanged | **102 / 102 / unchanged** |
| G1 | + MTP on the decode leg | 4/4, `accept len` 2.48–2.58, prefill view **51** | **4/4, 2.17–2.60, 51** |
| G2 | + a prompt spanning >1 prefill chunk | 5/5, split `8192+8192+1728` | **5/5, same split** |
| stress | conc=16 | 64/64 | **64/64** |
| stress | conc=128 | 1 CORRUPT / 256 | **1 CORRUPT / 256** |

The single conc=128 case is the same in both: the one response that ran to the
`max_tokens` cap, a plain repetition loop (`</think>` × 0) rather than the
chunk-boundary signature, and CLEAN when the identical prompt is replayed at
conc=1.

Verify a built image before running anything against it —
`packup/scripts/verify_built_image.sh` makes 18 bytecode/source assertions plus a
behavioural smoke test. A build log saying a patch printed success is not the same
as the interpreter running patched code.

Unit suite: **1162 passed, 1 skipped**.

The two numbers that would have gone red had a fix been absent or wrong:

- **kvd is serving, not merely wired.** A speed-up proves nothing — the in-GPU
  radix cache serves a repeated prefix without touching L3. Restarting the
  prefill engine empties that cache while the kvd daemon keeps running: 102
  reads afterwards, `sets` **unchanged**, zero misses.
- **The bigram fix produces the *right* hashes.** The router view reads 51
  blocks under MTP — byte-identical to the plain-int path in G0, which shows the
  flattened keys chain to the same hashes rather than merely being non-empty.
  Unfixed it reads 0.
