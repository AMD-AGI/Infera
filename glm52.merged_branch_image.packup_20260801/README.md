# Merged branch + built image: the three GLM-5.2 workstreams as one deliverable

**Ran:** 2026-07-31 14:30 – 2026-08-01 04:30 UTC
**Author:** yihou
**Nodes:** `chi2879` (prefill) + `chi2867` (decode), 8× MI355X gfx950 each, ionic RoCE
**Model:** GLM-5.2-MXFP4, 2-node PD over mooncake RDMA
**Status:** **PASS** — branch delivered, image built on both nodes, all gates green.

## What this kit is, and what it is not

The predecessor kit — `merge_kvaware_mtp_pd.packup_20260731/`, committed on the
branch — established that the three workstreams work together, by **patching a
running container**. It said so honestly, three times over, and named the gap:

> The merged image was never built. […] the Dockerfile applies the same scripts
> in the same order, but that is an argument, not a measurement.

**This kit closes that gap.** Two deliverables:

1. **A git branch** with the three workstreams as four independent commit groups,
   each droppable with one `git rebase --onto`.
2. **An image built from that branch's Dockerfile, on both cluster nodes**, with
   every gate re-run against it and **no in-container patching at all**.

## Result against the spec

`spec.md` is the mission file verbatim. Its four goals, and where each stands now:

| # | spec goal | status |
|---|---|---|
| 1 | new worktree + unified experiment branch | **done** — `yihou.dev.glm52.merged.experiment`, 25 commits, 4 groups |
| 2 | deliver the final Dockerfile | **done** — `deploy/docker/Dockerfile.sglang`, carries all three workstreams |
| 3 | deliver a built image on the cluster | **done** — built on chi2879 **and** chi2867 from branch source |
| 4.1 | kvd + kvaware still work | **done** — G0 |
| 4.2 | mtp+dpa+pd / pd+mooncake correct at 1 / 16 / 128 | **done** — G1, G2, stress |
| 4.3 | packup via the packup skill | **done** — this kit |

Goals 1 and 3 were deferred in the predecessor run and are now closed.

## The branch

`yihou.dev.glm52.merged.experiment`, branched from `main` @ `8692fb4`.

| group | what | commits | upstream |
|---|---|---|---|
| A | sglang DSA: PD + DP-attention + EAGLE MTP on gfx950 | 9 | PR #58 (ours) |
| B | kvaware + kvd | 7 | PR #59 (ours) |
| C | mooncake early-send + bigram kv-events | 2 | **PR #56 — temporary** |
| D | this merge's own fixes + docs + packups | 7 | new |

> Everything here describes the branch **at `330da16`, 25 commits** — the state
> that was built and validated. Committing this kit adds one commit on top, so a
> checkout shows 26. `branch/patches/` and `branch/commits.tsv` are the 25; the
> 26th is this kit and changes no code.

A and B are cherry-picked unmodified: after A the tree is **byte-identical** to
PR #58's branch, and B's files land identically too.

**Group C is reduced on purpose.** PR #56 has seven commits; two are here, and
those two are stripped of everything this experiment did not exercise:

| left out | why | status now |
|---|---|---|
| `Dockerfile.sglang.gfx942` early-send layer | the MI325X image — never built or run here | still out |
| `rust/router/src/kv_event.rs` bigram decode | every run used `--router-backend python` (the default), so a Rust-router deployment with MTP still had the original bug | **fixed later** — `d3c0d6f` |
| `dsv4_gfx942.py` arch detection | `apply_gfx942_dsv4` returns early on non-gfx942 → **no-op on MI355X** | still out |
| `INFERA_SGLANG_READY_TIMEOUT` | convenience; 1800 s sufficed | **added later** — `fd3540d` |
| `net.py` NodePort-range skip | bare metal, no kube-proxy, `ip_local_port_range` starts at 32768 — unreachable here. Also conflicts textually with B's `826619b`. | **hand-merged later** — `eef9bfc` |
| gfx942 v0.5.16 base bump | same image, same reason | still out |

The rule was: **only code this experiment exercised enters the branch.** Those
omissions are a boundary, not an oversight — see `branch/MERGE_BRANCH.md`.

> **Three of the six were backfilled after this kit was written**, as group E,
> each validated on its own terms rather than argued for — the Rust one by
> reverting the fix and watching the new ZMQ test fail `0 vs 2`. They are *not*
> on this kit's measured path, so every number below still stands as recorded.
> Only gfx942 work remains out. See `work.liying_rest_20260801/`.

**Verified droppable.** `git rebase --onto 7f2dac8 6e6fdb7` replays all 7 later
commits with no conflict, and with group C gone the bigram tests fail
*behaviourally* (`msgspec.ValidationError`, not `ImportError`) — which is also
the proof that C and D are independent.

## The image, and the gates

Built **on each node** from `git archive` of the branch — not built once and
shipped, because the claim under test is that the Dockerfile reproduces the run.

| gate | patched image (predecessor) | **built image (this kit)** |
|---|---|---|
| G0 correctness / prefix reuse | 4/4, 32/32 | **4/4, 32/32** |
| G0 kvd restart-replay | 102 gets / 102 hits / sets **unchanged** | **102 / 102 / unchanged** |
| G1 correctness | 4/4 | **4/4** |
| G1 router view, prefill | 51 | **51** |
| G1 `accept len` | 2.48–2.58 | **2.17–2.60** |
| G2 needles | 5/5, split `8192+8192+1728` | **5/5, same split** |
| conc=16 | 64/64 CLEAN | **64/64 CLEAN** |
| conc=128 | 1 CORRUPT / 256 | **1 CORRUPT / 256** |

`Traceback` on either leg across all gates: **0**. Unit suite: **1162 passed, 1
skipped**.

### The two numbers that actually discriminate

A green run proves little; these two would have gone red if a fix were absent.

**kvd is serving, not merely wired.** A speed-up proves nothing — SGLang's in-GPU
radix cache serves a repeated prefix without touching L3. Restarting the prefill
engine empties that cache while the kvd daemon and its L3 keep running:

| | gets | hits | sets | misses |
|---|---:|---:|---:|---:|
| after first reuse run | 0 | 0 | 102 | 0 |
| **after restart + replay** | **102** | **102** | **102 (unchanged)** | 0 |

102 reads, zero new writes, zero misses — reuse that could only have come from L3.

**The bigram fix produces the *right* hashes.** `is_eagle` is a global server arg,
so with MTP on the prefill leg's kv-events carry bigram pairs. Unfixed, the router
hashes the pairs and the view reads **0**. It reads **51 blocks — identical to the
plain-int path in G0**, which shows the flattened keys chain to the same hashes
rather than merely being non-empty.

### The one conc=128 case, stated plainly

Both images show exactly **1 `CORRUPT_REASONING` in 256**. The predecessor kit's
summary table claimed 0; its own raw JSON never supported that, and it is
corrected here and there. In both runs it is the single response that ran to the
`max_tokens` cap, and it is a plain repetition loop (`the the the…`, `</think>` ×
**0**) — not the chunk-boundary signature, which requires the tag to *cycle*.
Replaying the identical prompt at conc=1 returns CLEAN.

## The finding that cost the most time

**`temperature: 0` plus MTP is indistinguishable from KV corruption.**

G2 first read **3/5** on the built image: needle missing at two depths, `</think>`
× 1264, and depth 1.0 the only clean one — which is *exactly* the mooncake
early-send signature, because the final chunk goes through the sampling path that
already synchronizes. Every check said the fix was in: present in the bytecode of
all three files on both nodes, call sites wired, prompt genuinely split.

All four probes sent `temperature: 0`. GLM-5.2's own `generation_config.json` says
**temperature 1.0, top_p 0.95**. Greedy decoding sends a reasoning model into
repetition on a long prompt, and **EAGLE/MTP amplifies it**: the draft model
predicts the loop perfectly, so `accept len` pins at its maximum (4.00) and the
response runs to `max_tokens`. `accept len: 4.00` is a *symptom of the loop*, not
evidence MTP is healthy.

Settled by control, not argument — MTP off, everything else identical: **5/5**.
MTP back on with the official sampling: **5/5**, `accept len` back to 2.17–2.60.
The same defect inflated conc=128 to 5/256 BAD, *all at the cap*; raising OSL
alone dropped it to 1/256.

Full analysis: `notes.md` §1.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable: clean nodes → branch → image → all gates |
| `environment.md` | hardware, fabric, image digests, SHAs, external paths, secrets needed |
| `notes.md` | the traps, in the order they will bite — the most re-read file |
| `spec.md` | the originating mission file, verbatim |
| `branch/` | commit list, code-only diff, the 25 patches, `MERGE_BRANCH.md`, the Dockerfile |
| `scripts/` | every script needed, verbatim |
| `results/raw/` | needle + stress JSON from the built-image run |
| `env/` | per-node environment snapshots |

## Related

- `merge_kvaware_mtp_pd.packup_20260731/` — the predecessor (patched-image)
  validation. **Committed on the branch**, and still the reference for the
  patch-set rationale and the two earlier probe defects.
- `kvaware_kvd_pr.packup_20260731.pr.final` — workstream 1 alone
- `glm52.mxfp4.spur.mooncake.packup_20260731_main_converged` — workstream 2 alone
- `work.merge_20260731/` — the live scratch workspace (kept intact)
