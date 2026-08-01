# Backfilling the rest of PR #56 (everything that is not gfx942)

**Ran:** 2026-08-01 05:10 – 06:10 UTC
**Author:** yihou
**Nodes:** `chi2879` + `chi2867`, 8× MI355X gfx950 each
**Branch:** `yihou.dev.glm52.merged.experiment` → **31 commits** (was 26)
**Status:** **PASS** — three commits added as group E, each validated by
experiment, and the image rebuilt from the branch on both nodes.

## What this closes

`glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/` delivered the merged
branch and named, honestly, six things from liyingli's PR #56 it had left out —
including one it called a **real unfixed bug**:

> `rust/router/src/kv_event.rs` bigram decode — every run used
> `--router-backend python` (the default). **A Rust-router deployment with MTP
> still has the original bug.**

This kit closes every one of those that is **not gfx942**. Three commits:

| commit | PR #56 | what |
|---|---|---|
| `d3c0d6f` | `01b0534` (Rust half) | bigram decode in the Rust router — **the unfixed bug above** |
| `fd3540d` | `0bb23c7` | `INFERA_SGLANG_READY_TIMEOUT` |
| `eef9bfc` | `d63e48b` | NodePort-range skip, **hand-merged** with our `826619b` |

Plus two docs commits correcting the earlier kits, which said these were absent.

**gfx942 is out of scope and stays out** — three PR #56 commits (`6121189`,
`b2150c3`, `1ebdc7e`) touch only the MI325X image and an arch gate that is a
no-op on MI355X. This cluster cannot exercise them, and the branch rule is
unchanged: only code an experiment here ran enters it.

## Result

| patch | evidence | where |
|---|---|---|
| **P1** Rust bigram decode | **control**: revert the fix → the new real-socket test fails `left: 0, right: 2`; restore → passes. Whole crate **70 passed / 0 failed**. | `rounds/r4_rust_control/` |
| **P2** ready timeout | 5 new tests, **executing** (not marker-skipped) in the engine image | `rounds/r2_image_patch/` |
| **P3** NodePort skip | 10 tests; 40 live `free_tcp_port_block(8)` calls, none in 30000-32767, bases still spread | `rounds/r2_image_patch/` |
| all three, in the **built image** | rebuilt from the branch on both nodes, verified in bytecode + behaviour | `rounds/r5_built_image/` |

Unit suite locally: **1168 passed, 1 skipped** (was 1162; +6 NodePort tests).

## The number that discriminates

Everything else here is a green test that would also be green without the fix.
This one would not be:

```
CONTROL   (as_u32_any removed, as_u32_vec back to ints-only)
  subscriber_decodes_bigram_tokens_under_mtp ... FAILED
    left: 0
   right: 2

TREATMENT (fix restored)
  subscriber_decodes_bigram_tokens_under_mtp ... ok
```

`left: 0` is the bug verbatim, over a **real ZMQ socket**: SGLang under MTP
sends `token_ids` as the overlapping pairs `(t[i], t[i+1])`, `as_u32_vec` read
each element as an integer, a pair is not an integer, and `filter_map` dropped
all of them. The view stays permanently empty, `cache_hits` pins at 0, kv-aware
degrades to load balancing — and nothing anywhere errors.

The assertion is that the bigram view hashes to **the same two blocks** the
query side computes over the flat slice, not merely that it is non-empty.

## The wrong turn worth reading

**A live A/B of the two router binaries proved nothing, and looked like it did.**

Round 3 ran the shipped and the patched `infera-router` against the live PD pair
and read policy.rs's per-pick `cache_hits`:

| leg | picks |
|---|---|
| before (**unpatched**) | `0, 0, `**`51`**`, 0` |
| after (patched) | `0, 0, 0, 0` |

Read naively that says the fix *broke* something. In fact neither leg tested
anything: the live **prefill** leg runs without `--speculative-algorithm` — MTP
is on the decode leg, and `is_eagle` only switches the radix key to bigrams on a
leg that has it. The wire was carrying plain ints, which both binaries decode
correctly. The unpatched binary reading 51 is what exposes the run as vacuous.

That is why round 4 drives the decoder directly over a real socket with the
exact bigram shape, and runs it **both ways**. A test that never had a chance to
fail is not evidence.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable: patch a live container → test → control → rebuild → verify |
| `environment.md` | nodes, images, digests, SHAs, external paths, secrets needed |
| `notes.md` | the traps, in the order they bite — three of them cost real time here |
| `working_process.md` | the round-by-round record, including the wrong turn |
| `patches/` | the three commits as `git format-patch`, plus a combined diff and `commits.tsv` |
| `scripts/` | every script used, verbatim |
| `rounds/` | per-round logs — the raw evidence behind every number above |
| `results/` | the final `kv_event_zmq.rs` carrying the new test |

## Related

- `glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/` — the branch +
  built-image validation this extends. Its `notes.md` §9 and `README.md` table
  are annotated in place to point here.
- `merge_kvaware_mtp_pd.packup_20260731/` — the predecessor (patched-image) run;
  its `notes.md` §7 "Deferred from PR56" is likewise annotated.
- `work.liying_rest_20260801/` — the live scratch workspace (kept intact).
