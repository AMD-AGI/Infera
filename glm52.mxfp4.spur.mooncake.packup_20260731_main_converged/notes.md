# Notes — the convergence, the alternative, and what went wrong

## 1. Why a clean rebase was not enough

`git rebase origin/main` replayed all four commits with **zero conflicts**. That is exactly
the situation to distrust: this branch only ever *added* files, so git had nothing to
compare against changes main made to the same problem in different files.

Three overlaps existed and none of them produced a conflict marker:

**1.1 The nextn fix — a latent build failure, not just duplication.**

Main's `0d8d0ff` added `deploy/docker/patches/sglang/patch_glm52_nextn_quark_exclude.py`,
an idempotent Python patch making the quark-exclude check probe
`model.layers.{N}.eh_proj` instead of the bare layer prefix. Our
`deepseek_nextn_glm52_mtp_bf16.diff` made the *identical* edit as a context diff.

Had both shipped in one image, main's loop runs first (`Dockerfile.sglang` line ~52) and
our diff would then have failed at `--fuzz=0` against an anchor that no longer matches.
The `--fuzz=0` choice is deliberate elsewhere in this work — a fuzzy apply that "succeeds"
against a different base is worse than a clean failure — so this would have been a hard
build break the first time anyone combined the two paths.

Resolution: drop our diff, and **assert** main's edit as a prerequisite in
`apply_sglang_dsa_patches.sh`. The assert matters because main's script is idempotent: if
the anchor is ever absent it prints `skipped` and exits 0. Without the assert, a silent
skip would surface only at runtime, as GLM-5.2 dying at draft weight-load with
`3072 vs 6144`.

**1.2 `build_mooncake_dmabuf.sh` — a strict subset.**

Diffed line by line against main's `build_mooncake_sglang.sh` (comments stripped): same
logic throughout, and main additionally verifies the HIP-transport gate compiled in, cleans
the build tree, and prefixes its errors. Ours had no capability main's lacked.

**1.3 `Dockerfile.sglang.dmabuf` — superseded.**

Main's `Dockerfile.sglang` now compiles the dma-buf branch in and selects it at *runtime*
via `MOONCAKE_DISABLE_HIP_DMABUF`, which is precisely how our runs drove it (`=0`). The
only thing our separate Dockerfile still contributed was the DSA patch layer, so that layer
moved into `Dockerfile.sglang` with `ARG APPLY_SGLANG_DSA_PATCHES=1` — **default on** —
placed after the existing patch loop so the nextn prerequisite is satisfied by
construction.

Net effect on the PR: 8 files / 1181 lines → 6 files / 1048 lines, and one modified file
instead of two new Dockerfiles and a duplicated build script.

## 2. The configuration-only alternative: IndexShare off

This is worth stating precisely, because it is a genuine substitute for part of the patch
set and it is easy to overstate.

**What it is.**

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

**Why it works.** The deadlock comes from `eagle_worker_v2.py::draft()` deciding
graph-vs-eager **per rank**, and the two paths issuing different collectives. One term of
that guard is `draft_input.dsa_topk_indices is None`. On the PD decode leg that field is
seeded from RDMA-shipped per-request payloads (`eagle_disaggregation.py:54-59`), so it is
a function of *which requests each rank happens to hold* — a rank with an empty
`batch.reqs` gets `None`, a busy rank gets a tensor. That is rank divergence by
construction. In single-node mix the same field is seeded locally on every rank by the same
code, which is why only PD hangs.

So there are two ways out: make the decision uniform (patch 4 — vote it over the TP group),
or remove the seed so the term stops diverging (IndexShare off).

**Which patches it substitutes for — and which it does not.**

| patch | replaced by IndexShare-off? |
|---|---|
| 1 `dsa_indexer_hip_dp_padded_rows` | **No.** Independent bug — DP-padded vs real rows in the HIP indexer. Still required. |
| 2a `dsa_backend` DP host-sync | **No.** A host sync is invisible to the graph/eager decision either mechanism changes. |
| 2b `dsa_backend` page-table rows | **Yes, in effect** — the arm ran without it and passed. |
| 3 nextn `eh_proj` | **No.** Weight-load bug, unrelated. Still required. |
| 4 `draft_cuda_graph_dp_vote` | **Yes** — this is the one it genuinely targets. |

**Evidence.** `..._20260730_exp2_indexshare_off`, same cluster, same image: 4/4 probe,
32/32 twice, 64/64, zero tracebacks, with patches 2 and 4 asserted **absent from the
bytecode** (anti-markers, not assumed). Accept length 2.98–3.01, i.e. no measurable cost.

**Two conditions that are easy to miss.**

- MTP must be on the **prefill** leg too. Without it the prefill worker never fills
  `req.output_dsa_topk_indices`, the seed never reaches decode, and "IndexShare off" is
  untested rather than tested.
- conc=128 was **not** run in that arm (criterion was 32; 64 was headroom). The patch path
  is what has been taken to 128/128 twice.

**It has an expiry date.** Today the override is nearly free because IndexShare's consumer
is already disabled under PD by `should_use_dsa_fused_topk` — the seed is produced and then
not used. Upstream PR **#31477** (`[Spec][PD] Enable fused TopK for GLM-5.2 MTP
IndexShare`) exists specifically to remove that limitation; when it lands, disabling
IndexShare starts costing (~3 % TPOT, per llying — second-hand, not measured by us).

Checked with `gh` on **2026-07-31**: #31477 is **OPEN**, `reviewDecision =
REVIEW_REQUIRED`, last updated 2026-07-28 — no approval, unmerged. Timing unknown; do not
plan around it landing soon, nor around it not landing.

**Bottom line.** If IndexShare is not wanted, the override is a legitimate and cheaper
answer today for patches 2b and 4. If IndexShare is wanted — or once #31477 lands — the
patches are the durable path. Patches 1 and 3 are required either way. This is recorded in
the patch-set README and in the diff headers so the choice is visible at the point of use.

## 3. Traps hit during this run

### 3.1 Zombie sglang processes read as "still running"

After killing the previous round's servers, `pgrep -fc sglang::` returned 11 and 19 on the
two nodes. They were all `<defunct>` — zombies whose parent had already exited. VRAM was
already released; `docker rm -f` in the container start script reaps them.

The lesson is to look at `ps -eo pid,etimes,cmd` before killing again, not to trust a count.
A count cannot distinguish a live server from a zombie, and a second round of `kill -9`
against zombies does nothing while looking like it should have.

### 3.2 A derived script written from the wrong working directory

`router_conv.sh` and friends were derived from the previous kit's scripts with `sed`. One
`sed` was issued with a relative source path after the shell's cwd had been reset by an
earlier tool call, so it read a file from a different tree than intended. The output looked
plausible.

Caught by diffing the generated file against a freshly generated one built from an absolute
path. Anything derived by `sed` from another kit should be verified against its source, not
eyeballed — the failure mode is a script that runs and measures the wrong thing.

### 3.3 Binary bytes in logs

`grep -c "Traceback" decode.log` returns **0** on these logs because they contain binary
bytes and grep treats the file as binary. That reads as "clean" when it means "not
checked". Every check in this kit uses `strings … | grep` or `grep -a`.

### 3.4 Router circuit breaker (avoided this run, by construction)

A router restarted on a **reused** port can keep a circuit breaker open from before the
restart and return 503 in ~0.4 s — which looks exactly like a backend failure. The tell is
latency: ~0.4 s is the breaker, 12–23 s is a real fault. This run used a fresh port (8180;
8160 and 8170 belonged to earlier runs the same day) and hit no 503s.

## 4. What is still open

- **Patch 2a has no revert-style control** — but that is not the same as thin evidence, and
  an earlier revision of this kit wrongly implied it was. Its case was made from runtime
  state rather than from a revert: a py-spy dump caught a **busy** rank blocked inside
  `dsa_backend` on `.max().item()` while **idle** peers had already advanced into the next
  collective (the divergence itself, observed), and after the fix no rank appears in
  `dsa_backend` in a dump again with PD warmup passing on all 8 ranks. Its second half
  (removing two unconditional `.cpu()` syncs) was forced *by experiment*: with the
  `max_seqlen_k` change alone the hang persists. See
  `..._20260728/dpa_mtp_fix/{CODE_VERIFICATION_bug2.md,dpa.mtp.pd.debug.md}`. A revert
  control would still be a cheap addition, but it would confirm a conclusion that already
  rests on direct observation, not fill a hole.
- **The draft-graph replay count was not re-measured here** — it needs a probe, i.e. a
  different image. 92.0 % on all 8 ranks was measured on the immediately preceding build of
  the same patch set. Everything in this kit is consistent with the draft path being forced
  eager; only a replay count rules that out.
- **#32209's patch 2b port failure is unexplained** — 0/32 across seven runs, 17 candidate
  causes eliminated. Kit: `..._20260731_exp3a_32209_patch2b_unresolved`.
- **Prefill-leg MTP is off**, so the rank-split case patch 4 is designed for has never been
  exercised in this arm.
- **No performance comparison** against a DPA-only baseline.
