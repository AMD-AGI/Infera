# patches/ — sglang branch `dev/yihou/glm52_dcp_pd`

Base: sglang `origin/main` fda84f80b7. Head: 93034eba91 (worktree `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp`).
Branch = fda84f80b7 + merge 4792dc0e52 of PR #44322 (`yihou/pr44322`, triton sparse-MLA DSA DCP for ROCm) + 31 of our commits.

| Path | What |
|---|---|
| `sglang-branch/0000-merge-pr44322-onto-fda84f80b7.diff` | `git diff fda84f80b7 4792dc0e52`: the #44322 merge as one diff (format-patch skips merges) |
| `sglang-branch/0001..0031-*.patch` | `git format-patch 4792dc0e52..93034eba91` (linear, no merges) |
| `sglang-branch.full.diff` | `git diff fda84f80b7 93034eba91` (180 KB, 53 files, +2844 / -160) |

Apply (checked 2026-10-22 in a scratch worktree; result tree byte-identical to 93034eba91):

```bash
git checkout --detach fda84f80b7
git apply --index patches/sglang-branch/0000-merge-pr44322-onto-fda84f80b7.diff && git commit -m "merge PR #44322"
git am patches/sglang-branch/00*.patch
# or, one shot: git apply patches/sglang-branch.full.diff   (git apply --check passes on fda84f80b7)
```

The runs mount the tree as an overlay (`/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-<sha>/python/sglang` over
`/sgl-workspace/sglang/python/sglang`, see `scripts/harness/mounts.yihou.dcp-overlay.*.tsv`) on image
`infera-sglang:v0522-1020-yihou-dcp-mc4340`. Shas used on GPU: 4792dc0e52-pdrecipe overlay (early dryruns pdA/pdB/pdBe*-268, pdB*-054),
ae529c96bf (pdBe-r1 gate, pdBeperf-r1/r2/r4, mixed DCP4 AgentX), 93034eba91 (pdBe-r5g, pdBeperf-r5/r5b, E2 pdE2-*).
Sources: `scripts/harness/dryrun.pd*.txt`, ledger.pdrun P016, ledger.mixrun E2 header, RESULTS.md §1-2.

Most new behavior is env-gated and off by default: `SGLANG_DSA_DCP_PD_DROP_MTP_SEED`, `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER`,
`SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON`, `SGLANG_DSA_DCP_KEEP_KV_B_BF16`, `SGLANG_LOG_DECODE_SEQ_LENS`.

Side branch `dev/yihou/glm52_dcp_pd_dp2` exists (forks at 6ffb90ae6b; carries the original DP2 x DCP2 fixes
31309dc4f1, 075edb377e, fdf73b6fb2, the debug-only bounds check bd6a337ab7 `SGLANG_DEBUG_DSA_ROWS`, and fa810b9f69).
The fixes were cherry-picked onto this branch (0023, 0026, 0027, 0028); bd6a337ab7 was not. Not included as patches here.

## Commits (`git log --oneline --reverse 4792dc0e52..93034eba91`)

Verification sources: RESULTS.md (`yihou_ws/glm52_dcp_pd_20261021/RESULTS.md`), ledgers `ledger.{pdrun,mixrun,benchrun}.md`.
"CPU" = unit test in the commit, run on CPU; "fail-first" = test fails on the old code.

### Infera image recipe re-cut as commits (needed because the overlay mount shadows the image's patched sglang)

| # | Sha | Subject | What / why | Verification |
|---|---|---|---|---|
| 01 | e75479a189 | infera recipe: dsa_indexer HIP DP padded rows | Infera patch: paged aiter MQA logits path reconciles padded q rows with the real row count | The 8 recipe-touched files at 4d1d6a647e are sha256-identical to pdrun's recipe_series4 tree (ledger.pdrun, branch ae529c96bf note) |
| 02 | 9118404a4a | infera recipe: dsa_dp_sync | Infera patch: avoid per-rank D2H syncs in DSA metadata init that deadlock DP attention | same |
| 03 | 344765b3bc | infera recipe: dsa_page_table_rows | Infera patch: decode page table one row per token to match top-k under MTP | same |
| 04 | e11f97495d | infera recipe: prefill nonfinal-chunk wait event | Infera patch | same |
| 05 | 6a439488e0 | infera recipe: PD decode skips implicit ROCm rejection sampling | Infera patch | same |
| 06 | 3c733c7f5b | infera recipe: responses unstreamed tool args | Infera patch | same |
| 07 | 7abe9a8d5e | infera recipe: HiCache ROCm staged write-back off | Infera patch | same |
| 08 | 36bc86517d | infera recipe: kv-events metrics send noblock | 22k-kit runtime fix R001 | same |
| 09 | 4d1d6a647e | infera recipe: absorb-bmm config off on ROCm | 22k-kit runtime fix R003 | same |

### PD + DCP enablement

| # | Sha | Subject | What / why | Verification |
|---|---|---|---|---|
| 10 | a6bdc7894d | [PD] dcp: Relayout a DCP-sharded DSA draft KV like the target | On ROCm the MTP draft KV is DCP-sharded like the target; the DCP1 -> DCP-N relayout assumed a replicated draft and raised ("draft head slicing unsupported"). NIXL/Mori reject a sharded draft | CPU T1/T2 fail-first; pdBe-r1 PASS on GPU: math 29 x2, passkey 8K/34K/96K (RESULTS.md §1, §5) |
| 11 | 228263d116 | [PD] dsa: Allow a ROCm DCP decode behind a DCP1 prefill | main rejected DSA DCP + PD; allowed for a ROCm decode only | pdBe-r1 (RESULTS.md §5) |
| 12 | ae529c96bf | [PD] dsa: Add SGLANG_DSA_DCP_PD_DROP_MTP_SEED | A/B switch for the PD prefill's top-k seed on a DCP decode (default off) | CPU (RESULTS.md §5) |
| 13 | 906b0915a9 | [Test] dcp: DP-attention prefill fan-out to a DCP4 decode | Test only: one attn-TP1 prefill rank feeds 4 DCP decode ranks; each holds exactly its rows | CPU |

### Split decode indexer (each DCP rank scores 1/W of the pages, exact global top-k by merge)

| # | Sha | Subject | What / why | Verification |
|---|---|---|---|---|
| 14 | a78ff55a8b | [AMD] dsa: page-split DSA decode top-k for DCP over a replicated index K | Local top-k per rank + all-gathered merge into the exact global top-k | CPU: merge vs single-rank top-k with ties, partial pages, MTP verify rows (commit message) |
| 15 | a04c3f8e0c | [AMD] dsa: Gate the page-split decode indexer behind SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER | Off by default; ROCm + DCP + aiter preshuffle paged MQA only | benchrun T7: +6 ms at N >= 16, 34K (torch merge) (RESULTS.md §3) |
| 18 | 6ffb90ae6b | [AMD] dsa: owned pages and lens once per forward | Reuse across the 21 indexer layers of a verify step | benchrun T9: parity vs ref at 150K; 34K still +5.5-6 ms (ledger.benchrun) |
| 24 | 0347a6d47e | [AMD] dsa: Fuse the split decode indexer's pack and merge into Triton kernels | `SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON` (default off): one pack + one sort-merge launch per layer | benchrun T12: gate PASS (graph, sim off, 40/40); triton vs ref 34K +2.0..+2.6 ms, 150K -1.1 (N16) / -2.9 ms (N24); merge 102 us/layer (ledger.benchrun T12) |
| 30 | a6ea78e322 | [AMD] dsa: Merge candidates by radix select instead of a full sort | Four 8-bit histogram passes replace the bitonic sort | Failed to compile on GPU (triton 3.7 amd: constexpr reassigned) (ledger.benchrun T14); fixed by 31 |
| 31 | 93034eba91 | [AMD] dsa: Do not rebind a constexpr in the radix-select merge loop | Compile fix for 30 | benchrun T15/T16: gate PASS (40/40); merge 46 us/layer; rdx vs ref 34K +0.5..+1.5 ms, 150K -2.6..-3.1 (N16) / -4.1..-4.2 ms (N24), 200K -4.4 / -6.6 / -9.2 ms (N16/24/32) (ledger.benchrun T15, T16) |

### Other fixes, features, tests

| # | Sha | Subject | What / why | Verification |
|---|---|---|---|---|
| 16 | 646dc699f3 | [Test] dcp: PP prefill stages feeding a DCP4 decode | Test only | CPU |
| 17 | 6ee35f3617 | [PD] pp: Gate non-final prefill chunk sends on the forward stream | PP prefill (non-overlap) sent a non-final chunk without a wait event | CPU fail-first; PP never run on GPU (RESULTS.md §5) |
| 19 | c4eda0343e | [PD] decode: Release deferred KV and skip transfer polling in unison across attn-TP ranks | Rank-local deferred KV release after client cancels diverged attn-TP queues -> hang | CPU (4 of 5 new tests fail on old code); GPU cancel repro (`scripts/tools/pd_cancel_repro.yihou.py`) not run (RESULTS.md §5) |
| 20 | 5fb03011d6 | cherry-pick sgl-project/sglang#45483 | Init the active attention backend for eager IDLE batches (upstream open PR, unchanged) | upstream test (RESULTS.md §5) |
| 21 | c9394c114f | [AMD] offloader: Keep is_shuffled on CPU-offloaded weights | OffloaderV2 `.to()` dropped `is_shuffled`, so aiter fused_moe picked the wrong MXFP4 kernels | CPU fail-first; not run on GPU (RESULTS.md §5) |
| 22 | 636cc68dd0 | [AMD] dsa: Add SGLANG_DSA_DCP_KEEP_KV_B_BF16 | Skip gfx950 fp8 conversion of kv_b so `--dcp-replicate-q-proj` drops the per-layer Q all-gather (default off) | benchrun T10: -0.4 / -1.6 / -1.7 / -1.7 ms at N 8/16/24/32, 34K; weights +0.34 GB/GPU, KV -0.28 %; mixed AgentX 26,152 tok/s/GPU (RESULTS.md §2-3); pdBe-r5g logs "prepared full-head Q weights for 78 MLA layers" (ledger.pdrun P016) |
| 23 | b17da53725 | [Fix] attention: Pad a batch-major padded-extend output on its token dim | DP2 x DCP2 bug 1: 49 vs 50 rows (cherry-pick of 31309dc4f1) | crash gone (ledger.mixrun R13; RESULTS.md §4) |
| 25 | 9af2166f3f | debug: SGLANG_LOG_DECODE_SEQ_LENS | Local diagnostic: rows / sum / max context per decode log line | active in pdBe-r5g ("seq-lens rows: 1, sum: 100, max: 100", ledger.pdrun P016) |
| 26 | e9e1432ce6 | [Fix] dsa: Do not warn on an idle DP rank's DCP prepare | Log noise (cherry-pick of 075edb377e) | RESULTS.md §4 |
| 27 | da6e383513 | [Fix] dsa: Skip the indexer on an eager idle DP rank on ROCm too | DP2 x DCP2 bug 2: MAF in paged top-k v2 on the idle draft (cherry-pick of fdf73b6fb2) | idle fault gone (ledger.mixrun R18; RESULTS.md §4) |
| 28 | 5aacce1f1c | [Fix] forward-batch: Pad extend_seq_lens_cpu into a new list | DP2 x DCP2 bug 3: in-place pad of a shared list gave 42 q rows vs 36 metadata rows -> MAF (cherry-pick of fa810b9f69) | eager x2 + graph: check.py 5/5, 40/40, no fault (ledger.mixrun R24/R25; RESULTS.md §4) |
| 29 | efa41449fe | [Test] dcp: DP-attention prefill fan-out to a DP2 x DCP2 decode | Test only (E2 prep) | CPU |

Known open item tied to this code: PD E2 (DP2 x DCP2 decode, graph, sim off) returned garbled outputs on 3/200, 13/240
and 27/320 fresh-seed requests across rounds pdE2-r2/r5/r6; eager gate pdE2-r1 0/40. Not bisected (user decision,
recorded known risk). Source: ledger.mixrun E2 r2-r6.
