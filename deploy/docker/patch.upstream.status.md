# Patch ↔ upstream status

Every patch under `deploy/docker/patches/`, and where it stands relative to the
project it patches. Kept here so "why do we still carry this?" has one answer
per row, and so a patch that upstream has since merged gets dropped instead of
quietly outliving its reason.

**PR states and pinned source re-verified on 2026-09-17**, when the sglang mi35x
base moved to `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916`
(`sha256:eef7b70e015c94435d68dd92612229d52324b9e8d5edaac741802e7e78e66eb6`). Every
defect below was checked by reading that image's own sglang tree, and every
verdict was then confirmed by running the patch against it: applies, is
idempotent on re-run, and passes bytecode verification where the applier has a
marker. An anchor drift is recorded as a re-cut, not mistaken for an upstream fix.
State drifts; re-check before relying on a row. `gh search`
matches titles and bodies, **not diff content**, so "no upstream PR" means "none
found by search", not "none exists" — where a row could be checked by reading
upstream source instead, it says so.

**THE mi35x BASE IS A DATED NIGHTLY, NOT THE v0.5.19 RELEASE.** It reports
`0.5.19.dev20260916+ge7f7447333`, which is `main` several hundred commits past the
`release/v0.5.19` branch point. Some anchors below match the `v0.5.19` *tag* but
not this image, and the reverse also happens, so "verified on v0.5.19" is not a
statement about this base. Verdicts here are against the digest named above.

**THREE BASES ARE LIVE, AND THEY NO LONGER SHARE ONE PATCH SET.** `Dockerfile.sglang`
is on the nightly above; `Dockerfile.sglang.glm53` is on v0.5.18; and
`Dockerfile.sglang.gfx942` is on v0.5.16. Where a patch is needed on some bases
and fixed on others, the file stays put and the *image* stops copying it — see
the "applied by" note on each row. Where the anchor itself differs, the newer cut
lives in a `v0519/` subdirectory, which the older recipes' `*.py` globs do not
descend into, so those two images keep their exact previous patch set without
being edited. Both properties were re-checked on 2026-09-17 by running each
image's file list against its own base.

Column meanings:

- **ours?** — was the upstream PR opened by a contributor of this repo?
  `yes` = us; `no` = a third party; `—` = no PR.
- **PR state** — of the upstream PR named in the same row.

## sglang — `patches/sglang_dsa/` (baked by `Dockerfile.sglang` and `Dockerfile.sglang.gfx942`, `APPLY_SGLANG_DSA_PATCHES=1`)

**Upstream fixed none of these four.** All are still needed on the nightly base.

Patch 01 is an anchor script and is baked by both images. The other three are
`--fuzz=0` diffs verified against the mi35x base and are not applied by the
gfx942 image, which substitutes `dsa_page_table_rows` and `draft_cuda_graph_dp_vote`
at runtime with `--json-model-override-args '{"index_share_for_mtp_iteration":false}'`
— `patches/sglang_dsa/README.md` carries the reasoning.
The apply script dry-runs both directions before mutation, so rerunning it skips
an applied diff instead of allowing GNU patch to auto-reverse the fix.

Two of the four moved at the nightly, neither because the defect changed:

- **Patch 01 was re-anchored onto deliberately loose context.** The typing line it
  used lost `List`, and a nested helper was inserted between `weights.squeeze(2)`
  and the aiter branch. The replacements — the `contextlib`/`logging` import head,
  and the bare `if …is_aiter():` line — are unique on v0.5.16, v0.5.18 *and* the
  nightly, so one script still covers every base and the directory did **not**
  have to be split. Do not re-tighten them onto the adjacent code.
- **`draft_cuda_graph_dp_vote.diff` was re-cut, and the interesting part is not the
  rename.** Upstream took DP-sync slot 7 for `prefill_cuda_graph_max_prefix_len`,
  so the vote now rides slot 8; a mechanical re-anchor that kept slot 7 would have
  read a prefix length as a boolean and passed both compile and bytecode checks.
  The two `can_run_dp_*` fields it reads were also renamed. Only the `full` arm
  uses this diff, so no other image is affected.

Applied and bytecode-verified on 2026-09-17 against both trees each arm serves:
v0.5.16 (`indexer`, patch 01 only) and the nightly (`full`, all four).

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py` | HIP/aiter paged-MQA sizes its output from DP-padded rows while `lengths` is sized to real rows → `Expected lengths.size(0) == B` | none found | [sglang#33059](https://github.com/sgl-project/sglang/pull/33059) | **yes** (`dorado269`) | OPEN |
| ″ (same bug class, other platform) | — | none found | [sglang#32762](https://github.com/sgl-project/sglang/pull/32762) `[NPU] Fix DSA eager padding mismatch` — our diff is written in its shape | no (`stellaxcpeng`) | OPEN |
| ″ (anchor collision, **not** a fix) | — | — | [sglang#32738](https://github.com/sgl-project/sglang/pull/32738) pads heads for DeepGEMM at the same two aiter call sites; [#31480](https://github.com/sgl-project/sglang/pull/31480) extracts the paged-MQA backend and restructures the `is_aiter()` dispatch | no | both OPEN (re-read 2026-08-03) |
| `sglang_dsa/dsa_dp_sync.diff` | `seq_lens.max().item()` is a host sync on a branch only *some* DP ranks take → DP collectives desync → deadlock | none found | [sglang#33973](https://github.com/sgl-project/sglang/pull/33973) — this file **is** that PR's diff, so it drops by deletion when it merges | **yes** (`dorado269`) | OPEN |
| `sglang_dsa/dsa_page_table_rows.diff` | page table has one row per **request**, top-k one per **token** under MTP → `assert page_table.shape[0] == topk_indices.shape[0]` | none found | [sglang#32209](https://github.com/sgl-project/sglang/pull/32209) solves the same row mismatch by **trimming q/top-k**; porting that half here fails at conc=32 and is unresolved | no (`HZY-Wade`) | OPEN |
| `sglang_dsa/draft_cuda_graph_dp_vote.diff` | the draft graph/eager choice is per-rank, so under PD + DP-attention + MTP a DP group splits across the two paths and deadlocks on the first routed request | [sglang#32527](https://github.com/sgl-project/sglang/issues/32527) | [sglang#32209](https://github.com/sgl-project/sglang/pull/32209) carries the same vote at the same site; we take only that half | no (`HZY-Wade`) | OPEN |

Background, already present in the base and **not** patched by us:
[sglang#30378](https://github.com/sgl-project/sglang/pull/30378) /
[#30427](https://github.com/sgl-project/sglang/pull/30427) (MERGED) clamp padded-row
seq_lens **values**; our patch 01 fixes the HIP-side row **count**.
[#30839](https://github.com/sgl-project/sglang/pull/30839) /
[#31083](https://github.com/sgl-project/sglang/pull/31083) (MERGED 2026-07-14)
introduced the guard `draft_cuda_graph_dp_vote.diff` repairs — so that deadlock is a
regression in this baseline, not a legacy wart.
[#32722](https://github.com/sgl-project/sglang/pull/32722) (OPEN) adds a test for
PD + DP-attention + MTP, i.e. **no CI covers this topology today**.

## sglang PD — `patches/sglang_disagg/` (baked by all three SGLang recipes; `v0519/` only by `Dockerfile.sglang`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_disagg/patch_mooncake_early_send_wait_event.py` | `mooncake/conn.py` never waits on the forward's completion event and the overlap path records none, so chunked prefill hands a non-final chunk to the decode leg while the forward writing those pages is still running → prompts longer than one chunk come back **partially wrong**, with nothing in any log | [sglang#25583](https://github.com/sgl-project/sglang/issues/25583) reports the same corruption shape on GLM-5, but **aggregated** — no PD, no mooncake — so a shared root cause is unestablished; closed **inactive** 2026-07-18, no follow-up | [sglang#33970](https://github.com/sgl-project/sglang/pull/33970) | **yes** (`dorado269`) | OPEN |

> `prefill.py` already records the completion event this needs; only the `mori`
> backend ever read it, and the patch mirrors what `mori` does. **Drop it** once a
> base sglang synchronizes on that event itself — the script then reports "already
> present" and no-ops.
>
> Unlike the rest of this page, this row was **not** verified with `gh`: the issue
> state was read from the web UI on 2026-08-03, and no upstream PR search was run.
> "none found" here is weaker than elsewhere on the page.
>
> The import differs between supported bases (`Set` is added in v0.5.18), so the
> script accepts explicit v0.5.16 and v0.5.18 source shapes. Both apply and
> second-run idempotence checks pass.
>
> **Re-anchored at the nightly, and the defect is untouched:** `mooncake/conn.py`
> still contains no `wait_event` or `synchronize()` of its own, while `mori` has
> carried the full barrier since v0.5.18. Only Black moved — it reformatted the
> `assert` that used to anchor the `prefill.py` edit. That edit now anchors on the
> bare `send_kv_chunk(..., end_idx=req.tmp_end_idx)` call, which is unique on all
> three bases and immune to the next reformat.
>
> **One half of this patch has landed upstream, for the other backend.** The
> nightly declares `TransferKVChunk.wait_event` for `mori`'s benefit, so applying
> our field edit there would declare it twice. `_ALREADY_UPSTREAM` in the script
> skips `common/utils.py` when the field is present; the mooncake consumer edits,
> which are the ones that actually fix anything, still apply.

### PD decode vs the nightly's implicit ROCm rejection sampling

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_disagg/v0519/patch_pd_disable_implicit_rocm_rejection_sampling.py` | the nightly auto-enables `speculative_use_rejection_sampling` on HIP for any plain-EAGLE run. A PD **decode** worker cannot take that path: the handoff rebuilds `EagleDraftInput` without `draft_probs`, so `draft()` seeds `draft_probs_list` with `None` and dies at `torch.stack` during startup warmup | none found | none found | — | — |

**Applied only by `Dockerfile.sglang`**, from the `v0519/` subdirectory, because
only that base has the defect. This patch was first cut against the xiaobochen
GLM-5.2 fork and kept out of the shared glob as fork-specific; **that label is now
obsolete.** Official v0.5.18 has no implicit enable at all — `is_hip()` appears
once in its speculative hook, to choose a draft attention backend — but the
nightly adopted the same auto-enable via `_should_auto_enable_hip_rejection_sampling`,
whose guard list (EAGLE3, draft token map, topk != 1, non-default accept
thresholds, deterministic inference) omits PD decode. Confirmed absent from
upstream `main` too, so **this one is worth filing rather than only carrying.**

Costs nothing on a greedy deployment: rejection sampling exists so EAGLE verify
can honour temperature and top_p, and ROCm has no sampling-verify kernel, so
verify already falls back to argmax — which is exactly right at temperature 0.
`--speculative-use-rejection-sampling` still works if asked for explicitly.
**Drop it** when a base guards the auto-enable on `disaggregation_mode` or
transfers `draft_probs` across the handoff; the anchor stops matching in the
first case, so the drop is not silent.

## sglang Responses API — `patches/sglang_disagg/` and `patches/sglang_responses/`

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_disagg/patch_responses_pd_bootstrap.py` | `ResponsesRequest` drops `bootstrap_host/port/room` and its `GenerateReqInput` does not forward them, so every Responses request to a PD worker returns 400 | none found | none found | — | — |
| `sglang_responses/patch_responses_custom_encoder_prompt.py` | Responses takes empty prompt text instead of the authoritative prompt IDs for `kimi_k3`/`inkling`, so every request returns 400 | none found | none found | — | — |

Both defects are present in v0.5.16 and v0.5.18. The PD bootstrap script accepts
the explicit GenerateReqInput tail from each release and **still applies cleanly
to the nightly**, which is useful evidence in itself: the Responses area was not
fixed wholesale, so the two verdicts below are about their own defects.

**`patch_responses_custom_encoder_prompt.py` — DROPPED on the nightly, still
applied on v0.5.16 and v0.5.18.** Its own header set the exit criterion as "base
sglang routes both through one helper", and that is exactly what happened:
`OpenAIServingChat._engine_prompt` is new in the nightly and both Chat
(`serving_chat.py`) and Responses (`serving_responses.py`) now call it, so a
custom encoder's `prompt_ids` reach `GenerateReqInput` instead of an empty
`prompt`. `Dockerfile.sglang` no longer copies the file; the other two recipes
still glob it, so it stays where it is rather than moving to **Retired**.

**`patch_responses_unstreamed_tool_args.py` — RE-ANCHORED, and isolated by
directory.** The defect is untouched: `prev_tool_call_arr`, `streamed_args_for_tool`
and `_check_for_unstreamed_tool_args` are still absent from `serving_responses.py`,
which still treats the accumulated deltas as authoritative. But the nightly
restructured `_close_tool_call_state` for custom tool calls — `events` is now
initialised empty before a custom/function branch instead of being built as one
list literal after it — so the older shape needs two splices and the newer one
needs a single insertion. Rather than carry two anchor sets and two splice counts
in one script, whose error reporting could then no longer name which shape
drifted, the newer cut lives in `sglang_responses/v0519/`. It reconciles only the
function-call arm; the custom arm decodes through `decode_custom_tool_input` and
diffs its own remainder, so it neither needs nor wants a second delta.

## sglang ROCm — `patches/sglang_rocm/` (`host_alloc` by `gfx942`/`glm53` only; `staged_write_back` by `Dockerfile.sglang`/`gfx942`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_rocm/patch_hicache_rocm_host_alloc.py` | hicache allocates host pools with `mmap` + `hipHostRegister`, which on ROCm maps the pages at a device address ≠ the host VA, but the pools hand raw host `data_ptr()`s to GPU kernels via device-side pointer tables → `Memory access fault by GPU node-N on address <host VA>` on the first kvd write-back | none found | [sglang#33968](https://github.com/sgl-project/sglang/pull/33968) | **yes** (`dorado269`) | OPEN |
| `sglang_rocm/patch_hicache_rocm_staged_write_back.py` | On the v0.5.16 controller, `pool_host/mla.py` enables the staged write-back JIT on HIP while `DSAIndexerPoolHost` gates it on `_is_cuda`; group-level index movement then sends GPU indices to a JIT kernel requiring CPU/ROCm-host indices → scheduler exit −3. v0.5.18 adds per-pool index movement and no longer has that exact failure mode, but the shared patch keeps MLA on the conservative non-JIT path. | none found | [sglang#28534](https://github.com/sgl-project/sglang/pull/28534) `[AMD] Enable JIT staged HiCache write-back and fix CPU-index crash` | no (`AMD-yanfeiwang`) | **MERGED** 2026-07-09 |

**`patch_hicache_rocm_host_alloc.py` — DROPPED on the nightly, still applied on
v0.5.16 and v0.5.18. This is the one row where "it still applies" is a trap.** The
script exits 0 against the nightly, because its anchor sits in `ALLOC_MEMORY_FUNCS`,
a dict upstream never had to touch — the repair landed at the ~10 call sites this
patch's own header called "the more thorough shape".
[sglang#35233](https://github.com/sgl-project/sglang/pull/35233) `[AMD] Fix registered
HiCache host pointer aliases` (**MERGED 2026-09-14**, two days before this base was
cut) adds `make_kernel_ptr_table`, which translates every host-pool pointer table
through `hipHostGetDevicePointer`; its stated motivation is our fault verbatim, on
MI355X. [#39516](https://github.com/sgl-project/sglang/pull/39516) (**MERGED**) added
the guard that raises when the kernel op is missing. Confirmed in the image itself:
`torch.ops.sgl_kernel.get_device_accessible_ptr` is present. Our own
[#33968](https://github.com/sgl-project/sglang/pull/33968) is superseded in substance.

Keeping it would not crash — it would silently cost something. `alloc_with_pin_memory`
ignores the `allocator` argument, so a ROCm deployment configuring a mooncake / mori /
umbp storage backend would quietly get `torch.empty` instead, and teardown would warn
per buffer on an unregister that never registered. That is why `Dockerfile.sglang`
uses an explicit file list here instead of a directory glob.

The fault it fixed is **gfx950-only so far**: MI300X (amdgpu 6.14.14, ROCm 7.2.0)
measures the two addresses equal, so `Dockerfile.sglang.gfx942` carries it
preventively rather than to fix a crash. Don't read its row above as evidence the
fault was seen on both arches.

> Verified on 2026-08-03 by **reading upstream `main` directly** (contents API,
> `pool_host/common.py`): `ALLOC_MEMORY_FUNCS` still overrides only `"npu"` and
> `"musa"`, with no HIP entry — so main is affected too, and this is stronger
> than the usual "no search hit". [sglang#23361](https://github.com/sgl-project/sglang/pull/23361)
> (**MERGED**, MUSA) is the same one-line dispatch override for the same reason
> and is the shape this patch copies.
>
> **Drop this patch** when a base sglang routes HIP to `alloc_with_pin_memory`;
> the anchor stops matching and the script exits non-zero, so the drop is not
> silent. `ALLOC_MEMORY_FUNCS` still lists only `npu`/`musa` at v0.5.18. Note
> [#32503](https://github.com/sgl-project/sglang/pull/32503) /
> [#32792](https://github.com/sgl-project/sglang/pull/32792) (OPEN, Intel XPU
> HiCache) touch this same dict — expect an anchor conflict, not a fix.

**`patch_hicache_rocm_staged_write_back.py` — RE-ANCHORED, still needed.** The
precondition it guards is intact on the nightly: `pool_host/mla.py` still opts HIP
into the staged JIT with `_is_cuda or _is_hip`, `DSAIndexerPoolHost` still gates on
`_is_cuda` alone, and `HostPoolGroup` still ANDs over its members, so the group
still reads False on ROCm. sglang#30350 is still **CLOSED unmerged** and, checked
in the source rather than from the PR page, still absent: `_is_cuda_alike` appears
nowhere in `mem_cache/`. What moved is only the file — the pool was split out of
`mem_cache/memory_pool_host.py` into `mem_cache/pool_host/dsa.py` — so the script
now accepts both layouts and reads whichever one declares the pool. Its designed
drop signal still fires from the new location: simulating #30350 there makes it
refuse with exit 1 and leave `mla.py` byte-identical.

Whether to retire it is still the **GPU A/B that v0.5.18 already owed**, not an
anchor question, and it is now more worth running: #35233's thread reports the
kernel HiCache path measuring +16% AgentX throughput over `direct` on MI355X
GLM-5.2-MXFP4, so the staged path this patch forgoes is no longer hypothetical.

**Historical, on v0.5.16** — **not preventive there**:
without it the gfx942 prefill scheduler dies on the first reused prefix, so that
image needs it to run kvd. v0.5.18 still has the mixed pool gates, but its
`HybridCacheController._move_write_operation()` checks
`supports_per_pool_backup_indices` and moves host/device indices separately for
each pool. Therefore the gate mismatch alone is no longer proof that v0.5.18
needs this patch. The final v0.5.18 image retains the conservative MLA non-JIT
gate; removing it there requires a direct pristine-MLA JIT A/B, not merely an
anchor match.

> **That repair was proposed but did not merge (re-checked 2026-09-02):**
> [sglang#30350](https://github.com/sgl-project/sglang/pull/30350) `Add HiCache JIT
> test and benchmark for ROCm/HIP CI support` (**CLOSED, unmerged**, `Emmanuel0612`) adds
> `_is_cuda_alike = _is_cuda or _is_hip` and flips exactly the three CUDA-only gates
> (`DSAIndexerPoolHost`, `DeepSeekV4PagedHostPool`, `DeepSeekV4StateHostPool`), so the
> group AND stops reading False on ROCm — **including the V4 stack our patch does not
> cover**. It also teaches `staged_write_back.cuh` to accept kDLROCM/kDLROCMHost (the
> TensorMatcher check that emits our crash) and adds an AMD CI lane for the HiCache
> JIT; the author reports 47/47 on MI355X. Stalled rather than rejected: amd-bot
> called its AMD suites green on 07-09 alongside a merge conflict, conflicts were
> cleared 07-13, nothing since 07-16, and #28534 landed in between. **Our leverage is
> a gfx942 reproduction on that thread, not a competing PR** — it has no MI300X
> datapoint.
>
> **Anchor drift cannot be the drop signal.** #30350 never touches
> `pool_host/mla.py`, so our anchor would keep matching and the patch would keep
> applying on top of the fix — not a crash, since both gates read False again, but a
> silent forfeit of the staged kernel #30350 enables. `check_group_still_poisoned()`
> checks the *precondition* instead: once `DSAIndexerPoolHost` stops gating on
> `_is_cuda` alone, the script refuses and exits 1 telling the operator to drop it.
>
> **Exercised 2026-08-04** against the v0.5.16 *and* current-`main` copies of both
> files, on throwaway trees. Stock: applies, exit 0. Re-run: "already applied", exit
> 0. #30350 simulated by flipping the three CUDA-only gates: refuses, exit 1, with
> `mla.py` byte-identical to pristine. `pool_host/mla.py` removed (a base predating
> staged write-back): tolerated, exit 0. Anchor or pool renamed: exit 1.
>
> **V4 scope rechecked on v0.5.18 (2026-09-03):** the V4 stack does put CUDA-only
> `DeepSeekV4PagedHostPool` children under an always-true `LogicalHostPool` anchor,
> but the per-pool controller path handles those indices independently. A real
> DeepSeek-V4-Pro TP8 run with `hicache-ratio=0.01`,
> `page_first/kernel/write_back` attached all V4 sidecars, served a 5,888-token
> prefix twice, reused 5,632 tokens, and remained alive after write-back. V4 is
> therefore covered on the v0.5.18/gfx950 image; this MLA-only patch neither fixes
> nor participates in that V4 path.

## sglang carried — `patches/sglang_carried/` (**baked by nothing**)

Held, reviewable, and deliberately not built. No Dockerfile copies or runs this
directory, which is why these scripts are not in `sglang_rocm/` beside their
siblings: every other patch directory is consumed by an unconditional
`for f in .../*.py; do python "$f"; done`, so a file placed there is wired into
every image that copies it, with no per-file opt-out.
`patches/sglang_carried/README.md` carries the rule and the exit criteria.

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `sglang_carried/patch_glm_moe_gate_bias_fp32.py` | `MoEGate.__init__` allocates the MoE `e_score_correction_bias` as **bf16** whenever a quant_config is present and `_use_aiter`, and `biased_grouped_topk_gpu` casts it down again at the aiter call. GLM's bias is a narrow band at a large offset, so bf16 cannot hold it: measured on the checkpoints themselves, GLM-5.3 / -MXFP4 collapse **238 distinct fp32 values to 8**, which reorders `noaux_tc` top-k routing | none found | [sglang#37133](https://github.com/sgl-project/sglang/pull/37133) `[GLM-5.2] Keep GlmMoeDsa MoE e_score_correction_bias in fp32` (`xiaobochen-amd`) — same defect, **narrower gate**: see below | no (`xiaobochen-amd`) | OPEN |

**Status: carried, not applied.** Verified as text and as predicate logic only —
applies to copies of `v0.5.18` and `c821c425`, idempotent on re-run, and both
drift cases exit 1 having written neither file. **It has never been executed on a
GPU, and no accuracy or throughput delta has been measured.** Do not read the row
above as validation.

Not wired because the defect is a routing perturbation, not a crash: the server
starts, answers, and reports plausible numbers either way. Rebuilding the engine
image mid-comparison would leave one arm the only one running the fixed router
and destroy the like-for-like property those ratios rest on. The exit criterion
is a hardware run plus a measurement of the thing it claims to change.

> **Why the gate is broader than upstream's.** #37133 gates on
> `any("GlmMoeDsa" in arch for arch in config.architectures)`, which only fires
> when the config object `MoEGate` actually receives carries that arch string.
>
> This script's `_moe_gate_bias_wants_fp32()` is a union of three tests:
> `moe_router_dtype == "float32"` (the model author's own declaration — carried by
> the GLM-5.3 configs and GLM-5.2-FP8, and **sglang has zero references to the
> field**, which is the actual root cause); `model_type` (`glm_moe_dsa`, needed
> because GLM-5.2-MXFP4 and GLM-5.1-FP8 predate that field, and it survives the
> NextN rewrite in `configs/model_config.py:623`); and upstream's arch test, kept
> so this stays a superset of #37133 and becomes a no-op once it lands. Checked
> against every checkpoint available: GLM keeps fp32, non-GLM stays
> byte-identical.
>
> **Both edits or neither, and that is not stylistic.** aiter's launcher
> (`csrc/kernels/topk_softmax_kernels_group.cu:1156`) dispatches on
> `gating_output.dtype()` and then `reinterpret_cast`s the bias pointer to that
> same `scalar_t` **without checking the bias tensor's own dtype**. An fp32 bias
> under a bf16 gating tensor is neither an error nor a cast — it reads fp32 bytes
> as bf16. So applying the `deepseek_v2.py` half alone is strictly worse than the
> defect, and the script plans both files before writing either.
>
> **Drop this patch** when a base sglang keeps the GLM bias fp32 on both sides;
> the replacement text is then already present and the script reports "already
> present" and no-ops. If only #37133's narrower form lands, keep this — its gate
> stays a superset.

## Mooncake C++ — `patches/mooncake_cpp/`

SGLang now builds Mooncake `faae8dd4` directly and carries no private Mooncake
source patch. The one file below is still applied by the vLLM and ATOM builds, and
must be removed there once they enable upstream multi-protocol routing.

`Dockerfile.atom` stayed on `747003c` through the retirement of B.1 and B.3 below,
on the premise that nothing passed the `MOONCAKE_HIP_DMABUF=1` its rebuild was
gated on; `ci.yml`'s `INFERA_E2E_BUILD_ARGS` reaches every engine's build, and at
that ref B.1's wiring is absent, so the image failed its own dma-buf assertion.

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `mooncake_cpp/transfer_engine_impl.diff` | old single-protocol builds preferred HIP IPC for cross-host targets | none | [Mooncake#2753](https://github.com/kvcache-ai/Mooncake/pull/2753) | no | **MERGED**; still needed while the vLLM build leaves `ENABLE_MULTI_PROTOCOL` off |

Two files were deleted once the pinned ref carried them; see **Retired** below.
The upstream source includes a GPU dma-buf chunk test, but it is not registered
with CTest and does not cover a 1 GiB-aligned allocation on ionic. That
provider-specific case still needs a hardware regression test.

## vLLM — `patches/vllm/` (baked by `Dockerfile.vllm`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `vllm/patch_defer_kv_register.py` | registering the Mooncake KV pool before warmup trips a decode-boot crash at high util (`compile_or_warm_up_model` returns None → `AttributeError: 'NoneType' … language_model`); defer to the end of warmup | none found | none found | — | — |
| `vllm/patch_moriio_pagelen.py` | MoRIIO MLA derives per-block transfer size/stride from tensor **shape**, so block-scaled fp8 MLA + DSA transfers the wrong geometry → PD output is wrong while direct prefill is correct | none found | none found | — | — |
| `vllm/patch_moriio_write.py` | in WRITE (push) mode the decode addresses **itself** (`is_producer=True`), the consumer handler asserts, the notify thread dies and the request hangs | `AMD-AGI/Infera#67` | fixed on vLLM `main` / `v0.22.1rc0` **source**, but no AMD ROCm image ships it | no | n/a (source-only) |
| `vllm/patch_sched_guard.py` | decode EngineCore dies on `assert req_id in self.requests` when a KV-xfer-finished event arrives for an already-removed request | `AMD-AGI/Infera#69` | none found | — | — |
| `vllm/patch_vllm_mooncake_blocksize.py` | backends that force a kernel block size of 1 make the connector index logical pages at kernel granularity → RDMA moves empty rows, decode attends over zeros | none found | none found | — | — |
| `vllm/patch_vllm_mooncake_prom_metrics.py` | `MooncakeConnector` lacks `build_prom_metrics`, so `MultiKVConnectorPromMetrics.observe` asserts and kills the engine under `MultiConnector` | none found | internal PR #178 (the sibling fix for `InferaKvdConnector`) | **yes** | **MERGED** |

> The `AMD-AGI/Infera#NN` references above are quoted verbatim from the patches'
> own docstrings, which is the citation form this repo already uses. They resolve
> against an internal tracker, so treat a `404` as expected rather than as a
> stale number.

## vLLM DSv4 — `patches/vllm-dsv4/` (baked by `Dockerfile.vllm`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `vllm-dsv4/patch_aiter_flydsl_moe_memref_bufres.py` | aiter 0.1.16's `fx.ptrtoint` rejects flydsl memrefs → `MLIRError` in the 2-stage MoE GEMM (Kimi-K2.6 int4 W4A16) | none found | none found | — | — |
| `vllm-dsv4/patch_moriio_dsv4_hybrid_blocksize.py` | a global `block_size` equality check kills the prefill worker at KV registration, though DSv4 registers per-layer caches with different block sizes and offsets already use the per-layer map | none found | none found | — | — |
| `vllm-dsv4/patch_moriio_dsv4_noncontig_register.py` | `register_torch_tensor` rejects DSv4's 576B-aligned non-contiguous fp8_ds_mla KV view, and `.contiguous()` would detach from the buffer the forward writes into | none found | none found | — | — |
| `vllm-dsv4/patch_moriio_dsv4_sparse_backend.py` | the generic ROCm selector returns `ROCM_AITER_MLA_SPARSE` (no `fp8_ds_mla`) and raises, killing the prefill worker, though `backend_name` is only a P/D handshake tag | none found | none found | — | — |

### `patches/vllm-dsv4/legacy/` — archival, **not executed by any Dockerfile**

Kept for provenance; both are no-ops on the current verified stack
(vLLM `0.23.1rc1.dev748`, `amd-aiter 0.1.16.post2`).

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `vllm-dsv4/legacy/patch_dsv4_aiter_moe.py` | DSv4 MXFP4 MoE gate-mode plumbing, before aiter carried it | none found | [aiter#3123](https://github.com/ROCm/aiter/pull/3123) `[MoE] Align Swiglu MXFP4 fused quant paths` | no (`XiaobingSuper`) | **MERGED** 2026-05-12 |
| `vllm-dsv4/legacy/patch_dsv4_mhc_aiter.py` + `.diff` | `mhc_pre_gemm_sqrsum_kernel` store race → EngineCore dies | none found | [aiter#3033](https://github.com/ROCm/aiter/pull/3033) `Fix sqrsum store race condition` | no (`kkHuang-amd`) | **MERGED** 2026-05-06 |

## ATOM — `patches/atom/` (baked by `Dockerfile.atom`)

ATOM is an internal engine; there is no public upstream to file against.

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `atom/patch_gdn_pd_state_transfer.py` | hybrid GDN models never transfer the GatedDeltaNet recurrent state in PD, so decode starts from a zero state and cannot recall prompt context (5/5 mixed vs 0/5 PD) | n/a — internal engine | n/a | — | — |
| `atom/patch_minimax_m2_qknorm_rope.py` | stock `minimax_m2.py` fails to load: `get_rope()` rejects `dtype=`, and the TP fused QK-norm kernel needs a batch guard | n/a | n/a | — | — |
| `atom/patch_mooncake_consumer_slot.py` | `UnboundLocalError: consumer_staging_pool_idx` crashes the decode worker on the first PD request for models with slot state but no `slot_regions` | n/a | n/a | — | — |

## hipFile — `patches/hipfile_async/` (baked by `Dockerfile.vllm`)

| patch | fixes | upstream issue | upstream PR | ours? | PR state |
|---|---|---|---|---|---|
| `hipfile_async/hipfile_async.patch` (+ `file_async_fragment.py`, `patch_hipfile_async.sh`) | the ROCm hipFile binding exposes no async API; adds `write_async`/`read_async`, `Stream`, `supports_async()`. Cython stack-locals passed as `&local` die before the driver dereferences them → `bytes_done` reads 0 and intermittent `HipFileException 5022`, so the wrapper calls `libhipfile.so` via ctypes with heap-allocated slots | none found | none found | — | — |

## Not patches

Every file under `patches/` is covered above except these, which carry no fix of
their own: `mooncake_cpp/apply_mooncake_cpp_patches.sh`,
`sglang_dsa/README.md`, `sglang_disagg/README.md` and
`vllm-dsv4/legacy/README.md`.

## Retired

Dropped because upstream carries the fix. Kept as the record of what this repo
once had to work around; the strongest record of the ones we fixed ourselves is
the upstream PR itself.

| patch | fixed | superseded by | dropped at |
|---|---|---|---|
| `sglang/patch_glm52_nextn_quark_exclude.py` | GLM-5.2 MTP `eh_proj` built as an MXFP4 param because the quark-exclude check probed the bare layer prefix → draft weight-load died `3072 vs 6144` | [sglang#30265](https://github.com/sgl-project/sglang/pull/30265) (MERGED), whose `GlmMoeDsaForCausalLMNextN` overrides `_resolve_nextn_quant_config` outright | base v0.5.17. Keeping it would have been worse than useless: GLM no longer reaches the base-class path it edits, while DeepSeek-V4 still does. |
| `mooncake_cpp/rdma_auto_chunk_mr_2017.diff` | buffers over the device `max_mr_size` silently truncated by `ibv_reg_mr` → `IBV_WC_REM_ACCESS_ERR` past the boundary | [Mooncake#2644](https://github.com/kvcache-ai/Mooncake/pull/2644) (MERGED, ours — `jiejingzhangamd`) | Mooncake ref `faae8dd4` |
| `mooncake_cpp/rdma_transport_dmabuf_cmake.diff` | `USE_HIP_DMABUF` defined only on the `transfer_engine` target, so the `ibv_reg_dmabuf_mr` branch compiled out of `rdma_transport` | [Mooncake#2725](https://github.com/kvcache-ai/Mooncake/pull/2725) (MERGED) moved the wiring into `src/CMakeLists.txt`; the vLLM build now passes `-DUSE_HIP_DMABUF` | Mooncake ref `faae8dd4` |

## Maintenance

A row is ready to delete when its upstream PR is **merged and present in the
pinned base**. Merged-but-not-in-base still needs the local patch. The staged
write-back row is the exception: its `MERGED` PR (sglang#28534) is what
*introduced* the defect, so that patch drops on sglang#30350 instead.

Deleting a patch means moving its row to **Retired** in the same commit, not
removing it. A patch that stops applying is not evidence that upstream fixed it —
check the defect in the new base's source before dropping anything.

When adding a patch, add its row here in the same commit, and put the full
argument — evidence, alternatives, how it differs from our own upstream PR — in
the patch's own header. This table is the index, not the record.
