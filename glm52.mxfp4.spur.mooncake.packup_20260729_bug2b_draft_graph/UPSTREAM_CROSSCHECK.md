# Cross-check: our four patches vs upstream sglang, and vs llying's MI325X run

**Written 2026-07-30**, after reading `AMD-AGI/Infera` branch `llying/dev/glm5p2_fp8_exp`,
file `examples/sglang_glm5.2/REPORT.zh.md` (1287 lines), plus the actual upstream PR/issue
bodies and diffs pulled with `gh` this session.

This file exists because the PR #34 description written on 2026-07-29 contains **three
factual errors about upstream**, all of which came from quoting a prior research pass
without re-checking it. They are corrected below and must be fixed in the PR text.

---

## 0. Confidence labels — read these

Every claim below carries one. The failure this document is correcting was precisely the
loss of these labels when quoting.

| Label | Means |
|---|---|
| **VERIFIED-HERE** | I pulled the original this session (`gh pr diff`, `gh api .../contents/...?ref=<our baseline sha>`) and read it. Treat as fact. |
| **MEASURED-OURS** | Our own run data, raw evidence under `evidence/` or `results/`. Treat as fact. |
| **FROM-LLYING** | Stated in their report; I have not independently verified it. Their report is careful and self-consistent, but it is second-hand here. |
| **INFERRED** | Read-code reasoning, not executed. **Do not cite as fact.** |
| **UNKNOWN** | Genuinely open. Listed so nobody assumes it was settled. |

Our baseline for all `gh api` source reads:
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (v0.5.15.post1).

---

## 1. Upstream issues and PRs, and how each relates

### 1.1 The ones that matter

| # | What | State (2026-07-30) | Relation to us |
|---|---|---|---|
| **#32527** | issue, `[BUG] EAGLE + DP Attention + PD Disaggregation: Deadlock when index_share_for_mtp_iteration is enabled for GLM-5.2` | open, 2026-07-27 | **Same defect as our patch 4.** Reported **two days before** we fixed it. |
| **#32209** | PR, `Fix PD decode hang with DP attention and GLM-5.2 MTP`, +616/−25, 12 files | open, 2026-07-23, CI red (FROM-LLYING) | **Same fix strategy as our patch 4**, better placed. Also carries the padded-rows fix for TRT-LLM. |
| **#30839** | the PR that **introduced** the guard, "Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay" | merged 2026-07-14; v0.5.15 cherry-pick #31083 | FROM-LLYING. Means the guard is **new code**, not legacy — we had assumed legacy. |
| **#31477** | PR, remap IndexShare seeds to decode-local KV slots so fused top-k can stay on under PD | open, CI green, 1 approve (FROM-LLYING) | **The expiry date on the IndexShare workaround.** See §3. |
| **#31683** | PR, `[ROCm][MI35X] Enable GLM-5.2-MXFP4 MTP speculative decoding`, +2479/−102, 34 files | open, 2026-07-18 | Contains an **independent, differently-placed implementation of our patch 1**. Does not touch our patch 2 or 4 sites. |
| **#32175** | PR, `AMD ROCm enablement for GLM-5.x: DSA + prefill-CP + EAGLE and DFlash`, +785/−212 | open, 2026-07-23 | Contains a **semantically identical version of our patch 3**. Its `dsa_backend` hunks do not overlap ours. |
| **#32762** | PR, same padded-rows fix on NPU (`npu_lightning_indexer`) | open, CI red (FROM-LLYING) | Third independent expression of patch 1's bug class. |
| **#32120** | PR, GLM-5.2 MXFP4 1P1D CI recipes | draft, 2026-07-22 | **Zero product code.** Its `1p1d-dp8ep8-mtp.yaml` is our exact topology → upstream *wants* to test this and hasn't. |
| **#32722** | PR, `test: cover GLM-5.2 PD DP attention with MTP` | draft, 2026-07-29 | One 125-line test file, no product code. **Proves no CI covers PD+DPA+MTP today.** |
| **#30854** | issue, `dsa_seed_topk copy_` crashes in PD decode | open (FROM-LLYING) | Same family, unexamined by us. |

### 1.2 #32527 in detail — the one that invalidates "unreported"

VERIFIED-HERE (read the issue body and its one comment).

Reporter: Xavier1994, 2026-07-27. **8× B30Z (Blackwell), GLM-5.2-FP8** — a completely
different platform from both ours (MI355X/MXFP4) and llying's (MI325X/FP8). Their analysis
is the same as ours, arrived at independently:

* same 7-idle + 1-busy split;
* same localization to `not forward_batch.forward_mode.is_idle()` in `draft()`;
* same explanation of why PD differs from single-node (`output_dsa_topk_indices` is never
  set when the prefill worker does not run `_draft_extend_for_prefill`);
* they name the NCCL buffer-size mismatch (eager = per-rank counts, graph = padded to
  `max_len`) as the deadlock mechanism.

Their proposed fix is a **third strategy** we had not considered: instead of dropping to
eager, install a dummy all-zeros seed tensor so the guard's fourth term is false and *all*
ranks stay on the graph path. `draft_forward()` detects the all-zero seed and recomputes.

One comment, from **kpham-sgl** (an sglang maintainer):

> in PD Disagg + Spec, we should also enable Spec in the Prefill worker

**This applies to us.** VERIFIED-HERE in our own launcher: `scripts/pd_leg_spur.sh:72` reads
`if [ "$MTP" = "1" ] && [ "$ROLE" = "decode" ]` — **our prefill leg never enabled MTP.** So
our reproduction has the same configuration gap the maintainer is pointing at, and we are
**not** in a position to offer the counter-example ("we enabled MTP on both legs and still
deadlocked") that PR #34 currently claims. That claim must be removed.

llying **is** in that position: their two leg scripts force MTP to match, their prefill leg
completed PD warmup in 17 s, and the decode leg still deadlocked (FROM-LLYING).

### 1.3 #32209 in detail — what it actually does

VERIFIED-HERE (downloaded and read the full 899-line diff).

**PR #34 currently says #32209 "all-gathers `can_cuda_graph`, and we measured `can_cuda_graph`
to be uniform across ranks, so #32209's vote would not have fixed it." This is wrong.**

It does not vote on `can_cuda_graph`. It adds a **separate, new field**:

```python
# managers/scheduler_components/dp_attn.py
@dataclass
class MLPSyncBatchInfo:
    can_cuda_graph: bool
    can_draft_cuda_graph: bool          # <-- NEW
...
    self.can_draft_cuda_graph = bool(tp0_info[:, 7].min().item())
...
    batch.can_run_dp_draft_cuda_graph = mlp_sync_info.can_draft_cuda_graph
```

fed from the scheduler side by a new spec-worker hook:

```python
# speculative/eagle_worker_v2.py
def requires_dp_attention_eager_forward(self, batch) -> bool:
    if not self.draft_worker.seed_dsa_topk_from_draft_extend: return False
    draft_input = batch.spec_info
    if draft_input is None: return False
    if getattr(draft_input, "future_indices", None) is not None:
        has_seed = getattr(draft_input, "future_dsa_topk_indices_available", False)
    else:
        has_seed = getattr(draft_input, "dsa_topk_indices", None) is not None
    return not has_seed
```

called from `disaggregation/decode.py` before `maybe_prepare_mlp_sync_batch`, and consumed in
`eagle_draft_cuda_graph_runner.can_run_graph()`.

**This is the same idea as our patch 4** — vote draft-graph eligibility across the DP group —
with three engineering differences, all in its favour:

| | #32209 | ours |
|---|---|---|
| collective | rides the **existing** MLP-sync all-gather; none added | adds a 1-element gloo all-reduce per `draft()` |
| placement | scheduler side, before the forward | inside `draft()`, at the decision |
| scope | separates generic vs draft graph vote, so target-verify and draft-extend graphs stay captured | only the draft phase is affected (same effect, different route) |

Its stated motivation for the split: folding the seedless fallback into the generic DP graph
gate cost **9.7% of decode batches** their generic CUDA graph replay.

It also carries four other fixes we do not have: overlap-scheduling stale-seed detection
(`future_dsa_topk_indices_available` must be authoritative), eager draft postprocessing
slicing (`_slice_draft_output_to_local_tokens`), idle-rank DSA index short-circuit, and
TRT-LLM padded query/top-k row trimming — the last being the same bug class as our patch 1.

**Our advantage is size only**: 20 lines in one file, applicable to frozen `release/v0.5.15`.
#32209 is 12 files against `main`.

### 1.4 #31683 — an independent implementation of our patch 1

VERIFIED-HERE (downloaded the 3230-line diff, grepped the overlapping files).

Same bug, different placement. It pushes the slice **one layer down**, into
`jit_kernel/dsa/paged_mqa_logits.py::aiter_paged_mqa_logits`, adding a required `q_offset`
kwarg plus validation, and short-circuits the empty case:

```python
if q_offset == 0:
    return torch.empty((0, max_seq_len), device=q_fp8.device, dtype=torch.float32)
q_fp8 = q_fp8[:q_offset].unsqueeze(1)
weights = weights[:q_offset]
```

and removes the same `not _is_hip` gate on the padding restore that we removed.

That `q_offset == 0` branch is an independent solution to **our Bug 6** — the DP-idle-rank
case that our first `0 < q_offset` lower bound broke. Two independent parties reached the
same contract.

**What it does NOT cover** (VERIFIED-HERE, zero grep matches in its `dsa_backend.py` hunk for
`max_seqlen_k`, `.item()`, `.cpu()`, `page_table`, `repeat_interleave`, `topk_indices`):
our patch 2. And it does not touch the `draft()` guard at all — its `eagle_worker_v2.py` hunk
only adds an idle early-return to `_draft_extend_for_prefill`.

**Its most-cited hunk is dead code for us.** #31683 contains #31071's `eagle_utils.py`
broadcast fix, guarded by `if tp_group.world_size > 1` on `attn_tp_group` under DP-attention.
With `dp_size == tp_size == 8`, `attn_tp_size = tp/dp = 1` (verified at runtime in a prior
session, `compute_dp_attention_world_info(True,0,8,8,1)` → `(0,1,0,8)`), so that broadcast
**never executes** in our config. It fixes divergence *inside* an attention-TP group; ours is
*across* DP ranks.

### 1.5 What upstream's state tells us

Three things worth carrying forward:

1. **Nothing arrives by waiting.** Every item above is open. Our baseline sits on
   `release/v0.5.15`, which has exactly one commit after us (a cmake pin) — so every fix must
   be backported manually or it does not reach us.
2. **The padded-rows bug is being fixed on three backends independently**: NPU (#32762),
   TRT-LLM (#32209 item 4), and ours (aiter/HIP). **Nobody upstream has done the HIP one.**
   That makes our patch 1 a genuine contribution, and #32762 is the shape to mirror.
3. **No CI covers PD+DPA+MTP** (#32722 exists precisely to add it, #32120 to add the recipes).
   That is why this whole family of defects is alive.

---

## 2. What the four patches actually are, and how llying's run handles each

### 2.1 What llying actually applied — VERIFIED-HERE

Read from their branch: `examples/sglang_glm5.2/patch_sglang.sh`,
`infera_2_sglang_prefill.sh`, `infera_3_sglang_decode.sh`.

**They are not running stock sglang plus one flag.** `patch_sglang.sh` applies **two** diffs
on **both** legs, as a hard prerequisite:

1. `sglang_disagg/mooncake_early_send_wait_event.diff` — theirs (PD chunked-prefill KV race);
2. **`sglang_dsa/dsa_indexer_hip_dp_padded_rows.diff` — this is our patch 1.**

And the decode launcher hard-gates on it:

```bash
if [[ "$MTP" != "0" ]] \
   && ! grep -q q_fp8_mqa ".../dsa/dsa_indexer.py"; then
    echo "[decode] MTP=$MTP needs the DSA padded-row patch ..." >&2; exit 1
fi
```

Plus, when `MTP != 0`, both legs append
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'`.

So the accurate statement is: **llying reached PD+DPA+MTP with (our patch 1) + (their mooncake
patch) + (the IndexShare override).** The override replaces only our patch 4.

Note their platform: **GLM-5.2-FP8**, MI325X/gfx942, sglang **v0.5.16**, tilelang DSA
backend, `page_size 64`, `fp8_e4m3` KV. Ours: GLM-5.2-**MXFP4**, MI355X/gfx950, **v0.5.15.post1**.

> Doc inconsistency worth knowing: their §1.2 ends with "状态：已绕过，关掉 MTP 运行
> (`MTP=0`，现在是两个 leg 脚本的默认值)", but §6 item 1 opens with "MTP 已经跑通...PD +
> MTP=1 现在端到端可用". The §1.2 status line reads stale relative to §6 and to the scripts
> (which do implement the MTP=1 path with both prerequisites). Ask them which is current
> before quoting either.

### 2.2 The four patches, one line each

| # | essence | llying's coverage |
|---|---|---|
| **1** | HIP/aiter sizes the MQA-logits output from **DP-padded** rows while `lengths` is sized to **real** rows. Make HIP obey the contract CUDA already obeys. | **They apply our diff verbatim.** Hard prerequisite on both legs. |
| **2a** | `max_seqlen_k = int(seq_lens.max().item())` is a **host sync on a branch only some DP ranks take** → DP collectives desynchronize. Replace with the sync-free `req_to_token.shape[1]`. | **Not applied. They still pass.** Unexplained — see §3.2. |
| **2b** | `page_table_1` has one row per **request**; `topk_indices` has one row per **token** after `_pad_topk_indices`. Under MTP they differ → `assert page_table.shape[0] == topk_indices.shape[0]`. | **Not applied. Probably not reachable for them** — see §3.1. |
| **3** | GLM-5.2 **MXFP4**'s nextn layer is exported bf16 while the model is quantized; the quark `exclude` check must test `model.layers.{N}.eh_proj`, not the bare layer prefix. | **Not applied, and not needed**: they run FP8, which has no such export. #32175 carries the same fix upstream. |
| **4** | The graph/eager choice in `draft()` is computed **per rank from rank-dependent inputs**. Make it a DP-group decision. | **Replaced** by `index_share_for_mtp_iteration=false`, which makes the guard's third term permanently false. |

### 2.3 Minimum viable set

INFERRED, and this is exactly what the experiment in §5 is for:

* on **FP8**: patch 1 + IndexShare override (llying's actual configuration — FROM-LLYING);
* on **MXFP4**: patch 1 + patch 3 + IndexShare override *may* be enough. Patch 2 and patch 4
  would then both be **optional improvements, not requirements**.

If that holds, PR #34's framing ("each one is independently necessary") is wrong and must
change.

---

## 3. Patch 2 and 4 vs IndexShare — the relationship to be verified

### 3.1 Why turning IndexShare off probably bypasses patch 2b — INFERRED

VERIFIED-HERE at our baseline, `layers/attention/dsa/utils.py`:

```python
def should_use_dsa_fused_topk(server_args, seed_dsa_topk_from_draft_extend) -> bool:
    pd_index_share_seed = (
        server_args.disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend)
    # TODO(kpham-sgl): Transfer request-relative IndexShare seeds and remap them
    # to decode-local KV slots so fused top-k can remain enabled under PD.
    return envs.SGLANG_DSA_FUSE_TOPK.get() and not pd_index_share_seed
```

VERIFIED-HERE: `SGLANG_DSA_FUSE_TOPK` defaults to **True** (`environ.py:625`), and
`dsa_backend.py:442-447` resolves `self.use_fused_topk` from this once at init, printing
`"Disabling fused DSA top-k for IndexShare under PD disaggregation."` when it flips off.

VERIFIED-HERE: **both** of our patch 2b call sites are on the **non-fused** arm —
`dsa_backend.py:2133-2140` (`elif self.use_fused_topk: ... else: transform_index_page_table_decode(...)`)
and `:2686-2710` (same shape).

INFERRED chain: IndexShare off → `seed_dsa_topk_from_draft_extend = False` →
`pd_index_share_seed = False` → `use_fused_topk = True` → the `else` arm is never taken →
the assert patch 2b repairs is unreachable.

**Short chain, but not executed.** If true, patch 2b is unnecessary *whenever* fused top-k is
on, and necessary whenever it is off for any reason.

### 3.2 Patch 2a has no such story — UNKNOWN

VERIFIED-HERE, `dsa_backend.py:740-745`:

```python
if forward_batch.seq_lens_cpu is not None:
    max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
else:
    max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num   # <-- the sync
```

This sits in `init_forward_metadata` and depends on `seq_lens_cpu is None` (the spec-v2 relay
batch case). **There is no code path connecting it to `seed_dsa_topk_from_draft_extend`.**

Yet llying does not apply it and does not deadlock here. Three candidate explanations, none
confirmed:

1. their configuration never enters the `seq_lens_cpu is None` arm (v0.5.16 vs 0.5.15.post1,
   tilelang, FP8);
2. turning IndexShare off changes which batch types reach this line;
3. their load never produced the required rank asymmetry.

**This is the single most important open question in this document.**

### 3.3 Our own evidence for patch 2 is weak — MEASURED-OURS, and it is a gap

Re-read of `packup_20260728/dpa_mtp_fix/RESULTS_20260729.md`:

* **patch 4** has a proper same-node differential control: same container, same config, same
  router, same traffic, only the fix reverted (verified absent in bytecode) → **0/4, 120 s
  timeout on request 1**. Solid.
* **patch 1** has an A/B on the same node, and llying reproduced its necessity independently.
  Solid.
* **patch 2b** has a progression table (16-conc crash → after the fix 16/16 → 64/64 → 128/128).
  Suggestive, not a control.
* **patch 2a** has one sentence: *"after this fix no rank ever appears in `dsa_backend` again
  in a py-spy dump, and PD warmup passes on all 8 ranks."* **There was never a run with only
  patch 2a reverted.**

`CLAUDE.md` states 对拍 is mandatory for this bug class. **Patch 2 does not meet that bar.**
This was not disclosed in PR #34 or in the 20260728 kit's summary, and should have been.

### 3.4 Cost of the IndexShare workaround, and its expiry

FROM-LLYING, with the mechanism VERIFIED-HERE:

Under PD, IndexShare's *consumer* is already disabled by `should_use_dsa_fused_topk` — the
seed is produced and then not used by fused top-k. So switching it off currently costs
approximately nothing. Their measurement: accept length **3.78/4** with IndexShare on and off,
matching their single-node baseline; another reproducer on #32209 reported 3.239 vs 3.24.

**But #31477 exists to delete that TODO** — remap the seeds to decode-local KV slots so fused
top-k can stay on under PD. It is CI-green with one approval and may land before #32209. When
it does, IndexShare becomes genuinely useful under PD and the override starts costing
(~3% TPOT, FROM-LLYING). At that point the right move is #32209's vote or #32527's dummy seed,
not the flag.

Also FROM-LLYING, worth remembering: `SGLANG_DSA_FUSE_TOPK=false` does **not** fix the
deadlock. It only touches the consumer; `seed_dsa_topk_from_draft_extend` stays true and the
guard still fires. The switch has to be at the model-config level.

---

## 4. Better fixes for patches 2 and 4

### 4.1 Patch 2b — the current form is a workaround, not a fix

Current: `_glm52_match_page_table_rows()` expands the page table with `repeat_interleave` at
the two call sites, plus two fallbacks (trim when `n_rows > n_topk`, edge-pad otherwise) for
the non-integral-ratio case.

**The fallbacks are the problem.** They make shapes agree without establishing why they
disagreed. If either is ever taken, something is happening that we do not model — and we have
no instrumentation to tell us whether they have been taken. (Nothing crashed in 2540 requests,
which is weak evidence they were not, not evidence they cannot be.)

**Better fix** — mirror what upstream already does on the prefill side. VERIFIED-HERE,
`transform_index_page_table_prefill` takes `output_num_tokens=q.shape[0]` and
`page_table_is_expanded=(is_target_verify() or is_draft_extend_v2())`. The decode entry point
simply has no equivalent. Adding `output_num_tokens` to
`transform_index_page_table_decode` — and having the callee expand — is the change upstream
would accept, and removes the need to guess a ratio at the call site.

### 4.2 Patch 4 — adopt #32209's placement

Current: a fresh 1-element gloo all-reduce inside `draft()`. Correct, measured (LOCAL
diverges 38×, VOTED 0× over 2992 iterations, 190 averted deadlocks, graph retained 98.4%),
but it adds a collective whose cost **we have never measured**.

**Better fix** — #32209's shape: add `can_draft_cuda_graph` to `MLPSyncBatchInfo`, fill it
from a `requires_dp_attention_eager_forward()` hook, ride the existing all-gather, consume it
in `can_run_graph()`. No new collective, and it separates the draft vote from the generic
graph vote so target-verify and draft-extend graphs are not collateral damage.

Cost: it is a multi-file change against a frozen branch. A middle path is to keep our
one-file form for `release/v0.5.15` and propose the #32209 shape upstream — or simply support
#32209 and drop ours once it merges.

### 4.3 Three known-good strategies for the same defect

For the record, so nobody re-derives them:

| layer | strategy | who | trade-off |
|---|---|---|---|
| config | turn off `index_share_for_mtp_iteration` | llying | free today under PD; costs ~3% TPOT once #31477 lands |
| decision | vote draft-graph eligibility across the DP group | ours; #32209 | keeps IndexShare and the draft graph; #32209's placement is better |
| data | install an all-zeros dummy seed so the guard's 4th term is false | #32527 | 3 lines; all ranks stay on the graph path |

---

## 5. The experiment that settles §3

One boot answers every open question above.

**Arm A — minimal set.** On our MXFP4 / 0.5.15.post1 / MI355X stack, PD + DPA8 + MTP(3,1,4):

* apply **patch 1 + patch 3 only** (revert patch 2 and patch 4, verify absent in bytecode —
  `scripts/verify_pyc.sh` with an identifier, not a comment marker);
* add `--json-model-override-args '{"index_share_for_mtp_iteration":false}'` to **both** legs;
* **enable MTP on the prefill leg too** — `pd_leg_spur.sh:72` currently gates it to
  `ROLE = decode`, which is the configuration gap kpham-sgl flagged on #32527. Fixing this is
  required before any claim about prefill-side MTP can be made.

Outcomes:

| result | conclusion | action |
|---|---|---|
| passes | minimum set on MXFP4 is patch 1 + 3 + flag; patch 2 and 4 are optional | rewrite PR #34's "each independently necessary" framing |
| hangs in `init_forward_metadata` | patch 2a is genuinely required on our stack, independent of IndexShare | §3.2 answered; PR #34 keeps patch 2a with a real control at last |
| asserts in `transform_index` | §3.1's fused-top-k reasoning is wrong | re-derive patch 2b's necessity |

**Arm B — patch 2a's missing control.** Full patch set, revert **only** patch 2a. This is the
对拍 that was never run. Cheap once the node is up.

**Arm C (optional) — instrument patch 2b's fallbacks.** Add a counter to the trim / edge-pad
branches of `_glm52_match_page_table_rows` and run the concurrency sweep. If both stay zero
across a full sweep, the fallbacks can be replaced with an assert, which is honest; if either
fires, we have a real unknown to chase.

Cost: two spur nodes, ~8 min cold start each (jobs 11428/11429 were evicted 2026-07-29).

---

## 6. Corrections owed to PR #34

Three statements in the current PR body are wrong and were written by quoting a prior
research pass without re-checking it:

1. *"This mechanism appears unreported."* — **False.** #32527, 2026-07-27, two days earlier,
   independent platform, same analysis.
2. *"#32209 all-gathers that decision ... we measured `can_cuda_graph` to be uniform across
   ranks here, so #32209's vote would not have fixed it."* — **False.** #32209 adds a separate
   `can_draft_cuda_graph` field; it is the same strategy as ours, better placed.
3. The implied counter-example about enabling MTP on both legs — **we never enabled it on the
   prefill leg** (`scripts/pd_leg_spur.sh:72`). Remove it. llying holds that evidence, not us.

Additionally, "each one is independently necessary" for the four patches is
**unsubstantiated** pending §5, and patch 2's lack of a differential control should be stated
in the "Known limits of the validation" section.

---

## 7. Process note — why this document exists

The upstream facts corrected here were all reachable with a single `gh` command. `gh` was
configured and available; the questions were asked repeatedly. They were answered from a prior
session's research notes instead — notes which were themselves careful (every claim carried a
GIT-VERIFIED / SEARCH-ONLY style label) but whose labels were dropped in the retelling.

llying found #32527, #32209, #31477, #30839 and #32762 because they asked upstream directly.
The rule going forward: **any claim of the form "upstream has/has not done X" is either
sourced from an original pulled in this session, or it is explicitly marked second-hand.**
