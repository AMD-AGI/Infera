# Notes — Exp 1

## 1. Why patch 1 was reworked at all

**What.** Patch 1 trims DP-attention eager padding off `q_fp8`/`weights` before the aiter
paged-MQA call, then re-attaches it to the top-k result.

**Why it exists.** Under DP-attention the hidden states are padded to the largest token
count across ranks, so `q_fp8` carries more rows than this rank's batch really has, while
`lengths` (`dsa_seqlens_expanded`) is sized to the **real** count. The CUDA path
(`deepgemm_paged_mqa_logits_split`) slices for exactly this reason; aiter instead sizes
its `logits` output from `q_fp8.shape[0]`, so without the same slice the top-k sees
`score.shape[0] != lengths.shape[0]` and asserts
(`Expected lengths.size(0) == B to be true`).

**What was wrong with v1.** Nothing functionally — v1 passed 2540/2540. But its restore was
gated on:

```python
q_offset < q_fp8.shape[0] and topk_result.shape[0] == q_offset
```

The second conjunct is an assert doing a guard's job. If the kernel ever returned a
different row count, the padding would simply not be restored and the caller would receive
a **short tensor** — a wrong answer instead of a crash. That is the single worst property
of v1, and it is what this rework removes.

**How v2 differs** (following upstream PR #32762, `[NPU] Fix DSA eager padding mismatch in
PD MTP warm-up`, open as of 2026-07-30):

1. one boolean `_p1v2_trim` computed before the branch gates **both** the trim and the
   restore, instead of the condition being re-derived at each site;
2. the real row count is named explicitly (`_p1v2_real`) rather than reused from `q_offset`
   inline;
3. the post-kernel row count is **asserted** before padding is re-attached, so a shape
   drift is a loud failure.

**Context.** Reworking a passing patch is only worth it if it cannot regress. That is what
this arm measures, and it does not regress: 4/4 + 32/32 ×2 + 64/64, zero tracebacks.

## 2. Where this deliberately does NOT follow #32762

#32762 sources the real row count from `_original_num_tokens` / `num_token_non_padded_cpu`.
We keep `q_offset = sum(metadata.get_dsa_extend_len_cpu())`. Reasons, in order:

- `_original_num_tokens` **does not exist** in this baseline's `ForwardBatch` (verified
  absent from `forward_batch_info.py` at commit `0b3bb0c`).
- `num_token_non_padded_cpu` **does** exist, but it has **not been established** that it
  equals the indexer's real q-row count on the draft-extend / target-verify paths, where
  MTP contributes several rows per request and `adjust_num_token_non_padded_for_attn_tp`
  rewrites the tensor (but not obviously the `_cpu` mirror).
- `q_offset` **is** the quantity the kernel contract is written against: `lengths` is sized
  to it and the CUDA path slices to it.

So v2 reads `num_token_non_padded_cpu` only to **log** a disagreement under
`SGLANG_DEBUG_DSA_ROWS`. Asserting equality would stake the run on a belief with no
measurement behind it; logging turns it into data.

**Open, and not answered by this kit:** `SGLANG_DEBUG_DSA_ROWS` was **not enabled** during
this run, so we have **no data** on whether the two quantities agree. If a later run shows
they always do, the comparison can be promoted to an assert and #32762's source adopted
verbatim. Until then this is an unmeasured difference from upstream, not a settled
decision.

## 3. Traps hit or avoided this run

### 3.1 `/home` NFS was 100 % full — caches had to move (hit)

**What.** Mid-setup, writes to `/home` began failing with `EDQUOT`; the 10 TB NFS export
was at 100 %.

**Why it matters.** `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` defaulted under
`/home`. A failed JIT cache write is **silent** — it surfaces only as a much slower boot,
which on this stack is indistinguishable from normal (cold start is already ~8 min).

**How.** The whole workspace moved to `/shared_nfs/yihou_exp3way` (11 TB free) and
`boot.sh` now exports both cache dirs there explicitly.

**Context.** No files were deleted to make room — only the write target changed.

### 3.2 The `.pyc` staleness trap (avoided by construction)

A patch script that restores a backup with `shutil.copy2` preserves the original mtime, so
the `.py` can match the cached `__pycache__` entry and CPython runs the **unpatched**
bytecode. This has already invalidated one full experiment on this stack: the source
showed the fix, the runtime behaviour did not.

`apply_arm.sh` therefore purges `__pycache__`, recompiles, and greps the **bytecode** for
each marker. The v2 patch script also calls `os.utime(path, None)` and deletes the
module's `.pyc`.

Verification uses **identifiers**, never `#` comment markers — the compiler discards
comments, so a comment marker in a `.pyc` grep is a guaranteed false negative.

### 3.3 Server logs contain binary bytes (hit repeatedly)

Plain `grep` reports "binary file matches" and `grep -c` returns **0** — which reads as
"no errors" when it actually means "grep gave up". Always `strings <log> | grep` or
`grep -a`. Every log inspection in `REPRODUCE.md` uses `strings` for this reason.

### 3.4 Wrong patch source, caught before it cost a run (hit)

The first version of `apply_arm.sh` sourced the four baseline patches from the ad-hoc
`~/glm52_fix/fix_bug*.py` scripts. Those are from an **earlier round** and encode a
*different* fix for patch 2a than the kit shipped and verified: the scripts change an
empty-batch `.max()` guard and also edit `base_spec_worker.py`, whereas the verified kit
sets `max_seqlen_k = req_to_token.shape[1]` and touches `dsa_backend.py` only.

The kit diffs are the 2540/2540-verified artifacts; the scripts are not. `apply_arm.sh`
now applies the kit diffs, and says so in its header.

### 3.5 A 503 that is not a server failure (hit on the sibling arm)

A router whose circuit breaker is still open returns HTTP 503 in ~0.4 s. That looks
exactly like a dead backend but is not — the tell is the **latency**: a real backend
failure takes seconds to surface, a tripped breaker answers immediately. Restart the
router and re-probe before concluding anything about the servers.

This is why each arm of the three-way run got its own router port (E1: 8110). A breaker
left open by one arm can then never be mistaken for another arm's failure.

## 4. Reading the results honestly

- **`full tok` < request count is not a failure.** `max_new_tokens=512` is a cap; greedy
  decoding hits EOS earlier on some prompts. The criterion is the `ok` column.
- **`acc_len` must be > 1**, or MTP was silently bypassed and the run says nothing about
  speculative decoding. Both the probe and the stress driver print it for this reason.
- **`dp ranks: [0..7]`** confirms the whole DP group served traffic. A pass where only one
  rank worked would not exercise the DP-divergence class of bug this patch set addresses.
- **conc=128 was not run in this arm.** The bug-2b kit used 128; this three-way comparison
  used 32 (plus 64 as headroom). Do not cite this kit as a conc=128 result.
