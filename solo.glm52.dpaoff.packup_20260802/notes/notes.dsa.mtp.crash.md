# Decode-leg crash: `Expected lengths.size(0) == B` in the DSA indexer under MTP

**When.** 2026-08-01 12:42:27 UTC, 125 s into the Case A full run (`caseA_full`,
ramp 400 + sustain 3600). Image `infera/engine-sglang:merged-e`, decode leg on
chi2867, MTP=1, DP-attention 8/8, ROCm.

**Blast radius.** DP3's scheduler raised, SGLang SIGQUIT'd the whole process
group, the infera wrapper deregistered the worker from etcd, and the router
dropped to `active_workers: 1`. The driver saw 9 mid-stream
`TransferEncodingError` (responses cut off in flight) followed by 103 × HTTP 503
and 2 × HTTP 502. Run aborted at 407 s with 114 errors.

**This is an engine defect, not an operational error.** Both prior incidents this
session were mine (a regex-unsafe `pkill`, an over-large kvd disk budget). This
one is not: no operator action preceded it, and the prefill leg was untouched
and healthy throughout.

## The stack

    deepseek_nextn.py:271       forward            <- MTP draft model
      deepseek_v2.py:2227         self_attn
        forward_mla.py:413          forward_absorb_prepare
          dsa_indexer.py:1978         forward_cuda
            dsa_indexer.py:1017         _get_topk_paged
              dsa_backend.py:353          topk_transform
                dsa_topk_backend.py:89      topk_transform
                  top_k.py:41                 fast_topk_v2
    RuntimeError: Expected lengths.size(0) == B to be true, but got false.

`fast_topk_v2` requires `score` to be `(B, L)` and `lengths` to be `(B)`. The two
disagreed.

## Mechanism — and the patch that already exists for it

The image **already carries** a fix for this bug class, tagged `GLM52_P1V2` in
`dsa_indexer.py` (3 occurrences; verified present in the loaded `.pyc`, not just
in source). Its own comment states the problem exactly:

> Under DP-attention the hidden states are padded to the largest token count
> across ranks, so `q_fp8` carries more rows than this rank's batch really has,
> while `lengths` (`dsa_seqlens_expanded`) is sized to the REAL count. The CUDA
> path (`deepgemm_paged_mqa_logits_split`) slices q/weights for exactly this
> reason; aiter instead sizes its `logits` output from `q_fp8.shape[0]`, so
> without the same slice the top-k below sees `score.shape[0] != lengths.shape[0]`
> and asserts.

So on ROCm the trim is mandatory. And it is *always* on this path:
`paged_mqa_logits_backend.resolve()` returns `AITER` unconditionally under
`is_hip()` — `dsa_paged_mqa_logits_backend='auto'` cannot select anything else.

The trim gates on (`dsa_indexer.py:914-915`):

    _p1v2_real   = q_offset                       # = sum(dsa_extend_len_cpu)   :855
    _p1v2_padded = q_fp8.shape[0]
    _p1v2_trim   = _p1v2_real < _p1v2_padded

**The gap is that `q_offset` is not reliably the real row count on the MTP
draft-extend path.** The patch's own debug block says so, in as many words
(`:942-946`):

> Cross-check against the padding bookkeeping #32762 uses as its source. **NOT an
> assert: it has never been measured to agree with `q_offset` on the MTP
> draft-extend path.** Logged so a later revision can promote it if the data
> supports it.

That is precisely the path that crashed — the traceback enters through
`deepseek_nextn.py`, the MTP draft model. If `q_offset > ` the true count, the
trim under-trims (or is skipped entirely when `q_offset == q_fp8.shape[0]`), and
the mismatch reaches the kernel.

Note the failure is **shape-dependent, therefore load-dependent**: it needs a
draft-extend batch whose padded and real row counts differ in the wrong
direction. That is why it is not deterministic and why it did not fire earlier.

## Why Phase 1 never hit it

The fixlen sweep ran 8 rounds, 660 requests, MTP on, zero occurrences. Two
differences make Case A far more likely to expose it:

1. **Batch-shape diversity.** `--dataset-name random` with
   `--random-range-ratio 1.0` gives every request in a round an identical length,
   so the decode batches are shape-homogeneous. Case A samples input from a
   percentile triple (p50 74K / p90 155K / p99 235K) and output likewise, so
   ragged batches with mixed extend lengths are the norm.
2. **Concurrency profile.** Fixlen holds a fixed concurrency; Case A's population
   breathes (births/deaths, inter-turn gaps), so ranks routinely have unequal —
   including zero — request counts, which is exactly what drives DP padding.

At the crash instant the 8 ranks held 1–3 running requests each with `#token`
from 65K to 187K — heterogeneous across ranks, which is the padding-divergent
condition.

## Confirmed NOT the cause

| hypothesis | evidence against |
|---|---|
| memory pressure / OOM | `token usage` 0.02–0.06 on every rank; no `HSA_STATUS_ERROR`; the crash is a shape assert, not an allocation failure |
| retraction / KV exhaustion | `#retracted-req: 0` on all 8 ranks in the seconds before |
| MTP degenerate/looping | `accept len` 1.77–3.92 across ranks (healthy band; 4.00 would be the repetition-loop tell) |
| my own process management | no operator action in the window; prefill leg unaffected and still serving |
| stale bytecode | `GLM52_P1V2` verified present in the loaded module, count 3 |
| kvd / storage | kvd counters healthy (800 gets / 800 hits / 0 misses); crash is in the attention indexer |

## ROOT CAUSE — captured live on the second occurrence

The crash reproduced on attempt 2 (13:01:04 UTC, 766 s in, same signature) with
`SGLANG_DEBUG_DSA_ROWS=1` on. The last two `[dsa-rows]` lines in the log, on the
ranks that died:

    DP1 [dsa-rows] mode=ForwardMode.IDLE q_fp8=(1, 32, 128) q_offset=2 ntnp=0 agree=False lengths=(2,) -> mqa_q=(1, 32, 128)
    DP6 [dsa-rows] mode=ForwardMode.IDLE q_fp8=(1, 32, 128) q_offset=2 ntnp=0 agree=False lengths=(2,) -> mqa_q=(1, 32, 128)

Read the numbers against the trim logic:

| quantity | value | source |
|---|---|---|
| `_p1v2_padded` = `q_fp8.shape[0]` | **1** | actual query rows present |
| `_p1v2_real` = `q_offset` = `sum(dsa_extend_len_cpu)` | **2** | `dsa_indexer.py:855` |
| `lengths` = `dsa_seqlens_expanded` | **2** | what the kernel will index |
| `mqa_q` handed to aiter | **1** | `_p1v2_trim` was False |

`_p1v2_trim = _p1v2_real < _p1v2_padded` = `2 < 1` = **False**, so no trim
happened. aiter then sized `logits` from `q_fp8.shape[0] = 1`, giving
`score.shape[0] = 1` against `lengths.shape[0] = 2` — exactly the assert.

**The patch guards the wrong direction.** It assumes padding only ever makes
`q_fp8` *longer* than the real count (`real < padded`). Here the inequality is
**reversed**: `q_offset` (2) *exceeds* the rows actually present (1). The trim is
a no-op, the restore block is skipped, and the mismatch flows straight to the
kernel. A `>` case is not handled anywhere on this path.

**It happens in `ForwardMode.IDLE`** — a DP rank with no real work of its own,
participating only to keep the collective in lockstep. That is why `ntnp=0`
while `q_offset=2`: the two bookkeeping sources disagree precisely on idle
ranks, which is the disagreement the patch comment flagged as never measured.

Frequency in the sampled tail: 27,219 lines had `agree=False`, but almost all
are benign (`lengths == q_offset`, trim correct). Only the `mqa_q != lengths`
shape is fatal — **2 occurrences in the last 400 indexer calls**, both `IDLE`
`rows=1 qoff=2 len=2`, and both immediately precede the crash.

The dominant benign pattern is `IDLE rows=4 qoff=1 ntnp=0 len=1` (7,289×): there
`real(1) < padded(4)`, the trim fires correctly, `mqa_q=1` matches `lengths=1`.
So the trim is right for the common idle case and wrong for the inverted one.

## Reproduction and next step

`SGLANG_DEBUG_DSA_ROWS=1` (env, read at `dsa_indexer.py:63`) enables the
`[dsa-rows]` line that prints `q_fp8` shape, `q_offset`, `num_token_non_padded`,
whether they agree, and the `lengths` shape. Decode leg has been restarted as
**TAG p5 with this flag on** so the next occurrence yields the exact numbers
rather than another inference.

`start_leg.sh` now passes it through via `DSA_ROWS=1` (default 0 — it logs once
per indexer call).

## Upstream status

Related upstream work is referenced in the patch comments as **#32762** (NPU,
"same bug class") and **#2782**'s shape discussion. The AMD/ROCm aiter path fix
in this image is local (`GLM52_P1V2`) and incomplete for MTP draft-extend. Worth
filing: the trim's `q_offset` source disagrees with
`forward_batch.num_token_non_padded_cpu` on that path, and the code already
carries the instrumentation to prove it — it just never asserts.
