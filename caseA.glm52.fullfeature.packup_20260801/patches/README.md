# Patches

**One patch, and it is absolutely load-bearing: without it Case A does not run at
all.** This is not cleanup — it is the difference between a 67-minute measured
window and a decode leg that dies twice inside 13 minutes.

Phase 1's three patches (`MC_GID_INDEX` discovery, prefill `mem-fraction-static`
0.88→0.80, kvd `--long-bytes` 512G→64G) are all still in effect here; they ship in
`../fixlen.glm52.fullfeature.packup_20260801/patches/` and are already applied in
this kit's `scripts/`.

---

## 0004 — `dsa_indexer.py`: handle the REVERSED padding case (`GLM52_P1V3`)

**Applied by** `../scripts/apply_p1v3.py`, run inside the **decode** container.
Idempotent, anchors on exact source text, fails loudly if the image differs.
Not a `diff -u` because the target lives inside the image, not in this repo.

**Target:** `/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
(md5 before patch: `632f17acd38737459b43f830ee60ee89`).

### What

Three edits, all in `_get_topk_paged`:

1. `_p1v2_rows = min(_p1v2_real, _p1v2_padded)` becomes the single source of truth
   for the row count, replacing the one-sided `_p1v2_trim = _p1v2_real < _p1v2_padded`.
2. A new `_p1v2_clip` flag marks the inverted case, and when set, the top-k call
   passes `ke_offset=metadata.get_seqlens_expanded()[:_p1v2_rows]` so the lengths
   tensor is clipped to the rows that actually exist.
3. The existing restore-padding assert keys off `_p1v2_rows` instead of
   `_p1v2_real`.

### Why

The image **already carries** a fix for this bug class — `GLM52_P1V2`, verified
present in the loaded bytecode. Its own comment states the mechanism correctly:

> Under DP-attention the hidden states are padded to the largest token count across
> ranks, so `q_fp8` carries more rows than this rank's batch really has, while
> `lengths` (`dsa_seqlens_expanded`) is sized to the REAL count. […] aiter sizes its
> `logits` output from `q_fp8.shape[0]`, so without the same slice the top-k below
> sees `score.shape[0] != lengths.shape[0]` and asserts.

**But it guards only one direction.** `_p1v2_trim = _p1v2_real < _p1v2_padded`
assumes padding always makes `q_fp8` *longer*. On a DP-attention **IDLE** rank under
MTP draft-extend the inequality inverts. Captured live with the image's own
`SGLANG_DEBUG_DSA_ROWS=1`:

    mode=ForwardMode.IDLE q_fp8=(1, 32, 128) q_offset=2 ntnp=0 agree=False lengths=(2,) -> mqa_q=(1, 32, 128)

| quantity | value |
|---|---|
| `_p1v2_padded` = `q_fp8.shape[0]` | **1** — rows actually present |
| `_p1v2_real` = `q_offset` = `sum(dsa_extend_len_cpu)` | **2** |
| `lengths` = `dsa_seqlens_expanded` | **2** |
| rows handed to aiter | **1** (no trim ran: `2 < 1` is False) |

`fast_topk_v2` then gets 1 score row against 2 lengths entries and raises.

The patch's own debug block had flagged exactly this as unverified:

> **NOT an assert: it has never been measured to agree with `q_offset` on the MTP
> draft-extend path.** Logged so a later revision can promote it if the data
> supports it.

The data now says it does not agree. This patch is that later revision.

**A trim cannot fix it** — there are *fewer* query rows than lengths entries, so
there is nothing to cut. Both sides must be reconciled downward to
`min(real, padded)`.

### The symptom it cured

    RuntimeError: Expected lengths.size(0) == B to be true, but got false.
    [DP3 TP3 EP3] Scheduler hit an exception
    Subprocess scheduler_0 (pid=18750) crashed with exit code -3. Triggering SIGQUIT
    ERROR:infera.engine.base:engine subprocess exited (code=-9); deregistering worker

Stack: `deepseek_nextn.py:271` (the **MTP draft model**) → `deepseek_v2.py:2227`
→ `forward_mla.py:413` → `dsa_indexer.py:1978` → `_get_topk_paged` →
`dsa_topk_backend.py:89` → `top_k.py:41`.

Client-visible: 9 mid-stream `TransferEncodingError`, then 103 × HTTP 503, 2 × 502.
Router dropped to `active_workers: 1`.

Reproduced **twice** — 125 s into attempt 1 (stock, TAG p3) and 766 s into attempt 2
(stock + diagnostic, TAG p5). Both crash logs ship in `../logs/`.

### Why it needs all three of MTP + DPA + an idle rank

- **aiter is unconditional on ROCm.** `paged_mqa_logits_backend.resolve()` returns
  `AITER` whenever `is_hip()`; `dsa_paged_mqa_logits_backend='auto'` cannot pick
  anything else. So the buggy path is always taken on these nodes.
- **MTP** supplies the `DRAFT_EXTEND_V2` forward mode where `q_offset` and the real
  row count diverge.
- **DP-attention** supplies the padding and the idle ranks.

Phase 1 ran MTP + DPA for 8 rounds and 660 requests without hitting it, because
`--dataset-name random --random-range-ratio 1.0` makes every request in a round the
same length — batch shapes stay homogeneous and ranks rarely go idle mid-flight.
Case A's breathing session population produces ragged batches constantly.

Frequency: over the last 400 indexer calls before the crash, 2 had the fatal shape.
27,219 lines read `agree=False` but nearly all are benign (`lengths == q_offset`,
trim correct). The dominant benign case is `IDLE rows=4 qoff=1 len=1` (7,289×) —
`real < padded`, trim fires, correct.

### Verification

After the run with the patch: **0** occurrences of `Expected lengths.size` across the
full 4,006 s window, 0 scheduler exceptions, 0 retractions, `active_workers: 2`
throughout.

Verify the patch is in the **loaded** module, not just on disk (stale `__pycache__`
has invalidated an experiment in this tree before):

    docker exec bench_run python3 -c "
    import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect
    print('P1V3:', inspect.getsource(m).count('GLM52_P1V3'))"   # want 3

### Scope and risk — stated plainly

This modifies **engine code inside the image under test**, which is the artifact the
mission is measuring. Consequences, recorded rather than buried:

- Every Case A number in this kit is **`merged-e + P1V3`**, not stock `merged-e`.
- The prefill leg is **unpatched** — the bug is in the MTP draft path, decode-only.
- `ke_offset` is an existing parameter of `DSAIndexerMetadata.topk_transform`
  (`dsa_backend.py:333`) whose sole effect is to override `seq_lens_topk`, so the
  fix rides an intended extension point rather than reaching into internals.
- The original file is preserved at `/tmp/dsa_indexer.py.orig` inside the container
  for a revert-based A/B.

### Upstream status

**Not filed. It should be.** The patch comments reference upstream **#32762** (NPU,
"same bug class") and the local `GLM52_P1V2` is the ROCm/aiter analogue — incomplete
for MTP draft-extend.

The strongest argument for filing: the image already ships the instrumentation that
proves the bug (`SGLANG_DEBUG_DSA_ROWS`, `dsa_indexer.py:63`) and a comment
conceding the two bookkeeping sources were never verified to agree on this path. The
information needed was already there; it just never asserted.
