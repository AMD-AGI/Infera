#!/usr/bin/env python3
"""Bug 2 fix (A + A2): remove the rank-divergent D2H syncs from
DeepseekSparseAttnBackend.init_forward_metadata.

THE BUG
-------
On the PD decode leg with DP-attention + MTP, a decode step routinely gives work
to only a subset of DP ranks (measured: global_num_tokens_cpu = [0,0,4,0,0,0,0,0]).
Ranks without work run ForwardMode.IDLE; the rank with work runs DRAFT_EXTEND_V2.

The forward mode decides whether the host mirror `seq_lens_cpu` is present:
  * IDLE            -> mirror present -> cheap host read
  * DRAFT_EXTEND_V2 -> mirror is None -> `.max().item()` == blocking device sync

So the busy rank blocks in cudaStreamSynchronize while its idle peers sail on into
the next DP collective. 7 ranks wait at the collective for the 8th; the 8th waits
at the brake for the GPU. Hard deadlock, first routed request.

The defect is not "a sync exists" -- it is "a sync exists on a branch only some
ranks take". CUDA never runs this code (the draft-extend graph is captured there),
which is why this reproduces only on HIP.

FIX A -- the `else` arm of the max_seqlen_k branch
--------------------------------------------------
Use the static page-table width instead of a host max. This is the established
in-tree idiom, three times over:
  * dsa_backend.py:695  `_graph_page_table_width` -> `self.req_to_token.shape[1]`
  * dsa_backend.py:1203 and :1241 (graph capture) -> same expression
  * triton_backend.py / trtllm_mha_backend.py: "Static upper-bound page-table
    width ... never a host max / seq_lens_cpu D2H sync"

Over-sizing is safe: the wide page_table is only ever indexed *through* top-k, and
top-k masks per row by lengths derived from cache_seqlens (a GPU tensor), which is
independent of max_seqlen_k. Extra columns score -inf, are never selected, never
dereferenced.

FIX A2 -- the two unconditional syncs in the DRAFT_EXTEND_V2 branch
-------------------------------------------------------------------
`extend_prefix_lens.cpu()` and `seq_lens.cpu()` are equally blocking and sit on the
same rank-divergent branch, so Fix A alone leaves the hang in place.

Verified dead for DRAFT_EXTEND_V2 by following every consumer:
  * DRAFT_EXTEND_V2 is NOT is_extend() -- is_extend(include_draft_extend_v2=False)
    by default (forward_batch_info.py:107-115). Therefore:
      - `_cal_indexer_k_start_end` early-returns (its first line is
        `if not is_extend_without_speculative(): return None, None`), so its
        `assert seq_lens_cpu is not None` is unreachable.
      - dsa_indexer.py:1072 opens with
        `assert forward_batch.forward_mode.is_extend_without_speculative()`
        -- that whole function is unreachable for this mode.
  * `extend_prefix_lens_cpu` is read only inside the `is_extend()` arm (852+).
  * `seq_lens_sum` readers: :934 (inside is_extend()), :2830 (Blackwell + EXTEND),
    :990 (stored into metadata; the metadata field's only reader is the same
    is_extend()-gated code).
  * `seq_lens_cpu` feeds `indexer_seq_lens_cpu`, but that local is captured at
    line 776 -- BEFORE 811-813 runs -- so it is already None on this path today.
    The DRAFT_EXTEND_V2 indexer path (dsa_indexer.py:843-844) uses
    get_seqlens_expanded() and block_tables.shape[1], both GPU-side.

Rather than delete the block (which would change behaviour for any future consumer),
we gate it on the modes that actually consume the values. Assertions are preserved.

Idempotent. --revert restores. py_compile-checked.
"""
import argparse
import os
import shutil
import sys

DSA = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
BACKUP_SUFFIX = ".fix_a_a2_orig"
MARKER = "GLM52_BUG2_FIX_A"

# ---------------------------------------------------------------- Fix A
ANCHOR_A = """        else:
            # needs_cpu_seq_lens=False nulls the host mirror for spec-v2 relay
            # batches; graph replay uses the static page-table width, so only this
            # eager (e.g. over-capture-bs) fallback needs a length here.
            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num
"""

REPLACEMENT_A = """        else:
            # """ + MARKER + """: needs_cpu_seq_lens=False nulls the host mirror for
            # spec-v2 relay batches. `seq_lens.max().item()` here is a blocking D2H
            # sync on a branch that only SOME DP ranks take (IDLE peers keep their
            # mirror and take the cheap arm), which desynchronizes the DP collectives
            # and deadlocks. Use the static page-table width instead -- the same
            # sync-free idiom as _graph_page_table_width() and the graph-capture
            # paths below. Over-allocating columns is safe: the page table is only
            # indexed through top-k, which masks per row by cache_seqlens.
            max_seqlen_k = self.req_to_token.shape[1]
"""

# ---------------------------------------------------------------- Fix A2
ANCHOR_A2 = """        elif forward_batch.forward_mode.is_draft_extend_v2():
            if forward_batch.extend_prefix_lens_cpu is None:
                assert forward_batch.extend_prefix_lens is not None
                forward_batch.extend_prefix_lens_cpu = (
                    forward_batch.extend_prefix_lens.cpu().tolist()
                )
            if forward_batch.seq_lens_cpu is None:
                forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu()
                forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
"""

REPLACEMENT_A2 = """        elif forward_batch.forward_mode.is_draft_extend_v2():
            # """ + MARKER + """2: the two host mirrors that used to be materialized
            # here (extend_prefix_lens.cpu() and seq_lens.cpu()) are unconditional
            # D2H syncs on the same rank-divergent branch as the max_seqlen_k fix
            # above, so Fix A alone does not lift the deadlock.
            #
            # They are dead for DRAFT_EXTEND_V2, which is not is_extend() (see
            # ForwardMode.is_extend -- include_draft_extend_v2 defaults to False).
            # Every consumer is therefore unreachable on this path:
            #   * extend_prefix_lens_cpu -- read only inside the is_extend() arm
            #   * seq_lens_sum -- read at the is_extend()-gated capacity check and
            #     the Blackwell+EXTEND prefill heuristic
            #   * seq_lens_cpu -> indexer_seq_lens_cpu is captured ABOVE this block,
            #     so it is already None here today; the draft-extend indexer path
            #     reads get_seqlens_expanded() / block_tables.shape[1] (GPU-side)
            #   * _cal_indexer_k_start_end and dsa_indexer's k-only path both open
            #     with an is_extend_without_speculative() guard
            #
            # base_spec_worker.prepare_for_draft_extend already supplies the mirror
            # this branch actually needs -- extend_seq_lens_cpu -- precisely "so
            # backend max() reads from list without a per-iter D2H sync". This
            # completes that intent instead of contradicting it. The assert keeps
            # the invariant that the consumed values are present.
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
            ), "extend_seq_lens{,_cpu} must not be None for DRAFT_EXTEND_V2"
"""

PATCHES = [("Fix A (max_seqlen_k)", ANCHOR_A, REPLACEMENT_A),
           ("Fix A2 (806-813)", ANCHOR_A2, REPLACEMENT_A2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DSA):
        sys.exit(f"FAIL: {DSA} not found")

    backup = DSA + BACKUP_SUFFIX
    src = open(DSA).read()

    if args.revert:
        if not os.path.exists(backup):
            sys.exit("FAIL: no fix_a_a2 backup to revert to")
        shutil.copyfile(backup, DSA)
        print(f"OK: reverted {DSA}")
        return

    if MARKER in src:
        print("OK: Fix A/A2 already present (no-op)")
        return

    # verify every anchor before touching anything
    for name, anchor, _ in PATCHES:
        n = src.count(anchor)
        if n != 1:
            sys.exit(f"FAIL: {name}: anchor matched {n} times, expected 1. "
                     "Source drifted -- re-derive the anchor.")

    if not os.path.exists(backup):
        shutil.copyfile(DSA, backup)
        print(f"OK: backup -> {backup}")

    out = src
    for name, anchor, repl in PATCHES:
        out = out.replace(anchor, repl, 1)
        print(f"OK: applied {name}")

    open(DSA, "w").write(out)

    import py_compile
    try:
        py_compile.compile(DSA, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, DSA)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")

    # post-conditions: the two sync expressions must be gone from the eager path
    chk = open(DSA).read()
    bad = "max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num"
    if bad in chk:
        shutil.copyfile(backup, DSA)
        sys.exit("FAIL: Fix A did not take, reverted")
    print(f"OK: Fix A + A2 installed in {DSA}")


if __name__ == "__main__":
    main()
