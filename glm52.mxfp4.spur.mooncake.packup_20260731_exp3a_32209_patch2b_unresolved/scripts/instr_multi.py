#!/usr/bin/env python3
"""Multi-hypothesis probe: settle FOUR candidates in ONE boot.

WHY MULTI. Rounds 1-4 tested one hypothesis per boot (~25-40 min each) and
three of the four guesses were wrong. Serializing single guesses is the
expensive way to be wrong. This probe instruments four independent sites at
once so a single conc=32 run discriminates all of them.

THE ANOMALY, stable across four runs on three node pairs and a rebuilt image:

    buffer 24 rows = 8 ranks x 3 planned rows  (global_num_tokens_cpu=[3]*8)
    the faulting ranks always deliver EXACTLY 4 rows, never 2, never 5

  The value 4 is invariant while the faulting rank IDs are not (e3a: 0,5,7;
  e3c: 1,2), and 4 matches no rank's real token count. A constant that
  survives rank permutation is a FIXED-SIZE quantity, not a per-rank load.
  The obvious candidate is speculative_num_draft_tokens = 4.

UPSTREAM CONTEXT (queried, not recalled -- gh, 2026-07-31):

  PR #31760 "[DSA] Handle partial-DP padding in Decode page-table transform"
  (OPEN, REVIEW_REQUIRED, 2026-07-20) reports OUR EXACT NUMBERS from an
  independent site: "a 3-row page table and a 4-row top-k tensor whose last
  row is all -1", under GLM-5.2 DSA + MTP decode with partial-DP attention.
  Its position: partial-DP pads the local QUERY layout while the request page
  table stays unpadded, so `_pad_topk_indices` emits trailing -1 rows and the
  equal-rows assert in transform_index_page_table_decode_fast is simply the
  WRONG CONTRACT. Its fix relaxes the assert to `<=` and masks padded rows
  inside the Triton kernel.

  That matters here because BOTH our patch 2b and #32209's reconcile the two
  row counts BEFORE the transform -- ours by expanding the page table, #32209
  by trimming q/top-k. #31760 says the mismatch is legitimate and should be
  handled INSIDE the kernel instead. If #31760 is right, the 3-vs-4 mismatch
  is expected and neither reconciliation is addressing the real defect.

  Also open: #32209 (the hang fix we ported) and #32722 (a deliberately RED
  regression test for it). Three independent open reports of the same
  GLM-5.2 + PD + DPA + MTP cluster.

FOUR HYPOTHESES, four sites, one boot:

  H-A  "4 == speculative_num_draft_tokens": the row count is a fixed draft
       constant leaking into a DP-planned buffer.
       SITE: gather -- log num_draft_tokens and the spec/forward mode.

  H-B  "the rows come from TARGET_VERIFY, not the draft loop": verify runs
       bs*num_draft_tokens rows and is a different forward than the draft
       steps we have been watching.
       SITE: gather -- log forward_mode and capture-mode; DSTEP already
       covers the draft loop, so a gather with NO preceding DSTEP step on
       that rank implicates verify.

  H-C  "hidden_states is stale from a previous iteration": the tensor handed
       to prepare_mlp was produced by an earlier forward with a different
       row count.
       SITE: gather -- log id() of the local tensor plus a monotonic
       sequence number, so a row count that never matches the CURRENT
       forward's inputs is visible as a carried object.

  H-D  "the page-table/top-k mismatch #31760 describes is present here too":
       our reconciliation hides it rather than fixing it.
       SITE: the transform call -- log page_table rows vs topk rows BEFORE
       any reconciliation, which is the exact quantity pair #31760 argues
       about.

Every site is cheap, rank-tagged, capped and flushed. No behaviour changes.

Idempotent; anchors asserted unique; invalidates the .pyc.
"""
import os
import sys

MARK = "GLM52_MULTI"
CAP = 6000

DPA = "/sgl-workspace/sglang/python/sglang/srt/layers/dp_attention.py"
DSA = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

# ---------------------------------------------------------------- gather site
GATHER_HELPER = '''
# GLM52_MULTI -- see /shared_nfs/yihou_exp3way/e3/instr_multi.py
_MULTI_N = [0]
_MULTI_SEQ = [0]
_MULTI_CAP = {cap}


def _multi_gather_log(global_tokens, local_tokens, forward_batch):
    """H-A/H-B/H-C at the all-gather site."""
    _MULTI_SEQ[0] += 1
    if _MULTI_N[0] >= _MULTI_CAP:
        return
    _MULTI_N[0] += 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
        spec = getattr(forward_batch, "spec_info", None)
        # H-A: is the row count a fixed draft constant?
        ndt = getattr(forward_batch, "spec_num_draft_tokens", None)
        if ndt is None and spec is not None:
            ndt = getattr(spec, "num_tokens_per_req", None)
        print(
            "GLM52_MULTI gather seq={{}} rank={{}} local_rows={{}} global_rows={{}} "
            "plan={{}} orig={{}} bs={{}} fwd={{}} capture={{}} "
            "ndt={{}} spec_cls={{}} hs_id={{}} inp_rows={{}}".format(
                _MULTI_SEQ[0],
                rank,
                local_tokens.shape[0],
                global_tokens.shape[0],
                getattr(forward_batch, "global_num_tokens_cpu", None),
                getattr(forward_batch, "original_global_num_tokens_cpu", None),
                getattr(forward_batch, "batch_size", None),
                getattr(forward_batch, "forward_mode", None),
                getattr(forward_batch, "capture_hidden_mode", None),
                ndt,
                type(spec).__name__ if spec is not None else None,
                # H-C: object identity of the tensor being gathered.
                id(local_tokens) % 1000000,
                # the CURRENT forward's own input row count -- if local_rows
                # disagrees with this, the tensor did not come from this forward
                getattr(
                    getattr(forward_batch, "input_ids", None), "shape", [None]
                )[0],
            ),
            flush=True,
        )
    except Exception as _e:
        print("GLM52_MULTI gather probe-error {{}}".format(_e), flush=True)

'''.format(
    cap=CAP
)

GATHER_HELPER_ANCHOR = "def _dp_gather_via_all_gather(\n"

GATHER_ANCHOR = """    if get_attention_tp_size() == 1:
        _e3i_log(global_tokens, local_tokens, forward_batch)  # GLM52_E3INSTR
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return"""

GATHER_REPL = """    if get_attention_tp_size() == 1:
        _e3i_log(global_tokens, local_tokens, forward_batch)  # GLM52_E3INSTR
        _multi_gather_log(global_tokens, local_tokens, forward_batch)  # GLM52_MULTI
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return"""

# ------------------------------------------------------------- transform site
# H-D: the page_table-vs-topk row pair that #31760 argues about, logged BEFORE
# our reconciliation runs. Anchored on the patch-2b trim publish line, which is
# immediately before the page-table build.
DSA_HELPER = '''
# GLM52_MULTI -- see /shared_nfs/yihou_exp3way/e3/instr_multi.py
_MULTID_N = [0]
_MULTID_CAP = {cap}


def _multi_xform_log(page_table, topk_indices, forward_batch):
    """H-D: the exact row pair upstream PR #31760 says must be allowed to differ."""
    if _MULTID_N[0] >= _MULTID_CAP:
        return
    _MULTID_N[0] += 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
        print(
            "GLM52_MULTI xform rank={{}} pt_rows={{}} topk_rows={{}} fwd={{}} bs={{}}".format(
                rank,
                page_table.shape[0] if page_table is not None else None,
                topk_indices.shape[0] if topk_indices is not None else None,
                getattr(forward_batch, "forward_mode", None),
                getattr(forward_batch, "batch_size", None),
            ),
            flush=True,
        )
    except Exception as _e:
        print("GLM52_MULTI xform probe-error {{}}".format(_e), flush=True)

'''.format(
    cap=CAP
)

DSA_HELPER_ANCHOR = "def _p2bv2_trim_decode_dp_padding("

DSA_ANCHOR = """        # Published for the wrapper installed around forward_decode; see the
        # module docstring of patch2b_32209_style.py.
        self._p2bv2_pad_rows = _p2bv2_pad_rows
"""

DSA_REPL = """        # Published for the wrapper installed around forward_decode; see the
        # module docstring of patch2b_32209_style.py.
        self._p2bv2_pad_rows = _p2bv2_pad_rows
        _multi_xform_log(metadata.page_table_1, topk_indices, forward_batch)
"""


def die(msg):
    print("FAIL: {}".format(msg), file=sys.stderr)
    sys.exit(1)


def patch(path, pairs, pyc_name):
    src = open(path).read()
    if MARK in src:
        print("already patched: {}".format(path))
        return False
    for name, anchor, _ in pairs:
        n = src.count(anchor)
        if n != 1:
            die("{} anchor in {} matched {} times, expected 1".format(name, path, n))
    for _, anchor, repl in pairs:
        src = src.replace(anchor, repl, 1)
    open(path, "w").write(src)
    os.utime(path, None)
    pyc = os.path.join(os.path.dirname(path), "__pycache__", pyc_name)
    if os.path.exists(pyc):
        os.remove(pyc)
    print("patched {} with {}".format(path, MARK))
    return True


def main():
    if "_e3i_log" not in open(DPA).read():
        die("instr_e3.py must be applied first (this probe sits beside it)")
    if "_p2bv2_trim_decode_dp_padding" not in open(DSA).read():
        die("patch2b_32209_style.py must be applied first")

    patch(
        DPA,
        [
            ("gather helper", GATHER_HELPER_ANCHOR, GATHER_HELPER + GATHER_HELPER_ANCHOR),
            ("gather call", GATHER_ANCHOR, GATHER_REPL),
        ],
        "dp_attention.cpython-310.pyc",
    )
    patch(
        DSA,
        [
            ("xform helper", DSA_HELPER_ANCHOR, DSA_HELPER + DSA_HELPER_ANCHOR),
            ("xform call", DSA_ANCHOR, DSA_REPL),
        ],
        "dsa_backend.cpython-310.pyc",
    )


if __name__ == "__main__":
    main()
