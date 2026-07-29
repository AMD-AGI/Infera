#!/usr/bin/env python3
"""Bug 5 fix: pad page_table to match topk_indices in the DSA decode path.

Crash (job 9006, 2026-07-29 13:06, 16 concurrent requests):

    models/deepseek_nextn.py:271        <- the MTP/draft model
      -> layers/attention/dsa_backend.py:2154  forward_decode
      -> dsa/transform_index.py:14             transform_index_page_table_decode
      -> dsa/transform_index.py:138            transform_index_page_table_decode_fast
         assert page_table.shape[0] == topk_indices.shape[0]
    AssertionError   (all 8 DP ranks)

## The mismatch

In `forward_decode` (dsa_backend.py ~2140):

    topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])
    ...
    page_table_1 = transform_index_page_table_decode(
        page_table=metadata.page_table_1,     # NOT padded
        topk_indices=topk_indices,            # padded to q rows
    )

The two operands are sized from different quantities:

  * `metadata.page_table_1` is built in init_forward_metadata as
        req_to_token[forward_batch.req_pool_indices, :max_seqlen_k]
    so its row count is the number of REQUESTS in the batch.
  * `topk_indices` is padded up to `q_nope.shape[0]`, the number of TOKENS.

For plain decode those coincide (1 token per request). Under MTP they do not:
the draft model runs multiple tokens per request, and with `draft_token_num`
tokens per request the q row count exceeds the request count. The callee then
asserts they are equal.

The prefill sibling already handles exactly this: `transform_index_page_table_prefill`
takes an explicit `output_num_tokens=q_nope.shape[0]` and sizes its result
accordingly (dsa_backend.py:1947-1952). The decode path has no such parameter,
which is the gap.

## The fix

Expand `page_table` rows to match `topk_indices` rows before the call, by
repeating each request's row for the tokens belonging to that request. Rows are
laid out token-major per request, so a simple repeat_interleave by the
tokens-per-request factor reproduces the required mapping. When the counts
already agree (plain decode) this is a no-op and costs nothing.

Rows added by `_pad_topk_indices` are pure padding: their topk entries are all
-1, which the triton kernel masks out (`valid_topk_mask = mask & (loaded_topk_indices >= 0)`),
so whatever page-table row they line up against is never read. Only the row
COUNT has to match; the content of padded rows is irrelevant. That is what makes
a repeat-based expansion safe rather than merely convenient.

## Honest scoping note

This crash was first observed under Variant B, which forces the draft path to
run eager. The graph path may have been masking it, so it is NOT yet established
that this bug is independent of the CUDA-graph work. A concurrent run on the
eager control pair is in flight to settle that. The fix below is correct
regardless -- it restores an invariant the callee explicitly asserts -- but the
claim "Bug 5 is unrelated to the graph bug" must wait for that measurement.

Idempotent; handles the stale-.pyc trap.
"""

import os
import py_compile
import shutil
import sys

DSA = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
SUFFIX = ".fix_bug5_orig"
MARKER = "GLM52_BUG5"

# There are TWO call sites with the identical defect (dsa_backend.py:2154 and
# :2724). The traceback named 2154, but 2724 pairs the same unpadded
# metadata.page_table_1 with a topk_indices that _pad_topk_indices just widened
# to q rows, so it would fail the same assert on whichever impl routes there.
# Patch both via a shared helper rather than duplicating the logic inline.
OLD = """            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )"""

NEW = """            page_table_1 = transform_index_page_table_decode(
                page_table=_glm52_match_page_table_rows(
                    metadata.page_table_1, topk_indices
                ),
                topk_indices=topk_indices,
                page_size=1,
            )"""

# Module-level helper, inserted just above the class that owns forward_decode.
HELPER = '''

def _glm52_match_page_table_rows(page_table, topk_indices):
    """GLM52_BUG5: make page_table's row count agree with topk_indices'.

    `metadata.page_table_1` is `req_to_token[req_pool_indices, :max_seqlen_k]`,
    so it has one row per REQUEST. `topk_indices` has just been widened by
    `_pad_topk_indices` to `q.shape[0]`, one row per TOKEN. For plain decode
    those coincide (one token per request); under MTP the draft model runs
    several tokens per request and they do not, so
    `transform_index_page_table_decode_fast` trips

        assert page_table.shape[0] == topk_indices.shape[0]

    Observed as an 8-rank crash under 16 concurrent requests.

    Expanding by repeat_interleave reproduces the token-major-per-request
    layout. Rows added by `_pad_topk_indices` hold all -1, and the triton
    kernel masks those out (`valid_topk_mask = mask & (loaded_topk_indices >= 0)`),
    so their page-table counterparts are never read -- only the row COUNT has to
    match. The prefill sibling handles this with an explicit `output_num_tokens`
    argument; the decode entry point has no equivalent.
    """
    if topk_indices is None or page_table is None:
        return page_table
    n_topk = topk_indices.shape[0]
    n_rows = page_table.shape[0]
    if n_rows == n_topk:
        return page_table
    if n_rows > 0 and n_topk % n_rows == 0:
        return page_table.repeat_interleave(n_topk // n_rows, dim=0)
    # Non-integral ratio: trim or edge-pad so the shapes agree. Any row reached
    # only through padded (-1) topk entries is masked, so its content is moot.
    if n_rows > n_topk:
        return page_table[:n_topk]
    pad = page_table[-1:].expand(n_topk - n_rows, -1)
    return torch.cat([page_table, pad], dim=0)

'''


def bump_and_purge(path):
    """copy2 preserves mtime, so CPython can reuse a .pyc built from the
    UNPATCHED source. This already invalidated one full experiment."""
    os.utime(path, None)
    pycache = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(pycache):
        base = os.path.basename(path).rsplit(".", 1)[0]
        for fn in os.listdir(pycache):
            if fn.startswith(base + "."):
                os.remove(os.path.join(pycache, fn))


def main():
    bak = DSA + SUFFIX
    if "--revert" in sys.argv:
        if os.path.exists(bak):
            shutil.copy2(bak, DSA)
            bump_and_purge(DSA)
            print(f"reverted {DSA}")
        else:
            print("no backup")
        return

    if os.path.exists(bak):
        shutil.copy2(bak, DSA)
        print("restored from backup first")
    else:
        shutil.copy2(DSA, bak)
        print("backed up")

    with open(DSA) as f:
        src = f.read()

    n = src.count(OLD)
    if n != 2:
        print(f"FAIL: anchor found {n} times, expected 2 (sites 2154 and 2724)")
        sys.exit(1)
    src = src.replace(OLD, NEW)

    # Insert the helper above the first top-level class. NB: anchoring on
    # "\nclass " lands BETWEEN a decorator and the class it decorates -- the
    # first class here is preceded by @dataclass(frozen=True), so the decorator
    # ended up applied to this helper, and every scheduler died at import with
    # "AttributeError: 'function' object has no attribute '__mro__'".
    # Walk back over any contiguous decorator lines before inserting.
    idx = src.index("\nclass ")
    lines_before = src[:idx].split("\n")
    back = 0
    while lines_before and lines_before[-1 - back].lstrip().startswith("@"):
        back += 1
    if back:
        idx -= sum(len(lines_before[-1 - i]) + 1 for i in range(back))
    src = src[:idx] + "\n" + HELPER + src[idx:]

    with open(DSA, "w") as f:
        f.write(src)

    bump_and_purge(DSA)

    try:
        py_compile.compile(DSA, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"FAIL syntax: {e}")
        sys.exit(1)

    # py_compile only proves the file PARSES. The first version of this script
    # parsed fine but placed the helper between @dataclass and its class, so the
    # decorator was applied to the function and every scheduler died at import.
    # Actually import the module and call the helper before declaring success --
    # a 10-minute boot is far too expensive a way to discover a bad insert.
    import subprocess

    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '/sgl-workspace/sglang/python');"
            "import torch;"
            "from sglang.srt.layers.attention.dsa_backend import"
            " _glm52_match_page_table_rows as f;"
            "pt = torch.zeros(2, 8, dtype=torch.int32);"
            "tk = torch.zeros(8, 4, dtype=torch.int32);"
            "assert f(pt, tk).shape[0] == 8, 'expand failed';"
            "assert f(pt, torch.zeros(2, 4)).shape[0] == 2, 'noop failed';"
            "assert f(pt, torch.zeros(1, 4)).shape[0] == 1, 'trim failed';"
            "print('IMPORT+BEHAVIOUR OK')",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = (check.stdout + check.stderr).strip().splitlines()
    if "IMPORT+BEHAVIOUR OK" not in check.stdout:
        print("FAIL import/behaviour check:")
        for ln in tail[-8:]:
            print("   ", ln)
        sys.exit(1)
    print("OK import + behaviour check passed")

    print(f"OK applied + syntax clean: {DSA}")
    with open(DSA) as f:
        lines = f.readlines()
    i = next(i for i, ln in enumerate(lines) if MARKER in ln)
    print(f"  patched at line {i + 1}")


if __name__ == "__main__":
    main()
