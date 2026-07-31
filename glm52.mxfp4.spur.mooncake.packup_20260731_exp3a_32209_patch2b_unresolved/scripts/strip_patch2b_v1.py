#!/usr/bin/env python3
"""Remove patch 2b (v1) from dsa_backend.py, leaving patch 2a in place.

The kit ships 2a and 2b in one diff
(`dsa_backend_dp_sync_and_page_table_rows.diff`) because they were developed
together.  Arm E3 wants 2a as-is but 2b in PR #32209's shape, so this script
undoes exactly 2b's three pieces:

  * the `_glm52_match_page_table_rows` helper definition
  * its call at the `forward_decode` decode site
  * its call at the `_forward_trtllm` decode site

and leaves 2a (`max_seqlen_k = self.req_to_token.shape[1]` plus the
DRAFT_EXTEND_V2 host-mirror removal) untouched.

`patch2b_32209_style.py` refuses to run while the helper is still present, so
a failure of this script cannot silently produce a tree with both variants.
"""
import os
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

CALL = """_glm52_match_page_table_rows(
                    metadata.page_table_1, topk_indices
                )"""
CALL_REPL = "metadata.page_table_1"

HELPER_START = "\n\n\ndef _glm52_match_page_table_rows(page_table, topk_indices):"
HELPER_END = "\n    return torch.cat([page_table, pad], dim=0)\n"


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    src = open(TARGET).read()

    if "_glm52_match_page_table_rows" not in src:
        print("patch 2b (v1) already absent; nothing to do")
        return

    if "max_seqlen_k = self.req_to_token.shape[1]" not in src:
        die("patch 2a is not present -- refusing to strip 2b from a tree that "
            "does not have the kit diff applied")

    n = src.count(CALL)
    if n != 2:
        die(f"expected 2 call sites, found {n}")
    src = src.replace(CALL, CALL_REPL, 2)

    i = src.find(HELPER_START)
    if i < 0:
        die("helper definition not found")
    j = src.find(HELPER_END, i)
    if j < 0:
        die("helper end not found")
    src = src[:i] + src[j + len(HELPER_END):]

    if "_glm52_match_page_table_rows" in src:
        die("helper name still present after strip")

    open(TARGET, "w").write(src)
    os.utime(TARGET, None)
    pc = os.path.join(os.path.dirname(TARGET), "__pycache__")
    if os.path.isdir(pc):
        for f in os.listdir(pc):
            if f.startswith("dsa_backend."):
                os.remove(os.path.join(pc, f))
    print("stripped patch 2b (v1); patch 2a retained")


if __name__ == "__main__":
    main()
