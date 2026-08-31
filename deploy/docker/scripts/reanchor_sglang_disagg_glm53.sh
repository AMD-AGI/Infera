#!/usr/bin/env bash
# Re-anchor the two patches/sglang_disagg/ scripts onto the GLM-5.3-Flash source
# tree (sglang PR #36607 head, which is stacked on the model PR #36507).
#
# WHY: both scripts are literal-anchor rewriters that exit 1 when an anchor
# drifted, which is correct — an image whose PD path silently corrupts long
# prompts should not ship. Between the v0.5.17 base and the GLM-5.3 branch,
# exactly two anchor LINES moved, and neither fix moved upstream:
#
#   1. mooncake/conn.py    `from typing import List, Optional, Tuple, Union`
#                       -> `from typing import List, Optional, Set, Tuple, Union`
#      Only conn.py grew `Set`; disaggregation/common/utils.py did NOT, so this
#      is a per-file edit, not a global sed. conn.py at the GLM-5.3 head still
#      has no wait_event/synchronize() call, i.e. the fix is still needed.
#
#   2. openai/serving_responses.py  `background=request.background,`
#                                -> `background=request.background and not request.stream,`
#      The bootstrap trio is still absent at that head, i.e. still needed.
#
# The scripts themselves are copied, not edited in place, so the v0.5.17 Kimi-K3
# image keeps building from the untouched originals.
#
# Fails loudly if an old literal is gone: that means upstream moved again (or
# took the fix), and the Dockerfile should be re-derived rather than guessing.
set -euo pipefail

SRC="${1:?usage: reanchor_sglang_disagg_glm53.sh <src-patch-dir> <dst-patch-dir>}"
DST="${2:?usage: reanchor_sglang_disagg_glm53.sh <src-patch-dir> <dst-patch-dir>}"

mkdir -p "$DST"
cp "$SRC"/*.py "$DST"/

python3 - "$DST" <<'PY'
import sys, pathlib

dst = pathlib.Path(sys.argv[1])

EDITS = [
    (
        "patch_mooncake_early_send_wait_event.py",
        # Scoped to the conn.py entry of _EDITS. utils.py keeps the old import.
        '''    "disaggregation/mooncake/conn.py": [
        (
            "from typing import List, Optional, Tuple, Union",
            "from typing import Any, List, Optional, Tuple, Union",''',
        '''    "disaggregation/mooncake/conn.py": [
        (
            "from typing import List, Optional, Set, Tuple, Union",
            "from typing import Any, List, Optional, Set, Tuple, Union",''',
        1,
    ),
    (
        "patch_responses_pd_bootstrap.py",
        "                        background=request.background,\n",
        "                        background=request.background and not request.stream,\n",
        2,
    ),
]

for name, old, new, want in EDITS:
    p = dst / name
    s = p.read_text()
    got = s.count(old)
    if got != want:
        sys.exit(
            f"[reanchor] {name}: found {got}x of the v0.5.17 anchor, want {want}x.\n"
            f"           The script or upstream moved again — re-derive the anchors\n"
            f"           against the GLM-5.3 source instead of shipping a guess."
        )
    p.write_text(s.replace(old, new))
    print(f"[reanchor] {name}: {want} anchor(s) retargeted to the GLM-5.3 tree")
PY
