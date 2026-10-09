#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Idempotent applier for the GLM-5.2 NextN shared-experts-fusion fix
# (fork commit 4350d37c5). Run it INSIDE your engine container, on BOTH legs
# (prefill + decode), immediately before the engine starts.
#
# WHY A SCRIPT AND NOT `patch`/`git apply`. The class-attribute line moves
# between sglang trees (it shifted 36 lines between the fork's cut and one of
# our targets), so a line-addressed patch lands in the wrong class. This applier
# anchors on the CLASS STATEMENT instead, so it works regardless of line number.
#
# IDEMPOTENT. Safe to re-run against a container that already has the fix.
# FAILS LOUDLY. Exits non-zero if the marker is not present after applying, so
# you never launch an engine that silently lost the fix.
#
# Usage:
#   ./apply_nextn_fusion_fix.sh                 # uses the default path below
#   ./apply_nextn_fusion_fix.sh /path/to/sglang # sglang package root
#   SGLANG_ROOT=/path/to/sglang ./apply_nextn_fusion_fix.sh
#
# The path you pass (or SGLANG_ROOT) is the sglang PYTHON PACKAGE root, i.e. the
# directory that contains `srt/`. The default matches the sglang-rocm images
# (/sgl-workspace/sglang/python/sglang). Find yours with:
#   python3 -c 'import sglang, os; print(os.path.dirname(sglang.__file__))'
set -euo pipefail

# --- resolve the sglang package root -----------------------------------------
SGL="${1:-${SGLANG_ROOT:-/sgl-workspace/sglang/python/sglang}}"
F="$SGL/srt/models/glm4_moe.py"

if [[ ! -f "$F" ]]; then
  echo "[patch] FATAL: $F not found. Pass your sglang package root as \$1 or SGLANG_ROOT." >&2
  echo "[patch]        (the dir containing srt/; e.g. .../python/sglang)" >&2
  exit 1
fi

# --- the fix: GLM-5.2 NextN draft silently loses shared-experts fusion --------
# configs/model_config.py rewrites the draft's arch string to
# "GlmMoeDsaForCausalLMNextN"; the class inherited DeepseekV3ForCausalLMNextN's
# value, so the name compare in deepseek_v2.py could never match and fusion was
# disabled SILENTLY (the gate returns a reason string, not an error).
MARKER='fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"'
if grep -qF "$MARKER" "$F"; then
  echo "[patch] nextn-fusion: already present, skipping"
else
  python3 - "$F" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
# Anchor on the class statement itself, not a line number: the file shifts
# between builds and a line-addressed edit would land in the wrong class.
pat = r'(class GlmMoeDsaForCausalLMNextN\(DeepseekV3ForCausalLMNextN\):\n)'
new = (r'\1'
       '    # The loader rewrites the draft arch to this name, so the inherited\n'
       '    # DeepseekV3 value would never match and silently disable fusion.\n'
       '    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"\n')
out, n = re.subn(pat, new, src, count=1)
if n != 1:
    sys.exit(f"[patch] nextn-fusion FAILED: expected 1 match for the class statement, got {n}. "
             "The class was renamed or its base changed -- re-cut the patch, do not force it.")
open(p, 'w').write(out)
print("[patch] nextn-fusion: applied")
PY
fi

# Fail loudly rather than launching an engine that silently lacks the fix.
grep -qF "$MARKER" "$F" || { echo "[patch] FATAL: nextn-fusion not present after apply" >&2; exit 1; }
