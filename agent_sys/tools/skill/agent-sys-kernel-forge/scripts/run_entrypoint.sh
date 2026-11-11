#!/bin/sh
# Check the selected session runtime, then delegate to a generated entrypoint.
set -eu
: "${INFERA_REPO:?Set INFERA_REPO to the migrated Infera checkout}"
: "${HYPERLOOM_REPO:?Follow the canonical source/setup guide first}"
: "${AKLR_REPO:?Follow the canonical source/setup guide first}"
: "${FORGE_BACKEND:?Set a plain kernel backend such as triton or hip}"
: "${FORGE_ENTRYPOINT:?Generate a current workset or AKLR launcher first}"
: "${FORGE_PREFLIGHT_JSON:?Set an output file in the task directory}"
FORGE_PYTHON="${FORGE_PYTHON:-python3}"
FORGE_SESSION_JSON="${FORGE_SESSION_JSON:-$FORGE_PREFLIGHT_JSON}"
export FORGE_PYTHON FORGE_SESSION_JSON HYPERLOOM_REPO AKLR_REPO
[ -f "$FORGE_SESSION_JSON" ] || {
  echo "Prepare the session record and runtime through the source/setup guide first" >&2
  exit 2
}
case "$FORGE_ENTRYPOINT" in
  /*) ;;
  *) echo "FORGE_ENTRYPOINT must be absolute" >&2; exit 2 ;;
esac
"$FORGE_PYTHON" - "$FORGE_ENTRYPOINT" <<'CHECK'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
if "kernelforge.cli" not in source or "kernel-agents" in source:
    sys.exit("Generate a current Hyperloom launcher; the archived command is not supported")
CHECK
"$FORGE_PYTHON" "$INFERA_REPO/agent_sys/tools/skill/agent-sys-kernel-forge/scripts/check_forge.py" \
  --hyperloom "$HYPERLOOM_REPO" --aklr "$AKLR_REPO" --backend "$FORGE_BACKEND" \
  --session "$FORGE_SESSION_JSON" --out "$FORGE_PREFLIGHT_JSON"
exec "$FORGE_PYTHON" - "$FORGE_PREFLIGHT_JSON" "$FORGE_ENTRYPOINT" "$@" <<'LAUNCH'
import json
import os
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
env = os.environ.copy()
env["HYPERLOOM_REPO"] = report["hyperloom_repo"]
env["AKLR_REPO"] = report["aklr_repo"]
env["AKLR_KB_ROOT"] = report["aklr_knowledge"]["path"]
os.execvpe(sys.argv[2], sys.argv[2:], env)
LAUNCH
