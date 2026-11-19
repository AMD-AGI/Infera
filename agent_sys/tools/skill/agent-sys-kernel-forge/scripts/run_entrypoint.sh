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
"$FORGE_PYTHON" - "$INFERA_REPO/agent_sys/tools/skill/agent-sys-kernel-forge/scripts/check_forge.py" \
  "$HYPERLOOM_REPO" "$AKLR_REPO" "$FORGE_BACKEND" \
  "$FORGE_SESSION_JSON" "$FORGE_PREFLIGHT_JSON" <<'PREFLIGHT'
import json
import os
from pathlib import Path
import subprocess
import sys

helper, hyperloom, aklr, backend, session, output = sys.argv[1:]
installed_source = os.environ.get("FORGE_INSTALLED_SOURCE")
if not installed_source:
    # Read a saved identity before preflight can replace this same report/session.
    try:
        for name in dict.fromkeys((output, session)):
            path = Path(name)
            if path.is_file():
                installed = json.loads(path.read_text(encoding="utf-8")).get("installed")
                if installed:
                    installed_source = installed["source_path"]
                    if not isinstance(installed_source, str) or not installed_source:
                        raise ValueError("installed.source_path must be a nonempty string")
                    break
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        sys.exit(f"Forge entrypoint: cannot reuse installed-source identity: {error}")
command = [sys.executable, helper, "--hyperloom", hyperloom, "--aklr", aklr,
           "--backend", backend, "--session", session, "--out", output]
if installed_source:
    command.extend(["--installed-source", installed_source])
# The helper still verifies source bytes, module origin and CLI capabilities.
sys.exit(subprocess.call(command))
PREFLIGHT
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
env["FORGE_INSTALLED_SOURCE"] = report["installed"]["source_path"]
os.execvpe(sys.argv[2], sys.argv[2:], env)
LAUNCH
