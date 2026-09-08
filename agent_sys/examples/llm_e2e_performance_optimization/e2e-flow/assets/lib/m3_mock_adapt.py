#!/usr/bin/env python3
"""MOCK-MAP (A) and CONTRACT 3.4 for m3's two `structured_text` kinds.

PARKED HERE BECAUSE /home IS OUT OF QUOTA. Destination when space returns:
  e2e-flow/assets/lib/m3_mock_adapt.py

    m3_mock_adapt.py --kind kernel_worklist --out "$AGENT_SYS_OUTPUT_KERNEL_WORKLIST"

**An adaptation is a step after the copy, not a variant of the copy.** `mock.sh`
puts the sealed bytes down faithfully and on purpose; this adds only what the
sealed artefact could not have carried. Same division `deploy_and_prove.task/
mock_adapt.sh` makes for m1 and `build_workset.task/mock_adapt.py` for the
workset — **this is the third owner to need it, and it is one lesson rather than
three**: the mock copies bytes, and every per-kind adaptation is a step each
owner wires themselves.

Three gaps, each measured against the run at
`ws_handoff_refine/runroot/runs/20260903T150709-4be7ad`, not guessed:

1. **`items/env/environment.yaml` is absent.** `check_environment` FAIL. The
   sealed handoff predates the record entirely (mission G5 is this round's
   rule), so there is nothing to copy and it has to be rendered.
2. **`items/text.json` has no `schema_version`.** `check_worklist_shape` FAIL,
   and **a separate item rather than a consequence of the first** — reproduced
   on its own against the run's staged content before anything was changed. It
   is the one field this package's schemas require that the sealed documents
   lack, which the schema commit called out as "the single field a mock renders
   rather than copies".
3. **`items/schema` is the *sealed* schema, not this package's.** CONTRACT 3.4
   requires the carried copy to be byte-identical to `assets/schemas/`, and the
   sealed one is the older, weaker document — every interior `{"type":
   "object"}`. Copying it forward would make the artefact self-describing and
   describing something other than what graded it.

**Inherited verbatim, no `--set`.** In mock mode there is no bring-up of ours to
describe, and the upstream record *is* the deployment the sealed artefact would
have run against. m2 established that and it is right.

**The content type is named per kind rather than looped over.** A kind whose
content type changes then fails loudly here instead of quietly writing to the
path the old type used.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

PKG = pathlib.Path(os.environ.get("AGENT_SYS_TASK_PACKAGE")
                   or os.environ.get("AGENT_SYS_DEMO_PACKAGE")
                   or pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PKG / "assets" / "lib"))

#: kind -> (content type, the inputs whose environment record it may inherit).
#: Ordered: the first that is staged and carries a record wins. Named rather
#: than discovered, so a wiring change surfaces here.
KINDS = {
    "kernel_worklist": ("structured_text", ("AGENT_SYS_INPUT_PROFILING_EVIDENCE",)),
    "operator_identity": ("structured_text", ("AGENT_SYS_INPUT_KERNEL_WORKLIST",
                                              "AGENT_SYS_INPUT_DEPLOY_KIT")),
}

#: Where a staged input keeps its record, by content type. `deploy_kit` is
#: `code` and therefore the odd one.
_ENV_REL = ("items/env/environment.yaml", "items/codes/environment.yaml")


def _die(message: str) -> None:
    sys.exit(f"m3_mock_adapt: {message}")


def _interpreter() -> str:
    """An interpreter that can run `env_render.py`, which validates before it
    writes and therefore needs `yaml` and `jsonschema`.

    A task body never reaches `AGENT_SYS_DEMO_PYTHON` (`cli/main.py:668` puts it
    in `validation_env` only), so a bare `python3` is the policy PATH's. Probed
    rather than assumed — m1 measured that `/usr/bin/python3` here has `yaml`
    and `jsonschema`, so it will usually be fine, and "usually" is the reason to
    check rather than the reason not to.
    """
    for candidate in (os.environ.get("AGENT_SYS_DEMO_PYTHON"), sys.executable, "python3"):
        if not candidate:
            continue
        if subprocess.run([candidate, "-c", "import yaml, jsonschema"], capture_output=True).returncode == 0:
            return candidate
    _die("no interpreter here can import yaml and jsonschema; env_render validates before it writes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--out", required=True, help="the handoff's content directory")
    a = ap.parse_args()

    content_type, upstreams = KINDS[a.kind]
    out = pathlib.Path(a.out)
    items = out / "items"
    if not items.is_dir():
        _die(f"{out} has no items/ — was mock.sh run, and did it write here?")

    # (2) `schema_version`, first in the document so a reader meets it first.
    document = items / "text.json"
    if not document.is_file():
        _die("items/text.json is absent; the sealed structured_text artefact should carry one")
    data = json.loads(document.read_text(encoding="utf-8"))
    if "schema_version" not in data:
        document.write_text(json.dumps({"schema_version": 1, **data}, indent=2) + "\n", encoding="utf-8")

    # (3) the schema this package grades against, byte for byte.
    import schema as schema_lib

    (items / "schema").write_bytes(schema_lib.schema_path(a.kind).read_bytes())

    # (1) the environment record, inherited from whichever declared input has one.
    source = next(
        (pathlib.Path(os.environ[var]) / rel
         for var in upstreams if os.environ.get(var)
         for rel in _ENV_REL
         if (pathlib.Path(os.environ[var]) / rel).is_file()),
        None)
    if source is None:
        _die(f"no staged input among {list(upstreams)} carries an environment record. "
             f"In mock mode the upstream is the deployment this artefact would have run "
             f"against; without it there is nothing honest to write.")
    rendered = subprocess.run(
        [_interpreter(), str(PKG / "assets/lib/env_render.py"),
         "--inherit", str(source), "--content-type", content_type, "--out", str(out)],
        capture_output=True, text=True)
    if rendered.returncode != 0:
        _die(f"env_render --inherit failed:\n{rendered.stderr[-800:]}")

    print(f"m3_mock_adapt: {a.kind} — schema_version, items/schema, and environment inherited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
