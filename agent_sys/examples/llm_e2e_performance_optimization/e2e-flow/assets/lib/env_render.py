#!/usr/bin/env python3
"""Write the `environment.yaml` that mission rule G5 puts on every handoff.

> 整个流程的 handoff 都需要传递 env

One document, one schema (`../schemas/environment.schema.json`), three spellings
of where it lives, decided by the kind's content type (CONTRACT.md §2):

    reproducible      items/env/environment.yaml
    structured_text   items/env/environment.yaml
    code              items/codes/environment.yaml

**Two modes, and the split is the design rather than a convenience.**

``--new``      m1 only. Builds the record from the run's ``E2E_*`` variables
               plus the facts only a bring-up can discover (``gpu_arch``,
               ``image_id``), which arrive as ``--set``.
``--inherit``  every other stage. Copies m1's record forward unchanged, adding
               at most a ``runtime`` override and a ``warnings`` entry.

m1 is the sole producer because the flow's premise is that modules 1–4 are
talking about **one** container. A stage that re-derived the record could differ
from m1's and nothing would notice; a stage that inherits it cannot.

Nothing is written unless it validates. A handoff carrying a malformed
environment record is worse than one carrying none: `check_environment` would
fail either way, but the malformed one looks like a record to a human reader.

    # m1, after bring-up
    env_render.py --new --content-type code --out "$AGENT_SYS_OUTPUT_DEPLOY_KIT" \\
        --set fixed.gpu_arch=gfx950 --set fixed.gpu_count=8 \\
        --set fixed.image_id=sha256:92ed065bdc39 \\
        --set runtime.container="$E2E_CONTAINER" \\
        --set runtime.endpoint="http://$E2E_NODE_IP:$E2E_PORT_ROUTER"

    # every later stage
    env_render.py --inherit "$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml" \\
        --content-type reproducible --out "$AGENT_SYS_OUTPUT_PROFILING_EVIDENCE"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import schema as _schema  # noqa: E402

__all__ = ["ENV_PATH_BY_TYPE", "build", "inherit", "write", "main"]

#: Where the record goes, per content type. Not guessed by the body: a kind
#: whose type changes must change this lookup, not silently stop being checked.
ENV_PATH_BY_TYPE = {
    "reproducible": "items/env/environment.yaml",
    "structured_text": "items/env/environment.yaml",
    "code": "items/codes/environment.yaml",
}

#: `E2E_*` variable -> where it lands. Only the facts a *variable* can carry.
#: `gpu_arch`, `gpu_count` and `image_id` are absent on purpose: they are
#: discovered during bring-up, and a variable holding them would be a claim
#: rather than a measurement.
_FROM_ENV = {
    "E2E_NODE": ("fixed", "node"),
    "E2E_NODE_IP": ("fixed", "node_ip"),
    "E2E_IMAGE": ("fixed", "image"),
    "E2E_MODEL_NAME": ("fixed", "model_name"),
    "E2E_MODEL_PATH": ("fixed", "model_path"),
    "E2E_SERVED_NAME": ("fixed", "served_model_name"),
    "E2E_DEPLOY_MODE": ("fixed", "deploy_mode"),
    "E2E_JOBID": ("runtime", "slurm_jobid"),
    "E2E_CONTAINER": ("runtime", "container"),
    "E2E_TRANSPORT": ("runtime", "transport"),
}
_INT_FIELDS = {"gpu_count", "tp_size", "context_length"}


def _coerce(field: str, value: str):
    if field in _INT_FIELDS:
        return int(value)
    if value in ("null", "~"):
        return None
    return value


def _apply(doc: dict, assignment: str) -> None:
    """`fixed.gpu_arch=gfx950`, or `runtime.ports={"router":8101}` as JSON."""
    key, _, value = assignment.partition("=")
    if not _:
        raise SystemExit(f"--set wants SECTION.FIELD=VALUE, got {assignment!r}")
    section, _, field = key.partition(".")
    if section not in ("fixed", "runtime"):
        raise SystemExit(f"--set section must be `fixed` or `runtime`, got {section!r}")
    if value.startswith(("{", "[")):
        doc[section][field] = json.loads(value)
    else:
        doc[section][field] = _coerce(field, value)


def build(sets: list[str]) -> dict:
    """The `--new` path: the run's variables, plus what bring-up discovered."""
    doc: dict = {"schema_version": 1, "fixed": {}, "runtime": {}}
    for var, (section, field) in _FROM_ENV.items():
        value = os.environ.get(var, "")
        # `none` is this package's sentinel for "the operator said nothing", so
        # it is an absence rather than the string "none" (see shared.yaml).
        if value and value != "none":
            doc[section][field] = _coerce(field, value)
    for name, field in (("E2E_TP", "tp_size"), ("E2E_CTX", "context_length")):
        if os.environ.get(name):
            doc["fixed"][field] = int(os.environ[name])
    ports = {
        key: int(os.environ[var])
        for key, var in (("router", "E2E_PORT_ROUTER"), ("worker", "E2E_PORT_WORKER"), ("etcd", "E2E_PORT_ETCD"))
        if os.environ.get(var)
    }
    if ports:
        doc["runtime"]["ports"] = ports
    doc["runtime"].setdefault(
        "started_at", _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    # `served_model_name` empty means "same as the published name" — resolved
    # here rather than left blank, because a blank one downstream reads as a
    # deployment that registered no name at all.
    if not doc["fixed"].get("served_model_name"):
        doc["fixed"]["served_model_name"] = doc["fixed"].get("model_name", "")
    for assignment in sets:
        _apply(doc, assignment)
    return doc


def inherit(source: pathlib.Path, sets: list[str], warnings: list[str]) -> dict:
    """The `--inherit` path: m1's record, forward, with at most a note added."""
    import yaml

    doc = yaml.safe_load(source.read_text())
    if not isinstance(doc, dict) or "fixed" not in doc:
        raise SystemExit(f"{source}: not an environment record")
    for assignment in sets:
        _apply(doc, assignment)
    for note in warnings:
        field, _, rest = note.partition("=")
        expected, _, actual = rest.partition("!=")
        doc.setdefault("warnings", []).append(
            {"field": field, "expected": expected, "actual": actual, "stage": os.environ.get("E2E_STAGE", "")}
        )
    return doc


def write(doc: dict, out: pathlib.Path, content_type: str) -> pathlib.Path:
    import yaml

    try:
        rel = ENV_PATH_BY_TYPE[content_type]
    except KeyError:
        raise SystemExit(f"--content-type must be one of {sorted(ENV_PATH_BY_TYPE)}, got {content_type!r}")
    # Validate *before* writing. A malformed record on disk looks like a record.
    _schema.validate("environment", doc)
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the environment.yaml every handoff in this flow carries.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new", action="store_true", help="build from E2E_* variables (m1 only)")
    mode.add_argument("--inherit", metavar="PATH", help="carry an upstream environment.yaml forward")
    ap.add_argument("--out", required=True, help="the handoff's content/ directory")
    ap.add_argument("--content-type", required=True, choices=sorted(ENV_PATH_BY_TYPE))
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    ap.add_argument(
        "--warn",
        action="append",
        default=[],
        metavar="FIELD=EXPECTED!=ACTUAL",
        help="record a tolerated difference from the upstream environment (M4.3.5)",
    )
    a = ap.parse_args(argv)

    doc = build(a.set) if a.new else inherit(pathlib.Path(a.inherit), a.set, a.warn)
    try:
        path = write(doc, pathlib.Path(a.out), a.content_type)
    except _schema.SchemaError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "nothing was written: an environment record that does not validate "
            "is worse than none, because it reads like one",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
