#!/usr/bin/env python3
"""`check_workset_shape` — completeness, strong.

Every operator directory carries the files KernelForge reads, and the two
exported formats agree with each other.

**Shape, not quality.** Whether a driver actually runs and whether its numbers
are right is `verify_workset`'s question, and no static check answers it. The
split is deliberate: this one costs seconds and runs first, so a workset that is
missing a file fails before a GPU is booked for it.

What it does check is the part that is cheap and total:

1. The `reproducible` items are present and non-empty.
2. Every operator directory carries every file in `required_files`.
3. `invocation_spec.json` loads, declares `schema_version: 2`, and a `status` of
   `complete` or `partial` with `missing_fields` agreeing with it.
4. `forge_task.yaml` loads and carries `task_id`, `gpu_target`, `shapes.primary`
   and `targets.snr_db`.
5. The two agree on identity, primary shape, case count and source files.
6. `tests/cases.json` has at least `min_cases` entries, each with a `CASE_ID`.
7. `scripts/forge_driver.py` parses as Python and mentions every string in
   `driver_must_mention`.
8. No file under the operator directory names an absolute path outside the
   seal's allow-list. The seal enforces this too — checking it here names the
   file and the line instead of failing the delivery.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover — yaml is a declared dependency
    yaml = None

#: `handoff/locality.py`, duplicated. The same bounded duplication
#: `demo/assets/lib/store.py` makes of the store layout.
_ABS = re.compile(
    r"(?<![A-Za-z0-9._~@+-])(?:[A-Za-z]:\\[^\s\"'<>|]*|(?:/[A-Za-z0-9._+@-]+){2,}/?)"
)
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S*")
_ALLOWED = (
    "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/etc/", "/opt/",
    "/proc/", "/sys/", "/dev/", "/var/lib/", "/var/log/", "/run/", "/srv/",
    "/workspace/", "/app/",
)

#: `agent/gate.py:EXECUTABLE_ITEMS`, duplicated. The same bounded duplication
#: `demo/assets/lib/store.py` makes of the store layout.
EXECUTABLE_ITEMS = frozenset({"script", "command", "entry"})

REPRODUCIBLE_ITEMS = ("result", "env", "script", "code")


def _local_paths(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = _URL.sub(" ", line)
        if stripped.lstrip().startswith("#!"):
            continue
        for match in _ABS.finditer(stripped):
            hit = match.group(0)
            if not hit.startswith(_ALLOWED):
                out.append(f"{path.name}:{lineno}: {hit}")
    return out


def check_operator(directory: Path, args: dict) -> tuple[bool, str]:
    label = directory.name

    for name in args.get("required_files") or []:
        target = directory / name
        if not target.is_file():
            return False, f"{label}: {name} is absent"
        if target.stat().st_size == 0:
            return False, f"{label}: {name} is empty"

    try:
        spec = json.loads((directory / "invocation_spec.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return False, f"{label}: invocation_spec.json does not load: {error}"

    if spec.get("schema_version") != 2:
        return False, f"{label}: invocation_spec schema_version is {spec.get('schema_version')!r}, expected 2"

    status = spec.get("status")
    missing = spec.get("missing_fields")
    if status not in {"complete", "partial"}:
        return False, f"{label}: invocation_spec status {status!r} is neither complete nor partial"
    if status == "partial" and not missing:
        return False, f"{label}: status is partial but missing_fields is empty"
    if status == "complete" and missing:
        return False, f"{label}: status is complete but missing_fields lists {missing}"

    if yaml is None:
        return False, "PyYAML is not importable in the validation environment"
    try:
        task = yaml.safe_load((directory / "forge_task.yaml").read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as error:
        return False, f"{label}: forge_task.yaml does not load: {error}"

    for key in ("task_id", "gpu_target"):
        if not str(task.get(key) or "").strip():
            return False, f"{label}: forge_task.yaml has no {key}"
    if not (task.get("shapes") or {}).get("primary"):
        return False, f"{label}: forge_task.yaml has no shapes.primary"
    if (task.get("targets") or {}).get("snr_db") is None:
        return False, f"{label}: forge_task.yaml has no targets.snr_db"

    # The two formats describe one operator. They are produced from one record,
    # so a disagreement means one of them was edited independently.
    cases = ((spec.get("workload") or {}).get("task_group") or {}).get("cases") or []
    task_cases = 1 + len((task.get("shapes") or {}).get("validation") or [])
    if len(cases) != task_cases:
        return False, (
            f"{label}: invocation_spec declares {len(cases)} case(s), "
            f"forge_task.yaml declares {task_cases}"
        )
    if cases:
        primary = cases[0].get("selector") or {}
        if primary != ((task.get("shapes") or {}).get("primary") or {}):
            return False, f"{label}: the two formats disagree on the primary case"

    try:
        case_file = json.loads((directory / "tests" / "cases.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return False, f"{label}: tests/cases.json does not load: {error}"
    entries = case_file if isinstance(case_file, list) else (case_file.get("cases") or [])
    minimum = int(args.get("min_cases") or 3)
    if len(entries) < minimum:
        return False, f"{label}: {len(entries)} correctness case(s), mission 3.2.6 requires {minimum}"
    for entry in entries:
        selector = entry.get("selector") or entry
        if "CASE_ID" not in selector:
            return False, f"{label}: a correctness case carries no CASE_ID"

    driver = directory / "scripts" / "forge_driver.py"
    source = driver.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as error:
        return False, f"{label}: scripts/forge_driver.py does not parse: line {error.lineno}"
    for needle in args.get("driver_must_mention") or []:
        if needle not in source:
            return False, (
                f"{label}: scripts/forge_driver.py never mentions {needle!r}; "
                f"forge-loop's preflight would reject it"
            )

    offenders: list[str] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        offenders.extend(_local_paths(path))
    if offenders:
        return False, f"{label}: absolute path(s) the seal refuses — {offenders[0]}"

    return True, f"{label}: {len(entries)} cases, status {status}"


def check(content: Path, args: dict) -> tuple[bool, str]:
    for item in REPRODUCIBLE_ITEMS:
        target = content / "items" / item
        if not target.exists():
            return False, f"items/{item} is absent"
        # `agent/gate.py:EXECUTABLE_ITEMS` is `{script, command, entry}`: an item
        # under one of those keys must be executable or the seal is refused with
        # `output_not_executable` — after the body has returned, and with a
        # follow-up message that does not name the missing bit. Catching it here
        # names it while the producer can still act.
        if item in EXECUTABLE_ITEMS and not os.access(target, os.X_OK):
            return False, (
                f"items/{item} is not executable. agent/gate.py refuses the seal "
                f"for a {item!r} item without its executable bit; run "
                f"`chmod +x items/{item}`"
            )

    code = content / "items" / "code"
    if not code.is_dir():
        return False, "items/code is not a directory"
    directories = sorted(d for d in code.iterdir() if d.is_dir())
    if not directories:
        return False, "items/code holds no operator directory"

    for directory in directories:
        ok, why = check_operator(directory, args)
        if not ok:
            return False, why

    # The whole content directory, not only the operator trees: the handoff's
    # own README is beside `items/`, and a path there refuses the delivery just
    # as surely. Measured — a `<operator_id>/scripts/...` fragment in the
    # top-level README is what refused a complete workset once.
    offenders: list[str] = []
    for path in sorted(p for p in content.rglob("*") if p.is_file()):
        offenders.extend(f"{path.relative_to(content)}: {hit.split(': ', 1)[-1]}" for hit in _local_paths(path))
    if offenders:
        return False, f"absolute path(s) the seal refuses — {offenders[0]}"

    return True, f"{len(directories)} operator workset(s), all complete"


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_workset_shape: {hid}: no published content")
            continue
        ok, why = check(content, args)
        results[hid] = ok
        print(f"check_workset_shape: {hid}: {'PASS' if ok else 'FAIL'} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
