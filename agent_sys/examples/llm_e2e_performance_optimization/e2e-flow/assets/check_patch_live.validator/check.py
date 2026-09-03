#!/usr/bin/env python3
"""`check_patch_live` — trustworthiness, strong.

The one validator in this package that does not ask whether a record is
complete. It asks whether the thing the record describes actually happened.

**The failure it exists for.** A patch that is mounted and never executed
produces two arms with identical numbers and a comparison that reports "no
regression" — a green result for a change that was never tested. Nothing else in
the graph can see it: the deployment is healthy, the eval scores well, the replay
is fast, and every one of those is true of a deployment running stock code.

Three rules, in increasing order of what they prove.

**`docker inspect` shows the mount.** Cheapest, and it catches the plan not
reaching the `docker run` at all.

**The file hashes, inside the running container, to what the plan says.** This
is the rule people expect to be unnecessary and it is the only static one that
works: a bind mount does not change the path inside the container, so
`sglang.srt.models.glm5_next.__file__` reads identically on a patched and a stock
deployment. `__file__` proves nothing; the hash proves the bytes.

**The declared markers appear in the engine log.** The import marker says the
interpreter compiled the mounted bytes. The first-call marker says a real request
entered the patched code. Only the second answers the question this stage was
built to ask.

`runtime_marker` is optional in the contract, so a patch that declares none gets
the first two rules and a finding saying what could not be shown. That is a
deliberate hole and it is named rather than papered over: requiring markers would
mean refusing every KernelForge patch that does not know about this package.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def read_tsv(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    return [
        line.split("\t")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def check(content: Path, args: dict, reasons: list) -> bool:
    env = content / "items" / "env"
    try:
        record = json.loads((env / "deployment.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"deployment.json is not readable: {exc}")

    if record.get("arm") != "patched":
        return _fail(
            reasons,
            f"this validator is bound to the patched arm and the record says {record.get('arm')!r}",
        )

    overlay = record.get("overlay") or {}
    planned = overlay.get("mounts") or []
    if not planned:
        return _fail(reasons, "the deployment record carries no mount plan")

    ok = True

    # ---- 1. the mounts are on the container ----------------------------------
    if args.get("require_docker_mounts", True):
        try:
            inspected = json.loads((env / "docker_mounts.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ok = _fail(reasons, f"docker_mounts.json is not readable: {exc}")
            inspected = []
        destinations = {m.get("Destination") for m in inspected if isinstance(m, dict)}
        readonly = {
            m.get("Destination"): not m.get("RW", True)
            for m in inspected
            if isinstance(m, dict)
        }
        for entry in planned:
            # The plan is in placeholder form and docker reports real paths, so
            # the comparison is on the tail. Matching on the suffix rather than
            # expanding keeps this body free of the expansion table, which cannot
            # be published and therefore cannot be assumed present beside a
            # handoff someone else is reading.
            tail = entry["container_path"].split("@", 2)[-1].split("/", 1)[-1]
            hit = next((d for d in destinations if d and d.endswith("/" + tail)), None)
            if hit is None:
                ok = _fail(
                    reasons,
                    f"{entry['container_path']} is in the plan but the container has no mount "
                    "ending in that path",
                )
            elif not readonly.get(hit, False):
                ok = _fail(reasons, f"{hit} is mounted read-write; the plan says read-only")

    # ---- 2. the bytes inside the container -----------------------------------
    if args.get("require_container_hashes", True):
        observed = {row[0]: row[1] for row in read_tsv(env / "container_hashes.tsv") if len(row) >= 2}
        if not observed:
            ok = _fail(
                reasons,
                "container_hashes.tsv is empty — the bring-up recorded no hash from inside the "
                "running container, which is the only static proof a mount took",
            )
        for entry in planned:
            tail = entry["container_path"].split("@", 2)[-1].split("/", 1)[-1]
            hit = next((k for k in observed if k.endswith("/" + tail)), None)
            if hit is None:
                ok = _fail(reasons, f"no in-container hash was recorded for {entry['container_path']}")
                continue
            got = observed[hit]
            if got == "MISSING":
                ok = _fail(reasons, f"{hit} could not be hashed inside the container")
            elif got != entry["sha256_patched"]:
                ok = _fail(
                    reasons,
                    f"{hit} hashes {got[:12]}… inside the container but the plan says "
                    f"{entry['sha256_patched'][:12]}… — the deployment is not running the patch",
                )
            elif got == entry.get("sha256_stock"):
                ok = _fail(reasons, f"{hit} inside the container is still the stock file")

    # ---- 3. did it run? ------------------------------------------------------
    declared = overlay.get("runtime_marker") or {}
    hits = {row[0]: int(row[2]) for row in read_tsv(env / "marker_hits.tsv") if len(row) >= 3}
    if not declared:
        # **The hole, named in the findings rather than papered over.**
        # `require_runtime_marker` is `false` in the step yaml, so an
        # optimisation that declares no marker still gets the two static rules —
        # and this validator says what it could not show rather than reporting a
        # pass that means more than it does. Set it `true` at a site that can
        # require its optimiser to leave a marker; the cost is refusing every
        # KernelForge patch that does not know about this package.
        if args.get("require_runtime_marker", False):
            ok = _fail(
                reasons,
                "the patch declares no runtime_marker and args.require_runtime_marker is true. "
                "The mounts are proven and whether the patched code was ENTERED is not.",
            )
        else:
            print(
                "  note: the patch declared no runtime_marker. The mounts are proven; whether the "
                "patched code was ENTERED cannot be shown from this record, and two arms that "
                "never diverge would look identical for that reason rather than for a good one."
            )
    else:
        for key in ("import", "first_call"):
            if key not in declared:
                continue
            count = hits.get(key)
            if count is None:
                ok = _fail(reasons, f"the patch declares a {key!r} marker and none was searched for")
            elif count < 1:
                ok = _fail(
                    reasons,
                    f"the {key!r} marker never appears in the engine log. "
                    + (
                        "The mounted module was never imported."
                        if key == "import"
                        else "The patched code was never entered by a request, so this arm "
                        "measured the same code path as the stock arm."
                    ),
                )
            else:
                print(f"  {key} marker: {count} hit(s)")
        # A declared regex that is not a regex would silently never match.
        for key, pattern in declared.items():
            try:
                re.compile(str(pattern))
            except re.error as exc:
                ok = _fail(reasons, f"runtime_marker.{key} is not a valid regex: {exc}")

    return ok


def main() -> int:
    args = zone.args()
    results = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_patch_live: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
