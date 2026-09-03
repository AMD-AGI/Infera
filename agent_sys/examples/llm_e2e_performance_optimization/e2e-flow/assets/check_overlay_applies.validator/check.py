#!/usr/bin/env python3
"""`check_overlay_applies` — completeness, strong.

The mount plan is well formed, the files it would mount are in the handoff, they
hash as the plan says, and they are not the same bytes as what they replace.

**The last rule is the one worth having.** A patch that applies cleanly and
changes nothing gives two arms that are byte-identical, and everything downstream
then passes for the wrong reason: the pipeline compares the stock deployment
against itself and reports no regression. It is a failure mode with no symptom
anywhere else in the graph and it costs one hash comparison here.

This body runs on the login node, which cannot see the node-local files the
mounts actually point at. So it checks the plan's shape and the copies the
handoff carries; whether the node-local file is what ends up inside the running
container is `check_patch_live`'s question, asked of the container itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import patchkit  # noqa: E402
import zone  # noqa: E402


def check(content: Path, args: dict, reasons: list) -> bool:
    result = content / "items" / "result"
    if not (result / "mounts.json").is_file():
        reasons.append("items/result/mounts.json is missing — there is no mount plan")
        return False
    try:
        plan = patchkit.read_mounts(result)
    except ValueError as exc:
        reasons.append(f"mounts.json is not readable as JSON: {exc}")
        return False

    reasons.extend(
        patchkit.check_mounts(plan, require_difference=bool(args.get("require_difference", True)))
    )
    # The published copies, not the node-local ones: those are on a machine this
    # body cannot reach, and a validator that silently skipped its only checkable
    # evidence would be worse than one that had none.
    reasons.extend(patchkit.check_published_files(plan, result / "files"))

    if plan.get("apply_mode") != patchkit.APPLY_OVERLAY:
        reasons.append(
            f"the plan's apply_mode is {plan.get('apply_mode')!r}; this stage only "
            f"implements {patchkit.APPLY_OVERLAY!r}"
        )

    if args.get("compile_python", True):
        for i, entry in enumerate(plan.get("mounts", [])):
            rel = entry.get("repo_relative", "")
            if not rel.endswith(".py"):
                continue
            copy = result / "files" / str(i) / rel
            if not copy.is_file():
                continue
            try:
                compile(copy.read_text(encoding="utf-8"), rel, "exec")
            except (SyntaxError, ValueError) as exc:
                # Worth catching here even though apply_patch already compiled it:
                # the file this reads is the one the handoff published, and a
                # difference between that and what was compiled upstream is
                # exactly the kind of thing a seal-time rewrite could introduce.
                reasons.append(f"the published copy of {rel} does not compile: {exc}")

    for i, entry in enumerate(plan.get("mounts", [])):
        diff = result / "patches" / str(entry.get("patch", ""))
        if not diff.is_file():
            reasons.append(f"mounts[{i}]: the diff {entry.get('patch')!r} is not in the handoff")
            continue
        ok, why = patchkit.looks_like_unified_diff(diff.read_text(encoding="utf-8", errors="replace"))
        if not ok:
            reasons.append(f"mounts[{i}]: {entry.get('patch')}: {why}")

    return not reasons


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
        print(f"check_overlay_applies: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
