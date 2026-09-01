#!/usr/bin/env python3
"""`check_patch_shape` — completeness, strong.

Is this a patch set anything could apply? Every rule is decided by looking at a
file that either says the thing or does not, which is what makes it honestly
`strong`; nothing here is approximate.

The arithmetic lives in `assets/lib/patchkit.py` because `apply_patch` runs the
same check before it touches the image. Two implementations of "is this manifest
usable" is one of them being wrong.

The rule most worth having is the cheapest: `apply_mode`. A patch that has to be
compiled cannot be bind-mounted, and if it were quietly accepted it would produce
a deployment running stock code, two identical arms, and a green report saying
the change was safe. Failing at the first validator, naming the mode, is a much
better afternoon.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import patchkit  # noqa: E402
import store  # noqa: E402


def check(content: Path, args: dict, reasons: list) -> bool:
    codes = content / "items" / "codes"
    if not codes.is_dir():
        reasons.append("items/codes/ is missing — the handoff carries no patch set")
        return False
    manifest_file = codes / "manifest.json"
    if not manifest_file.is_file():
        reasons.append("items/codes/manifest.json is missing")
        return False
    try:
        manifest = patchkit.read_manifest(codes)
    except ValueError as exc:
        reasons.append(f"manifest.json is not readable as JSON: {exc}")
        return False

    supported = tuple(args.get("supported_apply_modes") or (patchkit.APPLY_OVERLAY,))
    # The roots the package knows, filtered by whatever this validator was
    # configured to accept. Configured rather than hard-coded because a site that
    # patches a fifth tree should say so in the spec, not in the body.
    allowed = args.get("container_roots")
    roots = None
    if allowed:
        wanted = {str(p).rstrip("/") for p in allowed}
        roots = {k: v for k, v in patchkit.CONTAINER_ROOTS.items() if v.rstrip("/") in wanted}
        unknown = wanted - {v.rstrip("/") for v in patchkit.CONTAINER_ROOTS.values()}
        if unknown:
            reasons.append(
                f"the spec allows container roots the package has no placeholder for: "
                f"{sorted(unknown)} — add them to assets/lib/container_roots.yaml"
            )

    bad = patchkit.check_manifest(manifest, codes, roots=roots, supported_modes=supported)
    reasons.extend(bad)

    for field in args.get("require_fields") or ():
        if field not in manifest:
            reasons.append(f"manifest.json has no {field!r}")

    version = args.get("schema_version")
    if version is not None and manifest.get("schema_version") != version:
        reasons.append(
            f"manifest.json declares schema_version {manifest.get('schema_version')!r}, "
            f"this stage reads {version!r}"
        )

    # Not a failure. A patch without markers can still be proven mounted, only
    # never proven to have run, and that limit belongs in the record rather than
    # in a refusal — a real KernelForge patch is not obliged to carry them.
    if not manifest.get("runtime_marker"):
        print(
            "  note: no runtime_marker declared. check_patch_live will be able to prove "
            "the bytes are mounted and not that the code ran."
        )
    return not reasons


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_patch_shape: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
