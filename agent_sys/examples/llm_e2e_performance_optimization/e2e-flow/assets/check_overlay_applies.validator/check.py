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

**`container_roots_from` is a whitelist and it is the third rule.** A patch that
names a host path has confused the machine it was cut on with the image it is to
be applied to, and nothing downstream can see the difference: the mount is made,
the container starts, the engine imports the file it always imported, and the
run reports no regression for a change that was never applied. The whitelist is
loaded from the package rather than from the handoff, so a producer cannot widen
its own bounds by shipping a roots file with `/` in it.
"""

import io
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import patchkit  # noqa: E402
import workset_io  # noqa: E402 — the shared report writer; see _report()
import zone  # noqa: E402


def container_roots(args: dict, reasons: list) -> dict[str, str] | None:
    """The `@NAME@ -> /path` table this plan's paths must resolve under.

    `None` means "use patchkit's own", which is the same file; naming it in
    `args` makes the whitelist a declared part of the validator rather than an
    import-time side effect, and lets a site swap it without editing a body.
    """
    name = args.get("container_roots_from")
    if not name:
        return None
    path = Path(__file__).resolve().parent.parent / "lib" / str(name)
    if not path.is_file():
        reasons.append(
            f"args.container_roots_from names {name!r}, which is not in the package's "
            f"assets/lib/. Without the whitelist this validator cannot tell a container "
            f"path from a host path, so it refuses rather than falling back."
        )
        return None
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {f"@{k}@": v["path"] for k, v in (loaded.get("roots") or {}).items()}


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

    roots = container_roots(args, reasons)
    if args.get("container_roots_from") and roots is None:
        return False

    reasons.extend(
        patchkit.check_mounts(
            plan,
            require_difference=bool(args.get("require_difference", True)),
            roots=roots,
        )
    )
    # The published copies, not the node-local ones: those are on a machine this
    # body cannot reach, and a validator that silently skipped its only checkable
    # evidence would be worse than one that had none.
    reasons.extend(patchkit.check_published_files(plan, result / "files"))

    # `APPLY_MODES_IMPLEMENTED`, not `APPLY_OVERLAY`. This read `!= APPLY_OVERLAY`
    # until 2026-09-04, which was correct while the enum had one value and became
    # a false refusal the moment m3's `patch_in_place` landed: the plan produced
    # by the one mechanism that can install a fragment would have been refused
    # here, one validator after `apply_patch` built it.
    #
    # The mount evidence this body grades is identical for both modes — a diff
    # per mount, a before hash and an after hash — because `apply.py` generates
    # the diff for a whole-file replacement and copies it for a patch, so the
    # published shape does not vary by mode. That is why widening is a one-line
    # change and not a second code path.
    if plan.get("apply_mode") not in patchkit.APPLY_MODES_IMPLEMENTED:
        reasons.append(
            f"the plan's apply_mode is {plan.get('apply_mode')!r}; this stage implements "
            f"{list(patchkit.APPLY_MODES_IMPLEMENTED)}"
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
    results: dict = {}
    findings: dict = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            # Captured and re-echoed: the lines that explain a PASS go to stdout,
            # and a zone keeps no stdout at all. A person watching the run still
            # sees them; so, now, does anyone reading the zone afterwards.
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    results[hid] = check(content, args, reasons)
            except Exception as exc:  # noqa: BLE001
                # A crash is not a refusal. verdict.json cannot express the
                # difference (todo.md T29); this text is the only place it exists.
                results[hid] = False
                reasons.append(f"THIS VALIDATOR DID NOT RUN: {type(exc).__name__}: {exc}")
            sys.stdout.write(buffer.getvalue())
            notes = [ln.strip() for ln in buffer.getvalue().splitlines() if ln.strip()]
            findings[hid] = ([] if results[hid] else list(reasons),
                             notes + (list(reasons) if results[hid] else []))
        findings.setdefault(hid, (list(reasons), []))
        print(f"check_overlay_applies: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    # Before write_verdict, deliberately: a crash in the writer must not be able
    # to take the reasons with it, and the verdict is what the phase reads.
    _report(findings, results)
    zone.write_verdict(results)
    return 0


def _report(findings: dict, results: dict) -> None:
    """`workset_io.write_report`, and never a second implementation of it.

    m3 measured that 16 of 21 validators persist nothing, and seven of those are
    this stage's. That matters most here because **stage 5 has never been
    reached**: every other stage has had refusals to learn from, and m5's first
    one would otherwise arrive with the diagnostics switched off.

    `verdicts` is passed rather than letting the heading infer from `problems`
    being non-empty — these bodies keep informational lines in the same
    `reasons` list, which is the case that made the argument exist.

    Wrapped so that a failure to write the report cannot fail the validation:
    the report is evidence *about* a verdict and must never become the reason
    there is not one.
    """
    try:
        workset_io.write_report("check_overlay_applies", findings, results)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print("check_overlay_applies: could not write the validator report: %s" % exc, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
