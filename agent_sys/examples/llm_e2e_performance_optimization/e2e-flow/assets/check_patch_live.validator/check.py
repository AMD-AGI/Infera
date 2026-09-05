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

**A marker is required by default since 2026-09-04, and the experiment that
changed it is the argument.** The old default was `false`, on the reasoning that
requiring markers would refuse every KernelForge patch that does not know about
this package. Then a control overlay on crsuse2-m2m-047 produced *perfect*
static evidence — in-container hash byte-equal to its own `sha256_patched`, the
file demonstrably holding a 2 ms sleep, the `.pyc` compiled that minute — and
measured **identical to stock**. Rules one and two both passed and neither could
tell *mounted and never executed* from *executed and had no effect*, which is
the single distinction this validator exists to draw. A third overlay carrying
markers settled it in one bring-up: import 18 hits, first_call 8.

So the hole was not a named cost, it was the check quietly not doing its job —
the same shape as `items_schema` validating a filename string. A patch that
truly cannot leave a marker passes `--var require_runtime_marker=false`, still
gets rules one and two, and gets a finding stating exactly what was not shown.
The refusal explains what a marker is and how to write one, because anyone who
meets it is someone who did not know the field existed.
"""

import contextlib
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import workset_io  # noqa: E402 — the shared report writer; see _report()
import zone  # noqa: E402



def _as_bool(value, default: bool) -> bool:
    """A `${...}` arg arrives as a STRING, and `bool("false")` is True.

    Measured on this package's own substitution: `'${bench_rounds:-1}'` resolves
    to `'1'`, never to `1`. So an arg written as `'${require_runtime_marker:-true}'`
    reaches here as `"true"` or `"false"`, and the obvious
    `args.get(name, True)` makes the documented escape hatch impossible —
    `"false"` is a non-empty string and therefore truthy. Caught while writing
    the flag that documents the hatch.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "0", "no", "off", "")


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
        # **The default is now `true`, and it was measured into being.**
        # 2026-09-04 on crsuse2-m2m-047: an overlay whose in-container hash was
        # byte-equal to its own `sha256_patched`, whose file demonstrably held a
        # 2 ms sleep, and whose `.pyc` had been compiled that minute — and the
        # arm measured **identical** to stock. The static evidence was perfect
        # and could not distinguish *mounted and never executed* from *executed
        # and had no effect*, which is the single distinction this validator
        # exists to draw. A third overlay carrying markers settled it in one
        # bring-up: import 18 hits, first_call 8, so the code did run and the
        # sleep was absorbed by overlap scheduling.
        #
        # A default that silently disables the check's own purpose is the same
        # shape as `items_schema` validating a filename string. So: declare a
        # marker, or say `--var require_runtime_marker=false` and own it.
        if _as_bool(args.get("require_runtime_marker"), True):
            ok = _fail(
                reasons,
                "the patch declares no `runtime_marker`, so whether the patched code was "
                "ENTERED cannot be shown. The mounts and the in-container hash are proven, "
                "and both are satisfied by a file that is never executed — measured on "
                "2026-09-04, where a hash-perfect overlay produced numbers identical to "
                "stock.\n"
                "  A marker is two regexes in the patch manifest's `runtime_marker`, matched "
                "against the engine log:\n"
                '    "runtime_marker": {"import": "MYPATCH_IMPORT\\\\s+<op>\\\\s+rev1",\n'
                '                       "first_call": "MYPATCH_FIRST_CALL\\\\s+<op>\\\\s+rev1"}\n'
                "  and two prints in the replacement — one at module scope, one guarded by a "
                "module-level flag at the top of the function the patch replaces. Reference "
                "implementation: `assets/lib/controls/` (the degraded control), which reads "
                "18 import hits and 8 first_call hits on an 8-rank deployment.\n"
                "  If this optimiser genuinely cannot leave a marker, pass "
                "`--var require_runtime_marker=false` — the static proof still runs and this "
                "validator will say in its findings exactly what it could not show.",
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
        print(f"check_patch_live: {hid} {'PASS' if results[hid] else 'FAIL'}")
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
        workset_io.write_report("check_patch_live", findings, results)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print("check_patch_live: could not write the validator report: %s" % exc, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
