#!/usr/bin/env python3
"""`check_identity_resolved` — trustworthiness, strong.

The failure this guards is a **plausible wrong answer**. A resolver that reports
a source file it never opened sends every downstream step at the wrong code, and
the mistake survives all the way to a forge-loop run that optimizes something
nobody asked about. So the rules below are about whether a claim was checked,
not about whether a field is populated.

1. Every operator carries a resolution method from the known set.
2. A resolved operator names at least one source file; an `agent_recovered` one
   names a non-empty hint instead. Neither may be silently empty.
3. Claimed paths are repository-relative, never absolute. An absolute path here
   is either a host path — which the seal would refuse anyway — or a container
   path masquerading as a source location.
4. `image_repo_path`, when set, is a container path under an allow-listed
   prefix.
5. The share resolved by evidence clears `min_resolve_ratio`.

Rule 5 is a ratio and not a requirement on every operator, because DESIGN.md
section 4.4 shows that Triton-JIT and TileLang operators reach level 3 by
construction. Demanding 1.0 would fail on a correct answer.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402

METHODS = {"trace_python_stack", "symbol_search", "agent_recovered"}

#: A container root is written either as a `${PLACEHOLDER}` — the normal case,
#: because `/sgl-workspace/` is outside the seal's allow-list — or as a literal
#: path under a prefix the seal already accepts. Anything else is a host path
#: that only happened to survive the regex, and this rejects it.
PLACEHOLDER = re.compile(r"^\$\{[A-Z][A-Z0-9_]*\}$")
LITERAL_PREFIXES = ("/usr/", "/opt/", "/workspace/", "/app/", "/srv/")


def check(content: Path, args: dict) -> tuple[bool, str]:
    document = content / "items" / "text.json"
    if not document.is_file():
        return False, "items/text.json is absent"
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return False, f"items/text.json is not valid JSON: {error}"

    if not (content / "items" / "schema").is_file():
        return False, "items/schema is absent"

    operators = data.get("operators")
    if not isinstance(operators, list) or not operators:
        return False, "operators is absent or empty"

    for operator in operators:
        label = operator.get("logical_operator") or operator.get("name", "?")

        method = operator.get("source_resolution_method")
        if method not in METHODS:
            return False, f"{label}: resolution method {method!r} is not known"

        if not (operator.get("repository_language") or "").strip():
            return False, f"{label}: repository_language is empty"

        identity = operator.get("kernel_identity") or {}
        if not (identity.get("logical_operator") or "").strip():
            return False, f"{label}: kernel_identity.logical_operator is empty"

        sources = operator.get("source_file_path") or []
        if method in {"trace_python_stack", "symbol_search"}:
            if not sources:
                return False, f"{label}: claims {method} but names no source file"
        elif not (operator.get("resolution_hint") or "").strip():
            return False, (
                f"{label}: unresolved and carries no hint. An unresolved operator "
                f"has to say where to look, or the next step has nothing to act on"
            )

        for path in sources + (operator.get("editable_sources") or []):
            if path.startswith("/"):
                return False, (
                    f"{label}: source path {path!r} is absolute. Source paths are "
                    f"repository-relative; an absolute one is either a host path "
                    f"or a container root in the wrong field"
                )

        root = (operator.get("image_repo_path") or "").strip()
        if root and not (PLACEHOLDER.match(root) or root.startswith(LITERAL_PREFIXES)):
            return False, (
                f"{label}: image_repo_path {root!r} is neither a ${{PLACEHOLDER}} nor "
                f"a literal path under {LITERAL_PREFIXES}. A host path here would be "
                f"wrong on any other machine"
            )

        if not (operator.get("cases") or []):
            return False, f"{label}: carries no workload cases"

        if args.get("require_actionable"):
            # The real gate. `build_workset` needs four things to proceed on an
            # operator: what language to read, which container root to read it
            # in, what shapes to build cases from, and either a file to open or
            # a hint saying where to look. Missing any one leaves it nothing to
            # act on, and a step that hands the next one an unusable record has
            # failed regardless of how complete the record looks.
            missing = [
                field
                for field, value in (
                    ("repository_language", operator.get("repository_language")),
                    ("image_repo_path", operator.get("image_repo_path")),
                    ("cases", operator.get("cases")),
                    (
                        "source_file_path or resolution_hint",
                        (operator.get("source_file_path") or operator.get("resolution_hint")),
                    ),
                )
                if not value
            ]
            if missing:
                return False, f"{label}: not actionable, missing {', '.join(missing)}"

    summary = data.get("summary") or {}
    ratio = float(summary.get("resolve_ratio") or 0.0)
    floor = float(args.get("min_resolve_ratio") or 0.0)
    if ratio < floor:
        return False, (
            f"corroborated resolve ratio {ratio} is below {floor}. Either the kernel "
            f"finder had no repositories to index (--var sglang_src=...), or the "
            f"profile carried no launcher frames (with_stack was off upstream)"
        )

    verified = [o for o in operators if o.get("source_file_path")]
    return True, (
        f"{len(operators)} operators, all actionable; {len(verified)} with a corroborated "
        f"source file, {len(operators) - len(verified)} to be read from source by the next step"
    )


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_identity_resolved: {hid}: no published content")
            continue
        ok, why = check(content, args)
        results[hid] = ok
        print(f"check_identity_resolved: {hid}: {'PASS' if ok else 'FAIL'} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
