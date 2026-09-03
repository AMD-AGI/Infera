#!/usr/bin/env python3
"""`check_env_report_shape` — completeness, **strong**.

The handoff is one well-formed JSON report covering all seven capabilities, each
with a level, a status, a token of the right shape, a written account of how it
was obtained, and the proof key its capability owes. Plus a README with four
non-empty sections and an install report that is not empty.

**It checks no token against anything.** That is `check_capabilities_genuine`,
it costs subprocesses, and it runs second precisely so that a report which is
malformed — the cheap failure — is rejected before anything is started
(`validator` spec §5.3). The two are separate so neither borrows the other's
claim.

The readme beside this file argues each rule. This module is their exact form.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import envchk  # noqa: E402 — the path insert above is what makes it importable
import zone  # noqa: E402

#: The README sections. The first two are `structured_text`'s own
#: (`handoff/content.py:CONTENT_TYPES`) and the last two are the kind's
#: `readme_sections` in `steps/check.yaml`. Listed here rather than derived,
#: because deriving them would mean this body reading `handoff`'s table, and a
#: body is not a place to put a second reader of another package's constants.
README_SECTIONS = ("Purpose", "Schema", "Method", "Limits")

#: A well-formed token. The value is `check_capabilities_genuine`'s business;
#: the *shape* is this one's, because a `how` field pasted into `token` is a
#: shape fault and reporting it here saves a subprocess.
TOKEN = re.compile(r"\AENVCHK-[A-Z_]+-[0-9a-f]{12}\Z")

#: `nonce_digest`, as `envchk.nonce_digest` produces it.
DIGEST = re.compile(r"\A[0-9a-f]{12}\Z")

#: A heading line, and a fence marker. Same definitions as
#: `single_real_task`'s shape check, for the same reason: a document that is
#: four `##` lines with nothing under them is exactly what a content floor is
#: for, and counting the headings would let it pass.
HEADING = re.compile(r"\A\s*#")
FENCE = re.compile(r"\A\s*(?:```|~~~)")

#: Placeholder tokens — a section that was templated and not written. The `<…>`
#: form is matched only when the brackets wrap something that is not a URL or an
#: e-mail address, so a legitimate `<https://…>` does not trip it.
PLACEHOLDER = re.compile(
    r"""
      \b (?: TODO | TBD | FIXME | XXX ) \b
    | to \s+ be \s+ filled \s+ in
    | < (?! [A-Za-z][A-Za-z0-9+.-]* : // ) (?! [^<>@\s]+ @ ) [^<>\n]{2,} >
    """,
    re.IGNORECASE | re.VERBOSE,
)


def content_lines(text: str) -> list[str]:
    """Lines that carry content: non-blank, not a heading, not a fence marker."""
    return [
        line
        for line in text.splitlines()
        if line.strip() and not FENCE.match(line) and not HEADING.match(line)
    ]


def check_readme(text: str | None, floor: int) -> list[str]:
    """The handoff's own README: four sections, substance, no placeholder."""
    if text is None:
        return ["README.md: missing or unreadable"]
    faults = []
    for section in README_SECTIONS:
        if not re.search(rf"^\s*#{{1,6}}\s*{section}\b", text, re.IGNORECASE | re.MULTILINE):
            faults.append(f"README.md: no `## {section}` heading")
    lines = content_lines(text)
    if len(lines) < floor:
        faults.append(f"README.md: {len(lines)} content lines, needs {floor}")
    found = PLACEHOLDER.search(text)
    if found:
        faults.append(f"README.md: unfilled placeholder {found.group(0)!r}")
    return faults


def check_section(label: str, section: object, min_how: int) -> list[str]:
    """One capability's entry. Every fault it has, not just the first.

    Every rule here is a key that is present or absent, a string that matches a
    pattern or does not, or a count against a floor. Nothing in it is a
    judgement, which is what makes this validator `strong` without qualification.
    """
    capability = envchk.BY_LABEL[label]
    where = f"capabilities.{label}"
    if not isinstance(section, dict):
        return [f"{where}: is {type(section).__name__}, needs an object"]

    faults: list[str] = []

    level = section.get("level")
    if level != capability.level:
        faults.append(
            f"{where}.level: {level!r}, needs {capability.level!r} — "
            f"{capability.what}. The level is which install route is being "
            f"claimed, not a label"
        )

    status = section.get("status")
    if status not in envchk.STATUSES:
        faults.append(f"{where}.status: {status!r}, needs one of {list(envchk.STATUSES)}")

    # **A token is required exactly when `status` is `ok`, and forbidden
    # otherwise.** A token beside a non-`ok` status is a contradiction the
    # report should not be able to carry: it says both that the capability did
    # not work and that its salt was reached.
    token = section.get("token")
    if status == "ok":
        if not isinstance(token, str) or not TOKEN.match(token):
            faults.append(f"{where}.token: {token!r} is not ENVCHK-<LABEL>-<12 hex>")
        elif not token.startswith(f"ENVCHK-{label.upper()}-"):
            faults.append(
                f"{where}.token: {token!r} carries another capability's label. "
                f"Two capabilities have two salts and reporting one for both is "
                f"the mistake this pairing exists to catch"
            )
    elif token is not None:
        faults.append(f"{where}.token: {token!r} beside status {status!r}; needs null")

    how = section.get("how")
    if not isinstance(how, str):
        faults.append(f"{where}.how: missing")
    else:
        weight = len("".join(how.split()))
        if weight < min_how:
            faults.append(
                f"{where}.how: {weight} non-whitespace characters, needs {min_how}. "
                f"This is the field a human reads when the token mismatches"
            )

    # The proof key, for the sections that owe one. **Only when `status` is
    # `ok`** — a capability that did not arrive has nothing to prove, and
    # demanding a proof object for it would push the author towards inventing
    # one, which is worse than the absence.
    expected = envchk.PROOF_KEYS.get(label)
    if status == "ok" and expected is not None:
        proof = section.get("proof")
        if not isinstance(proof, dict):
            faults.append(f"{where}.proof: missing or not an object")
        elif expected not in proof:
            faults.append(f"{where}.proof.{expected}: missing — {capability.what}")
        elif not proof[expected]:
            faults.append(f"{where}.proof.{expected}: empty")

    return faults


def check_report(payload: dict, min_how: int, min_entries: int) -> list[str]:
    """The whole report. Returns every fault; empty means pass."""
    faults: list[str] = []

    for key in envchk.REPORT_KEYS:
        if key not in payload:
            faults.append(f"items/{zone.REPORT_ITEM}: no {key!r}")

    digest = payload.get("nonce_digest")
    if digest is not None and not (isinstance(digest, str) and DIGEST.match(digest)):
        faults.append(f"nonce_digest: {digest!r} is not 12 hex characters")

    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        faults.append(f"capabilities: is {type(capabilities).__name__}, needs an object")
    else:
        missing = [label for label in envchk.LABELS if label not in capabilities]
        if missing:
            faults.append(f"capabilities: missing {missing} of the seven")
        extra = sorted(set(capabilities) - set(envchk.LABELS))
        if extra:
            faults.append(f"capabilities: {extra} is not one of the seven")
        for label in envchk.LABELS:
            if label in capabilities:
                faults += check_section(label, capabilities[label], min_how)

    # **The install report is not optional and an empty one is a fault.** It is
    # the only independent account of what `env_mgr` did, and section 7's
    # `unavailable` exemption is decided against it — so a report that omits it
    # is a report whose one permitted excuse cannot be checked.
    installs = payload.get("install_report")
    if not isinstance(installs, list):
        faults.append(f"install_report: is {type(installs).__name__}, needs a list")
    elif len(installs) < min_entries:
        faults.append(
            f"install_report: {len(installs)} entries, needs {min_entries} — "
            f"this run declares one recipe (L1) and one component (L2), and a "
            f"report naming neither is a report that never looked"
        )

    source = payload.get("install_report_source")
    if not isinstance(source, str) or not source.strip():
        faults.append("install_report_source: missing — say where the report came from")

    return faults


def main() -> int:
    parameters = zone.args()
    min_how = int(parameters.get("min_how_chars", 1))
    min_readme = int(parameters.get("min_readme_lines", 1))
    min_entries = int(parameters.get("min_install_report_entries", 1))

    results: dict[str, bool] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            results[hid] = False
            print(f"check_env_report_shape: {hid}: no staged content")
            continue
        faults = check_readme(zone.readme_of(content), min_readme)
        payload, why = zone.report(content)
        if payload is None:
            faults.append(why)
        else:
            faults += check_report(payload, min_how, min_entries)
        results[hid] = not faults
        for fault in faults:
            print(f"check_env_report_shape: {hid}: FAIL: {fault}")
    zone.write_verdict(results)
    print(f"check_env_report_shape: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
