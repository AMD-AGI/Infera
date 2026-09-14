#!/usr/bin/env python3
"""`check_workset_shape` — is this workset one an optimizer could actually use?

Every rule here is decided by looking at a file that either is there or is not,
or at a string that either occurs or does not. Nothing is judged. That is why
the spec calls it `strong` without qualification: it cannot be *approximately*
right about whether `driver.py` exists.

**Why the driver's stdout tokens are checked rather than assumed.** The whole
optimization loop reads correctness and timing off `driver.py`'s stdout. A
workset whose driver does not print `case_ms:` cannot be benchmarked, and the
failure surfaces hours later as a campaign that ran and measured nothing. Four
substring checks buy that back for free.

**What this cannot catch, stated so nobody assumes otherwise.** It does not run
the driver, so a driver that prints the right tokens and measures the wrong
thing passes. It does not verify that the baseline numbers in
`baseline_measurement.md` are true, only that a cross-check against a profile
figure is *present and claimed*. Establishing that the driver measures the
traced kernel is `optimize_kernel`'s first job and this body is not a substitute
for it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Resolved from this file's own location, not from a package environment
# variable. That matters here more than it does in the template: this validator
# runs in **both** phases, and the two export different package variables
# (`AGENT_SYS_TASK_PACKAGE` on the output phase, `AGENT_SYS_DEMO_PACKAGE` on the
# input one). `__file__` is correct under either.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable

#: A line that counts as substance: not blank, not a heading, not a fence.
_FENCE = re.compile(r"^\s*```")


def _content_lines(path: Path) -> int:
    total = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or _FENCE.match(raw):
            continue
        total += 1
    return total


def _check(content: Path, args: dict, problems: list[str]) -> bool:
    workset, why = zone.find_one_dir(content)
    if workset is None:
        problems.append(why)
        return False

    required = args.get("required_files") or []
    for rel in required:
        target = workset / rel
        if not target.is_file():
            problems.append(f"missing {rel}")
        elif target.stat().st_size == 0:
            problems.append(f"{rel} is empty")

    driver = workset / "kernel" / "driver.py"
    if driver.is_file():
        text = driver.read_text(encoding="utf-8", errors="replace")
        for token in args.get("required_driver_tokens") or []:
            if token not in text:
                problems.append(f"driver.py never prints/accepts {token!r}")

    # At least N distinct correctness cases. The driver reports one
    # `case_snr:`/`case_allclose:` pair per case, and the case ids are built
    # from the shapes in `_CASES`, so the tuple is what is counted -- reading
    # the source rather than running it, which is this validator's whole remit.
    floor = int(args.get("min_correctness_cases") or 0)
    if floor and driver.is_file():
        text = driver.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"_CASES\s*=\s*\((.*?)\)\s*\n", text, re.DOTALL)
        cases = len(re.findall(r"\(\s*\d+\s*,", block.group(1))) if block else 0
        if cases < floor:
            problems.append(f"driver.py declares {cases} correctness cases, needs >= {floor}")

    # The cross-check is the one claim that makes a speedup meaningful, so its
    # *presence* is mandatory even though its truth cannot be checked here.
    baseline = workset / "baseline_measurement.md"
    if baseline.is_file():
        if _content_lines(baseline) < 5:
            problems.append("baseline_measurement.md has almost no content")
        text = baseline.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\d", text):
            problems.append("baseline_measurement.md carries no numbers at all")

    # A workset that is still a template is not a workset.
    for rel in ("README.md", "program.md", "integration.md"):
        target = workset / rel
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        for marker in ("TODO", "TBD", "FIXME", "XXX", "to be filled in"):
            if marker in text:
                problems.append(f"{rel} still carries a {marker} placeholder")

    return not problems


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            # Staged nothing is *no content*, and it is never a pass.
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            verdicts[hid] = _check(content, args, problems)
        for problem in problems:
            print(f"{hid}: {problem}")
    zone.write_verdict(verdicts)
    print(f"check_workset_shape: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
