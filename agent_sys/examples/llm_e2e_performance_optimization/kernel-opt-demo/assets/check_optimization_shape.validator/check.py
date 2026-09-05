#!/usr/bin/env python3
"""`check_optimization_shape` — is this a real campaign, or a description of one?

The cheap half of the output gate. It runs before
`check_speedup_substantiated` (`cost: seconds` against `cost: minutes`, and a
phase is ordered cheap-first), so a handoff missing `forge_result.json` fails in
a second instead of after a re-measurement that had nothing to measure.

**The rule that is not about shape.** Everything else here counts files and
lines; the mock-consistency rule reads two documents against each other and
fails a handoff whose `verification.json` says `"mock": true` while its prose
claims a speedup. That is here rather than in the expensive validator because it
costs nothing and because a mock that is not visibly a mock is the single most
misleading artefact this package can produce — it looks exactly like a success.

**What this cannot catch.** It does not run anything and does not check that any
number is true. `forge_result.json` carrying `mean_case_speedup: 99.0` passes
here. Substantiating a number is the next validator's job, and the two are
deliberately separate so the cheap one can fail first.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable

_FENCE = re.compile(r"^\s*```")
_PLACEHOLDERS = ("TODO", "TBD", "FIXME", "XXX", "to be filled in")

# **There is deliberately no `<...>` template-slot rule, and its absence was
# measured rather than assumed.** The sibling package's shape check has one, so
# this body had one too; run against a real, complete, honest packup it fired
# twice — on `<workset>` in a REPRODUCE.md command and on `<project_root>` in a
# sentence describing another tool's default path. Both are documentation
# metavariables and both are *correct writing*.
#
# A regex cannot separate "a slot the author forgot to fill" from "a
# metavariable the author meant", because they are the same characters. Given
# that, the choice is which error to make. This validator is `strong` and its
# PASS is not qualified, so a false failure is the worse one: it teaches an
# author to write vaguer documentation to get past the gate, which is the exact
# opposite of what the gate is for. The explicit markers below catch the cases
# that are unambiguous.

#: Fields `forge_result.json` must carry for a reader to know what happened.
_FORGE_FIELDS = ("baseline_ms", "best_ms", "mean_case_speedup", "improved")
#: Fields the producer's own re-measurement must carry.
_VERIFY_FIELDS = ("mock", "rounds", "iters", "mean_case_speedup", "correctness_passed")


def _lines(path: Path) -> tuple[int, int]:
    """(content lines, command lines inside fenced blocks)."""
    content = commands = 0
    fenced = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _FENCE.match(raw):
            fenced = not fenced
            continue
        line = raw.strip()
        if not line:
            continue
        if fenced:
            # A comment inside a fence is documentation, not a command.
            if not line.startswith("#"):
                commands += 1
            continue
        if line.startswith("#"):
            continue
        content += 1
    return content, commands


def _load_json(path: Path, problems: list[str]) -> dict | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{path.name} does not parse: {exc}")
        return None
    if not isinstance(loaded, dict):
        problems.append(f"{path.name} is a {type(loaded).__name__}, expected an object")
        return None
    return loaded


def _check(content: Path, args: dict, problems: list[str]) -> bool:
    packup, why = zone.find_packup(content)
    if packup is None:
        problems.append(why)
        return False

    floors: dict = args.get("min_content_lines") or {}
    for name, floor in floors.items():
        target = packup / name
        if not target.is_file():
            problems.append(f"missing {name}")
            continue
        content_lines, command_lines = _lines(target)
        if content_lines < int(floor):
            problems.append(f"{name} has {content_lines} content lines, needs >= {floor}")
        if name == "REPRODUCE.md":
            need = int(args.get("min_command_lines") or 0)
            if command_lines < need:
                problems.append(f"REPRODUCE.md has {command_lines} command lines, needs >= {need}")
        text = target.read_text(encoding="utf-8", errors="replace")
        for marker in _PLACEHOLDERS:
            if marker in text:
                problems.append(f"{name} still carries a {marker} placeholder")

    for name in ("scripts", "results"):
        directory = packup / name
        if not directory.is_dir():
            problems.append(f"missing {name}/")
        elif not any(p.is_file() for p in directory.rglob("*")):
            problems.append(f"{name}/ holds no files")

    readme = packup / "README.md"
    if readme.is_file():
        if not re.search(r"^##\s+Result\b", readme.read_text(encoding="utf-8", errors="replace"), re.M):
            problems.append("README.md has no `## Result` section")

    environment = packup / "environment.md"
    if environment.is_file() and not re.search(r"\d", environment.read_text(encoding="utf-8", errors="replace")):
        problems.append("environment.md carries no numbers at all — nothing is pinned")

    for rel in args.get("required_evidence") or []:
        target = packup / rel
        if not target.is_file():
            problems.append(f"missing evidence {rel}")
        elif target.stat().st_size == 0:
            problems.append(f"evidence {rel} is empty")

    forge = packup / "results" / "forge_result.json"
    verification = packup / "results" / "verification.json"
    forge_data = _load_json(forge, problems) if forge.is_file() else None
    verify_data = _load_json(verification, problems) if verification.is_file() else None

    if forge_data is not None:
        for field in _FORGE_FIELDS:
            if field not in forge_data:
                problems.append(f"forge_result.json has no {field!r}")
    if verify_data is not None:
        for field in _VERIFY_FIELDS:
            if field not in verify_data:
                problems.append(f"verification.json has no {field!r}")

        # --- the mock-consistency rule ---
        is_mock = bool(verify_data.get("mock"))
        if is_mock:
            head = ""
            if readme.is_file():
                head = readme.read_text(encoding="utf-8", errors="replace")
            if "MOCK" not in head.upper():
                problems.append(
                    "verification.json says mock=true but README.md never says MOCK — "
                    "a mock that is not visibly a mock reads as a success"
                )
            claimed = verify_data.get("mean_case_speedup")
            if isinstance(claimed, (int, float)) and claimed > 1.0:
                problems.append(
                    f"verification.json is mock=true and still claims a speedup of {claimed}"
                )
        # A degraded (smoke) campaign really ran, so it is not a mock — but it
        # is just as mistakeable for a full one, and the same rule applies: a
        # run that could be read as a complete campaign must be impossible to
        # read as one.
        if bool(verify_data.get("degraded")):
            head = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
            if "SMOKE" not in head.upper():
                problems.append(
                    "verification.json says degraded=true but README.md never says SMOKE — "
                    "a degraded budget produces a campaign that reads like a full one"
                )

        if not is_mock:
            if verify_data.get("correctness_passed") is not True:
                problems.append("verification.json reports correctness_passed != true on a real run")
            rounds = verify_data.get("rounds")
            if isinstance(rounds, int) and rounds < 5:
                problems.append(
                    f"verification.json reports {rounds} rounds; the baseline protocol is 5 and "
                    "a comparison across differently-sized samples is not one"
                )

    return not problems


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            verdicts[hid] = _check(content, args, problems)
        for problem in problems:
            print(f"{hid}: {problem}")
    zone.write_verdict(verdicts)
    print(f"check_optimization_shape: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
