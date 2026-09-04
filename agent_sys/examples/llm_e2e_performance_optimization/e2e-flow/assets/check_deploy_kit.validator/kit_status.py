#!/usr/bin/env python3
"""What a `deploy_kit` slot holds, whether or not anyone sealed it.

    python3 kit_status.py <run-root-or-handoff-dir> [...]

**Why this exists.** A `deploy_and_prove` run on 2026-09-04 was killed at 58
minutes by an actor nobody has identified. The kit was **complete on disk twenty
minutes before the kill** — a real bring-up, a real record, `gpu_devices` from an
obeyed operator directive — and *nothing in the graph knew it*. Establishing that
the run had produced anything at all took reading the tree by hand: finding the
handoff directory, noticing it had `claim` and `content` but no `manifest.yaml`,
and parsing the record to see whether it described a real deployment.

That is the gap: **an unsealed handoff is not nothing, and the system offers no
way to say so.** A run that loses its verdict should not also lose its kit.

**This changes no producer behaviour.** It is a reader, run by a person after the
fact, over a directory that already exists. The alternative shapes — a marker the
producer writes, an early record write — all require touching
`deploy_and_prove.task/readme.md`, and the next real bring-up is the acceptance
test for two producer changes already in flight (`c1c10ba`, `a32f06d`). Changing
a third thing between attempts would make a failure unattributable. **If the
graph should learn this live, that is a different change and it should wait for a
rung that is not also testing something else.**

**It must not look like a seal, and that is the first design constraint.**
A partial artefact that reads as complete is worse than one that reads as absent
— `check_overlay_applies`'s `require_difference` argument, one layer out. So:

  * the word **UNSEALED** leads every unsealed report and is never softened;
  * this tool **never prints `PASS`, `valid`, or `ok`** for a kit. It reports
    what is *present*, not whether it is *correct*. Correctness is
    `check_deploy_kit`'s answer and it needs a zone;
  * a sealed slot is reported as sealed and then left alone — its verdict is
    the authority and this tool does not second-guess it.

**What it deliberately does not do.** It does not validate against
`deploy_kit.layout.yaml`. That is `check_deploy_kit`'s job, it needs `zone.args()`
and a staged input, and a second half-implementation of it here would be the
"six owners each fixed it in their own stage" pattern this package keeps logging.
This answers a narrower question — *is there a kit here, and what does it say
about itself* — which is the question nobody could answer quickly at 09:43.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

#: A slot is sealed when the store has written its manifest. `claim` and
#: `content` appear as soon as the slot is opened, so their presence says the
#: producer started, not that it finished.
MANIFEST = "manifest.yaml"
VERDICT = "verdict.json"

#: The fields worth printing from a `code` handoff's environment record: enough
#: to tell a real bring-up from a replay or a stub, and no more.
RECORD_FIELDS = (
    ("fixed", "node"),
    ("fixed", "image"),
    ("fixed", "gpu_devices"),
    ("fixed", "tp_size"),
    ("runtime", "container"),
    ("runtime", "endpoint"),
    ("runtime", "started_at"),
    ("runtime", "replayed_from"),
)


def slots(root: Path):
    """Every versioned handoff slot under `root`, whether or not it is sealed.

    Accepts a run root, a `handoffs/` directory, a single handoff or a single
    version directory — because the person running this has just been handed a
    path by someone else and should not have to know which of the four it is.
    """
    if (root / MANIFEST).is_file() or (root / "content").is_dir():
        return [root]
    found = sorted(p.parent for p in root.rglob("content") if p.is_dir())
    return [p for p in found if p.name.startswith("v")] or found


#: **All three of CONTRACT §2's spellings, not just this stage's.** The record
#: is at `items/codes/` for a `code` handoff and `items/env/` for `reproducible`
#: and `structured_text`. Looking only where `deploy_kit` keeps it would make
#: this tool report *"no kit was written here yet"* over a perfectly good
#: unsealed `profiling_evidence` — a false statement, from the tool whose one
#: job is not to mislead someone reading a wreck. Caught by pointing it at a
#: sealed `kernel_table` slot and noticing it would have lied had that slot been
#: open.
RECORD_PATHS = (
    ("items", "codes", "environment.yaml"),
    ("items", "env", "environment.yaml"),
)


def record_of(version: Path) -> dict | None:
    """The environment record wherever its content type keeps it, or None."""
    for parts in RECORD_PATHS:
        candidate = version.joinpath("content", *parts)
        if candidate.is_file():
            path = candidate
            break
    else:
        return None
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # a record that will not parse is worth saying so
        return {"_unparseable": str(exc)}


def describe(version: Path) -> int:
    """One slot. Returns 1 if it is unsealed, 0 otherwise — see `main`."""
    sealed = (version / MANIFEST).is_file()
    kind = "?"
    if sealed:
        try:
            import yaml

            kind = (yaml.safe_load((version / MANIFEST).read_text()) or {}).get("kind", "?")
        except Exception:
            kind = "?"

    label = "SEALED  " if sealed else "UNSEALED"
    print(f"\n{label}  {version}")

    if sealed:
        verdict = version / VERDICT
        state = "absent"
        if verdict.is_file():
            try:
                state = json.dumps(json.loads(verdict.read_text()))
            except Exception:
                state = "unreadable"
        print(f"          kind={kind}  verdict={state}")
        print("          Sealed: check_deploy_kit's verdict is the authority, not this tool.")
        return 0

    # ---- the case this tool exists for -------------------------------------
    print("          No manifest. The producer opened this slot and never closed it —")
    print("          the run was cut, or is still running. NOTHING BELOW IS VALIDATED.")

    record = record_of(version)
    if record is None:
        print("          no environment.yaml at either items/codes/ or items/env/: nothing written yet.")
        return 1
    if "_unparseable" in record:
        print(f"          environment.yaml does not parse: {record['_unparseable']}")
        return 1

    print("          A record IS present. What it says about itself:")
    for section, field in RECORD_FIELDS:
        value = (record.get(section) or {}).get(field)
        print(f"            {section + '.' + field:<24} {value!r}")

    replayed = (record.get("runtime") or {}).get("replayed_from")
    packups = sorted(
        p.name for p in (version / "content" / "items" / "codes").glob("*.packup_*") if p.is_dir()
    )
    print(f"            packup dirs              {packups or 'NONE'}")

    # **Would it seal?** This tool was written to stop a reader over-claiming
    # from a wreck, and its first output over-claimed: it printed "a REAL
    # bring-up's kit" over an artefact that could not have been sealed, the claim
    # was relayed, and another owner built a route on it. The missing thing was
    # one file — `content/README.md`, which `handoff/content.py` requires of a
    # `code` handoff and which nothing in `deploy_kit.layout.yaml` looked for
    # either. **A tool that cannot report the failure it exists to prevent is the
    # same shape as a probe that passes on the artefact that failed.**
    handoff_readme = (version / "content" / "README.md").is_file()
    print(f"            content/README.md        {'present' if handoff_readme else 'MISSING'}")

    # The one inference worth drawing, because it is the question that was asked
    # at 09:43 and it took a person twenty minutes: is this a real bring-up?
    if replayed:
        print("          -> a REPLAY. It stands in for a sealed kit; it is not a bring-up.")
    elif packups and handoff_readme:
        print("          -> a REAL bring-up's kit, written and never sealed, and it")
        print("             WOULD seal on this evidence. Worth keeping. Still NOT a")
        print("             validated handoff — no consumer should take it as one, and")
        print("             the container it names may be long gone.")
    elif packups:
        print("          -> a REAL bring-up's kit, and it WOULD NOT SEAL AS IT STANDS:")
        print("             content/README.md is absent and handoff/content.py requires")
        print("             it of every `code` handoff. The deployment happened; the")
        print("             artefact is one file short of being a handoff at all.")
        print("             Do not plan on this as an input until that file exists.")
    else:
        print("          -> a record with no packup beside it: started, not finished.")
    return 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    unsealed = 0
    for raw in argv:
        root = Path(raw).expanduser()
        if not root.exists():
            print(f"kit_status: {root}: no such path", file=sys.stderr)
            return 2
        found = slots(root)
        if not found:
            print(f"kit_status: {root}: no handoff slot under here")
            continue
        for version in found:
            unsealed += describe(version)
    # **Exit 1 when anything is unsealed, and that is not a failure.** It is the
    # difference a caller in a script needs, and the two-causes-one-exit-code
    # trap this package has now hit four times is avoided by the text above
    # saying which: 0 = every slot sealed, 1 = at least one unsealed, 2 = this
    # tool could not run.
    return 1 if unsealed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
