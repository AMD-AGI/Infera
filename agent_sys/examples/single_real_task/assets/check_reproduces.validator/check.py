#!/usr/bin/env python3
"""`check_reproduces` — usability, **weak**.

Hand the handoff's packup to a fresh Claude Code session, tell it to follow
`REPRODUCE.md` for real, and pass iff it reports that it reproduced the run
**and** every artefact its report names exists and is non-empty.

The readme beside this file argues why that is `weak` and what the corroboration
is worth. This module is the exact form of it.

**No credential is read, echoed or written here.** `claude` is inherited from
this body's environment and consumes `ANTHROPIC_API_KEY` itself; nothing in this
file names a value.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable

#: What the reproducer must leave behind, inside the copy of the kit it worked
#: in. Named for what it is rather than `verdict.json`, which is the validation
#: zone's own file and belongs to `PhaseRunner`'s seam.
REPORT = "reproduction.json"

#: The reproducer's transcript. Kept beside the copy so a failing verdict can be
#: read rather than guessed at.
TRANSCRIPT = "claude.log"

#: The task given to the reproducer. It is deliberately silent about *what*
#: success is: `REPRODUCE.md` says that, and a prompt that restated it would be
#: this validator marking its own homework — the kit would then be checked
#: against the check's idea of the experiment instead of its own.
PROMPT = f"""\
You are reproducing an experiment from a reproduction kit. The kit is the
current working directory and you may read and run anything in it.

1. Read README.md, then REPRODUCE.md, then notes.md if it is there.
2. Actually carry out the steps in REPRODUCE.md, in order, on this machine.
   Run the commands. Do not simulate them, do not paraphrase their output, and
   do not report a result you did not observe.
3. Decide whether the run reproduced, judged against REPRODUCE.md's own
   "Expected output" section, or against README.md's "Result" section if
   REPRODUCE.md has none. Those sections are the criterion; nothing else is.
4. Clean up anything you started that is still running.
5. Write ./{REPORT}, a JSON object with exactly these three keys:

   {{"reproduced": true or false,
     "evidence": "one paragraph: what you ran and what you observed",
     "artifacts": ["paths, relative to this directory, of files you produced
                   that back the claim - at least one when reproduced is true"]}}

Rules:
- Every path in "artifacts" must exist and be non-empty when you finish. It is
  checked. A path you did not create is a failed check, not a stronger claim.
- If it did not reproduce, say so: "reproduced": false with the evidence of
  what went wrong is the correct and useful answer, and is not a failure on
  your part.
- Never print the value of an environment variable, and never copy one into a
  file. Naming a variable is fine; its value is not.
"""


def claude_command() -> list[str] | None:
    """`claude`, from `PATH` or from the one place it is installed.

    `PATH` on the producer row is `env_mgr`'s policy-derived one rather than the
    operator's, so `shutil.which` is a route and not the route. `~/.local/bin`
    is the fallback and is where this box has it.
    """
    found = shutil.which("claude")
    if found is None:
        candidate = Path(os.environ.get("HOME", "")) / ".local" / "bin" / "claude"
        found = str(candidate) if candidate.is_file() else None
    if found is None:
        return None
    # `--dangerously-skip-permissions` because the reproducer's whole job is to
    # run the kit's commands, and under `-p` an unapproved tool call is denied
    # rather than prompted — so a narrower flag would fail every kit for a
    # reason that has nothing to do with the kit. This stage runs with
    # `agent_sys`'s own permission enforcement off by default, and this is the
    # same decision one layer down.
    return [found, "-p", PROMPT, "--dangerously-skip-permissions"]


def run_reproducer(work: Path, timeout: float) -> tuple[bool, str]:
    """Drive `claude` inside `work`. Returns `(started, why_not)`."""
    command = claude_command()
    if command is None:
        return False, "claude is not on PATH and not at $HOME/.local/bin/claude"
    transcript = work / TRANSCRIPT
    try:
        completed = subprocess.run(  # noqa: S603 — argv form, no shell
            command,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        transcript.write_text("claude timed out\n", encoding="utf-8")
        return False, f"the reproducer did not finish inside {timeout:.0f}s"
    # The whole transcript goes to a file and **not** to this body's stderr:
    # `ScriptBodyRunner` folds a body's stderr tail into an exception message,
    # so anything written there travels into the event stream.
    transcript.write_text(
        f"$ claude -p <prompt> --dangerously-skip-permissions\n"
        f"exit: {completed.returncode}\n\n--- stdout ---\n{completed.stdout}"
        f"\n--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return False, f"the reproducer exited {completed.returncode}; see {TRANSCRIPT}"
    return True, ""


def read_report(work: Path) -> tuple[bool, list[str]]:
    """The reproducer's own report, and every fault it has."""
    path = work / REPORT
    if not path.is_file():
        return False, [f"the reproducer wrote no {REPORT}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, [f"{REPORT} is not readable JSON: {exc}"]
    if not isinstance(data, dict):
        return False, [f"{REPORT} is not a JSON object"]

    faults: list[str] = []
    claimed = data.get("reproduced")
    if not isinstance(claimed, bool):
        return False, [f"{REPORT}: `reproduced` is {claimed!r}, not a boolean"]
    evidence = data.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        faults.append(f"{REPORT}: `evidence` is missing or empty")

    # **The corroboration, and the only part of this verdict this body owns.**
    # A claim of success has to be backed by files that are actually there.
    artifacts = data.get("artifacts")
    if claimed:
        if not isinstance(artifacts, list) or not artifacts:
            faults.append(f"{REPORT}: `reproduced` is true with no artifacts")
        else:
            for entry in artifacts:
                if not isinstance(entry, str) or not entry:
                    faults.append(f"{REPORT}: artifact {entry!r} is not a path")
                    continue
                # Confine the lookup to the kit copy: an artefact outside it is
                # not evidence this validation can stand behind.
                target = (work / entry).resolve()
                if not str(target).startswith(str(work.resolve())):
                    faults.append(f"artifact {entry!r} points outside the kit")
                elif not target.exists():
                    faults.append(f"artifact {entry!r} does not exist")
                elif target.is_file() and target.stat().st_size == 0:
                    faults.append(f"artifact {entry!r} is empty")
                elif target.is_dir() and not any(target.iterdir()):
                    faults.append(f"artifact {entry!r} is an empty directory")
    else:
        faults.append(f"{REPORT}: the reproducer reports it did not reproduce")

    return (claimed and not faults), faults


def check_one(hid: str, content: Path, timeout: float) -> bool:
    packup, why = zone.find_packup(content)
    if packup is None:
        print(f"check_reproduces: {hid}: {why}")
        return False

    # A writable copy. The staged content is what the phase validates and this
    # body has no business writing into it, and the reproducer must be able to
    # create files beside the kit it is following.
    work = Path.cwd() / "reproduce" / hid
    if work.exists():
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(packup, work)

    started, why_not = run_reproducer(work, timeout)
    if not started:
        print(f"check_reproduces: {hid}: FAIL: {why_not}")
        return False

    passed, faults = read_report(work)
    for fault in faults:
        print(f"check_reproduces: {hid}: FAIL: {fault}")
    return passed


def main() -> int:
    timeout = float(zone.args().get("timeout_seconds", 5400))
    results: dict[str, bool] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            results[hid] = False
            print(f"check_reproduces: {hid}: no staged content")
            continue
        results[hid] = check_one(hid, content, timeout)
    zone.write_verdict(results)
    print(f"check_reproduces: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
