#!/usr/bin/env python3
"""`check_command_parses` — a `reproducible` handoff's `command` must be a script.

`agent.gate` requires the `command` item to be **executable**. Nothing anywhere
requires it to **parse**. Found by m2 while sweeping for a fault class they had
just fixed in their own generators, and confirmed here first-hand:

    11 of the 14 sealed `items/command` scripts under `cheat_for_mock/`
    do not parse.

Six in stage 2, one in stage 3, four in stage 5. One cause in every case: an
apostrophe inside a `${VAR:?word}` message opens a single-quoted string that
runs to end of file —

    : "${SCRIPTS:?export SCRIPTS=<the package's assets/load directory>}"
                                              ^ opens a quote that never closes

`bash -n` says `unexpected EOF while looking for matching '`, and the line it
names is the end of the file rather than the apostrophe.

**Eleven artefacts were sealed, validated and shipped PASS carrying the one
script a reproducer would run first.** That is the same shape as the `/bin/sh`
finding earlier in this effort: a property everyone assumed was checked, checked
by nothing.

`sh -n` and not `bash -n`, because `agent_sys` invokes a body as
`["/bin/sh", entry]` and `/bin/sh` here is dash (CONTRACT.md §3.2a) — grading a
script by a shell that will not run it is the mistake one level up.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))
_pkg = os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ.get("AGENT_SYS_DEMO_PACKAGE", "")
if _pkg:
    sys.path.insert(0, str(pathlib.Path(_pkg) / "assets" / "lib"))

import workset_io as W  # noqa: E402 — `write_report`; nothing else from it
import zone  # noqa: E402

#: Where a script that a reproducer runs may live. `command` is the
#: `reproducible` type's own item; `script` is its alternative, and a kind may
#: legitimately carry either (`handoff/content.py:_ALTERNATIVES`).
CANDIDATES = ("items/command", "items/script")


def judge(path: pathlib.Path) -> str:
    """The shell to check with: **the one this file's own shebang names.**

    Measured, and it is why the first version of this validator passed a broken
    script. On `stage2-profiling/aiperf_baseline`:

        shebang   #!/usr/bin/env bash
        sh -n     clean            — and it *runs*, aborting correctly at line 6
        bash -n   line 9: unexpected EOF while looking for matching `''

    dash tolerates the unterminated quote; bash does not. So **the script is
    broken under the shell its own shebang names and works under the shell it
    does not** — and a reproducer running `./command` gets the shebang, hence
    the syntax error, while one running `sh command` gets the intended
    behaviour. Checking with whichever shell happens to be tried first decides
    the verdict by luck.

    No shebang means the caller picks, and `/bin/sh` is what `agent_sys` itself
    uses (CONTRACT.md §3.2a).
    """
    try:
        first = path.read_text(errors="replace").splitlines()[:1]
    except OSError:
        return "sh"
    if not first or not first[0].startswith("#!"):
        return "sh"
    line = first[0][2:].strip()
    words = line.split()
    if not words:
        return "sh"
    # `#!/usr/bin/env bash` names the shell in the second word.
    name = pathlib.Path(words[0]).name
    if name == "env" and len(words) > 1:
        name = pathlib.Path(words[1]).name
    return name if name in ("sh", "bash", "dash", "zsh", "ksh") else "sh"


#: `${VAR:?message}` / `${VAR:=message}` — the span whose message must not
#: contain an apostrophe. Non-greedy to the closing brace, which is what these
#: expansions actually use.
_GUARD = __import__("re").compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:[?=]([^}]*)\}")


def quoted_guards(path: pathlib.Path) -> list[str]:
    """Apostrophes inside a `${VAR:?…}` message. **Parsing cannot find these.**

    Found by m5 while fixing their own generators, and it is the sharper half of
    the fault this validator was written for.

    An **odd** number of apostrophes leaves a quote open to end of file and
    `bash -n` says so. An **even** number pairs up: the first opens a quote, the
    second closes it, the parse is clean — and the script is still wrong. On
    `apply_patch`'s emitted `command`, which had exactly two:

      * the error message reads `<this handoffs items/result/patches directory>}"`
        — apostrophe eaten, closing brace leaked out;
      * and **the second guard is swallowed into the first one's quoted region,
        so `ROOTS` is never checked at all.** A reproducer with it unset gets no
        error and the script proceeds.

    A deleted guard is worse than a syntax error: the syntax error stops, and
    this does not. So the rule is about **the character in that position**, not
    about the parse outcome — which is why counting is right here and parsing
    cannot be made to do it.
    """
    bad: list[str] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return bad
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in _GUARD.finditer(line):
            if "'" in m.group(1):
                bad.append(f"line {lineno}: {m.group(0)[:90]}")
    return bad


def parses(path: pathlib.Path) -> tuple[bool, str]:
    shell = judge(path)
    try:
        done = subprocess.run([shell, "-n", str(path)], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return False, f"the shebang names {shell}, which is not on PATH here"
    except subprocess.TimeoutExpired:
        return False, f"{shell} -n did not finish in 30s"
    if done.returncode != 0:
        last = ((done.stderr or "").strip().splitlines() or ["no message"])[-1]
        return False, f"{shell} -n (from its own shebang): {last}"[:400]
    return True, f"{shell} -n clean (shell taken from its shebang)"


def main() -> int:
    args = zone.args()
    required = bool(args.get("require_present", True))

    verdict: dict[str, bool] = {}
    findings: list[str] = []

    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            findings.append(f"{hid}: nothing staged — treated as no content, never as a pass")
            verdict[hid] = False
            continue

        found = [content / rel for rel in CANDIDATES if (content / rel).is_file()]
        if not found:
            # A `code` or `structured_text` kind has no `command`, and grading it
            # for the absence would fail a correct artefact. Only a kind that
            # declares one is held to this.
            verdict[hid] = not required
            findings.append(
                f"{hid}: no {' or '.join(CANDIDATES)}"
                + ("" if required else " (not required for this kind)")
            )
            continue

        ok = True
        for path in found:
            good, why = parses(path)
            rel = path.relative_to(content)
            if not good:
                ok = False
                findings.append(
                    f"{hid}: {rel} does not parse — {why}. "
                    f"A reproducer runs this first, so an unparseable one makes the "
                    f"handoff unreproducible however complete the rest of it is."
                )
            else:
                findings.append(f"{hid}: {rel} {why}")
            # Executable is `agent.gate`'s rule and is checked there; reported
            # here because a script that parses and cannot be run is the same
            # dead end arrived at differently.
            if not os.access(path, os.X_OK):
                findings.append(f"{hid}: {rel} is not executable")
                ok = False
        verdict[hid] = ok

    for line in findings:
        print(line, file=sys.stderr)
    # **The reasons have to outlive stdout**, and here they are printed to
    # *stderr*, which is kept even less than stdout. This validator's whole
    # output is a per-file verdict — "`items/command` does not parse, line 9,
    # unterminated quote" — and that sentence is the entire value of the check;
    # the boolean alone tells a reader nothing they can act on.
    #
    # `findings` is a flat list here rather than per-handoff, because one
    # handoff can carry several scripts. Re-keyed by the id each line names so
    # the report matches the shape `write_report` publishes, and lines that
    # name no handoff are kept under `""` rather than dropped.
    by_hid: dict[str, tuple[list[str], list[str]]] = {hid: ([], []) for hid in verdict}
    for line in findings:
        hid = line.split(":", 1)[0]
        by_hid.setdefault(hid if hid in verdict else "", ([], []))[0].append(line)
    # Before the verdict, so a crash in the writer cannot take the reasons with it.
    W.write_report("check_command_parses", by_hid, verdict)
    zone.write_verdict(verdict)
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
