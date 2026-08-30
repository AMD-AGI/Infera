# check_packup_shape — completeness, strong

The `runbook` handoff carries **exactly one** `<name>.packup_<YYYYMMDD>/`
directory under `items/codes/`, that directory holds every entry this package
makes mandatory, and each mandatory document carries substance rather than a
heading and a blank line.

The mission's words for this check are *"packup规定的目录都存在"* — the directories
the packup mandates all exist. This file records two decisions that phrase does
not settle: **which** entries are mandatory here, and what stops an empty file
from satisfying "exists".

## Which entries are mandatory, and why

`~/.claude/skills/experiment-result-packup/references/deliverable_layout.md`
splits the layout in two. Three files are never optional — *"never omit
`README.md`, `REPRODUCE.md`, or `environment.md`"* — and the rest are *"as the
work requires"*, with the explicit instruction to *"omit empty folders rather
than shipping hollow scaffolding"*.

So the layout hands over three, and the other five are this package's call. It
is a call that has to be made: "as the work requires" is not a rule a program
can apply, and a check that accepts whatever it finds is not a check.

| entry | here | why |
|---|---|---|
| `README.md` | **mandatory** | The layout says never omit it |
| `REPRODUCE.md` | **mandatory** | The layout says never omit it, and it is the file `check_reproduces` executes |
| `environment.md` | **mandatory** | The layout says never omit it, and calls it *"the #1 repro trap"* |
| `scripts/` | **mandatory** | This task's whole substance *is* a launch procedure. The layout requires scripts *"copied VERBATIM… so byte-level flags survive"*; a bring-up whose flags exist only as prose inside `REPRODUCE.md` is one paraphrase away from not reproducing. Must hold at least one file |
| `results/` | **mandatory** | The task's success criterion is that the server answered. `results/` is where the evidence of that answer lives. Without it the kit asserts a result and shows none, and `README.md`'s `## Result` becomes an unbacked claim. Must hold at least one file |
| `notes.md` | **mandatory** | Not a judgement call on this box. The measured traps here — `/home/yihou` is NFS root_squash and sglang fails on it *silently*; a cold start spends ~150–200 s in AITER JIT while the log repeats `Health check failed`; ports are shared with seven other worktrees — are exactly what `notes.md` is for, and a reproducer who does not know the second one kills the server at 60 seconds and reports that it does not reproduce |
| `patches/` | optional | The image already carries a working infera and a matching sglang (`scratch/single-real-task-2026-08/FINDINGS.md` §4), so a run needing no patch is the expected case, not a lapse. The layout's own rule — omit rather than ship empty — applies |
| `logs/` | optional | The layout makes it opt-in and warns it is often large |

Two entries are therefore mandatory here that the layout leaves open, and both
are argued from *this* task rather than from tidiness. If this package is ever
retargeted at an experiment where a launch script or a results file genuinely
does not exist, the honest change is a row in this table and a name in
`REQUIRED_FILES`, not a looser check.

## What "substance" was settled on, and why it is counted rather than judged

**A check that passes an empty `REPRODUCE.md` is not a check.** Presence is
trivially satisfiable — `touch REPRODUCE.md` satisfies it — and an agent that
produced a kit under time pressure will satisfy exactly what is measured. So
each mandatory document is measured, and every measure below is a count or a
match, never a judgement:

| rule | applies to | the number |
|---|---|---|
| **content lines** — non-blank, not a heading, not a code fence marker | all four `.md` files | `README.md` 5, `REPRODUCE.md` 8, `environment.md` 8, `notes.md` 3 |
| **no unfilled placeholder** | all four | `TODO`, `TBD`, `FIXME`, `XXX`, `to be filled in`, and the packup templates' own `<…>` form |
| **command lines** — non-blank lines inside a fenced or four-space-indented code block | `REPRODUCE.md` | 5 |
| **a `## Result` heading** | `README.md` | — |
| **at least one digit** | `environment.md` | — |

The floors live in the validator spec's `args` block, not in this code, so the
number a reader sees in `steps/serve.yaml` is the number that is enforced.

**Headings do not count as content.** A document that is four `##` lines with
nothing under them is precisely the failure the floor exists to catch, and
counting the headings would let it through. The same reasoning puts fence
markers outside the count.

**`REPRODUCE.md` is measured in commands and the other three are not.** It is
the file a reproducer executes; eight paragraphs describing a bring-up are not a
reproduction kit, and the layout says as much — *"ordered, copy-pasteable
commands from zero to result"*. The counter does **not** try to decide whether a
line is valid shell: that is a shell parser, and a check that guesses at syntax
starts failing honest kits. It counts lines that an author put inside a code
block, which is what "copy-pasteable" reduces to.

**`environment.md` must contain a digit.** Crude on purpose, and exact: the
layout calls for *"pinned versions, image digests, git SHA, driver"*, and a file
with no digit anywhere in it has pinned none of them. It is the cheapest rule
that separates a real environment note from a paragraph about one.

**`README.md` must carry `## Result`.** One heading, required by name, because
it is the section that answers the question this whole package is about — did
it work. The other headings in the packup's README template are not required:
they organise a document, and requiring them would be this check having an
opinion about layout it cannot defend.

## Why exactly one packup directory

`items/codes/` must hold exactly one directory matching
`^.+\.packup_\d{8}$`. Zero means the agent wrote its kit somewhere the layout
does not put it — the layout fixes the folder name, so a differently named
folder is a kit a reproducer has to go looking for. Two means a reproducer has
to guess which of them is the one that worked, and guessing is the failure mode
the whole artefact exists to remove.

The eight digits are checked as digits: `something.packup_soon` is not a packup
and refusing it costs nothing.

## Why this is `strong`

Every rule above is decided by looking at a file that either is there or is not,
or by counting lines. There is nothing approximate in it and no gap between what
it is named after and what it measures. `strength` qualifies a PASS
(`validator` spec §5.4), and a PASS here means exactly: *this handoff has the
shape and the bulk a packup is required to have.*

It emphatically does **not** mean the kit reproduces. That is
`check_reproduces`, it is `weak`, and the two are separate validators precisely
so that neither borrows the other's claim.

## Failure reporting

Every fault the packup has is reported, not just the first, because a producer
retrying against a check that reports one fault per attempt pays a round trip
per fault. The per-handoff verdict is `no faults`.

An empty `items/codes/`, a missing staged content directory and a packup with
zero mandatory files all fold to `False` rather than to a vacuous pass —
`all([])` is `True`, which is how an empty artefact passes a check nobody wrote
carefully.

## Layout

`entry.sh` is the command, `check.py` is the implementation, and
`assets/lib/zone.py` is the four body-facing zone files
(`args.json`, `inputs.json`, `materials.json`, `verdict.json`) shared with
`check_reproduces`.
