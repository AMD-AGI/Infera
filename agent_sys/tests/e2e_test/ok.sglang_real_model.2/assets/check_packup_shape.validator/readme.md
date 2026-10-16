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
| `notes.md` | **mandatory** | Not a judgement call for this kind of work. A bring-up of this shape has a standing set of traps — a home directory on a network filesystem with `root_squash`, where the engine fails to write its logs *silently*; a cold start that spends minutes in a JIT compile while the log repeats `Health check failed`; ports already taken by whoever else is on the host — and they are exactly what `notes.md` is for. A reproducer who does not know the second one kills the server after a minute and reports that it does not reproduce |
| `patches/` | optional | `$SRT_IMAGE` is specified as already carrying a working infera and a matching sglang, so a run needing no patch is the expected case, not a lapse. The layout's own rule — omit rather than ship empty — applies |
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

**Numbers go in `args`; sets go in the body.** `REQUIRED_FILES`,
`REQUIRED_DIRS`, `PLACEHOLDER` and the binding-flag list below are all sets, and
they are all module constants documented in this file. The split is not
cosmetic: a floor is a dial a site might reasonably turn, and a set is a rule
whose *reason* is an argument in this readme and does not survive being moved
into a yaml with no room for one.

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

## Shared-namespace identifiers, and why this one is a program and not prose

The task brief tells the agent that **every identifier it binds on a shared host
is a parameter, not a constant** — container names, host ports, the container
workdir, the GPU index. This check enforces the second part of that rule: the
kit must be re-pointable without editing it.

**It is a check because the prose version already failed once, measurably.** The
brief said *"Pick your ports; do not assume them… record the ports you actually
used"* before B5 ran. B5's kit hardcodes all four ports, both container names
and the workdir. So a rule the executor is asked to honour, with nothing
checking it, is worth what that run showed it to be worth. This system's claim
is that quality comes from standardising what crosses the boundary rather than
from trusting the producer, and a validator is where that claim is cashed.

### The rule, exactly

Over every file under `scripts/`, two facts are collected, and a fault needs
both:

| | |
|---|---|
| **frozen** | the name is assigned with a plain `X=…` / `export X=…`, and what remains of the value after every `${…}` and `$X` reference is removed still contains an alphanumeric character |
| **bound** | the name reaches one of `--name`, `--publish`, `--volume`, `--mount`, `--port`, `-p`, `-v` |

A name is exempt the moment it appears as `${X:=…}`, `${X:-…}` or `${X:?…}`
**anywhere in the kit**, because that is all it takes to let a caller re-point
it. Separately, a bare literal at `--name`, `--publish`, `--port` or `-p` is a
fault on its own, with no variable involved.

Three details that are decisions rather than details:

**The value is not inspected, only its shape.** `"${HOST}:${PORT}"` is built
entirely out of other variables and punctuation, so it is as parameterised as
they are and is not frozen. `"${SRT_WORK_ROOT}/srt_qwen36_mix"` has a fixed leaf
and is — which is what catches the workdir, the one that survives fixing the
container names.

**No attempt is made to recognise "a container name" from its text.** That is a
heuristic over values, it has no stopping condition, and it fails honest kits in
ways their authors cannot predict. What is recognised is a *flag*, which is a
fact about the command being run.

**`-p` and `-v` count only inside a command that mentions `docker`.** `mkdir -p`
is the counter-example, and it appears in this very kit.

**`--volume` is absent from the bare-literal rule** while `--name` and the port
flags are in it. A read-only mount of an input path — `-v /data/models:/models`
— is legitimately fixed and only its host side is shared at all; a container
*name* never legitimately is.

### What it cannot catch

Written here rather than left implicit, because a check that reads as complete
is worse than one whose edges are known.

- **A port embedded in a URL.** B5's `ETCD_CLIENT_PORT` and `ETCD_PEER_PORT`
  reach `--listen-client-urls "http://0.0.0.0:${…}"`, not `--port`. They are
  frozen and genuinely host-shared, and this check does not see them. Widening
  the flag set to "any flag whose value contains a port-looking number" is the
  heuristic road above.
- **The GPU index.** `GPU_ID` reaches `export HIP_VISIBLE_DEVICES=${GPU_ID}`
  inside a `docker exec` heredoc. It is an environment variable, not a flag.
- **A name computed at runtime**, e.g. `CTR_NAME=$(basename "$PWD")`. The value
  has no literal remainder, so it is not frozen — correctly, since two copies in
  two directories would differ, but a kit that computed a *constant* would slip
  through.
- **Anything not in `scripts/`, and anything not shell.** A kit driving docker
  from a Python or Makefile driver is unparsed. `scripts/` is where the packup
  layout puts the scripts and where this looks.
- **`docker rm -f <name>` before `docker run --name <name>`** — the destructive
  act, and the sharpest of the brief's three sub-rules. It is deliberately **not
  checked here**, and the reason is structural rather than an oversight: the
  PASS control for this check is B5's kit with its assignments rewritten to
  `:=`, and that kit still contains its two `docker rm -f` lines. A check that
  also refused those would refuse the control, which would mean this rule and
  that one were being measured together and neither one isolated. It is a second
  check, with its own control, and it is not written yet.

Each of these is a **false negative**: the check does not report a kit that has
the problem. There is no configuration under which it reports a kit that does
not — the two conjuncts have to both hold, and the exemption is generous.

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
