# `check_deploy_kit` — completeness, **strong**, program

**Program, not AI** (mission G4.1: 能程序化的，尽量不用 ai). Every rule it applies
is decided by opening a file that either is there or is not, and every threshold
is counted rather than judged. There is nothing here a model would be better at.

## Where the rules live

**In `../schemas/deploy_kit.layout.yaml`, and nowhere else.** Mission M1.1 asks
for the kit's file and directory requirements to be written into their own yaml
and for a program validator to check against it; `check.py` is an interpreter for
that yaml and holds no rule of its own. `args.layout` names it, so a package
standardising on a different layout re-points it instead of forking this file.

Read the layout, not this readme, to know what is required — the layout carries
the reason beside each rule. What follows is only what `check.py` adds as
machinery.

## The six primitives the layout is expressed in

| primitive | what it does |
|---|---|
| **presence** | `entries[]` — a file or directory, required or not; `non_empty`; `min_files` over a glob |
| **substance** | `min_content_lines` counts lines that are non-blank, not a heading and not a fence marker. Headings are excluded on purpose: a document that is four `##` lines with nothing under them is exactly the failure a floor exists to catch. `min_command_lines` counts lines inside code blocks, which is what "copy-pasteable" reduces to when it has to be counted |
| **placeholders** | a templated document that was never written |
| **evidence** | predicates over `results/`: `forbid` (no file may match), `require_each` (each pattern in *some* file), `require_together` (all patterns in *one* file) |
| **runtime contract** | each declared parameter is read in a `${X:=…}` form somewhere under `scripts/` |
| **schema** | `codes/environment.yaml` validated against `environment.schema.json` through `../lib/schema.py` |

## The one substitution mission M1.1.1 asks for

The previous stage checked the environment with three regexes over
`environment.md` — *"does the word `image` appear on some line"*
(`../../../deploy-demo/assets/check_deploy_kit.validator/check.py:71-80`). That
is what M1.1.1 objects to and it is gone.

The record is now `codes/environment.yaml`, validated as a document. Its
`environment.md` counterpart is checked as a **rendering**: the fields the layout
lists under `rendered_from.must_render` must appear in it verbatim, so a human
can read the environment without a schema in hand and the two cannot drift into
disagreeing. Prose around them is unconstrained.

Everything else is carried across from that file **in substance**, because each
of its rules was a fault observed in a real kit rather than a rule somebody
liked: `require_served_name_not_a_path`, `require_mode_readback`,
`require_completion_evidence`, `min_json_results`, `require_expected_output`,
and the frozen-and-bound identifier scan.

## The frozen-and-bound rule, since it is the subtle one

A name is **frozen** when a plain assignment gives it a value that is not built
purely out of other variables, and **bound** when it reaches one of the layout's
`binding_flags`. Frozen *and* bound is the shape a second copy of the kit cannot
run beside the first — which matters because `check_deploy_serves` starts a
deployment from this same kit while the run that produced it may still be up, and
it cannot edit the scripts.

A name is exempt the moment it appears in a `${X:=…}` form **anywhere** in the
kit: that is all it takes to let a caller re-point it.

It is a list of flags rather than an attempt to recognise "a container name" from
its text. The flags are where a value enters a host-wide namespace, they are few,
and they are literal; guessing at the semantics of a string would fail honest
kits. `--volume` is deliberately not in `literal_forbidden_flags`: a read-only
mount of an input path is legitimately fixed, and only the host side of it is
shared. The two short flags `-p` and `-v` only count inside a command that
mentions `docker`, because `mkdir -p` is not a port binding.

## Gate — both directions, against the real sealed kit

Run against `$E2E_MOCK_ROOT/stage1-deploy/deploy_kit` (38 files, a real bring-up
on this cluster on 2026-09-02):

- **positive** — the untouched kit plus a conforming `codes/environment.yaml` and
  the `runtime_contract` parameters: **passes**. Nothing this validator inherited
  from the previous stage fires on a kit that stage produced.
- **negative** — a copy with faults planted one per rule: a `gpu_arch` of
  `MI355X`, a missing `image_id`, a missing `runtime.endpoint`, an
  `environment.md` disagreeing with the record, a `"model": "/models/…"` in
  `results/`, the router-side mode reading deleted, a frozen container name bound
  to `--name`, a literal `--publish 8106:8106`, and `Expected output` renamed:
  **each is reported, naming the file and the condition.**

A validator that has only been shown to reject is a validator that may reject
everything; a validator that has only been shown to accept is worse. Both
directions, or it is not gated.

## Failure modes this body chooses on purpose

- **A layout it cannot load refuses every input**, and says which file. Passing
  because the rules were unreadable is the failure this package was built
  against.
- **A missing `environment.yaml` is reported once**, by its own entry, not a
  second time by the rendering check that depends on it.
- **Every fault, not the first.** One problem at a time turns a five-field
  mistake into five runs.
