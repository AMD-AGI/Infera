# publish_workset — put the workset into the graph

A `kind: program` leaf. It copies one workset directory into this task's output
handoff and writes the `code` content type's `README.md`. There is no judgement
in it, which is why it is a program and not an agent.

**This readme is not an instruction channel.** A program body reads `entry.sh`,
not prose. It is here because `task.schema.json` requires `body.readme` for
every task, and because the next person to read the package needs to know why
this leaf exists.

## What it does

```
assets/worksets/$KFO_WORKSET/        (or $KFO_WORKSET_DIR, if set)
  └─copytree─▶ $AGENT_SYS_OUTPUT_WORKSET/items/codes/$KFO_WORKSET/
               $AGENT_SYS_OUTPUT_WORKSET/README.md   (Purpose / Interface / Boundary)
```

`items/codes/` is not decoration: `content_type: code` requires exactly one
top-level item named `codes` and rejects anything else before a reader gets to
it. The workset keeps its own directory name inside, so nothing is renamed and
`check_workset_shape` can look for the files the layout mandates.

## Why it exists at all

Stage 3 of the series produces worksets. Stage 3 does not exist yet. Rather than
have stage 4 reach outside the graph for its input — which would make the
handoff a fiction — the workset ships as package data and this leaf publishes
it. When stage 3 lands, one of two things happens and neither touches
`optimize_kernel`:

- `--var workset_dir=<what stage 3 produced>`, or
- this entry is replaced in `main.yaml`'s subgraph by stage 3's own.

## Variables

| variable | source | meaning |
|---|---|---|
| `KFO_WORKSET` | `--var workset=`, default `sampler_vocab_softmax` | which workset under `assets/worksets/` |
| `KFO_WORKSET_DIR` | `--var workset_dir=`, default empty | an absolute directory to publish instead of package data |

## Failure behaviour

It refuses, loudly, on: an unset `AGENT_SYS_OUTPUT_WORKSET`; an empty
`KFO_WORKSET`; a source that is not a directory; and a destination that already
exists.

That last one is deliberate. An existing destination means this version was
already written, and overwriting a sealed artefact is worse than failing. **The
body deletes nothing** — not the destination, not the source, not a stale
partial copy. If a re-run is genuinely wanted, the run gets a new version, which
is the store's job and not this body's.
