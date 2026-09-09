# packup

Assemble the five upstream handoffs into one directory a colleague can be handed,
in the layout `experiment-result-packup` defines.

## It touches no cluster

Everything it needs was published by the tasks before it. That is the property
that makes the packup a deliverable rather than a second copy of the run: if it
needed the machine, the machine would be part of the artefact.

## What it carries, and what it does not

- **Scripts, verbatim, from the package.** The layout reference asks for the real
  thing so byte-level flags survive. They come from the package rather than from
  a handoff because the handoffs carry `command` items with placeholders instead
  — see `temp/bugs/002` for why a copied bring-up script cannot pass the locality
  seal.
- **The per-round invocations**, one `command.<kind>.sh` each.
- **Results**: both rounds' AIPerf summaries and exports, the kernel ranking, the
  trace manifest, the smoke evidence.
- **Logs**, gzipped, as the layout reference suggests.
- **Not the torch traces.** 462 MB per capture, and the ranking derived from them
  is here instead. `results/trace_manifest.json` identifies them by SHA-256, so a
  copy held elsewhere can be matched to this run.
- **No `patches/`.** Nothing in this pipeline needed a patch to the engine, and
  the reference says to omit a folder rather than ship it empty.

## The three mandatory documents are generated, not templated

`README.md`, `REPRODUCE.md` and `environment.md` are never omitted, and
`environment.md` is called the number-one reproduction trap. They are rendered
from the handoffs by `assets/kit/render.py`, so every number in them was read
out of a published artefact and a document that disagrees with the run cannot be
produced.

A missing metric renders as `-` rather than as `0`. The two are different facts,
and a table that shows them the same is how a gap becomes a claim.
