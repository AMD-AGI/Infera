# ROCm Infera user manual (Sphinx)

The user-facing documentation site for Infera. Markdown (MyST)
authored, built with Sphinx + the Furo theme.

## Build

```bash
# The diagrams are Graphviz DOT; Sphinx shells out to the `dot` binary, which
# is a system package, not a pip one. `make html` runs with -W, so a missing
# `dot` is a build failure, not a skipped image.
sudo apt-get install -y graphviz   # macOS: brew install graphviz
cd manual
pip install -r sphinx/requirements.txt
make html
# open _build/html/index.html
```

Live-reload preview while editing: `pip install sphinx-autobuild && make serve`
(serves on :8800).

## Layout

```
manual/
├── index.md                 # landing page (cards + toctrees)
├── getting_started/         # overview, installation, quickstart
├── serving/                 # server & API, engines, deployment
├── features/                # the topic one-pagers
└── reference/               # cli, environment, troubleshooting, glossary
```

## Scope

This manual is the **front door**: clear, task-oriented, one self-contained page
per feature. It is **self-contained** — no links into the source tree that a
manual reader can't open. If deep material is useful to a user, inline the
relevant part here rather than linking out. Keep code-internal detail (source
paths, function names) out of the manual.

## Conventions

- One feature = one page under `features/`, with a `{admonition} One-pager`
  summary box at the top.
- Commands use the canonical `python -m infera.<thing>` form (no console
  scripts in this build).
- Cross-link liberally with `{doc}` / relative `.md` links so the sidebar and
  inline references stay coherent.
