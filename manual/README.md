# ROCm Infera user manual (Sphinx)

The user-facing documentation site for Infera. Markdown (MyST)
authored, built with Sphinx and the ROCm documentation theme.

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

The sidebar is defined in `sphinx/_toc.yml.in`, with five groups: Install,
Conceptual, Reference, Examples, and About. Existing source paths are kept stable
when pages move between sidebar groups. Recipes are grouped under Examples and
simulation pages under Conceptual.

The release version is set in `conf.py`. Use `{{ infera_version }}` in prose to
keep it consistent; update literal version pins in installation commands when
preparing the next release. Documentation release notes live in
`about/release-notes.md`.


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
