# main — the whole of this package, as one task

This task runs nothing itself. Its work is its subgraph, and it carries this
readme and **no `entry.sh`** — the rule is *`entry.sh` versus a subgraph*, not
*a body versus a subgraph*, so a non-leaf still has to say what it is for or it
is a step nobody can review.

## What it does

| | |
|---|---|
| `directions` | an agent node. Picks the CLRS topics this round will cover, from the closed list in `assets/catalog/clrs_topics.json`, and writes `directions` |
| `problems` | an agent node. Turns those directions into a concrete problem set drawn from `assets/catalog/leetcode_index.json`, and writes `problems` |

## Why this shape

Two AI leaves and one handoff between them. That is the smallest graph in which
an agent and a dependency edge are both exercised: remove the second task and
nothing consumes a handoff; make either task a program and no model is called.

Three validators hang off the two kinds, so output validation comes with the
shape rather than being added to it — and `check_directions` runs in both
positions, `directions`' output phase and `problems`' input phase, which is the
cheapest demonstration that a phase is a position and not a kind of validator.

## What it needs

Credentials for the Claude backend and a working sandbox. No compiler, no GPU,
no cluster, and no network access during the run: the two catalogues under
`assets/catalog/` are shipped with the package precisely so that the closed
lists the validators match against do not depend on reaching leetcode.com.
