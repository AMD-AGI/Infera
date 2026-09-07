# General validator specs

Workflow-independent validators. Main spec §4.5: *"the degenerate case of a
template — one whose `config` is empty"*, in **their own directory**, so the
distinction from a task package's specs is visible in a directory listing rather
than only in a registry.

Three, and the set is chosen so that **each of the three quality dimensions is
represented** (criterion 15). A registry of forty completeness checks with
nothing checking trustworthiness is the failure that criterion exists to make
answerable.

| Folder | Dimension | Strength | Body |
|---|---|---|---|
| `schema_conformance` | completeness | `strong` | `entry.sh` — a schema either matches or it does not |
| `downstream_loads` | usability | `strong` | `entry.sh` — the next task's loader is the criterion |
| `production_grade` | trustworthiness | `weak` | `readme.md` only — an agent judges, and the label says so |

`production_grade` is the case the withdrawn Python callable could not express:
there is no function to register, only a description an agent carries out. It is
labelled `weak` because it cannot state, in advance, the number or the comparison
that decides it — which is spec §5.6's observable test, and a weak check labelled
strong is worse than no check.

> **Placement is reported, not settled.** Main spec §4.5 says general specs live
> in "their own directory" and does not say where. They are here because this is
> the module that owns validators; if the repository grows one general-spec
> directory for all five spec kinds, they move there unchanged.
