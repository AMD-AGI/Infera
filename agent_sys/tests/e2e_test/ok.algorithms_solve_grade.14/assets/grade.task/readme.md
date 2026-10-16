# grade — marking three students' answers, as a subgraph

Like `main`, this task runs nothing itself: it carries this readme and **no
`entry.sh`**, and its work is the six steps below. It is the package's depth-2
case — a subgraph nested inside a subgraph — and the only reason it is a
subgraph rather than one large step is that grading genuinely is six things,
each of which can fail on its own terms.

## What comes in and what goes out

In: `problems`, and the three independent answer sets `solutions_a`,
`solutions_b`, `solutions_c`.

Out: `scores`, and **only** `scores`. The other five kinds this subgraph
produces — `review_x`, `review_y`, `review`, `harness`, `extra_tests` — are
internal, and the engine enforces that: a handoff produced inside a subgraph
leaves only through the end entry's outputs, and anything outside consuming one
of the five is a fatal load error naming both sides. Cancelling `grade` must not
silently block a step that had quietly reached inside it.

## The six steps

| | |
|---|---|
| `review_x` | an agent node. Reads the problems and all three answer sets and writes one reviewer's opinion, `review_x` |
| `review_y` | an agent node. A second reviewer, independently, on the same material — `review_y` |
| `reconcile` | a program node. Merges the two opinions into one `review`, recording where they disagreed rather than hiding it |
| `harness` | a program node. Builds a C++ test harness for the problem set and compiles it — `harness` |
| `extra_tests` | an agent node. Writes additional cases the students' own tests would miss — `extra_tests` |
| `score` | a program node. Runs every answer against the harness and the extra tests and writes `scores` |

`review_x` and `review_y` are the second fan-out in this package, and they are
also where a rule becomes visible: **both declare `froms: []`**, and that is
correct. Every kind they consume — `problems` and the three solution kinds — is
an *input of `grade` itself*, produced outside this subgraph, so no edge is
derived and there is no earlier entry to point at. A kind nobody inside a
subgraph produces resolves to the parent's own input.

`reconcile` is where the two reviews join, and it is a program rather than a
third opinion on purpose: reconciling two structured documents is a merge, and
handing a merge to a model would make the one deterministic step in the middle
of this graph non-deterministic for no gain.
