# main — the whole of demo2, as one task

This task runs nothing itself. Its work is its subgraph, and it carries this
readme and **no `entry.sh`** — the rule is *`entry.sh` versus a subgraph*, not
*a body versus a subgraph*, so a non-leaf still has to say what it is for or it
is a step nobody can review.

## What it does

It runs a small algorithms course, end to end.

| | |
|---|---|
| `directions` | an agent node. Picks the CLRS topics this round will cover, from the closed list in `assets/catalog/clrs_topics.json`, and writes `directions` |
| `problems` | an agent node. Turns those directions into a concrete problem set drawn from `assets/catalog/leetcode_index.json`, and writes `problems` |
| `solve_a` | an agent node. One student's C++ answers — `solutions_a` |
| `solve_b` | an agent node. A second student's, independently — `solutions_b` |
| `solve_c` | an agent node. A third student's — `solutions_c` |
| `grade` | **a non-leaf.** Six more steps: two reviewers, a reconciliation, a test harness, extra tests, and a scoring pass. Emits `scores` |
| `optimise` | an agent node. Takes the problems, the three sets of answers and the scores, and writes a faster set — `optimised` |

## Why this shape

`examples/demo/` is a chain of three and proves the format loads. It cannot
show three things this one is built around, and each is here for exactly that:

- **Fan-out.** `solve_a`, `solve_b` and `solve_c` all consume `problems`,
  become eligible in the same scheduler pass, and run in parallel threads.
- **Fan-in.** `grade` joins all three. They have three *different* output
  kinds because the engine records one producer per kind per subgraph, so a
  single `solutions` kind would let the last writer erase the other two —
  silently, with no error anywhere.
- **Depth-2 nesting.** `grade` is a subgraph inside a subgraph. Only its end
  entry's output, `scores`, leaves it; `review_x`, `review_y`, `review`,
  `harness` and `extra_tests` are produced and consumed entirely within it.

Ten validators hang off the twelve handoff kinds. Four of them compile and run
real C++ through `assets/lib/cpp.py`, so a submission that does not build, or
that loops for ever, is a recorded verdict rather than a hang.

## What it needs

Credentials for the Claude backend, `g++` on `PATH`, and a working sandbox.
Nothing else — no GPU, no cluster, no network access during the run. The two
catalogues under `assets/catalog/` are shipped with the package precisely so
that the closed lists the validators match against do not depend on reaching
leetcode.com.
