# check_faster — usability, weak

For every problem directory in an `optimised` handoff:

1. `solution.cpp` and `report.json` are both present, and the report carries
   all eight keys with none of them empty.
2. `solution.cpp` compiles with `g++ -O2 -std=c++17`.
3. It answers **every** worked example the `problems` artefact carries for that
   problem id, each within 30 seconds.
4. Either it has fewer non-blank non-comment lines than `baseline_lines`, or
   `new_algorithm` differs from `baseline_algorithm`.

An `optimised` handoff with no problem directories fails. `all(())` is `True`,
and an empty artefact passing every check above would be the loudest possible
case of an instrument pointed at nothing.

## `new_lines` is recounted, not read

Step 4 counts the lines itself, from the source, and ignores the `new_lines`
the producer wrote. A field the producer fills is a field the producer can fill
wrongly, and this validator's entire subject is a claim the producer made about
its own work. A disagreement between the count and the report is printed,
because a producer whose arithmetic is off is worth knowing about, but the
verdict is decided on the count.

## Why this is `weak`, said plainly

**It cannot measure that the new implementation is faster.** The name says
`check_faster` and this is the thing it does not do. Three separate reasons,
and none of them is fixable inside this validator:

- **The inputs are tiny.** It runs the worked examples from the problem
  statement, which exist to illustrate the IO format. An asymptotic improvement
  does not show up on an input of size five; a wall-clock difference at that
  size is dominated by process start-up.
- **Wall clock on a shared machine is noise.** This runs beside eleven other
  tasks in the same graph, on whatever the run happens to land on. Two
  measurements a few milliseconds apart are not two facts.
- **There is no baseline binary here.** A `faster than` claim needs both
  programs run under the same conditions, and this handoff carries only one.
  Reaching into `solutions_a/b/c` for the other would mean this validator
  guessing at another kind's item layout, which is the sort of coupling that
  breaks quietly when the other author changes their mind.

So `baseline_lines` — the only number the comparison in step 4 rests on that is
not recomputed — is taken on trust from the report. That is a second, smaller
version of the same weakness, and it is stated here rather than papered over.

**What it does establish** is worth having: the rewrite still compiles, it still
answers every example it was given, and the improvement claim is *structurally*
supported — the source really is shorter, or the producer really did name a
different algorithm. A producer that wanted to fake a pass would have to write a
genuinely shorter correct program, or write down an algorithm name it did not
use, and the second of those is a different act from optimising.

The line count is crude in a stated direction: comment markers are stripped
without parsing string literals, so a line holding only `puts("// hi");` is
counted as blank and the total comes out *low*, which flatters the producer. It
is enough to tell 41 lines from 22 and not enough to tell 41 from 40.

## Why not call it `strong` anyway

Because `strength` qualifies a PASS, and it is the one field a consumer reads
before deciding how much a green tick is worth. A sophisticated check that is
silently approximate is worse than a crude one that says what it is —
`examples/demo/steps/describe.yaml` is the long-form argument, and it is the
failure mode this repository cares most about. Naming this `strong` would make
every downstream reader believe an optimised solution had been benchmarked.
None of them have been.
