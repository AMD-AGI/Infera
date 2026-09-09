# measure_stock

Five measurements against the unpatched deployment, in a fixed order, producing
two handoffs.

## Why one task and not three

Correctness and performance have to run against the same deployment instance and
must not overlap in time — a saturating trace replay during an eval invalidates
both. As sibling tasks with the same `froms`, agent_sys would schedule them
concurrently, and there is no synchronisation primitive between them; lining them
up would need a rendezvous file, which is an edge the graph cannot see. The
framework's rule is one producer per kind per subgraph, not one kind per task.

## Why two handoffs and not one

Correctness and performance are different kinds of evidence, they bind different
validators, and `compare` judges them on different lines. Merging them would
force `check_acceptance` to also understand AIPerf's CSV.

## The order is part of the measurement

smoke, needle, probe, llm-eval, then the replay rounds. "Round 1 is cold for this
trace" is only true if the same things happened before it in both arms, so the
sequence is recorded with timestamps in `env/steps.json` and `compare` fails if
the two arms disagree.

## The three directions of text

The mission asks for long and short text plus a needle. Split three ways, because
each fails in a way the other two cannot see: a short prompt with a short answer
catches fluent nonsense; a short prompt with 512 tokens of output catches
degeneration into repetition, which produces a score rather than an error; and a
long prompt with a short answer is the needle.
