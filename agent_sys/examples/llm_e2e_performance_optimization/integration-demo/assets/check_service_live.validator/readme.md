# check_service_live

Completeness, `strong`. Five rules over a deployment record, one boolean per
handoff id.

`strong` is claimed honestly here: every rule is decided by reading a file that
either says the thing or does not. There is no rule that is approximately right.

## The rules

1. **Shape.** The items `reproducible` declares are present with substance in
   them. "Non-empty" is two bytes rather than zero, because a zero-byte
   `workers.json` is the shape of a curl that timed out, and an empty file that
   exists would otherwise pass a presence check.
2. **Registration.** Exactly one worker, `disagg_mode` `mixed`, status `active`.
   Two workers is not a stricter version of one — it means a registration from an
   earlier round is still in etcd, and the router would then split load across a
   worker that exists and one that does not.
3. **Arithmetic.** The smoke probe asked for `17 * 23` and got `391`. Both the
   marker `SMOKE_ARITHMETIC_OK` and the value are checked: the marker alone would
   pass a stale file from a previous run, and the value alone would pass a model
   that emitted it inside an apology.
4. **Faults.** The engine log tail carries none of `memory access fault`,
   `HIP error`, `CUDA error`, `Traceback`.
5. **Round agreement.** The record's own `cuda_graph` and `profiling_enabled`
   match what its round name means.

## Why rule 5 is here

It is the rule that is easiest to leave out and the one that catches the
expensive mistake. The two rounds differ only in those two flags. A
`serve_profiled` that came up with graphs on would still serve, still pass rules
1 through 4, and still produce a trace — one in which every decode step is a
single graph launch and no kernel is attributable. Three steps later
`kernel_scan` would emit a plausible-looking CSV describing nothing. Checking the
record against its own declared round is what makes that failure loud at the
point it happens.

## It reads files, and does not call the endpoint

A validator that re-probed the live service would pass or fail on the state of
the cluster at validation time rather than on the artefact it was handed, so the
same handoff would get different verdicts on different days. Everything checked
here is in the record, which is also what makes the verdict reproducible by
somebody who was not there.

## Why it does not short-circuit

All five rules run even after one fails. A deployment costs a quarter of an hour
to reproduce, so reporting the whole set of problems at once is worth the
microseconds; stopping at the first would make the operator pay that quarter hour
once per problem.
