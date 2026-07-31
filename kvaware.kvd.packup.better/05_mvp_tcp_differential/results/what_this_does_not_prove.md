# What r4/r5 does not prove — the boundary of the claim

This experiment produces a result that is easy to over-read in both directions.
It is worth being precise about where the evidence stops, because the whole
point of running a differential is to get an *attribution*, and an attribution
you overstate is worse than none.

## The claim, stated exactly

**Supported:**

> On a same-host PD pair over `MC_FORCE_TCP`, output was garbled with
> kvaware+kvd ON and equally garbled with them OFF. The garbling is therefore
> not attributable to kvaware or kvd.

**Not supported, and not addressed:**

> kvaware+kvd produce correct output.

Arm A scored 0/4. Arm B scored 0/4. A comparison in which both arms fail can
tell you the two arms are alike; it cannot certify either one.

## The three over-readings to avoid

**1. "The features passed."** They did not pass anything. There was nothing to
pass. The experiment establishes an absence (no *difference*), not a presence
(no *correctness*). "No regression on a broken substrate" is the whole claim.

**2. "The features were exercised."** Four short, prefix-disjoint prompts give a
KV-offload path nothing to do. Even in the arm where kvd was wired and
connected, there is no evidence in this round that it stored or served a single
block — its counters were not read. "Connected" is not "serving", and "serving"
would need a workload with a long shared prefix plus the daemon's own counters
before and after.

**3. "Same-host PD is the cause"** or **"`MC_FORCE_TCP` is the cause."** Neither
was convicted. They are entangled: `MC_FORCE_TCP` is in use *only because*
same-host mooncake RDMA fails on this fabric. No single-node experiment can
separate them, and none was attempted — moving to two nodes removed both at
once. They were eliminated, not diagnosed.

## What would be needed to close each open question

| Question | What it would take |
|---|---|
| Are kvaware+kvd correct? | The same probe on a substrate that is known-good with the switches OFF — i.e. two nodes, real RDMA, a baseline that scores 4/4 first. Then flip the switches and re-run. |
| Does kvd actually serve? | A workload with a long shared prefix across sessions, plus the kvd daemon's `gets`/`hits`/`misses`/`sets` counters snapshotted before and after. |
| Is kvd *the thing* doing the work, or is the GPU radix cache? | Kill the engine, keep the daemon, verify VRAM is back to idle, then replay. Hits with `sets` flat means the new engine read blocks it never wrote. |
| Does kv-aware routing actually route? | At least two workers in the same role — with one, the scorer has no alternative to choose between and can only be confirmed *loaded*, not *effective*. |
| What causes the garbling? | Not answered here and not answerable on one node. Would need same-host PD and `MC_FORCE_TCP` varied independently, which requires a fabric where same-host RDMA works. |

## Why the round was still worth running

It cost one extra cold start of about two minutes and it removed kvaware and
kvd from the suspect list for a failure that would otherwise have swallowed the
next several rounds. The alternative — reading the KV plane's source looking for
a corruption bug that was never there — is the expensive path.

That is the general lesson: when a system produces wrong output and you have a
switch, flip the switch before you open the source. And when the baseline is
equally broken, the correct response is to *fix the substrate*, not to keep
bisecting on top of it.
