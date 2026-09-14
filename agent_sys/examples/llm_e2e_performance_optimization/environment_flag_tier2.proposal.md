# PROPOSAL, UNAPPLIED — make `compare.py --environment` tier 2 instead of tier 3

Written 2026-09-06 by m35 at the leader's request, while run 4's m1 is running.
**Not applied, and not proposed for tonight.** m5 is not tonight, and the
package is being launched from repeatedly — a change to `compare.py` between
launches is exactly the stale-copy hazard this file's neighbours record.

Context: `M5-REACHABILITY.md` § AMENDMENT 2026-09-06T15:04:58Z.

---

## The gap, stated in the three tiers this project uses

`assets/compare.py:799-834` writes an environment record — but only under
`if args.environment:`. Without the flag the report ships with no record and
`check_environment` refuses it.

| tier | what exists today |
|---|---|
| 1 — changes the command | **nothing** |
| 2 — changes what the artefact must contain | **nothing** |
| 3 — requires someone to remember | `integrate_and_verify.task/readme.md:296`, *"`--environment` is not optional in practice"*, plus a `print(..., file=sys.stderr)` note at the moment of omission |

**Tier 3 is the only one populated, and tier 3 is the one that decays.** The
stderr note is unusually good for its tier — `compare.py:829-834` explains that
STEP 10 is a shell command the agent runs and reads, so it reaches the one actor
who can add the flag, in the same turn. **That is why this is a proposal and not
a defect report.** It is still tier 3.

---

## Option A — default the flag on

`--environment` defaults to `$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml`;
supplying it explicitly stays legal; supplying `--no-environment` is the way to
opt out.

**Blast radius, named:**
- **A default that equals the value everybody would have passed hides a broken
  flag.** This file's neighbours record that exact failure twice today
  (`--var gpu_devices`, `expect_ranks`). If the default path is ever wrong,
  nothing distinguishes "resolved correctly" from "resolved to the same string
  by accident".
- It changes behaviour for any caller relying on the current default-off, and
  `compare.py` has callers outside m5's step that I have not enumerated.

## Option B — name the omission IN the artefact

Keep the flag optional. When it is absent, write the absence into
`items/text.json` as a declared field rather than only to stderr — e.g.
`environment_record: {present: false, reason: "--environment not supplied"}`.

**Why this is the better half:** a consumer can then tell **absent by choice**
from **absent by failure**, which is precisely the distinction
`check_environment`'s current refusal cannot express. It also survives the
producer's stderr being discarded, which is a documented hazard in this package
(*"a validator's stdout is kept nowhere"*).

**Blast radius, named:**
- It changes the artefact's shape, so `structured_text`'s schema copy must change
  with it, and `check_environment` would need to learn to read the declaration —
  otherwise a report that honestly declares "no record" still refuses, and the
  only thing gained is a better error.
- **That second half is the real cost and it is why B is not free:** the value of
  B is entirely in the validator learning to distinguish the two, and that is a
  contract change, not a producer change.

## Option C — do nothing, deliberately

The stderr note reaches the right actor in the right turn, and the readme says
it. **If STEP 10 is reliably executed, tier 3 is sufficient here in a way it is
not elsewhere**, because the reminder is delivered *at the moment of the
omission* rather than in a document read beforehand.

**What would settle it:** whether any real m5 run has ever omitted the flag.
**Zero real `integration_report` has been produced on this cluster**, so the
sample is empty and C cannot currently be defended by evidence — only by
argument.

---

## Recommendation

**B, and not before an `integration_report` has been produced here at least
once.** A contract change to `check_environment` justified by a producer that
has never run is the shape this project has paid for repeatedly. **Produce one
first; then the question has a sample.**

**A is the tempting one and I would refuse it on the record above:** the
default would equal the value every caller passes, which is the exact
configuration in which a broken flag is invisible.
