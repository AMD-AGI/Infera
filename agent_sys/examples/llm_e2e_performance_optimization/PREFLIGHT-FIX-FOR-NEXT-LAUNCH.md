# The instruction change the next launch needs — drop-in replacement text

Written 2026-09-06T14:54:08Z by m35, after run `20260906T140831-026b96` died at
`_off` preflight. Diagnosis: `bug.record.2026-09-06.md` §13, gate:
`RUNG5-CHECKLIST.md` P2d.

**Why it goes in the instruction and not in a patch:** the kit is written by
m1's agent from `--var instruction=…`. The sealed `preflight.sh` that aborted
lives inside a dead run's handoff. The next run's m1 writes a fresh one, so the
only durable place to fix it is the sentence that produced it.

**Replaying m1 does NOT avoid this.** m2 runs `deploy.sh`/`preflight.sh` *from
the kit*, so a replayed kit carries the same abort. The kit must change.

---

## What the current instruction says (the clause that fired)

> *"If a foreign container holds a GPU, abort immediately as a stranger."*

## What it must say instead

> **A container counts as holding a GPU only if GPU memory is attributable to
> it. Mapping `/dev/kfd` into a container is NOT evidence that it holds a GPU —
> long-lived development containers on this host map `/dev/kfd` permanently and
> hold zero VRAM, so a test on `HostConfig.Devices` names them on every run and
> your stranger branch fires every time a card reads busy for any reason. Use
> `rocm-smi` occupancy, and when you cannot attribute the memory to a specific
> container, the honest classification is `unknown` — and `unknown` WAITS, it
> does not abort. Abort as a stranger only when you can name a container AND
> attribute non-zero VRAM to it. Your abort message must state which of the two
> conditions was met and how you measured the second.**

## And add this, because it caused the busy reading in the first place

> **The validator `check_deploy_serves` performs its own bring-up on the same
> cards and its container is removed while VRAM is still draining. Expect a
> busy card with NO container to attribute it to for up to ~60 seconds after a
> preceding stage finishes. That is the single most likely thing your preflight
> will meet, and it is precisely the case the wait exists for.**

---

## Verification the next kit must pass before it is trusted

A known-answer test, on this host, costing nothing:

```
# Instrument 2 must return EMPTY right now — all eight cards read 0.
# If it names rc_26_7_902 or xiaoming-dev, the kit has the old defect.
```

Measured 2026-09-06T14:49:43Z: cards `0 0 0 0 0 0 0 0`, both containers running.
**So on a quiet host the correct answer for instrument 2 is the empty set, and
the old kit returns two names. One command tells the two kits apart.**

---

## What is NOT proposed here

- No edit to the sealed artefact of the dead run.
- No widening of `DK_VRAM_BUSY_BYTES`. The threshold was never the problem; the
  classification was.
- No removal of the stranger branch. A genuine stranger must still abort — the
  change is what qualifies as one.
