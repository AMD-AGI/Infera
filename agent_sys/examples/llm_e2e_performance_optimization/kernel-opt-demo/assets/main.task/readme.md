# main — stage 4 of the end-to-end optimization flow

This package runs **one** stage of the six-stage
`llm_e2e_performance_optimization` flow: **kernel optimization**. It takes a
workset describing one operator, hands it to KernelForge, and hands back a
reproduction kit carrying the optimized kernel and the evidence for it.

```
main                        non-leaf: readme, no entry.sh, NO agent
│                           inputs [] · outputs [kernel_optimization]
│
├── publish_workset         program: workset_publisher
│                           out: workset                    [code]
│                             check_workset_shape    program · seconds · strong
│
└── optimize_kernel         is_end · ai: kernel_opt_lead (opus)
                            in:  workset
                            out: kernel_optimization        [code]
                              check_optimization_shape      program · seconds · strong
                              check_speedup_substantiated   re-measure · minutes · weak
```

## Why two leaves

`publish_workset` looks like overhead — it copies a directory. It is here for
three reasons and each one is load-bearing:

1. **It is the seam where stage 3 plugs in.** Stage 3's deliverable *is* a
   workset. When it exists, this leaf is re-pointed with `--var workset_dir=` or
   replaced by stage 3's own output entry, and nothing downstream changes.
2. **It makes the handoff real.** With one leaf there is no transfer between
   tasks, so `froms` is empty, the input-validation phase is empty, and the
   thing the mission calls "handoff" is a declaration nothing exercises. With
   two, `workset` is produced by one task and consumed by another, and
   `check_workset_shape` runs on both sides of the boundary.
3. **It gives the expensive leaf something to fail early against.** A malformed
   workset is caught by a seconds-long program check before an hours-long,
   dollars-costing agent is ever dispatched.

## The one grant, and where it lives

`main` owns the vocabulary: both kinds, both `write`. Permissions are inherited
downwards, so the root is the one place that has to know the whole set, and
**WRITE covers READ** — which is why `optimize_kernel`'s read of `workset` needs
nothing extra here.

## What `main` itself does

Nothing. Its work *is* its subgraph, so it carries a readme and no `entry.sh`,
and it names no agent. There is nothing of its own to validate, which is why its
`validators` list is empty — and separately, a closure's `validators` list has
no runtime consumer today, so every real validator in this package hangs off a
handoff **kind** instead.

## What is not here

- **Stages 1, 2, 3 and 5.** No serving, no profiling, no workset *generation*,
  no re-integration. The workset this package consumes was produced by hand.
- **No end-to-end claim.** A kernel-level speedup is not a service-level
  speedup, and this package measures only the former. See
  `assets/worksets/<name>/integration.md` for what stage 5 would still owe.
- **No GPU reservation.** `--var gpu=` names a card; nothing enforces it.
