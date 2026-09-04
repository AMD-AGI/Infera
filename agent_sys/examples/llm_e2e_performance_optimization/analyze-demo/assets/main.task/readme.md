# main — analyse a kernel profile into KernelForge worksets

A non-leaf. Its work is its subgraph and it executes nothing itself, which is
why this folder carries a `readme.md` and no `entry.sh`.

Stage 3 of `temp/mission.md`: analyse performance data, produce the list of
operators worth optimizing, and build a complete workset for each one.

## The six leaves

| leaf | kind | needs | produces |
|---|---|---|---|
| `seed_table` | program | a CSV on disk | `kernel_table` — mock of the upstream profiling handoff |
| `rank` | program | nothing beyond its input | `kernel_worklist` — the ranked operator list |
| `identify` | program | network, to index kernel source repos | `operator_identity` — where each operator lives |
| `build_workset` | **ai** | an Anthropic endpoint | `operator_workset` — the material KernelForge reads |
| `verify_workset` | program | 8×MI355X | `workset_evidence` — the measured correctness and performance |
| `packup` | program | nothing | `analyze_packup` — the deliverable |

The split follows failure mode and cost class. `identify` can fail because a
repository is unreachable; `build_workset` can fail because a model wrote a
driver that does not run; `verify_workset` can fail because a kernel does not
compile. Merging any two of them would put two unrelated failure modes in one
attempt.

## What this stage produces, and what it does not

It produces the **measuring apparatus**, not the thing measured. The kernel
source already exists inside the framework; forge-loop edits it in place in a
git worktree, so a copy in a handoff would be useless to it. What this stage
writes is the driver that decides whether an optimized kernel is still correct
and how much faster it is — and forge-loop treats that driver as a protected
file the optimizing agent may not modify.

See `DESIGN.md` section 5.4.
