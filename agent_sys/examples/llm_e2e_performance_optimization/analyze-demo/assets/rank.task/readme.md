# rank — classify, filter, rank

Turn a kernel-level GPU time table into the ranked list of operators worth
optimizing.

## What it does

**Classify first, rank second.** Every row of the input lands in exactly one of
five buckets, driven by the symbol-name rules in
`assets/lib/kernel_taxonomy.yaml`:

| bucket | in the candidate pool | why |
|---|---|---|
| `collective` | no | NCCL/RCCL/MSCCL; forge-loop's single-GPU driver cannot measure one |
| `vendor_tuned` | no | Tensile/rocBLAS/rocPRIM assembly, tuned by table rather than by source edit |
| `framework_native` | no | PyTorch ATen, shared by everything on the machine |
| `routable` | yes | has an identifiable owner and an editable source |
| `unknown` | no | no rule matched; recorded so the taxonomy can be extended |

Then within `routable`: drop below `$AD_MIN_PCT` and `$AD_MIN_CALLS`, drop rows
the profiler recorded no shapes for, sort by share of GPU time, take the top
`$AD_TOP_N`.

## Why the order cannot be reversed

In the sample profile a single collective is **78.98%** of GPU time. A plain
top-N by percentage puts it first, and it is the one entry in the table that
cannot become a forge-loop task. Classification is what stops the ranking from
leading with an unusable answer.

## What it does not decide

Whether these are the *right* kernels to optimize. `check_worklist_shape` checks
that the document is complete, ordered and bounded. The selection quality is a
question about the profile, and mission.md is explicit that the ranking
algorithm is not the focus of this round.

## Watch out

Every row of the input reaches the output, including the excluded ones, each
carrying an `excluded_reason`. Dropping them would make the list unauditable: a
reader could not tell a kernel that was considered and rejected from one the
profile never recorded.
