# check_kernel_table — usability, strong

The gap-analysis table is readable by everything downstream.

## What it checks

1. The header is one of the two shapes Magpie writes: the six base columns, or
   those six plus the thirteen `KernelSourceInfo` columns that
   `--find-kernel-sources` appends.
2. At least `min_kernel_rows` data rows.
3. `items/gap_analysis/` holds the CSV as Magpie wrote it.
4. `% Total` sums into `[pct_total_min, pct_total_max]`.
5. No row has an empty kernel name.

## Why rule 4 is a range and not an equality

Magpie rounds each percentage to two decimals. Over the ~143 rows of the sample
profile the rounding accumulates and the sum lands at 99.94 rather than 100. The
upper bound catches a double-counted table; the lower bound is deliberately
loose so that a table truncated by `--top-k` still passes.

## Why `strong` is honest here

Every rule is decided by counting or by comparing against a constant. None of
them is approximate, and none of them is a judgement about the profile.
