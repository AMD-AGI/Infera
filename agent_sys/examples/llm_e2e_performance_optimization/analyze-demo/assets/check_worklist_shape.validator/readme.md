# check_worklist_shape — completeness, strong

The worklist is a complete, auditable, ordered document.

## What it checks

1. Every kernel carries a `bucket` from the known set.
2. `selected` and `excluded_reason` are complements: a selected row carries an
   empty reason, an excluded one carries a non-empty reason. A row that is
   neither is a hole in the audit trail.
3. The selected count is between `min_selected` and the declared `top_n`.
4. Selected rows are ranked `1..N` contiguously and ordered by descending share
   of GPU time.
5. Every selected row carries at least one case, and every case selector carries
   a `CASE_ID`. A row with no shapes cannot become a workset.
6. The unclassified share is at or below `max_unknown_ratio`.

## What it does not check

Whether the selection is a good one. That is a question about the profile, not
about the document, and mission.md is explicit that the ranking algorithm is not
the focus of this round.

## Which rule ages

Rule 6. A workload the taxonomy has not seen pushes rows into `unknown`, and
past the threshold the rules in `assets/lib/kernel_taxonomy.yaml` have fallen
behind the workload. Measured on the sample profile: 0.077 against a ceiling of
0.3, so there is room before it starts failing.
