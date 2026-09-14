# main

A non-leaf whose subgraph has two independent reviewers, a reconcile step,
and a consumer. The reviewers deliberately disagree on student_a, so
check_agree fails the merged review and downstream never starts.
