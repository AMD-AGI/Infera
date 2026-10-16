# dangling — the body of a task that never loads

This exists so that the *only* fault in this package is the one it is for.

`main.yaml` names a handoff kind, `nonexistent`, that no loaded package
declares. That is a cross-registry fault: no schema can see it, and
`check_closures`' check 1 is what reports it, naming this package's `main.yaml`.

If this readme were missing, `body` would fail its schema too — `task.body` is
required and the assets convention would have found nothing to fill it with —
and criterion 11's message would carry two faults where the criterion asks for
one, named by file.

Nothing runs it. There is no `entry.sh` here for the same reason.
