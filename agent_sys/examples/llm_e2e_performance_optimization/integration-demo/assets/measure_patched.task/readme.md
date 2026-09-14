# measure_patched

The same five measurements, in the same order, against the patched deployment.

It runs `assets/accept/measure.sh` with `IT_ARM=patched` and changes nothing
else. A second implementation here would make the comparison meaningless: any
difference it introduced would be indistinguishable from a difference the patch
caused.

Read `measure_stock/readme.md` for why the five steps are one task, why they
produce two handoffs, and why the order is recorded.

## What the numbers here mean

Nothing on their own. They are interpretable only against the stock arm measured
in the same session, on the same node, against the same trace. The handoff's
`watchout` says so, and `compare` is the only thing that should be quoted from.
