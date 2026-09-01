# main

A non-leaf. Its work is its subgraph, so it runs nothing itself and declares no
agent.

Eight leaves, and the shape is a chain because the thing being built is a
controlled comparison:

```
seed_patch -> apply_patch -> serve_stock -> measure_stock
                          -> serve_patched -> measure_patched -> compare -> packup
```

Two deployments of GLM-5.3-Flash on one node, in one session, differing in
exactly one thing: whether the patched files are bind-mounted over their paths in
the image. Everything else — node, image, weights, engine argv, trace, the order
of the five measurements — is held identical, because any second difference would
be indistinguishable from the one under test.

Three edges carry an argument rather than a dependency:

- `serve_stock` waits for `apply_patch`, not because it needs the overlay (it
  applies none of it) but because building and checking the overlay costs seconds
  while the stock arm costs twenty minutes. A patch that will not apply should
  fail before the measurement.
- `serve_patched` waits for `measure_stock`, not `serve_stock`. Its first act is
  to tear the stock deployment down, so the edge has to mean "the stock arm has
  been measured"; wiring it to `serve_stock` would let agent_sys schedule it
  alongside `measure_stock` and destroy the deployment mid-measurement.
- `packup` waits for everything, including `compare`, because the deliverable
  carries the verdict.

`seed_patch` is a mock and is the first leaf to delete. When the
kernel-optimization stage lands, its producer takes over `kernel_patch` and
`apply_patch`'s `froms` points at it instead.
