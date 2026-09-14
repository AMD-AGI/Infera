# main — one real task

This package exists to run **one** real piece of work end to end and hand back
something a stranger can re-run. Its subgraph is a single entry, `serve_qwen`,
which is also its `is_end` — so `runbook`, that entry's output, is the one kind
that leaves this graph.

```
main                        non-leaf: readme, no entry.sh, NO agent
│                           inputs [] · outputs [runbook]
│
└── serve_qwen              is_end · ai: deployer
                            out: runbook                   [code]
                              check_packup_shape   program · seconds    · strong
                              check_reproduces     claude  · gpu_hours  · weak
```

`main` is a **non-leaf**: its work *is* its subgraph, so it carries a readme and
no `entry.sh`, and it names no agent — `closure.schema.json` requires one of a
leaf and of nothing else.

There is nothing for `main` itself to do and nothing of its own to validate. It
is here because a non-leaf root is the shape `examples/demo` and
`examples/demo2` use and the shape the suite exercises, and because the day a
second step is added — a benchmark after the bring-up, a teardown check — it
goes in the subgraph beside `serve_qwen` with no change to anything else.

The one thing `main` owns is the **grant**: `handoffs/runbook`, write.
Permissions are inherited downwards, so the root is the one place that has to
know the whole vocabulary, and `write` covers `read` for whatever consumer
arrives later.
