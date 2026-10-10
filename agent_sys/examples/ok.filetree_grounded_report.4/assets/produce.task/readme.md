# produce — collect a file manifest

Walk this package's directory tree and write one row per file:

```json
{"path": "main.yaml", "lines": 62, "sha256_prefix": "3f9c1a4b"}
```

plus a `totals` object carrying the file count and the line count.

Reproducible by hand, outside the system:

```bash
AGENT_SYS_DEMO_CONTENT=/tmp/out sh assets/produce.task/entry.sh
```

Every number in the output can be recomputed from the tree, which is what lets
`check_facts` be honestly `strong` rather than a rubric.

**It does not measure how long it took**, and nothing here records a duration.
That absence is deliberate — it is what `describe`'s goal asks for and cannot
be given, and therefore what `check_grounded` catches.
