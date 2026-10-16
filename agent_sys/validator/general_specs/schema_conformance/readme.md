# schema_conformance

Read `inputs.json` for the handoff ids under check and `args.json` for the
configured schema, then write `verdict.json` mapping each id to `true` or
`false`.

A schema check is the easiest dimension to make `strong`: the schema exists, so
the criterion can be stated in advance. That is exactly the test spec §5.6 gives
for the label being honest.

`entry.sh` runs it. This file is required anyway — a check nobody can read is a
standard nobody can review.
