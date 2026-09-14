# check_facts — completeness, strong

Every row of the `facts` manifest carries the keys `args.json` names, and the
`totals` object agrees with the rows.

`strong`, and honestly so: the check is **total** over the document. It reads
every row and recomputes both totals. There is nothing approximate in it, which
is exactly what the label claims and the only thing that makes the label worth
carrying.

## How it runs

`entry.sh` is the body. `validator.ScriptBodyRunner` runs it in a freshly
allocated zone with `args.json` and `inputs.json` beside it, and it writes
`verdict.json` — one boolean per handoff id in `inputs.json`.
