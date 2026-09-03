# check_env_report_shape — completeness, strong

The `env_report` handoff is **one well-formed JSON report covering all seven
capabilities**, each with the level this package assigns it, a status from a
closed set, a token of the right shape, a written account of how it was
obtained, and the proof key its capability owes — plus a README with four
non-empty sections and an install report that is not empty.

It verifies **no token against anything**. That is `check_capabilities_genuine`,
and the split is `validator` spec §5.3's cheap-first ordering: a malformed
report is the cheap failure, and rejecting it here means the expensive body
never starts two subprocesses to discover the same thing.

## The rules, in full

| rule | applies to | the number |
|---|---|---|
| four headings — `Purpose`, `Schema`, `Method`, `Limits` | `README.md` | — |
| **content lines** — non-blank, not a heading, not a fence marker | `README.md` | 10 |
| no unfilled placeholder — `TODO`, `TBD`, `FIXME`, `XXX`, `to be filled in`, `<…>` | `README.md` | — |
| the four top-level keys | `items/text.json` | — |
| `nonce_digest` is 12 hex characters | `items/text.json` | — |
| exactly the seven capability keys, no more and no fewer | `capabilities` | 7 |
| `level` equals the level this package assigns that capability | each section | — |
| `status` is `ok` or `unavailable` | each section | — |
| `token` matches `ENVCHK-<LABEL>-<12 hex>` **and carries its own label**, iff `status` is `ok` | each section | — |
| **`how`** — non-whitespace characters | each section | 80 |
| the section's proof key is present and non-empty, when `status` is `ok` | six of seven | — |
| `install_report` is a list with entries | `items/text.json` | 2 |
| `install_report_source` is a non-empty string | `items/text.json` | — |

The three numbers live in the validator spec's `args` block in
`steps/check.yaml`, so the number a reader sees in the YAML is the number that
is enforced. **Numbers go in `args`; sets go in the body** — `README_SECTIONS`,
the token pattern and the placeholder set are module constants argued here,
because a set's reason does not survive being moved into a YAML with no room
for one.

## The decisions behind them

**A token is required exactly when `status` is `ok`, and forbidden otherwise.**
A token beside `unavailable` is a contradiction the report should not be able to
carry: it says both that the capability did not work and that its salt was
reached. Allowing it would let a section hedge, and a hedged section is one a
reader has to interpret.

**The token must carry its own label.** `ENVCHK-SKILL-…` under
`capabilities.plugin` is the specific mistake the skill/plugin pairing exists to
catch — two routes, two salts, and a package that copies a `skills/` directory
without installing a plugin looks identical from the outside until the tokens
differ. Catching it here, on the shape, means the message names the confusion
rather than reporting a value mismatch the author then has to diagnose.

**`level` is checked against a fixed expectation, not merely for being one of
three.** The level is *which install route is being claimed*. A report that
puts `mcp_external` at L3 is describing a package that carries its own external
server, which is not the package that ran — and the whole point of this example
is which level delivered what.

**`how` is measured in non-whitespace characters, not in words or lines.** It is
the field a human reads when `check_capabilities_genuine` says a token
mismatched, and `"ok"` satisfies presence while telling that reader nothing.
Eighty is about one written sentence. It is a floor on effort, not a judgement
of content, which is what keeps this validator `strong` — there is nothing
approximate in counting characters.

**The proof key differs per capability and two sections owe none.** What counts
as evidence is not the same across seven routes: an MCP tool has a response, a
hook has the file it wrote, a plugin has an install record. A single universal
`proof` string would flatten that away. `skill` owes no key because a skill
invocation leaves nothing behind but its answer — requiring one would be
requiring the agent to invent a field, and an invented proof is worse than an
absent one. `how` and its floor carry that section instead.

**`install_report` is mandatory and an empty one is a fault.** It is the only
independent account of what `env_mgr` did, and `check_capabilities_genuine`
decides serena's one permitted `unavailable` against it. A report that omits it
is a report whose only excuse cannot be checked. The floor is 2 because this run
declares one recipe (L1) and one component (L2); a report naming neither did not
look.

**Every fault is reported, not just the first.** A producer retrying against a
check that reports one fault per attempt pays a round trip per fault.

**An absent staged content directory folds to `False`, not to a vacuous pass.**
`all([])` is `True`, which is how an empty artefact passes a check nobody wrote
carefully.

## Why this is `strong`

Every rule is a key that is present or absent, a string that matches a pattern
or does not, or a count against a floor. There is no gap between what this
validator is named after — the *shape* of the report — and what it measures.

`strength` qualifies a PASS (`validator` spec §5.4), and a PASS here means
exactly: *this handoff is a complete, well-formed report of the right shape*. It
means nothing whatever about whether the seven capabilities worked. That is the
next validator, and the two are separate so that neither borrows the other's
claim.

## Layout

`entry.sh` is the command, `check.py` is the implementation,
`../lib/zone.py` is the four body-facing zone files, and `../lib/envchk.py` is
the capability register and the token scheme — shared with
`check_capabilities_genuine`, so the two cannot disagree about which seven
capabilities exist or what a token looks like.
