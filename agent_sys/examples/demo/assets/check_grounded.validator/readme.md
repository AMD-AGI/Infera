# check_grounded — trustworthiness, strong

Every numeral appearing in the `summary` also appears in the `facts` it
summarises.

That is the `summary` kind's contract, stated as a check. It is `strong`
because the rule is exact — set inclusion over the numerals — and the
extraction is **deliberately crude**: digits, not a parser. A crude check that
is honestly described is a strong validator; a sophisticated one that is
silently approximate is not.

## This is the one that fails, and why that is not rigging

`describe`'s goal asks for the wall-clock time the collection took. `facts`
carries no duration — `assets/produce.task/collect.py` does not measure one and the kind does
not declare one. So the agent is asked, in good faith, for a figure its input
cannot ground; it produces something, and this finds a numeral that is not in
the facts.

The failure depends on the task having been specified with a gap in it, which
is the failure the whole system exists to catch. If the model ever answers
*"the facts do not record a duration"*, there is no ungrounded numeral, this
passes, and the demo must report that **loudly** — it is a strict expected
failure, and an expected failure that passes is a failure.
