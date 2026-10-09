# check_analysis — trustworthiness, weak

Every `notes.json` in a `solutions_a` / `solutions_b` / `solutions_c` handoff
names a non-empty `algorithm`, and carries a `time_complexity` and a
`space_complexity` that parse as big-O.

## Why this is `weak`, and why calling it `strong` would be the real failure

**It checks the form of a claim, never its truth.** It does not read
`solution.cpp` at all. A program that is `O(n^2)` and writes `O(n log n)` in
its notes passes this check, and so does one that writes `O(1)` for a sort.
The thing it measures is not the thing its name suggests.

That gap is not fixable with a better regex — deciding a program's asymptotic
cost is not what a regex does — so the honest move is to declare the strength
that matches what is actually checked. `examples/demo/steps/describe.yaml`
puts the rule in one line: *a crude check that is honestly described is a
`strong` validator; a sophisticated one that is silently approximate is not*.
This one is crude **and** approximate, so it is `weak`, and a `strong` here
would be a trust claim the check cannot back — the failure mode this
repository cares most about.

`strength` qualifies a PASS and never a failure. A `False` from this validator
is a real finding: the note is missing, empty, or not a complexity expression.
A `True` means only *the student wrote something of the right shape*.

## What the pattern accepts

An optional `O` or `o`, parentheses, and inside them a **sum of products of
factors**. A factor is one of:

| form | examples |
|---|---|
| a constant | `1`, `26` |
| a bare name | `n`, `m`, `V`, `E`, `k` |
| a name to an integer power | `n^2`, `n^3` |
| a constant to a name power | `2^n`, `3^n` |
| a factorial | `n!` |
| a logarithm | `log n`, `log(n)`, `log^2 n` |
| a root, or the inverse Ackermann | `sqrt(n)`, `alpha(n)` |

Factors are multiplied by writing them side by side or with `*`, and products
are added with `+`. So all of these are accepted:

```
O(1)   O(n)   O(n log n)   O(n^2)   O(2^n)   O(n!)
O(m + n)   O(V + E)   O(n * m)   O(n log^2 n)   O(sqrt(n))
```

Whitespace is free anywhere. The separator between two factors is **not**
optional: `O(n log n)` is accepted, and the pattern requires the space or the
`*` rather than silently accepting a run-together expression as one name.

## What the pattern rejects, including things that are not wrong

It is a closed grammar, so a legitimate expression written outside it fails:

- `Θ(n)` and `Ω(n)` — only `O` and `o` are accepted. Tight and lower bounds are
  meaningful and this refuses them, because the students were asked for big-O
  and admitting three symbols would make the claim ambiguous.
- amortised or expected notations — `O(1) amortised`, `expected O(n)` — because
  the trailing or leading word is not part of the expression. Put that in
  `rationale`.
- prose — `linear`, `quadratic in n`, `about n log n`.
- a range or a disjunction — `O(n) to O(n^2)`.

If one of these ever turns out to be what the problems genuinely need, the fix
is a row in the table above and a line in the pattern, not a looser match: a
pattern that accepts everything is the placeholder check this file exists to
avoid.

## Folding

The verdict is per handoff id, so the per-problem results are folded with
`all`, and an empty `items/codes/` is a fail rather than a vacuous pass. The
per-problem detail goes to stdout so a reviewer can see which note was
malformed.
