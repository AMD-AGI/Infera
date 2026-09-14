# check_solvable — trustworthiness, weak

Every problem ships at least `args.min_examples` (default 2) **worked**
examples: `input` and `output` both non-empty, the output not a verbatim copy of
the input, no placeholder text in it, and — where `output_format` promises a
one-line answer — an output that is one line.

## What `weak` means here, in plain words

**This does not check that the problems are solvable, and the name is the most
that could honestly be claimed for it.** There is no solver behind it, no
reference implementation, nothing that compiles or runs. It checks that a worked
answer *was supplied at all*, and that the answer is shaped like an answer.

A problem with two beautifully formatted, confidently wrong outputs passes this.
A problem that is impossible as stated passes this, as long as somebody wrote
two plausible-looking outputs underneath it. That is the gap, it is not
closeable by anything running in a validator zone with no compiler, and the
label is the place to say so.

`examples/demo/steps/describe.yaml` makes the argument this file is obeying: a
crude check that is honestly described is a strong validator, and a
sophisticated one that is silently approximate is not. This check is neither
crude nor total — it is *partial*, in a way a reader cannot audit from its
output — so `strong` would be a lie about coverage rather than about precision.
`weak` is the accurate word.

**`weak` qualifies a PASS and never a failure** (`validator` spec §5.4). When
this returns False, the examples really are missing or malformed; there is
nothing approximate about that direction.

## The four rules, exactly

An example is *worked* when all of these hold:

1. `input` is a non-empty string.
2. `output` is a non-empty string.
3. `output` contains none of `...`, `…`, `TODO`, `TBD`, `<answer`, `<output`,
   `N/A` — case-insensitive. That list is what "I did not write an answer here"
   looks like when it is typed into a template.
4. `output.strip() != input.strip()`. An answer identical to its question is
   the shape a filled-in template takes when nobody solved anything.
5. If `output_format` contains any of `one line`, `a single line`,
   `single line`, `one integer`, `a single integer`, `one number`, then
   `output` has exactly one non-empty line.

Rule 4 has a **known false negative**: a handful of real problems legitimately
echo their input — already sorted, identity transforms — and one of their
examples will have to be phrased differently to pass. That trade is made
deliberately, because rule 4 is what catches the common template, and it is
recorded here rather than discovered by whoever hits it.

Rule 5 has a **known gap**: an `output_format` that promises a single line
without using any of those six phrases is not shape-checked at all. The phrase
list is crude on purpose — the alternative is parsing an English sentence, which
would be the sophisticated-and-silently-approximate check this file exists to
refuse.

## How it runs

`entry.sh` is the body; `validator.ScriptBodyRunner` runs it in a fresh zone
holding `args.json`, `inputs.json` and `materials.json`, and it writes
`verdict.json` — one boolean per handoff id.

It reads **only the artefact it was handed**: no catalogue, no second handoff,
no store. It therefore behaves identically in `problems`' output phase and in
the three students' input phases, which is not true of its neighbour
`check_problems`.

No published content is False, not a pass. The weak label qualifies what a pass
is worth; it does not soften an absence.

Every failing example is printed with its reason, not just the first, because a
run here costs several model calls.
