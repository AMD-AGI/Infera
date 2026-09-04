# An unparseable `${...}` is passed through as a literal, with no diagnostic

**Found:** 2026-09-02, by `integration-demo` on `crsuse2-m2m-276`, while
localising `mix_worker.sh`'s hard-coded GLM flags into package variables.
**Severity:** a **silent wrong value reaching a subprocess**. The run does not
fail at load; it fails much later, somewhere else, as a malformed argument.
**Status:** worked around by the package. The gap is `agent_sys`'s.

## What happened

`integration-demo/shared.yaml` declared, using what its author believed was the
shell's familiar "default if unset" form:

```yaml
IT_DSA_ARGS: '${dsa_args---dsa-prefill-backend tilelang --dsa-decode-backend tilelang}'
```

That is a **bare dash** (`${name-default}`). `agent_sys` does not accept it. But
nothing said so — the reference was neither substituted nor rejected. It was
copied through **verbatim**, survived every layer between the spec and the
engine, and arrived at sglang as:

```
--dsa-decode-backend 'tilelang}'
```

A trailing `}` inside an argument value, on a flag that only a GLM model reads,
in a container, several minutes into a bring-up.

## Why

`spec_loader/variables.py:81` accepts exactly two forms and no third:

```python
_REF = re.compile(
    r"""
    \$\$                                  # an escaped dollar
    |
    \$\{
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        (?: :- (?P<default>[^}]*) )?
    \}
    """,
    re.VERBOSE,
)
```

`${dsa_args-...}` reaches `dsa_args`, then finds `-` where the grammar allows
only `:-` or `}`. The alternative fails, the regex simply does not match at that
position, and **a non-match is not an error** — the scanner moves on and the
text stays as it was.

So the two supported forms are `${NAME}` and `${NAME:-default}`. **`:-` is
mandatory; a bare `-` is not the same thing**, even though it is in POSIX shell,
and this is a file full of shell fragments written by people fluent in shell.

## Why it is worth a bug note rather than a shrug

The package's own contract is that **an unfilled variable with no default is a
load-time fault naming the file, the line and the variable** — that is how a
package demands a `--var`, and it is stated in every one of these READMEs. This
case defeats that contract from the other side: a reference the loader cannot
understand is treated as ordinary text rather than as a fault, so the one
mechanism the author is relying on for safety is silently absent exactly when
they have made a typo.

The failure also travels a long way from its cause. Nothing between
`spec_loader` and the engine has any reason to inspect a string for a stray `}`,
so the first observer is a subprocess that does not know what a package variable
is.

## What would fix it upstream

Scan for `${` that does **not** match `_REF` and refuse the load, naming the
file, the line and the malformed reference — the same treatment an unfilled
variable already gets. The information is all present at the same point; only
the check is missing. `$$` already exists as the escape for a literal dollar, so
a package that genuinely wants the text `${x-y}` has a way to say so, and the
refusal would not take anything away.

## What the package does instead

Uses `:-` everywhere, which is the supported form:

```yaml
IT_DSA_ARGS: '${dsa_args:---dsa-prefill-backend tilelang --dsa-decode-backend tilelang}'
```

Note the `:---`: the `:-` separator followed by a default that itself begins
with `--`. It reads badly and it is correct.

**Checked across the whole example tree on 2026-09-02** — no other bare-dash
reference remains in any of the five stage packages:

```sh
grep -rnoE '\$\{[a-zA-Z_][a-zA-Z0-9_]*-[^:}][^}]*\}' \
  agent_sys/examples/llm_e2e_performance_optimization/ --include='*.yaml'
```

Anyone adding a variable to these packages should re-run that grep, until the
loader does it for them.

## A second form: a nested default. **Also silent — this section was wrong**

Found 2026-09-02 by `kernel-opt-demo` while threading a scratch directory
through as a variable. **A default may not contain another reference:**

```yaml
scratch_dir: '${verify_scratch_dir:-${scratch_root}/verify_scratch}'
```

`variables.py:87` matches the default as `[^}]*`, so the first `}` ends the
reference and the rest is left behind. The module's own comment says the
restriction is deliberate — *"a nested reference would make this a grammar, and
the alternative to a grammar is a parser nobody asked for"* — and it is stated
in the docstring.

> **Corrected 2026-09-04.** This section previously said *"this one **is**
> diagnosed — as a load error"* and *"it is refused at load"*. **That is false,
> and it is false in the direction that hurts:** it told the reader the loader
> would catch this, so anyone who consulted this note before reaching for a
> nested default was told to expect a refusal that never comes. The claim was
> written from reading the docstring's *"not supported"* and assuming
> unsupported meant refused. Nobody ran it. Measured now:

```python
>>> t = {"a": '${verify_scratch_dir:-${scratch_root}/verify_scratch}'}
>>> V.substitute(t, {}, origin="t")       # this note's own example
[]                                        # <- ZERO problems
>>> t
{'a': '${scratch_root/verify_scratch}'}   # mangled, passed through
```

So the nested default belongs to **exactly the same family as the bare dash
above** — unmatched text walked past in silence — and this note had the one
member it documented in detail filed on the wrong side of its own thesis.

### The variant that is worse than the one recorded

Found 2026-09-04 by m5, reaching for a nested default to give
`check_bench_report` its own bar while keeping m2's as the fallback:

```python
>>> t = {"a": '${integration_min_requests:-${min_requests:-50}}'}
>>> V.substitute(t, {}, origin="t")
[]                                        # <- ZERO problems
>>> t
{'a': '${min_requests:-50}'}              # <- *valid syntax*, never expanded
```

The two differ in the residue, and the difference is the whole severity:

| form | residue | can a later reader tell? |
|---|---|---|
| `${A:-${B}/suffix}` | `${scratch_root/verify_scratch}` | **maybe** — malformed, a stray `${…}` with no `:-` and a `/` inside the name |
| `${A:-${B:-50}}` | `${min_requests:-50}` | **no** — a perfectly well-formed reference |

The second residue is **indistinguishable from text a package meant to write
literally**. It survives `show` with no diagnostic, and `show` is the gate
everyone on this package trusts for exactly this class of mistake — a sub-second
load-and-typecheck of every yaml. It then arrives at the body as the *string*
`"${min_requests:-50}"`, where `float()` raises `ValueError` at whatever moment
that validator first grades something.

### And the *working* path is broken, not only the fallback

Found 2026-09-04 by m2, checking the claim above rather than taking it, and it
is the row that matters most. All four states of the same string, zero problems
reported in every one:

```
${integration_min_requests:-${min_requests:-50}}   unset            -> '${min_requests:-50}'
${integration_min_requests:-${min_requests:-50}}   min_requests=20  -> '${min_requests:-50}'
${integration_min_requests:-${min_requests:-50}}   OUTER=7          -> '7}'
${integration_min_requests:-${min_requests:-50}}   OUTER=7,inner=20 -> '7}'
```

**Setting the outer variable — the entire point of adding it — yields `'7}'`**:
the value with a trailing brace glued on, because the outer reference's match
ended at the inner `}` and the outer `}` was never part of it.

So the damage is not confined to the fallback that nobody exercises. The path a
package author would actually take — declare the nested default, then pass the
new `--var` — produces a corrupted value **only when the knob is used**, which
is the worst possible distribution: it works in every dry run where the variable
is left unset (silently reading the wrong one), and breaks the first time
somebody sets it.

`float("7}")` throws, so this one at least fails loudly at the far end. A field
consumed as a string — a container name, a path, a served model name — would not
throw at all.

`str`-not-`int` is already the known hazard here (CONTRACT.md:403,
`interpreter_sweep.py:131`); this makes the string one that cannot be parsed at
all, and moves the failure from the load to the grading.

**Same fix as above, and it covers both:** scan for a `${` that `_REF` does not
match and refuse the load. `$$` already provides the escape for a literal
dollar, so nothing legitimate is taken away. Until then, the grep in the section
above finds the bare dash; this finds a nested default:

```sh
grep -rnoE '\$\{[A-Za-z_][A-Za-z0-9_]*:-[^}]*\$\{' \
  agent_sys/examples/llm_e2e_performance_optimization/ --include='*.yaml'
```

**It returns four hits on 2026-09-04 and all four are prose, not values** — a
`#` comment is still yaml text, and three of these files warn against the very
pattern they are matched on:

| hit | what it is |
|---|---|
| `e2e-flow/shared.yaml:58` | comment: *"**Not** `${served_name:-${model_name}}`"* |
| `e2e-flow/steps/m2_profiling.yaml:95` | comment: *"`${expect_ranks:-${tp:-8}}` is not spellable"* |
| `e2e-flow/steps/m5_integration.yaml:350` | comment: this note's own example |
| `kernel-opt-demo/steps/kernel_optimization.yaml:348` | comment: the case above |

So: **no live nested default anywhere in the example tree**, and the grep cannot
tell you that by its exit status — read the four lines. Recorded because the
first version of this paragraph claimed the grep came back clean, which was
written before it was run; running it is what produced the table. A checker that
reports four hits and means zero is a checker whose next reader either panics or
stops believing it.

### The four hits are all *warnings*, and that is the finding

Observed by m2, 2026-09-04, from the same four lines I had already read and
walked past. **Every one of the four is somebody warning the next reader off the
pattern.** Not one is a live value; not one was written by somebody who had been
told.

`git blame` cannot separate them — the whole team commits under one git identity
— but the commit each line arrived in can:

| line | arrived in | episode |
|---|---|---|
| `e2e-flow/shared.yaml:58` | *"freeze the cross-module contract"* | the leader's freeze |
| `e2e-flow/steps/m2_profiling.yaml:95` | *"m3: validator readmes"* | **m3**, writing in m2's file |
| `kernel-opt-demo/…:348` | *"debug the five e2e stage packages"* | the solo phase |
| `e2e-flow/steps/m5_integration.yaml:350` | this note's own example | m5, today |

**Four separate discoveries, four separate people paying for the same fact.**
The third row is the sharp one: the warning in `m2_profiling.yaml` was not
written by that file's owner, so even *within* one file the person who hit the
restriction and the person who owns the place it is recorded are different. m2
and I both misread that line as m2's until it was checked.

This is the same shape as the hazard that produced this section — the shared
`--var` at `m2_profiling.yaml:46` and `m5_integration.yaml:334`, where each
owner read a correct floor in their own file and neither could see the other.
**Knowledge that is correct in every local file and absent from the package.**
Four warnings scattered across four files are four people's savings that nobody
can spend; one note that a grep finds is the difference.

That is the argument for this record existing, and it is a better one than *the
old section was wrong*.

The package's fix is to write
two entries rather than one nested one — which is what
`steps/m5_integration.yaml`'s `integration_min_requests` does, at the cost of
duplicating a measured default in two files.

Note the resemblance to the publication seal's `@NAME@` rule, which exists for
the same underlying reason: `}` is not in the seal's lookbehind class, so
`${VAR}/path` offers `/path` as a fresh absolute-path candidate. Both rules say
the same thing — **`}` is not a character this system composes across.**
