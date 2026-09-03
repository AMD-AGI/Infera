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

## A second form that fails, and this one *is* diagnosed — as a load error

Found 2026-09-02 by `kernel-opt-demo` while threading a scratch directory
through as a variable. **A default may not contain another reference:**

```yaml
scratch_dir: '${verify_scratch_dir:-${scratch_root}/verify_scratch}'   # load error
```

`variables.py:87` matches the default as `[^}]*`, so the first `}` ends the
reference and the rest is garbage. The module's own comment says this is
deliberate — *"a nested reference would make this a grammar, and the
alternative to a grammar is a parser nobody asked for"* — and the restriction
is stated in the docstring.

**This one is not the silent failure above**: it is refused at load. Recorded
here because the two are the same family and are hit by the same instinct — a
package author reaching for shell semantics that this loader deliberately does
not implement. The fix is to write two entries rather than one nested one.

Note the resemblance to the publication seal's `@NAME@` rule, which exists for
the same underlying reason: `}` is not in the seal's lookbehind class, so
`${VAR}/path` offers `/path` as a fresh absolute-path candidate. Both rules say
the same thing — **`}` is not a character this system composes across.**
