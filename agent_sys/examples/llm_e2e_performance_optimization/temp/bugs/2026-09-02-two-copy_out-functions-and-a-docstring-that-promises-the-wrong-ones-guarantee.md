# Two `copy_out` functions, and a docstring that promises the wrong one's guarantee

**Found:** 2026-09-02 by `kernel-opt-demo` while testing whether delivered
handoffs work as fixtures; the correction and the call sites verified by the
leader.
**Severity:** **a safety property is documented and absent on the path that
matters.** Post-seal damage to a handoff's content reaches a consuming task
silently.
**Status:** reported, not fixed — outside this round's agreed scope.

## The two functions

| | verifies the digest? |
|---|---|
| `handoff.store.copy_out(hid, version, dst)` | **yes** — recomputes and raises `DigestMismatch` |
| `env_mgr.fs.layout.copy_out(src, dst)` | **no** — a plain `shutil.copytree` |

`env_mgr/fs/layout.py`:

```python
def copy_out(src: str, dst: str) -> str:
    """Copy a stored artefact to `dst`. Spec §6.3 rule 2: an agent works on a copy.
    ...
    """
    shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=True)
```

Same name, one level apart, different guarantees. The lower one even documents
its kinship with the upper one — *"`handoff`'s own `copy_out(hid, version, dst)`
has no default for `dst`… the same reasoning applies one level down"* — which
makes them read as the same operation.

## Proved by experiment, not only by reading

`kernel-opt` built a correctly-laid-out store from a real delivered handoff,
called `env_mgr.fs.layout.stage()`, then damaged every file's mode exactly as
`chmod -R 777` would, and called it again:

```
A. pristine, correctly laid out
   digest matches manifest = True
   stage() -> <root>/into/a/ea05f78e-.../v1   (no exception)

B. after chmod 777 on every file
   digest matches manifest = False
   stage() -> <root>/into/b/ea05f78e-.../v1
   *** stage() RAISED NOTHING ***
   staged tree digest = 9768a9f2d2a8c713...
   manifest  records  = e6ae4461ee32b7c2...
```

**A** also confirms the layout question: `stage()` accepts
`<root>/<hid>/v<N>/`, so that is the correct shape for a delivered fixture.

**B** is the finding. `env_mgr/fs/layout.py` imports **no** `copy_out` and no
`handoff`, so the `copy_out(src, dst)` at `stage()`'s line 99 is the plain
`shutil.copytree` defined at line 327 of the same file.

## Which one is on which path

- **Input staging uses the non-verifying one.** `env_mgr/material.py:20` imports
  `copy_out` **from `env_mgr.fs.layout`** and calls it at `:87`.
- **The verifying one runs on the producer side.** `agent/gate.py:226` calls
  `store.copy_out(hid, version, …)`, and `gate.py:211` explains why it always
  runs: *"`copy_out` is the only way to learn the item keys"*. That is the
  **output gate**, checking a handoff being sealed.

So a handoff's digest is verified **when it is produced and never when it is
consumed.**

## The documentation defect

`env_mgr/grants.py:177-180` justifies not granting a consumer its input's
manifest like this:

> *"spec §6.3 has a consumer work on a copy and `copy_out` verifies the digest
> **before returning**, so integrity arrives as content the body can trust
> rather than a manifest it must check. A body that verified its own input
> would reimplement `copy_out`'s one job in the one place an agent could skip
> it."*

That reasoning is sound **about `handoff.store.copy_out`**, and the staging path
does not use it. Meanwhile `layout.py`'s own `stage()` docstring says the
opposite in as many words — *"`handoff.get_manifest` verifies a digest where a
staged copy does not"*. Both sentences are correct about different functions,
and together they lead a careful reader to the wrong conclusion. `kernel-opt`
read `grants.py` and reported that a damaged fixture *"fails the consuming
task"*; it does not.

## What it actually costs

A handoff damaged **after sealing** — the realistic case being a `chmod -R 777`
over a delivered copy, which rewrites every entry of the git-shaped
`tree_digest` (`handoff/digest.py:82`) — stages without complaint. The consuming
task runs against content the manifest no longer describes, and nothing on that
path looks. The failure has no error, no log line and no obvious cause.

This is what makes the fixture instruction stronger than it first appeared:
**verify the digest yourself before delivering, because nothing downstream
will.**

```python
import yaml
from handoff import digest
m = yaml.safe_load((v / "manifest.yaml").read_text())
assert digest.tree_digest(str(v / "content")).hex() == m["digest"]["sha256"]
```

## What would fix it upstream

Either rename one of the two, or make the staging path verify. Renaming is the
cheaper half and would have prevented this report on its own: two functions
called `copy_out`, one level apart in the same import graph, with opposite
safety properties, is a trap independent of any docstring.

At minimum, correct `grants.py:177-180` so the justification names the function
it is true of. The argument it makes — that a body checking its own input would
reimplement `copy_out`'s one job — is exactly right for the verifying route and
exactly backwards for the one in use.

## The general shape, worth keeping

Two findings in this round have it:

- a validator that checks **internal consistency** cannot catch a **wrong
  premise** (`kernel-opt`'s gfx942 baseline: `check_speedup_substantiated`
  substantiates the ratio, never the denominator);
- a guarantee that is **documented and never applied** is a premise nobody
  re-derives either (this note, and `handoff/locality.py` — see below).

## The locality check: not called, and that is deliberate

Resolved 2026-09-02 by `analyze-demo`, verified at the source by the leader.
Two modules — `kernel-opt` and `analyze` — independently hunted for a call site
narrower than the docstring implied. **There is none, and the absence is a
decision rather than an oversight.**

`handoff/store.py:447`, inside `seal()`:

```python
readme.check(content_dir, content_mod.required_sections(ctype))
# locality.check — NOT CALLED. User-ruled 2026-08-31; ROADMAP §6.4.
content_mod.check_items(...)
```

and `:494`, inside `put()`, with the reason:

> **`locality.check` is not called, and criterion 17 is therefore not
> enforced.** User-ruled 2026-08-31 after it refused a correct artefact: the
> shape heuristic read an HTTP access-log line as a filesystem path, and the
> brief that produced the artefact *required* that line. Measured 97% false
> positive on a real kit.

**So the stale artefact is `handoff/protocols.py:294`**, which still promises
*"Runs the README check and the locality check before anything is…"*. It is the
only thing left in the tree claiming the check runs, and it is what sent both
modules looking. One line to fix, by whoever owns `handoff/`.

**This is not `temp/bugs/002`.** That one assumes the check runs and is merely
under-informed by `dependencies`. This is the check not running at all.

**The consequence worth broadcasting:** several packages in this tree tell their
agents *"the seal will refuse an absolute path"*. **Nothing enforces that at
publication.** Keep the advice — a kit naming one machine's paths is still a bad
kit — but drop the claim that the framework will catch it.

### And the checker itself has the defect that got it disabled

`analyze-demo`'s offline `check_locality.py` reproduced the same 97% problem:
its lookbehind excludes word characters and nothing else, so **any relative path
following a closing delimiter matches** —

| text | what matched | the delimiter |
|---|---|---|
| `<operator_id>/scripts/…` | `/scripts/…` | `>` |
| `"$PACKUP"/scripts/…` | `/scripts/…` | `"` |
| `${AITER_ROOT}/ops/…` | `/ops/…` | `}` |

All three are relative paths under a substitution. It now reports them as
composition artefacts without setting the exit code, and still fails on genuine
host paths.

**The general rule, which has now bitten in three separate places:** this system
does not compose across a `}` or a quote before `/`. Same break as the seal's
`@NAME@` placeholder rule and the variable grammar's `[^}]*` default.

**The only defence in both cases is that the premise be re-derived where it is
used, rather than carried in.**
