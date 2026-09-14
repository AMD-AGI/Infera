#!/usr/bin/env python3
"""What a Python file defines at module level — one rule, two ways of running it.

**Why this is a shared file and not a helper each caller copies.** Three places
need the same answer and they cannot run the same code the same way:

| caller | how it runs | needs |
|---|---|---|
| `identify.task/identify.py` | the **image's** python, inside a container | source *text* |
| `build_workset.task/mock_adapt.py` | same | source *text* |
| `check_optimization_shape.validator` (m4) | its own interpreter, in a zone | a *callable* |

A validator cannot import a container payload and a container cannot import this
package — nothing of the repository is mounted inside the image. So the same
rule has to be available as a string *and* as a function, and the only safe way
to do that is to make the string **be** the function's own source.

`SNIPPET` is `inspect.getsource` of the two functions below. It is not a second
copy kept in step by discipline; it is the same bytes the interpreter compiled,
so the two cannot disagree. `exec` was the alternative and this is better: the
functions stay ordinary, importable and testable, and nothing has to be trusted.

### History, because the shape of the bug argues for the shape of the fix

Until 2026-09-04 this text existed **twice**, inline in the two producers, and
took `def` / `async def` / `class` only. Two consequences, both found the same
day:

- **The narrow extraction was wrong.** `srt/layers/sampler.py`'s `logger`,
  `SGLANG_RETURN_ORIGINAL_LOGPROB` and `SYNC_TOKEN_IDS_ACROSS_TP` are
  module-level assignments and were absent, so `check_workset_shape` reported a
  `public_symbol` naming one of them as *"not defined at module level"* — false,
  and a refusal of a legitimate workset. m4 traced it from a 12-vs-9
  disagreement against m5's import surface.
- **It had two producers** (`todo.md` T34) that agreed only because neither had
  been changed. Fixing one would have made the mock and the real path disagree
  about what a file defines — a mock-only refusal, worse than the original bug.

`51af864` unified the two producers on one constant; this file is the third step,
on m4's argument: their gate is about **the importable surface**, so it must
mirror the same rule, and a third copy would undo the unification on the day it
was made.

### The boundary, which decides whether a refusal is honest

**Included** — every form that binds a *new* module-level name:

    def f  /  async def f  /  class C
    x = 1                       simple
    a, b = f()                  tuple, and nested/list forms
    first, *rest = xs           starred
    c = d = 3                   chained
    x: int = 1                  annotated

**Excluded, deliberately:**

- `obj.attr = x` and `d[k] = x` — these *mutate*; there is no module-level name
  for an overlay to replace. The obvious implementation, `ast.walk` over the
  target, adds `obj` and `d` and reports names that do not exist. That would be
  a false **pass**, which is the dangerous direction, and it is why `_tgt`
  recurses explicitly instead.
- **imports.** `import torch` does bind a module-level name, but including them
  changes the question from *what does this file define* to *what does its
  namespace contain* — a different and much longer answer. m5's count of 12
  against the old list's 9 is exactly the three assignments, which is the
  evidence that defs + classes + assignments is the surface every reader meant.
  A consumer needing the import surface must read the file itself.

Source order is preserved and duplicates collapse to their first appearance
(`dict.fromkeys`), because a reader comparing this list against the file expects
to walk down it.
"""

from __future__ import annotations

import ast
import inspect


def _tgt(t):
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, ast.Starred):
        return _tgt(t.value)
    if isinstance(t, (ast.Tuple, ast.List)):
        return [n for e in t.elts for n in _tgt(e)]
    return []


def _syms(src):
    out = []
    for x in ast.parse(src).body:
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(x.name)
        elif isinstance(x, ast.Assign):
            out += [n for t in x.targets for n in _tgt(t)]
        elif isinstance(x, ast.AnnAssign):
            out += _tgt(x.target)
    return list(dict.fromkeys(out))


#: The same rule as source text, for a payload that crosses into a container.
#:
#: **`inspect.getsource`, not a hand-kept string.** The two producers base64 this
#: into `docker run ... bash -c`, where the image's python compiles it — so the
#: text is what actually runs there, and it is byte-for-byte what runs here.
#:
#: A caller must have `import ast` already in its payload; both do. The functions
#: reference nothing else in this module, which is what makes the text
#: self-contained — keep it that way.
SNIPPET = inspect.getsource(_tgt) + inspect.getsource(_syms)

#: The callable, for anything running in a normal interpreter — a validator in a
#: zone, or a test.
module_symbols = _syms

__all__ = ["SNIPPET", "module_symbols"]
