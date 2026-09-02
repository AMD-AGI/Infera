"""`${NAME}` and `${NAME:-default}`, expanded on the parsed tree.

## Why this is written here and not adopted

Mission rule 5 says use a de facto standard, wrap it if a thin wrapper suffices,
write it only when nothing fits **and say why**. Three candidates were measured
(`scratch/ui-yaml-2026-08/w3/probe_substitution_libraries.py`):

| Candidate | Measured |
|---|---|
| `string.Template` (stdlib, free) | `safe_substitute("${inputs:-any}")` returns the string `'${inputs:-any}'` unchanged. It has no default-if-absent form, which is one of the three needs |
| `os.path.expandvars` (stdlib, free) | Same — `${NOPE:-fallback}` comes back literal — and it reads the **process** environment, which is not the set a package declares |
| `OmegaConf` 2.3.1 | Has both interpolation and a default form (`${oc.env:V,d}` resolved correctly). `OmegaConf.create(tree)` **raises `ValidationError: Object of unsupported type: 'CommentedMap'`** — it will not accept a position-carrying tree at all, and adopting it costs `antlr4-python3-runtime` and a second bundled PyYAML |

The OmegaConf result answers for a **class** of library rather than one package:
`dynaconf`, `hydra` and `pydantic-settings` all own their container types too, so
any of them means either parsing twice or parsing without positions. Positions
are the reason main spec §7 adopted `ruamel.yaml`, so a substituter that costs
them is not a candidate however good its interpolation is.

## Why the expansion happens *after* the parse

The obvious implementation is a regex over the source text before parsing. Main
spec §7 rejects Jinja2 with exactly the argument against it — text templating
*"renders strings and can emit a document that is not valid YAML at all"* — and
that argument is about the operation, not about Jinja2. Checked rather than
assumed, over six values a package author could reasonably supply
(`probe_substitute_shape.py`):

    a plain path            parsed, ok
    a colon in prose        BROKE THE DOCUMENT: ScannerError
    a leading dash          BROKE THE DOCUMENT: ScannerError
    a newline               BROKE THE DOCUMENT: ScannerError
    a hash                  parsed, WRONG VALUE: 'run'      <- the bad one
    a windows-ish root      parsed, ok

Four of six. Three are loud; the fourth is not — `run #3` becomes `run`, because
`#` starts a comment, and the document validates and admits with a truncated
value. Post-parse, all six substitute correctly, a value containing `a: b\\n- c`
stays one scalar, and the positions are byte-identical before and after.

## The whole surface, and why it stops here

Measured over every non-comment line of all 21 jsonnet sources, the computation
was constants, path concatenation, and default-if-absent — and every reference
was to a **caller-supplied** set (`config.package_root`, `config.outside`,
`config.inputs`). Two forms cover all three needs, because concatenation is what
substituting into the middle of a string *is*:

    ${NAME}              the value, or a fault naming the file and line
    ${NAME:-default}     the value, or `default` — which may be empty
    $$                   a literal `$`

There is no `${a.b}`, no arithmetic, no conditional, and no reference to another
variable's value. Each of those is a language, and the measurement is that no
package in this tree wanted one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import Any

from .protocols import Problem
from .yaml_source import position_of

__all__ = ["ASSETS_VAR", "substitute"]

#: The user's own spelling, from `refine.task_package.define.md` §2.2, kept
#: verbatim. It is their interface and renaming it — to `ASSETS`, which is what
#: it means — is not this wave's call to make (`docs/ui-stage.md` §4 W2 says the
#: same thing about the same token).
ASSETS_VAR = "TASK_PACKAGE_ASSERT_DIR"

#: `${NAME}` or `${NAME:-default}`, plus `$$` for a literal dollar.
#:
#: The default may be empty (`${NAME:-}`) and may itself contain no `}`. That is
#: the one restriction, and it is deliberate: a nested reference would make this
#: a grammar, and the alternative to a grammar is a parser nobody asked for.
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


def substitute(
    tree: Any,
    variables: Mapping[str, str],
    *,
    origin: str,
    path: str = "$",
) -> list[Problem]:
    """Expand every `${...}` in `tree`, in place. Returns the faults.

    **In place, and that is the point.** A rebuilt tree would be plain `dict`s
    and `list`s, and `lc` would be gone with them — which is the same failure
    OmegaConf was rejected for, reintroduced by hand. Assignment back into a
    `CommentedMap` preserves the positions exactly; measured, before and after
    are equal.

    A reference to a variable that is neither supplied nor defaulted is a fault
    rather than a value left literal. `${NOPE}/readme.md` left alone is a path
    that resolves to nothing later, in another module, with nothing to say why —
    and `examples/demo`'s own history has the version of that bug where an
    unfilled value concatenated to `'' + "/leak.txt"` and produced a plausible
    absolute path that demonstrated nothing (`demo/lib/demo.libsonnet`).

    `path` is the JSONPath prefix for the faults, so a diagnostic points at the
    field rather than at the document.
    """
    problems: list[Problem] = []
    _walk(tree, variables, origin=origin, path=path, problems=problems)
    return problems


def _walk(
    node: Any,
    variables: Mapping[str, str],
    *,
    origin: str,
    path: str,
    problems: list[Problem],
) -> None:
    if isinstance(node, MutableMapping):
        for key in list(node):
            child = node[key]
            here = f"{path}.{key}"
            if isinstance(child, str):
                node[key] = _expand(
                    child,
                    variables,
                    origin=origin,
                    path=here,
                    at=position_of(node, key),
                    problems=problems,
                )
            else:
                _walk(child, variables, origin=origin, path=here, problems=problems)
    elif isinstance(node, MutableSequence) and not isinstance(node, (str, bytes)):
        for i, child in enumerate(node):
            here = f"{path}[{i}]"
            if isinstance(child, str):
                node[i] = _expand(
                    child,
                    variables,
                    origin=origin,
                    path=here,
                    at=position_of(node, i),
                    problems=problems,
                )
            else:
                _walk(child, variables, origin=origin, path=here, problems=problems)
    # A key is never substituted. A package that computed its own key names
    # would be deciding the *shape* of a document rather than filling one in,
    # and the schema is the only thing entitled to say what shapes exist
    # (main spec §4.4).


def _expand(
    value: str,
    variables: Mapping[str, str],
    *,
    origin: str,
    path: str,
    at: Any,
    problems: list[Problem],
) -> str:
    def one(match: re.Match[str]) -> str:
        if match.group(0) == "$$":
            return "$"
        name = match.group("name")
        if name in variables:
            return str(variables[name])
        default = match.group("default")
        if default is not None:
            return default
        problems.append(
            Problem(
                origin=origin,
                path=path,
                keyword="variable",
                message=(
                    f"no value for ${{{name}}}, and it declares no default. "
                    f"Supply it to the package, or write ${{{name}:-<fallback>}}.\n"
                    f"  supplied: {', '.join(sorted(variables)) or 'nothing'}"
                ),
                line=at.line if at else None,
                column=at.column if at else None,
            )
        )
        return match.group(0)

    return _REF.sub(one, value)
