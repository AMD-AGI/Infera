"""`assets/` — finding a body's files by filename convention.

A package root must hold `assets/` (main spec §4.3), and a document may leave
`body.readme` and `body.entry` out: the file is found by matching its name
against the object's `name` and its `type`.

A filename is a `.`-separated sequence of tokens (the object's `name`,
optionally its `type`, optionally the role word `readme`/`entry`) plus a
mandatory extension (`.md` readme, `.sh` entry); order does not matter. A
folder named `${name}`, `${name}.${type}` or `${type}.${name}` scopes a
lookup to itself and makes `name` optional inside it; for an `agent`, that
folder *is* the binding, since an agent has no `body`. Two paths matching
one query raises `SpecInconsistent`; filling a path never changes any other
field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

from .protocols import Problem, SpecInconsistent

__all__ = [
    "ASSETS_DIRNAME",
    "AssetIndex",
    "fill_agent_assets",
    "fill_agent_env_recipe",
    "fill_body",
]

#: The mandatory directory (main spec §4.3). Not configurable: the whole point of
#: fixing the name is that a document's unqualified paths have something to be
#: relative to without the loader inferring it from the tree.
ASSETS_DIRNAME = "assets"

#: Role to the extension it requires; the extension is mandatory in every row.
#: `env_recipe` reuses `_stems`'s permutation generator, so it costs one row
#: rather than a second mechanism. `.yaml` only — a `.yml` recipe is silently
#: absent rather than an error.
_ROLES: Mapping[str, str] = {"readme": ".md", "entry": ".sh", "env_recipe": ".yaml"}


@dataclass(frozen=True)
class _Found:
    """One matching path, and how it matched. The `how` is for the message: a
    conflict between a folder-scoped file and a flat one is confusing until you
    are told which was which."""

    path: Path
    how: str


class AssetIndex:
    """Every file under `assets/`, ready to be asked for one object's body.

    Built once per package and queried per object, because the alternative —
    walking the tree per query — is O(objects x files) and turns a conflict into
    something only some queries notice.
    """

    def __init__(self, assets_root: Path) -> None:
        self._root = assets_root
        self._files: list[Path] = (
            sorted(p for p in assets_root.rglob("*") if p.is_file()) if assets_root.is_dir() else []
        )

    def resolve(self, role: str, *, name: str, type_: str | None) -> Path | None:
        """The package-relative path for one object's `readme` or `entry`, or
        `None`. Raises `SpecInconsistent` when more than one path matches."""
        found = sorted(self._candidates(role, name=name, type_=type_), key=lambda f: f.path)
        if not found:
            return None
        if len(found) > 1:
            raise SpecInconsistent(
                f"{len(found)} files under {self._root.name}/ could be "
                f"{name!r}'s {role}, and a conflict is not resolved by guessing:\n"
                + "\n".join(f"  {f.path}  ({f.how})" for f in found)
                + f"\n  Rename all but one, or bind body.{role} explicitly."
            )
        # Package-relative, because that is what `_common.schema.json` says a
        # body path is and what `agent` resolves against the *staged* copy
        # (`interfaces.md` §4.16). An absolute path here would be the F-D18
        # regression: `Path(staged) / "/abs"` is `/abs`, so a staged body would
        # never be reached.
        return Path(ASSETS_DIRNAME) / found[0].path.relative_to(self._root)

    def resolve_folder(self, *, name: str, type_: str | None) -> Path | None:
        """The package-relative path of this object's own directory, if any.

        Matches the same three spellings `_folder_names` gives `resolve`.
        `None` when nothing matches; raises `SpecInconsistent` on a conflict.
        """
        if not self._root.is_dir():
            return None
        wanted = _folder_names(name=name, type_=type_)
        found = sorted(p for p in self._root.iterdir() if p.is_dir() and p.name in wanted)
        if not found:
            return None
        if len(found) > 1:
            raise SpecInconsistent(
                f"{len(found)} directories under {self._root.name}/ could be "
                f"{name!r}'s assets, and a conflict is not resolved by guessing:\n"
                + "\n".join(f"  {p.name}/" for p in found)
                + "\n  Merge them, or rename all but one."
            )
        # Package-relative for `resolve`'s reason, restated because it is the
        # one that bit (F-D18): `agent` resolves this against the **staged**
        # copy, and `Path(staged) / "/abs"` is `/abs`.
        return Path(ASSETS_DIRNAME) / found[0].name

    # -- matching ----------------------------------------------------------- #

    def _candidates(self, role: str, *, name: str, type_: str | None) -> Iterable[_Found]:
        suffix = _ROLES[role]
        free = _stems(role, name=name, type_=type_, name_required=True)
        scoped = _stems(role, name=name, type_=type_, name_required=False)
        folders = _folder_names(name=name, type_=type_)

        for path in self._files:
            if path.suffix != suffix:
                continue
            stem = path.name[: -len(suffix)]
            if stem in free:
                yield _Found(path, "matched by filename")
                continue
            if stem in scoped and _under_a_folder(path, self._root, folders):
                yield _Found(path, f"matched inside {_folder_of(path, self._root, folders)}/")


def _stems(role: str, *, name: str, type_: str | None, name_required: bool) -> frozenset[str]:
    """Every `.`-joined permutation of the tokens this role admits.

    `name_required=False` is the folder-scoped form: the folder already named the
    object, so `readme.md` and `task.readme.md` are enough. The empty stem is
    excluded — a file called `.md` names nothing.
    """
    optional = [t for t in (type_, role) if t]
    out: set[str] = set()
    for k in range(len(optional) + 1):
        for chosen in permutations(optional, k):
            tokens = [name, *chosen] if name_required else list(chosen)
            for order in permutations(tokens):
                if order:
                    out.add(".".join(order))
    if not name_required:
        # The name may also appear inside its own folder — `produce/produce.md`
        # is redundant but not wrong, and rejecting it would be a rule nobody
        # stated.
        out |= _stems(role, name=name, type_=type_, name_required=True)
    return frozenset(out)


def _folder_names(*, name: str, type_: str | None) -> frozenset[str]:
    """`${name}`, `${name}.${type}`, `${type}.${name}` — the user's three, and
    only those three. Unlike a filename this is not a free permutation set,
    because the user wrote the folder rule as a closed list."""
    out = {name}
    if type_:
        out |= {f"{name}.{type_}", f"{type_}.{name}"}
    return frozenset(out)


def _folder_of(path: Path, root: Path, folders: frozenset[str]) -> str:
    for part in path.relative_to(root).parts[:-1]:
        if part in folders:
            return part
    return ""


def _under_a_folder(path: Path, root: Path, folders: frozenset[str]) -> bool:
    """Whether `path` is anywhere below a matching folder, not only directly in it."""
    return bool(_folder_of(path, root, folders))


# --------------------------------------------------------------------------- #
# Filling a document


def fill_body(
    doc: MutableMapping[str, Any],
    index: AssetIndex,
    *,
    kind: str,
    name: str,
    origin: str,
    line: int | None,
) -> list[Problem]:
    """Fill `body.readme` / `body.entry` from the index, or warn if bound by hand.

    Only `closure` and `validator` own a `body` (`_body_owner`), a closure's being
    the nested task's; other kinds are untouched. An explicit binding warns.
    """
    body_owner, type_ = _body_owner(doc, kind)
    if body_owner is None:
        return []

    problems: list[Problem] = []
    body = body_owner.get("body")
    if not isinstance(body, MutableMapping):
        body = {}

    for role in ("readme", "entry"):
        if body.get(role):
            problems.append(
                Problem(
                    origin=origin,
                    path=f"$.body.{role}",
                    keyword="explicit-binding",
                    message=(
                        f"body.{role} is bound by hand to {body[role]!r}. That is legal "
                        f"and it is not what this package format is for: name the file "
                        f"by convention under {ASSETS_DIRNAME}/ and drop the key."
                    ),
                    fatal=False,
                    line=line,
                )
            )
            continue
        found = index.resolve(role, name=name, type_=type_)
        if found is not None:
            body[role] = found.as_posix()

    if body:
        body_owner["body"] = body
    return problems


def fill_agent_assets(
    doc: MutableMapping[str, Any],
    index: AssetIndex,
    *,
    kind: str,
    name: str,
    origin: str,
    line: int | None,
) -> list[Problem]:
    """Fill an agent's `assets` from the index, or warn if bound by hand.

    Only `kind == "agent"`; other kinds return `[]` untouched. Nothing here
    reads what is inside the directory.
    """
    if kind != "agent":
        return []

    if doc.get("assets"):
        return [
            Problem(
                origin=origin,
                path="$.assets",
                keyword="explicit-binding",
                message=(
                    f"assets is bound by hand to {doc['assets']!r}. That is legal "
                    f"and it is not what this package format is for: name the "
                    f"directory by convention under {ASSETS_DIRNAME}/ and drop the key."
                ),
                fatal=False,
                line=line,
            )
        ]

    found = index.resolve_folder(name=name, type_="agent")
    if found is not None:
        doc["assets"] = found.as_posix()
    return []


def fill_agent_env_recipe(
    doc: MutableMapping[str, Any],
    index: AssetIndex,
    *,
    kind: str,
    name: str,
    origin: str,
    line: int | None,
) -> list[Problem]:
    """Fill an agent's `recipes` from an `env_recipe` file it carries.

    Scoped to `assets/` only. A declared `recipes` wins and warns instead.
    Only `kind == "agent"`; other kinds return `[]` untouched.
    """
    if kind != "agent":
        return []

    if doc.get("recipes"):
        return [
            Problem(
                origin=origin,
                path="$.recipes",
                keyword="explicit-binding",
                message=(
                    f"recipes is bound by hand to {doc['recipes']!r}. That is legal "
                    f"and it is not what this package format is for: name the file "
                    f"by convention under {ASSETS_DIRNAME}/ and drop the key."
                ),
                fatal=False,
                line=line,
            )
        ]

    found = index.resolve("env_recipe", name=name, type_="agent")
    if found is not None:
        doc["recipes"] = [found.as_posix()]
    return []


def _body_owner(doc: MutableMapping[str, Any], kind: str) -> tuple[MutableMapping | None, str]:
    """Where this kind's `body` lives, and the `type` token its filenames use.

    A closure's body is the task's, one level in, because a package author
    writes `module: task` and never `closure` (`closure` spec §2) — so the token
    in a filename is `task`, which is the word they typed, not `closure`, which
    is the schema's.
    """
    if kind == "closure":
        task = doc.get("task")
        return (task if isinstance(task, MutableMapping) else None), "task"
    if kind == "validator":
        return doc, "validator"
    return None, kind
