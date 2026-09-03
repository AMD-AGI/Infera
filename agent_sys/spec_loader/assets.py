"""`assets/` — finding a body's files by filename convention.

A package root must hold `assets/` (main spec §4.3), and a document may leave
`body.readme` and `body.entry` out: the file is found by matching its name
against the object's `name` and its `type`.

## The convention, as the user wrote it

`refine.task_package.define.md` §2.3, restated as a rule rather than as a list of
examples. A filename is a `.`-separated sequence of **tokens** plus an extension:

| Role | Tokens | Extension |
|---|---|---|
| readme | the object's `name`, **mandatory**; its `type`, optional; the literal word `readme`, optional | `.md`, mandatory |
| entry | the same, with the literal word `entry` | `.sh`, mandatory |

**Order does not matter** — `collect.readme.md`, `readme.collect.md`,
`task.collect.readme.md` and `collect.task.md` all name `collect`'s readme. The
user's list is the permutations of a set, so this generates the permutations of
the set rather than transcribing the list; transcribing it is how the tenth
spelling gets forgotten.

`type` is the `module:` word — `task`, `agent`, `handoff`, `validator`.

## Folders do not fill a field; they scope a lookup

The third rule is *"a `${name}.${type}` or `${name}` or `${type}.${name}` folder"*,
and it is the one that needed working out, because **no schema field takes a
folder path.** `task.body` and `validator.body` have `readme`, `entry` and
`materials`, and `materials`' own description says nothing reads it yet
(`interfaces.md` §5.1b). So a folder could not be bound to anything.

What it does instead is what `validator` spec §9.1 already means by *"a validator
is a folder"*: it groups one object's files, and inside it the name is implied.

    assets/check_facts.validator/readme.md      -> check_facts' readme
    assets/produce/entry.sh                     -> produce's entry

Inside a matching folder the object's `name` becomes optional, because the folder
already said it. Everywhere else it is mandatory, which is what keeps a bare
`assets/readme.md` from being every object's readme at once.

**For an `agent` the folder itself is the answer**, and that is the one place a
folder does fill a field. `agent` has no `body`, so `fill_body`'s note below
records it as *"a gap, not an omission"* — the user's rule named agents and the
schema had nothing for them to bind. `agent.assets` closes that gap: what an
agent carries is not one readme and one entry but a **directory** — a `.claude/`
tree in Claude Code's canonical layout — and a directory is what the third rule
was already scoping. So the same `_folder_names` set that scopes a task's lookup
*is* an agent's binding, and no fourth spelling rule was invented for it.

## Conflicts crash

Two paths matching one query is a fault, and the error is
`SpecInconsistent` — `registry.py`'s policy for the same situation one layer up,
where *"two specs claiming one name is a fault"*. Adopting the existing shape
rather than inventing one is `docs/ui-stage.md` §4 W4's instruction.

## Paths are derived; semantics are not

`user_interface.ai.draft.md` §4.9's one surviving rule, and it came from a real
failure (opa#6509): finding an `entry.sh` fills `body.entry` and **changes
nothing else**. It does not make a task programmatic behind the author's back,
because it was always the *presence of the file* that said so — the author moved
the declaration from a YAML key to a filename and kept the same control. The
existing named check that `entry` and a subgraph are mutually exclusive
(`closure` spec §2.6) then runs unchanged over the filled document, so a non-leaf
that acquires an `entry.sh` fails loudly rather than quietly becoming a leaf.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

from .protocols import Problem, SpecInconsistent

__all__ = ["ASSETS_DIRNAME", "AssetIndex", "fill_agent_assets", "fill_body"]

#: The mandatory directory (main spec §4.3). Not configurable: the whole point of
#: fixing the name is that a document's unqualified paths have something to be
#: relative to without the loader inferring it from the tree.
ASSETS_DIRNAME = "assets"

#: Role to the extension it requires. The extension is mandatory in both rows —
#: it is the only token the user's rules never allow to be dropped.
_ROLES: Mapping[str, str] = {"readme": ".md", "entry": ".sh"}


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
        """The package-relative path for one object's `readme` or `entry`.

        `None` when nothing matches, which is not an error here: `readme` is
        required by the schema and `entry` is not, and the schema is the only
        enforcement point (main spec §4.4). An absent required body path fails
        there, with the message that field already has.

        Raises `SpecInconsistent` when more than one path matches.
        """
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
        """The package-relative path of this object's own **directory**, if one exists.

        The same three spellings `_folder_names` gives `resolve` — `X`,
        `X.agent`, `agent.X` — asked as a question in their own right rather
        than used to scope a filename match. One function, one answer: a second
        set of folder spellings here is how the two would come to disagree about
        what `agent.X/` means (`engineer_principle.md` §1, *never let an
        invariant have two writers*).

        **Directly under `assets/`, not anywhere below it.** `_under_a_folder`
        recurses because a *file* may sit deep inside its object's directory;
        the directory itself is a top-level member of `assets/`, and admitting a
        nested one would make `assets/a.agent/b.agent/` two answers for two
        different agents in a tree neither of them owns.

        `None` when nothing matches, which is not an error: L3 material is
        **undeclared and auto-detected** (`env_mgr/agent_assets.py`), so an
        agent that carries nothing has no directory and that is its normal
        shape. Raises `SpecInconsistent` when two spellings both exist —
        `resolve`'s rule, for the same reason: two paths matching one query is a
        fault, and picking one would silently drop half an agent's material.
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
    """Anywhere below a matching folder, not only directly inside it.

    The user asked for recursion under `assets/` and said nothing about stopping
    at a folder boundary; stopping would mean `check_facts.validator/logic/entry.sh`
    is invisible for no reason a reader could reconstruct.
    """
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
    """Fill `body.readme` / `body.entry` from the index, or warn that they were
    bound by hand.

    **Two of the four kinds have no body and that is a gap, not an omission.**
    The user's rule says a readme is found for *"某个命名的 agent/task/handoff/
    validator"*, and measured against the schemas only `task.body` and
    `validator.body` exist — `agent` has `knowledge` / `rules` / `skills` and
    `handoff` has `readme_sections`, neither of which is a path. So there is
    nothing to fill for those two, and inventing a field to fill would be
    `engineer_principle.md` §2's failure mode. Reported rather than built.

    **Half of that gap is now closed, and by the report rather than around it.**
    `agent.assets` exists because per-agent component install needed a directory
    to detect L3 material in, and the report above is what said where it goes.
    It is filled by `fill_agent_assets`, not here: this function's unit is a
    `body` mapping with two path roles in it, and an agent has neither.

    **Explicit binding is legal and warns** (`refine.task_package.define.md`
    §2.3.5), as a non-fatal `Problem` — the mechanism `closure/check.py`'s check
    3 already uses for a report-severity finding, so a warning reaches the same
    report and the same log rather than a second channel.

    This reaches into `doc["task"]["body"]`, and that is allowed *here* and not
    in `load_package`: a package owns its own content and the loader does not
    (`access.py`'s note on the line this package does not cross). The seam is
    what separates them, and this runs on the package's side of it.
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
    """Fill an agent's `assets` from the index, or warn that it was bound by hand.

    **Only `kind == "agent"`.** Every other kind returns `[]` untouched, because
    `assets` is not in their schemas and writing it would put a key there that
    `additionalProperties: false` then rejects — an error attributed to the
    author for something the loader did.

    The value is what `env_mgr` resolves L3 material against: the directory is
    copied into the zone with the rest of the package, and `<assets>/.claude/`
    is auto-detected there. Nothing is read here and nothing is checked to
    exist beyond the directory itself — this module fills paths and does not
    learn what is in them, the same line `fill_body` holds.

    Explicit binding warns for `fill_body`'s reason and through its mechanism: a
    non-fatal `Problem`, so it reaches the report and the log rather than a
    second channel.
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
