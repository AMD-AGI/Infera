"""Read a task package the way agent_sys does, without building a registry.

`spec_loader.YamlPackage.documents()` already does the scan, the `${VAR}`
substitution and the body-by-convention fill, so every field here is the value
the runtime would see. What this module adds is the graph a reader needs:
leaves in a valid execution order, each with the leaves it waits for.

The graph is derived the way `task_graph/models.py` unfolds a subgraph: a leaf
that consumes kind K waits for the last leaf listed before it that produces K,
and an entry's `froms` adds an ordering edge to every leaf of the named
sibling. A kind nothing produces is an *external input* the caller must seed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillkit import AGENT_SYS
from spec_loader.package import YamlPackage
from spec_loader.report import format_problems

#: `tags.cost` order, cheapest first — `validator/phase.py` sorts the same way.
COST_RANK = {"seconds": 0, "minutes": 1, "gpu_hours": 2}


class PackageError(Exception):
    """The package does not load; the message lists every problem."""


@dataclass
class Leaf:
    name: str
    agent: str
    agent_kind: str  # ai | program | human
    goal: str
    inputs: list[str]
    outputs: list[str]
    readme: str | None
    entry: str | None
    ancestors: list[str]
    deps: list[str] = field(default_factory=list)
    external_inputs: list[str] = field(default_factory=list)


@dataclass
class Package:
    root: Path
    variables: dict[str, str]
    closures: dict[str, dict]
    agents: dict[str, dict]
    validators: dict[str, dict]
    kinds: dict[str, dict]
    warnings: list[str]

    # -- lookups ------------------------------------------------------------ #

    def kind(self, name: str) -> dict:
        try:
            return self.kinds[name]
        except KeyError:
            raise PackageError(f"no handoff kind {name!r} (have: {sorted(self.kinds)})") from None

    def task(self, name: str) -> dict:
        return dict(self.closures[name].get("task") or {})

    def is_leaf(self, name: str) -> bool:
        return not self.task(name).get("subgraph")

    def validators_for(self, closure: str | None, kind: str) -> list[str]:
        """Validators that judge `kind`: the kind's own, plus `closure`'s phase
        validators whose `inputs` name it — `closure/query.py` folds the same two."""

        def judges(v: str) -> bool:  # `validator/phase.py`: a validator runs only on kinds it lists
            return kind in (self.validators.get(v, {}).get("inputs") or [])

        names = [v for v in self.kind(kind).get("validators") or [] if judges(v)]
        if closure is not None:
            for v in self.closures[closure].get("validators") or []:
                if v not in names and judges(v):
                    names.append(v)
        return sorted(names, key=lambda n: (self.cost_rank(n), n))

    def cost_rank(self, validator: str) -> int:
        tags = self.validators.get(validator, {}).get("tags") or {}
        return COST_RANK.get(str(tags.get("cost")), 9)

    def producer_of(self, kind: str) -> str | None:
        """The (last-listed) leaf that writes `kind`, or None."""
        found = None
        for leaf in self.leaves():
            if kind in leaf.outputs:
                found = leaf.name
        return found

    # -- the graph ---------------------------------------------------------- #

    def leaves(self, root: str = "main") -> list[Leaf]:
        if root not in self.closures:
            raise PackageError(f"no task {root!r} (have: {sorted(self.closures)})")
        out: list[Leaf] = []
        producer: dict[str, str] = {}
        self._unfold(root, [], [], out, producer)
        return out

    def _unfold(
        self,
        name: str,
        ancestors: list[str],
        extra_deps: list[str],
        out: list[Leaf],
        producer: dict[str, str],
    ) -> list[str]:
        """Append `name`'s leaves to `out`; return their names."""
        task = self.task(name)
        if self.is_leaf(name):
            agent = str(self.closures[name].get("agent") or "")
            body = task.get("body") or {}
            leaf = Leaf(
                name=name,
                agent=agent,
                agent_kind=str(self.agents.get(agent, {}).get("kind", "?")),
                goal=str(task.get("goal", "")),
                inputs=list(task.get("inputs") or []),
                outputs=list(task.get("outputs") or []),
                readme=body.get("readme"),
                entry=body.get("entry"),
                ancestors=list(ancestors),
            )
            deps = list(extra_deps)
            for k in leaf.inputs:
                if k in producer:
                    deps.append(producer[k])
                else:
                    leaf.external_inputs.append(k)
            leaf.deps = sorted(set(deps) - {name})
            for k in leaf.outputs:
                producer[k] = name
            out.append(leaf)
            return [name]

        subtree: dict[str, list[str]] = {}
        mine: list[str] = []
        for entry in task.get("subgraph") or []:
            child = entry["closure"]
            ordering = [leaf for f in entry.get("froms") or [] for leaf in subtree.get(f, [])]
            got = self._unfold(child, [*ancestors, name], [*extra_deps, *ordering], out, producer)
            subtree[child] = got
            mine += got
        return mine


#: The loader's message for a required `${VAR}` with no value.
_MISSING_VAR = re.compile(r"no value for \$\{([A-Za-z_][A-Za-z0-9_]*)\}")
UNSET = "__astk_unset__"


def missing_variables(root: str | Path, variables: Mapping[str, str] | None = None) -> list[str]:
    """Required `${VAR}`s the package needs and `variables` does not supply."""
    contents = YamlPackage(Path(root).resolve(), dict(variables or {})).documents()
    return sorted(
        {m.group(1) for p in contents.problems for m in [_MISSING_VAR.search(p.message)] if m}
    )


def load(
    root: str | Path, variables: Mapping[str, str] | None = None, *, allow_unset: bool = False
) -> Package:
    """Load `root` with `variables` (the `--var` set). Raises `PackageError`.

    `allow_unset` fills every missing required variable with a visible
    sentinel — for commands that only judge a handoff, where a variable that
    feeds the producer's `env:` is irrelevant. A validator whose `args` used one
    sees the sentinel, which is why it is reported as a warning.
    """
    root = Path(root).resolve()
    if not (root / "main.yaml").is_file() or not (root / "assets").is_dir():
        raise PackageError(f"{root} is not a runnable task package (needs main.yaml and assets/)")
    variables = dict(variables or {})
    missing = missing_variables(root, variables)
    filled: list[str] = []
    if missing and allow_unset:
        variables.update({k: UNSET for k in missing})
        filled = missing
    elif missing:
        raise PackageError(
            f"package needs --var for: {', '.join(missing)}\n"
            "  pass each as --var NAME=VALUE (every astk command that loads the package needs them),"
            " or --allow-unset-vars when only judging a handoff"
        )
    contents = YamlPackage(root, variables).documents()
    fatal = [p for p in contents.problems if getattr(p, "fatal", True)]
    warnings = [format_problems([p]) for p in contents.problems if not getattr(p, "fatal", True)]
    warnings += [f"--var {k} not given; filled with {UNSET!r}" for k in filled]
    if fatal:
        raise PackageError("package does not load:\n" + format_problems(fatal))
    by: dict[str, dict[str, dict]] = {"closure": {}, "agent": {}, "validator": {}, "handoff": {}}
    for d in contents.documents:
        doc = _plain(d.doc)
        by.setdefault(d.kind, {})[doc["name"]] = doc
    return Package(
        root=root,
        variables=variables,
        closures=by["closure"],
        agents=by["agent"],
        validators=by["validator"],
        kinds=by["handoff"],
        warnings=warnings,
    )


def show(root: str | Path, variables: Mapping[str, str] | None = None) -> tuple[int, str]:
    """agent_sys's own static check (`cli.main show`), on this checkout's code."""
    cmd = [sys.executable, "-B", "-m", "cli.main", "show", "--package", str(root)]
    for k, v in (variables or {}).items():
        cmd += ["--var", f"{k}={v}"]
    proc = subprocess.run(
        cmd,
        cwd=str(AGENT_SYS),
        env={**_env_for_show(), "PYTHONPATH": str(AGENT_SYS)},
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _env_for_show() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _plain(node: Any) -> Any:
    """ruamel's round-trip containers -> plain dict/list/str, for json dumps."""
    if isinstance(node, Mapping):
        return {str(k): _plain(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(v) for v in node]
    if isinstance(node, (bool, int, float)) or node is None:
        return node
    return str(node)


def env_name(kind: str) -> str:
    """`AGENT_SYS_{INPUT,OUTPUT}_<this>` — `env_mgr/grants.py`'s rule."""
    return "".join(c if c.isalnum() else "_" for c in kind).upper()
