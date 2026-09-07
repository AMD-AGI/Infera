"""Finding `examples/demo/`, and handing it over as an ordinary task package.

**Criterion 16 in one line: there is no privileged path because there is no
path at all** — only a directory argument, resolved here and handed to
`spec_loader.YamlPackage` exactly as the whole-system CLI would hand it any
other package.

The three-step resolution and its refusal are `demo` design §12, and step 3 is
what keeps it honest. Measured (`materials/08-demo.md` §6): a console script
pointing into an unpackaged directory is **not an install-time error** — `pip`
writes the script, the install reports success, and the failure arrives when a
reviewer runs it. And setuptools ships `.py` only, so from a wheel a task
package's YAML simply is not there.

**What changed with the YAML front end.** `DirectoryPackage(root, config=...)`
became `YamlPackage(root, variables=...)`, and the `config` fill — a jsonnet
`extVar` holding whatever a caller liked — became a flat `str -> str` map
expanded as `${name}` and `${name:-default}`. Two of the three keys the demo used
to pass are gone rather than renamed: `package_root` filled body paths, which the
assets convention now finds, and `store_root` was measured to be referenced by no
spec in the package. `outside` is the one that survives, because it is per-run and
absolute and no static string can name it.
"""

from __future__ import annotations

from pathlib import Path

from spec_loader import YamlPackage

__all__ = ["BROKEN", "PackageNotFound", "broken_package", "locate", "task_package"]

#: The task package, relative to the repository checkout root.
_RELATIVE = Path("agent_sys") / "examples" / "demo"

#: The deliberately broken package, a **sibling directory** of the demo rather
#: than a directory inside it. `YamlPackage` scans every `*.yaml` under a root
#: except `assets/`, so a broken document anywhere under `examples/demo/` would
#: be loaded on every ordinary run — which is criterion 13's *"two runs, no
#: hand-editing"* gone. Not reached by the ordinary pass, so criterion 13 is
#: unaffected by its existence.
BROKEN = "demo-broken"


class PackageNotFound(RuntimeError):
    """The demo task package is not where a checkout would have put it.

    Its own message names both paths tried and says why, because the most likely
    reader is somebody who installed a wheel and typed the command.
    """


def locate(explicit: str | Path | None = None) -> Path:
    """The demo task package's directory. `demo` design §12's three steps."""
    tried: list[Path] = []

    if explicit is not None:
        # 1. `--package DIR` **always wins, including when it is wrong.** Falling
        #    through to the checkout would silently run a different package than
        #    the one a reviewer named, which is the worst kind of helpful.
        chosen = Path(explicit).expanduser().resolve()
        if _is_package(chosen):
            return chosen
        raise PackageNotFound(
            f"--package {explicit} is not a task package: {chosen} has no "
            f"`main.yaml` and no `assets/`.\n"
            f"  the demo task package is not installed with the wheel; run from "
            f"a checkout, or point --package at one"
        )

    # 2. The editable-install case, which criterion 1 is written against:
    #    `cli/package.py` -> `cli/` -> `agent_sys/` -> the checkout root.
    checkout = Path(__file__).resolve().parent.parent.parent
    for candidate in (
        checkout / _RELATIVE,
        Path(__file__).resolve().parent.parent / "examples" / "demo",
    ):
        if _is_package(candidate):
            return candidate
        tried.append(candidate)

    # 3. Refuse, naming what was tried. A wheel install of `agent_sys` gives a
    #    working `agent-sys` command that refuses to run, and that is the
    #    correct behaviour: packaging the specs as package data would make the
    #    example behave differently depending on how it was installed. It is
    #    still a refusal, so it names why (design §16.1 D7).
    raise PackageNotFound(
        "the demo task package is not installed with the wheel; run from a checkout.\n"
        + "".join(f"  tried: {path}\n" for path in dict.fromkeys(tried))
        + "  or pass --package DIR"
    )


def _is_package(path: Path) -> bool:
    """A directory holding the two names that make one.

    Not "the directory exists": an empty `examples/demo` left behind by a partial
    checkout would then resolve and load zero specs, and a package that loads
    nothing is indistinguishable from a package that loaded — which is
    `docs/interfaces.md` §4.11's rule, applied to discovery.

    **The test is the loader's own, not a demo convention.** It used to look for
    a `closures/` directory, which was this package's layout and nobody else's;
    `main.yaml` and `assets/` are what `YamlPackage._structural_problems` refuses
    a package for, so a directory that passes here is one the loader will accept
    on structure.
    """
    return path.is_dir() and (path / "main.yaml").is_file() and (path / "assets").is_dir()


def task_package(root: Path, **variables: str) -> YamlPackage:
    """The package, with the variables it declares a default for.

    `outside` is the only one, and it is criterion 8's leak target: per-run and
    absolute, so `assets/describe.task/readme.md` cannot name it as a literal and
    the `describe` agent's `env` block carries `${outside:-...}` instead. `show`
    and `--dry-run` pass nothing and get the visibly-unfilled default, which is
    the point of writing a default rather than leaving the reference bare.
    """
    return YamlPackage(root=root, variables={str(k): str(v) for k, v in variables.items()})


def broken_package(root: Path, **variables: str) -> YamlPackage:
    """`examples/demo-broken/`, for `--dry-run --with-broken` only."""
    return task_package(root.parent / BROKEN, **variables)
