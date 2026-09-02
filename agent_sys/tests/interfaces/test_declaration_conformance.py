"""Every implementation against its own `protocols.py` — signatures, defaults included.

`docs/interfaces.md` §8 ships each package's contract twice: the `.py` carries the
reasons and is importable, the `.pyi` is what a type checker reads.
`test_stub_agreement.py` keeps those two in step. **Nothing kept either of them in
step with the code**, and this file is that third edge.

Three defects in two days came through it, and all three were silent:

| | |
|---|---|
| `closure.check_closures` | carried `handoff_report: HandoffLoadReport \\| None = None` where the declaration had no default. The early return on `None` meant an escape-hatch admission went unreported **in the assembled system** while three packages' suites stayed green |
| `agent.select_backend` | carries a fourth parameter, `assignment`, that the declaration does not have — and it is how the executor receives its readme, entry point and zone, so a caller using the declared signature builds an agent that starts and does nothing |
| `handoff` | the composition root reached for `load_report()` and the registry named it `report()`; `getattr(..., lambda: None)()` returned `None` and nothing failed |

The rule that makes this test worth having, and the reason the first one
survived a signature test that **already existed and was already green**:

> **A signature test that compares parameter names and not defaults is not a
> signature test. The default is the drift.**

`closure`'s own `test_check_closures_matches_its_declaration` compared names
only. It passed over the defect it was named for, which is worse than no guard,
because a guard that passes is counted.

---

**What this does not check, stated so nobody reads more into a green run.**
Parameter *names*, *kinds* (positional / keyword-only / variadic) and *defaults*,
for module-level functions a `protocols.py` declares and its package exports.
Not types — annotations are strings under `from __future__ import annotations`
and the two files legitimately spell one type two ways (`Any` where a package may
not name a neighbour's class). Not classes: a Protocol's methods are a contract an
implementation satisfies *structurally*, and asserting shape there would forbid
the widening that Protocols exist to permit.

Import failures are reported rather than skipped. A package that cannot be
imported has no conformance, and `pytest.skip` on an `ImportError` is how a test
suite stops noticing that a package is broken.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Protocol

import pytest

#: The seven packages carrying a `protocols.py`. `docs/interfaces.md` §8.
PACKAGES = ("spec_loader", "handoff", "validator", "agent", "closure", "env_mgr", "monitor")

#: **All seven conform** on the function half. On the run that introduced this file exactly one
#: package did not: `agent.select_backend` took a fourth parameter, `assignment`,
#: that `agent/protocols.py` did not declare — review finding R3, and not
#: cosmetic, because `assignment` is how the executor receives its `readme`,
#: `entry`, `zone`, `environment` and `wrap_argv`. A caller using the declared
#: signature built an executor with `Assignment()` and got an agent that started
#: and did nothing.
#:
#: It was quarantined for about an hour with `xfail(strict=True)` rather than
#: tolerated, because the instruction was that nobody loosens the comparison to
#: make it pass — and then `agent` and `main` closed it from both sides and the
#: strict marker XPASSed, which pytest reports as a failure, which is what made
#: removing it somebody's diff rather than nobody's. **If a drift is found that
#: cannot be fixed the same day, that is the shape to use again**: quarantine the
#: package, never the comparison.


def _module(name: str) -> ModuleType:
    return importlib.import_module(name)


def _declared_functions(protocols: ModuleType) -> dict[str, object]:
    """Module-level functions a `protocols.py` declares, from its `__all__`.

    `__all__` rather than `dir()`: a protocols module imports names it re-exports
    — `closure/protocols.py` takes `Problem` and `Registries` from `spec_loader`
    — and those are somebody else's declarations, checked in their own package.
    """
    exported = getattr(protocols, "__all__", ())
    return {
        name: obj for name in exported if inspect.isfunction(obj := getattr(protocols, name, None))
    }


def _implementation(package: ModuleType, protocols: ModuleType, name: str) -> object | None:
    """The package's live object for a declared name, if it has one.

    A declaration the package does not export is not a violation *here* — that is
    `docs/interfaces.md` §4's question and `test_import_rules.py`'s — so this
    returns `None` and the caller skips the pair rather than failing it. What it
    must not do is compare the declaration against itself, which is what
    `getattr(package, name)` gives for a package whose `__init__` re-exports
    straight from `protocols.py`.
    """
    impl = getattr(package, name, None)
    if impl is None or impl is getattr(protocols, name, None):
        return None
    return impl


def _compare(declared: inspect.Signature, actual: inspect.Signature) -> list[str]:
    """Every way the two disagree, all of them, rather than the first."""
    faults: list[str] = []
    want, got = declared.parameters, actual.parameters

    if list(want) != list(got):
        faults.append(f"parameters {list(got)} against declared {list(want)}")
        return faults  # positions are meaningless once the names differ

    for name in want:
        if want[name].kind is not got[name].kind:
            faults.append(
                f"{name!r} is {got[name].kind.description} and is declared "
                f"{want[name].kind.description}"
            )
        if want[name].default != got[name].default:
            faults.append(
                f"{name!r} defaults to {got[name].default!r} and is declared "
                f"{want[name].default!r}"
                + (
                    "  <- a default the declaration does not have is the drift this test exists for"
                    if want[name].default is inspect.Parameter.empty
                    else ""
                )
            )
    return faults


def _declared_constructors(protocols: ModuleType) -> dict[str, object]:
    """Classes in `__all__` whose own body defines `__init__`.

    Two exclusions, and the first cost me a wrong measurement before I made it.

    `"__init__" in vars(cls)` rather than `hasattr`, because everything inherits
    `object.__init__`. But that alone is not enough: **`typing.Protocol` injects
    a synthetic `_no_init_or_replace_init` into every Protocol class**, so the
    naive check reports a declared constructor for `Executor`, `Monitor`,
    `Pushable` and a dozen others that declare none. Filtering on
    `__init__.__module__` is what tells a written constructor from an injected
    one — a `grep "def __init__"` over the source and this check disagreed, and
    the source was right.

    A dataclass's generated `__init__` is a real declaration, and those fall out
    at `_implementation` instead: a value type exported straight from
    `protocols.py` *is* its own implementation, so there are not two things to
    compare.
    """
    out: dict[str, object] = {}
    for name in getattr(protocols, "__all__", ()):
        cls = getattr(protocols, name, None)
        if not inspect.isclass(cls) or "__init__" not in vars(cls):
            continue
        if getattr(vars(cls)["__init__"], "__module__", None) == "typing":
            continue  # Protocol's synthetic, not a declaration
        out[name] = cls
    return out


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_declared_constructor_matches_its_implementation(package_name: str) -> None:
    """The half a Protocol's *methods* do not cover, and `validator` found it.

    **It compares exactly one constructor, and that one drifted on its first
    run** — which I had not expected from a test written as armed-and-empty.
    `validator.PhaseOutcome` was declared as a dataclass whose five fields had no
    defaults and implemented as one where four of them did, so
    `PhaseOutcome(kind)` was a `TypeError` against the declaration and
    `empty=True` against the implementation. `validator` closed it within
    minutes; the strict quarantine XPASSed and the marker went with it.

    Every other declared constructor is a dataclass on a *value type* exported
    straight from `protocols.py`, which is its own implementation, so there is
    nothing to compare. **No Protocol declares a constructor at all**, which is
    the gap below.

    Why it is worth an armed-and-empty test rather than nothing. `validator`
    shipped `PhaseRunner.__init__(package_root: Path | None = None)` falling back
    to `Path.cwd()`, so a package-relative body path resolved against wherever the
    process started — finding nothing with a puzzling message, or a different file
    of the same name. Their note on why the sibling test could not see it is the
    finding:

        `protocols.PhaseRunner` is a Protocol declaring only `run_phase`, so
        there is no `__init__` to compare against. A constructor never named in
        a Protocol is outside its reach — and anything wired by the composition
        root is exactly that shape.

    They are right, and it is wider than one class. `docs/interfaces.md` §4.3 and
    §4.6 show constructors **in prose** — `PhaseRunner(strict_level)` and
    `EnvManager(ctx)` — and §2's root constructs twenty-odd types, while **no
    Protocol in the tree declares a constructor**. That is finding C1's shape
    applied to construction rather than to resolution, and closing it is a
    decision for `main` rather than a test I can write alone. This is the half
    that costs nothing and waits.
    """
    package = _module(package_name)
    protocols = _module(f"{package_name}.protocols")

    faults: list[str] = []
    for name, declaration in _declared_constructors(protocols).items():
        impl = _implementation(package, protocols, name)
        if impl is None or not inspect.isclass(impl):
            continue
        for fault in _compare(
            inspect.signature(declaration.__init__), inspect.signature(impl.__init__)
        ):
            faults.append(f"{package_name}.{name}.__init__: {fault}")

    assert not faults, (
        f"{package_name} and {package_name}/protocols.py disagree on a "
        f"constructor:\n  " + "\n  ".join(faults)
    )


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_declared_function_matches_its_implementation(package_name: str) -> None:
    """One failure names every disagreement in the package, not the first.

    Reporting them one at a time would make fixing a drifted seam an N-round
    trip, which is the same argument `load_package` makes for collecting problems
    rather than raising on the first bad spec.
    """
    package = _module(package_name)
    protocols = _module(f"{package_name}.protocols")

    faults: list[str] = []
    for name, declaration in _declared_functions(protocols).items():
        impl = _implementation(package, protocols, name)
        if impl is None:
            continue
        for fault in _compare(inspect.signature(declaration), inspect.signature(impl)):
            faults.append(f"{package_name}.{name}: {fault}")

    assert not faults, (
        f"{package_name} and {package_name}/protocols.py disagree:\n  "
        + "\n  ".join(faults)
        + f"\n\n{package_name}/protocols.py is the frozen side. If the "
        f"declaration is wrong, `docs/interfaces.md` §1.1 says say so and name "
        f"both sides — do not change a cross-module signature quietly, and do "
        f"not loosen this comparison."
    )


@pytest.mark.parametrize("package_name", PACKAGES)
def test_the_package_imports_at_all(package_name: str) -> None:
    """Stated separately so a broken import fails as itself.

    Folded into the test above it would read as a conformance failure, and the
    two want different people.
    """
    assert _module(package_name) is not None
    assert _module(f"{package_name}.protocols") is not None


def test_this_test_would_catch_the_defect_it_was_written_for() -> None:
    """The meta-test, and it is the price of trusting a green run.

    `closure`'s name-only comparison passed over exactly this. Two synthetic
    signatures differing *only* in a default must be reported — if `_compare`
    ever stops seeing that, every green run above means nothing.
    """

    def declared(regs, report, *, skip=frozenset()): ...

    def drifted(regs, report=None, *, skip=frozenset()): ...

    faults = _compare(inspect.signature(declared), inspect.signature(drifted))
    assert len(faults) == 1
    assert "defaults to None" in faults[0]
    assert "the drift this test exists for" in faults[0]

    def renamed(regs, handoff_report, *, skip=frozenset()): ...

    assert _compare(inspect.signature(declared), inspect.signature(renamed))

    def keyword_only(regs, *, report, skip=frozenset()): ...

    assert _compare(inspect.signature(declared), inspect.signature(keyword_only))

    assert not _compare(inspect.signature(declared), inspect.signature(declared))


def test_no_protocol_declares_a_constructor_today() -> None:
    """The measurement behind the test above, asserted rather than described.

    Not a rule — a **statement of the gap**, written as a test so that closing it
    is a visible event rather than a quiet one. `docs/interfaces.md` shows
    `PhaseRunner(strict_level)` and `EnvManager(ctx)` in prose and declares
    neither; §2's composition root constructs twenty-odd types and the checkable
    half describes none of their constructors.

    When a Protocol declares its first, this fails, and **the fix is to delete
    this test** rather than to exempt the package: at that point the sibling above
    is doing real work and this has stopped saying anything true.

    Value types are excluded because they are not the gap: a dataclass in
    `protocols.py` *is* the thing it declares, so its generated `__init__` has no
    second side to drift from.
    """
    declaring = {}
    for name in PACKAGES:
        protocols = _module(f"{name}.protocols")
        found = [
            cls_name
            for cls_name, cls in _declared_constructors(protocols).items()
            if issubclass(cls, Protocol) and getattr(cls, "_is_protocol", False)
        ]
        if found:
            declaring[name] = sorted(found)

    assert not declaring, (
        f"{declaring} declares a constructor on a Protocol — good, that closes "
        f"the gap. Delete this test; "
        f"test_every_declared_constructor_matches_its_implementation is live."
    )
