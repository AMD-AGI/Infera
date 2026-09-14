"""The CI half. `demo` design §11, and criteria 1, 6, 11, 13, 15, 16.

Spec §5 says *"the demo is not a test, and CI does not run it"*; spec §1 says the
demo is *"the first thing to break when one of them drifts"*. Both cannot be
true of an artefact nothing checks.

**Airflow has the answer and it is not a compromise.** Every shipped example DAG
is covered by three tests and not one of them executes a DAG —
`test_should_be_importable`, `test_should_not_do_database_queries`,
`test_should_not_run_hook_connections`. CI **loads** the example on every
commit; a human **runs** it. Our load-time half already has a name — `--dry-run`
— and it needs no credentials, no sandbox, no model and no network, so putting
it in CI contradicts nothing: the thing spec §5 excludes from CI is the *run*.

The counter-example is worth the sentence: dbt's `jaffle_shop`, the most-copied
example in its ecosystem, 544 stars, archived, and `.github/workflows` is a 404.
"""

from __future__ import annotations

import ast
import io
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import spec_loader
from cli import main as cli_main
from cli import package
from cli.environment import CredentialsMissing, preflight_credentials
from cli.events import EventKind
from cli.render.human import HumanRenderer
from cli.stream import Stream

REPO = Path(__file__).resolve().parents[2]

#: Every component package. `cli` imports all of them; **none imports `cli`**.
COMPONENTS = (
    "spec_loader",
    "handoff",
    "validator",
    "agent",
    "closure",
    "env_mgr",
    "monitor",
    "task_graph",
)


def _run_cli(*argv: str) -> tuple[int, Stream]:
    """`main`, with the human renderer pointed at a buffer.

    Calling `main` rather than a helper is deliberate: the exit code is part of
    what criteria 9 and 11 assert, and a test that called `_dry_run` directly
    would not see it.
    """
    stream = Stream()
    buffer = io.StringIO()
    real = cli_main.Stream
    captured: list[Stream] = []

    class _Capturing(real):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            captured.append(self)

    cli_main.Stream = _Capturing  # type: ignore[misc]
    real_renderer = cli_main.HumanRenderer
    cli_main.HumanRenderer = lambda out=None: HumanRenderer(buffer)  # type: ignore[misc, assignment]
    try:
        code = cli_main.main(list(argv))
    finally:
        cli_main.Stream = real  # type: ignore[misc]
        cli_main.HumanRenderer = real_renderer  # type: ignore[misc]
    return code, captured[0] if captured else stream


# --------------------------------------------------------------------------- #
# Criterion 16 — it loads through the ordinary task-package path


def test_loads_as_an_ordinary_package(package_root: Path, registry: Any) -> None:
    """YAML, discriminated and schema-validated like any package's.

    **No privileged import and no private loader path**: the runner hands
    `spec_loader.YamlPackage` a directory, which is the same call the
    whole-system CLI would make for a package from anywhere else. Criterion 16
    is that sentence being structurally true, and the evidence is that this test
    can name every step without naming anything in `demo/` except `locate`.

    **`problems == []` is the assertion that grew teeth.** Under the old format
    a `SpecSource` was a file and a kind, and a package with a bad body path
    still discovered cleanly. `documents()` now substitutes variables and fills
    every body from `assets/`, and both report through `problems` — so an
    unresolved `${...}`, a body the convention could not find, and a body bound
    by hand where it could have (non-fatal, but still a `Problem`) all land here.
    An empty list is therefore also the assertion that **nothing in this package
    binds a filename**, which is what the demo is the reference for.
    """
    pkg = package.task_package(package_root)
    assert isinstance(pkg, spec_loader.YamlPackage)
    contents = pkg.documents()
    assert contents.documents, "the package declares nothing"
    assert list(contents.problems) == [], contents.problems
    assert {d.kind for d in contents.documents} == {"handoff", "validator", "agent", "closure"}
    assert all(Path(d.origin.split("#")[0]).suffix == ".yaml" for d in contents.documents)

    # Several objects out of one file, which is what the discriminator buys and
    # what the per-kind directory layout could not express.
    per_file: dict[str, int] = {}
    for document in contents.documents:
        per_file[document.origin.split("#")[0]] = per_file.get(document.origin.split("#")[0], 0) + 1
    assert max(per_file.values()) > 1, per_file

    # And every one of them was admitted through the schema for its kind.
    assert sorted(registry.get("closures").names()) == [
        "consume",
        "describe",
        "main",
        "produce",
    ]
    assert sorted(registry.get("handoff_specs").names()) == ["facts", "summary"]
    assert sorted(registry.get("validator_specs").names()) == ["check_facts", "check_grounded"]
    # **`compose` is gone, not renamed.** A non-leaf declares no agent, so the
    # hand-written spec whose description was "it executes nothing itself" has
    # no reason to exist; `task_graph` supplies the built-in.
    assert sorted(registry.get("agent_specs").names()) == ["collect", "describe"]


def test_the_non_leaf_declares_no_agent(package_root: Path) -> None:
    """The authoring half of the rule, asserted where an author would break it.

    `test_loads_as_an_ordinary_package` shows `compose` is not in the catalogue;
    this shows the reason — `main` names nobody — and it is the assertion that
    fails if someone reintroduces an agent on the root to quiet a `KeyError`.
    Every leaf still declares one, which is the other half: `closure.schema.json`
    requires `agent` of a leaf and of nothing else, so a leaf silently losing its
    agent must not read as "the same rule".
    """
    closures = registry_docs(package_root)
    assert "agent" not in closures["main"], closures["main"].get("agent")
    assert closures["main"]["task"]["subgraph"], "main is not a non-leaf"
    for leaf in ("produce", "describe", "consume"):
        assert closures[leaf]["agent"], leaf
        assert "subgraph" not in closures[leaf]["task"], leaf


def registry_docs(package_root: Path) -> dict[str, Any]:
    """The closure documents, keyed by name, straight off the package."""
    return {
        str(d.doc["name"]): d.doc
        for d in package.task_package(package_root).documents().documents
        if d.kind == "closure"
    }


def test_package_declares_no_schema(package_root: Path) -> None:
    """It has no schema of its own, and could not use one if it wrote one.

    The five schemas live in `spec_loader/schemas/` and `spec_loader.validate`
    is the single enforcement point; a `.json` schema in the package would be a
    file nothing reads, which is worse than an error.
    """
    assert not list(package_root.rglob("*.schema.json"))
    assert set(spec_loader.KINDS) >= {"handoff", "validator", "agent", "closure"}


def test_every_spec_loads_with_no_variables_supplied(package_root: Path) -> None:
    """A spec must load with **no** variables, because `show` and `--dry-run`
    supply none and a reviewer reading the package by hand supplies none either.

    An unsupplied `${name}` with no default is a fatal `Problem` by design — a
    reference left literal is a path that resolves to nothing later, in another
    module, with nothing to say why. So this is the test that keeps every
    reference in the package defaulted, and there is exactly one: `${outside}`,
    criterion 8's leak target.
    """
    contents = spec_loader.YamlPackage(root=package_root).documents()
    assert list(contents.problems) == [], contents.problems
    for document in contents.documents:
        problems = spec_loader.validate(
            document.doc, spec_loader.schema_for(document.kind), origin=document.origin
        )
        assert problems == [], f"{document.origin}: {problems}"
        assert document.doc["name"]


def test_the_only_variable_is_the_leak_target(package_root: Path) -> None:
    """One variable, and the test that says so is what keeps the list from
    growing back.

    `package_root` and `store_root` were passed beside it and are gone: the
    assets convention finds body paths, and no spec ever referenced the second.
    A new `${...}` in the package is a new thing the runner must supply, and
    every one of them is a way for `show` to disagree with `run`.
    """
    import re

    from ruamel.yaml import YAML

    def strings(node: Any) -> Any:
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from strings(value)
        elif isinstance(node, list):
            for item in node:
                yield from strings(item)

    # **The parsed tree, not the file text.** `substitute` walks values, so a
    # `${name}` written in a comment — this package explains the assets
    # convention in several — is not a variable and a text scan would report one.
    found = {
        match.group(1)
        for path in package_root.rglob("*.yaml")
        for value in strings(YAML().load(path.read_text()))
        for match in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", value)
    }
    assert found == {"outside"}, found


# --------------------------------------------------------------------------- #
# Criterion 11 — `--dry-run` resolves everything and dispatches nothing


def test_dry_run_dispatches_nothing(tmp_path: Path) -> None:
    code, stream = _run_cli("run", "--dry-run", "--demo-root", str(tmp_path))
    assert code == cli_main.OK

    done = [dict(e.fields) for e in stream.of_kind(EventKind.RUN_COMPLETE)][-1]
    # Asserted rather than assumed: "dispatched nothing" is exactly the kind of
    # claim that stays true by accident until it does not.
    assert done["dispatched"] == 0
    assert done["agents_instantiated"] == 0
    assert done["tasks"] == 4
    # And it resolved everything on the way: four closures, each reported.
    assert stream.count(EventKind.CLOSURE_RESOLVED) == 4
    assert stream.count(EventKind.SPEC_REJECTED) == 0


def test_broken_closure_names_its_file(tmp_path: Path) -> None:
    """Criterion 11's second half, and it collides with criterion 13 if the
    broken file is in the package — so it is a **sibling package** the ordinary
    discovery pass does not reach, behind one flag. Two runs, no editing.

    It used to be `examples/demo/broken/`, a subdirectory. `YamlPackage` scans
    every `*.yaml` under a root except `assets/`, so that would now load on every
    ordinary run; a sibling directory with its own `main.yaml` and `assets/` is
    the smallest thing that is unambiguously not part of the demo."""
    code, stream = _run_cli("run", "--dry-run", "--with-broken", "--demo-root", str(tmp_path))
    assert code == cli_main.LOAD_ERROR == 1

    rejected = stream.of_kind(EventKind.SPEC_REJECTED)
    assert len(rejected) == 1
    message = rejected[0].message
    assert "demo-broken/main.yaml" in message
    assert "nonexistent" in message  # the kind, named
    # A cross-registry fault: no schema can catch it, and the message says which
    # kinds *do* exist rather than only that this one does not.
    assert "facts" in message and "summary" in message


def test_the_broken_package_is_not_in_the_ordinary_pass(package_root: Path) -> None:
    """Criterion 13 depends on this. If the broken document were discovered,
    every run would fail and no amount of not-hand-editing would help.

    **Two assertions, because the scan rule changed under this test.** The first
    is that nothing named `dangling` is admitted. The second is the one that
    would have caught the move going wrong: the broken package's root is not
    inside the demo's, so no rule about *which* files are scanned is being relied
    on — the file is simply not under the tree that is walked.
    """
    contents = package.task_package(package_root).documents()
    assert contents.documents
    assert "dangling" not in {d.doc.get("name") for d in contents.documents}
    broken = package.broken_package(package_root)
    assert broken.root.is_dir(), broken.root
    assert package_root not in broken.root.parents and broken.root != package_root


# --------------------------------------------------------------------------- #
# Criterion 13 — running twice succeeds without hand-editing


def test_two_runs_do_not_collide(tmp_path: Path) -> None:
    first_code, first = _run_cli("run", "--dry-run", "--demo-root", str(tmp_path))
    second_code, second = _run_cli("run", "--dry-run", "--demo-root", str(tmp_path))
    assert first_code == second_code == cli_main.OK

    def root_id(stream: Stream) -> str:
        return dict(stream.of_kind(EventKind.GRAPH_BUILT)[0].fields)["root"]

    # Two runs, two sets of ids. Nothing is reused and nothing was edited.
    assert root_id(first) != root_id(second)


# --------------------------------------------------------------------------- #
# Criterion 1 — the overhead budget


def test_overhead_budget(tmp_path: Path) -> None:
    """*Under one minute excluding model latency, no GPU, no cluster, no setup.*

    Asserted at **30 s — half the budget** — over the part CI can measure: the
    whole load path, twice. The other half is `git clone`, which
    `materials/08-demo.md` §7 measured at ~0.56 s per zone and which needs a
    sandbox this test does not have. Measured here at well under a second, so
    the headroom is two orders of magnitude and the assertion is a guard against
    something going quadratic rather than a stopwatch.
    """
    started = time.monotonic()
    for _ in range(2):
        assert _run_cli("run", "--dry-run", "--demo-root", str(tmp_path))[0] == cli_main.OK
    assert time.monotonic() - started < 30.0


# --------------------------------------------------------------------------- #
# Criterion 6 — no credentials means a clear message, and no fake fallback


def test_missing_credentials_message(tmp_path: Path, monkeypatch: Any) -> None:
    """The backend's own words, surfaced — **including stdout**.

    Measured with an empty config directory and `ANTHROPIC_*` scrubbed: `rc=1`,
    0.6 s, and `'Not logged in · Please run /login'` **on stdout**. So the demo
    invents no credentials check; it runs the preflight, catches the non-zero
    exit, and reports what the backend said. A demo that printed only stderr
    would lose the one message this criterion is about, which is why the test
    puts the sentence on stdout and nothing on stderr.
    """
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\necho 'Not logged in · Please run /login'\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)

    with pytest.raises(CredentialsMissing) as raised:
        preflight_credentials(cli="claude")
    assert "Not logged in" in str(raised.value)
    assert "/login" in str(raised.value)
    assert "stdout:" in str(raised.value)


def test_a_missing_backend_is_also_a_precondition(monkeypatch: Any) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(CredentialsMissing, match="not on PATH"):
        preflight_credentials(cli="claude")


def test_credentials_failure_exits_two_not_one(tmp_path: Path, monkeypatch: Any) -> None:
    """§8.3 separates 2 from 1: the **precondition** failed, not the run, and a
    reviewer needs to see which without reading prose."""
    monkeypatch.setattr(
        cli_main,
        "preflight_credentials",
        lambda: (_ for _ in ()).throw(CredentialsMissing("no key")),
    )
    monkeypatch.setattr(cli_main, "confinement", lambda *a, **k: "landlock")
    assert _run_cli("run", "--demo-root", str(tmp_path))[0] == cli_main.PRECONDITION == 2


def test_no_fake_backend_exists(package_root: Path) -> None:
    """*It does not fall back to a fake agent*, and there is nothing to fall back
    to: the package declares one real backend and the runner has no branch that
    would substitute another.

    Asserted over the source rather than over a run, because a fallback is a
    thing that is *absent*, and absence is a property to check rather than to
    arrange (SWE-bench's answer key was physically absent for two years and
    still leaked).
    """
    declared = {
        str(d.doc["name"]): d.doc
        for d in package.task_package(package_root).documents().documents
        if d.kind == "agent"
    }["describe"]
    # The loaded document, not the source text: the source says the words
    # "no fake" in a comment, which is the file being clear rather than the
    # package declaring one.
    assert [b["backend_entry"] for b in declared["backends"]] == [
        "agent.backends.claude_sdk:ClaudeSdkBackend"
    ]
    # And the runner substitutes nothing: `demo/` never names the fake, so there
    # is no branch that could reach for it.
    for source in (REPO / "demo").rglob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "FakeRunner" not in imported, f"{source} would substitute a fake runner"


# --------------------------------------------------------------------------- #
# Criterion 15 — the components import nothing from the demo


@pytest.mark.parametrize("component", COMPONENTS)
def test_no_component_imports_the_cli(component: str) -> None:
    """The wall, checked by reading every import in every component package.

    A grep for the token would also find it in a comment, which is why this
    walks the AST: the rule is about the **import graph**, and an import graph is
    checkable.
    """
    offenders: list[str] = []
    for source in (REPO / component).rglob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [
                    f"{source}: import {a.name}" for a in node.names if _is_the_cli(a.name)
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                # `from .cli import ...` is `env_mgr`'s own submodule, not this one.
                and node.level == 0
                and _is_the_cli(node.module or "")
            ):
                offenders.append(f"{source}: from {node.module} import ...")
    assert offenders == []


def _is_the_cli(module: str) -> bool:
    return module == "cli" or module.startswith("cli.")


def test_no_component_names_the_token_at_all(tmp_path: Path) -> None:
    """The grep half, kept beside the AST half rather than instead of it.

    `cli` is an ordinary word — `env_mgr` has a module of that name — so this
    permits it in a comment or a docstring and fails on it in code, which is
    the line the AST test cannot draw and this one can, cheaply. A relative
    `from .cli import ...` inside a component is its own module and does not
    match.
    """
    pattern = re.compile(r"^\s*(?:from\s+cli\b|import\s+cli\b)", re.MULTILINE)
    for component in COMPONENTS:
        for source in (REPO / component).rglob("*.py"):
            assert not pattern.search(source.read_text()), source


def test_examples_has_no_init(package_root: Path) -> None:
    """`examples/demo/` is **data**, not a Python package.

    This is what makes spec §1.1's rule checkable rather than a promise: the
    moment it holds an `__init__.py` it is importable, and the example stops
    looking like what an out-of-repository task package looks like. Main design
    §2's projection — *"`demo/` docs only"* — could not carry a console script
    (measured: it installs and dies with `ModuleNotFoundError` when run), and
    `demo` design D2 is the split that resolves both.
    """
    assert not list(package_root.rglob("__init__.py"))
    assert (package_root.parent.name, package_root.name) == ("examples", "demo")


def test_the_examples_tree_is_not_installed() -> None:
    """`examples*` is absent from `packages.find`, deliberately and permanently."""
    text = (REPO / "pyproject.toml").read_text()
    block = text[text.index("[tool.setuptools.packages.find]") :]
    block = block[block.index("include = [") :]
    block = block[: block.index("]") + 1]
    # The quoted entries only. The comment inside the list says the word
    # "examples" several times, which is the file explaining why it is absent
    # rather than the file shipping it — and a substring check over the raw
    # block cannot tell those apart.
    entries = re.findall(r'"([^"]+)"', block)
    assert "cli*" in entries
    assert not [e for e in entries if e.startswith("examples")], entries


def test_the_package_imports_no_component(package_root: Path) -> None:
    """The other direction of the wall, and the one design §2 states as a rule:
    *`examples/demo/` contains no file that `demo/` imports as a module* — and,
    symmetrically, nothing in it reaches into `agent_sys`.

    The programs there are run as subprocesses and told what they need through
    the environment. That is what an out-of-repository task package can do, and
    it is the whole bound on the exception spec §1.1 grants this one.
    """
    for source in package_root.rglob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                root = name.split(".")[0]
                assert root not in {*COMPONENTS, "cli"}, f"{source} imports {name}"


# --------------------------------------------------------------------------- #
# The load path needs neither a credential nor a sandbox


def test_loading_needs_no_credentials_and_no_sandbox(tmp_path: Path, monkeypatch: Any) -> None:
    """Airflow's *"what loading must not do"* shape, and it is what lets CI run
    this on every commit.

    `PATH` is emptied so no backend binary can be found, every `ANTHROPIC_*` is
    scrubbed, and the confinement probe is replaced with one that raises if it
    is called at all — so a load that reached for either fails here rather than
    on a reviewer's machine.
    """
    for name in [key for key in os.environ if key.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the load path probed for a sandbox")

    monkeypatch.setattr(cli_main, "confinement", refuse)
    monkeypatch.setattr(cli_main, "preflight_credentials", refuse)

    assert _run_cli("show", "--demo-root", str(tmp_path))[0] == cli_main.OK
    assert _run_cli("run", "--dry-run", "--demo-root", str(tmp_path))[0] == cli_main.OK


def test_the_console_script_is_declared() -> None:
    """Criterion 1 is written against `pip install -e agent_sys` then a verb, so
    the verb has to be a real console script."""
    assert 'agent-sys = "cli.main:main"' in (REPO / "pyproject.toml").read_text()


def test_a_missing_package_is_a_precondition_naming_both_paths(tmp_path: Path) -> None:
    """Design §12 step 3, and it is a deviation worth its own test: a **wheel**
    install of `agent_sys` gives a working `agent-sys` that refuses to run.

    That is the correct behaviour — packaging the specs as package data would
    make the example behave differently depending on how it was installed — but
    it is a refusal, so it names why and what it tried.
    """
    with pytest.raises(package.PackageNotFound) as raised:
        package.locate(tmp_path / "nowhere")
    named = str(raised.value)
    assert "not installed with the wheel" in named
    assert str(tmp_path / "nowhere") in named  # the path it was given, named

    # And with nothing given, both derived candidates are named. Patched rather
    # than run from a wheel, because a wheel install is the one thing a test in
    # a checkout cannot arrange.
    import cli.package as mod

    original = mod._is_package
    mod._is_package = lambda _path: False
    try:
        with pytest.raises(package.PackageNotFound) as fell_through:
            package.locate()
    finally:
        mod._is_package = original
    message = str(fell_through.value)
    assert "tried:" in message
    assert "--package DIR" in message


def test_the_entry_scripts_are_executable_and_shell(package_root: Path) -> None:
    """Every `entry.sh` the package declares is a real shell script.

    `validator.ScriptBodyRunner` and `agent.backends.program.ProgramExecutor`
    both run `/bin/sh <entry>`, so the executable bit is not what makes it work
    — but a body that is not a shell script at all is a load-time-invisible
    fault that appears only under an executor, and this is the cheapest place to
    notice it.
    """
    scripts = sorted(package_root.rglob("entry.sh"))
    assert len(scripts) == 4, [str(s) for s in scripts]
    for script in scripts:
        assert script.read_text().startswith("#!/bin/sh")
        assert (
            subprocess.run(  # noqa: S603 — `sh -n`, a syntax check, runs nothing
                ["/bin/sh", "-n", str(script)], capture_output=True
            ).returncode
            == 0
        ), script


def test_python_target_is_310(package_root: Path) -> None:
    """Python ≥ 3.10 is the target and this machine runs 3.13, so a 3.10-only
    failure would not surface locally.

    Checked over **names the code actually uses**, not over the source text: the
    3.11-and-later constructs get named in comments explaining why they are not
    used, and a substring check cannot tell those apart. That mistake was made
    twice in this file before it was made structural.
    """
    assert sys.version_info >= (3, 10)
    banned = {"StrEnum", "Self", "assert_never", "TypeVarTuple", "override"}
    for source in [*(REPO / "demo").rglob("*.py"), *package_root.rglob("*.py")]:
        tree = ast.parse(source.read_text(), filename=str(source))
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert not (used & banned), f"{source}: {sorted(used & banned)}"


# --------------------------------------------------------------------------- #
# The two seams that block the `run` verb, as strict-xfail handshakes.
#
# `docs/interfaces.md` §8.1 names this mechanism: **`xfail(strict=True)` as a
# handshake between two packages** — the test lands green today and goes red the
# instant the other side changes, so nobody has to remember anything. Both of
# these are red-when-fixed on purpose: the day one of them XPASSes, the demo's
# `run` verb can be finished, and this is what says so.


@pytest.mark.xfail(
    strict=True,
    reason="F-D1: no component calls HandoffStore.put, so nothing publishes a "
    "handoff and agent/gate.py reports OUTPUT_ABSENT for every task. Reported "
    "to `handoff` and `agent`; when it is closed this XPASSes and the demo's "
    "run path gets its publication step.",
)
def test_something_publishes_a_handoff() -> None:
    """Measured, not predicted: three `HandoffStore.put` call sites in the tree
    and all three are in `tests/handoff`."""
    callers: list[str] = []
    for component in COMPONENTS:
        for source in (REPO / component).rglob("*.py"):
            tree = ast.parse(source.read_text(), filename=str(source))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "put"
                    and any(
                        isinstance(kw.arg, str) and kw.arg == "producer" for kw in node.keywords
                    )
                ):
                    callers.append(f"{source}:{node.lineno}")
    assert callers, "nothing publishes a handoff"


def test_a_task_is_given_to_a_monitor_by_set_task() -> None:
    """F-D8, **closed by `task_graph` while this was xfailing.**

    It landed as `Scheduler._watch` — `self._r.get("monitor_for")(task,
    self._r).set_task(task.id)`, guarded by `if "monitor_for" not in self._r`, so
    a declaration-only `monitor` still works. That is the right site: the
    scheduler is one of the two places that sees every task at birth, and the
    subtasks are why it had to be one of them — `Task.unfold` creates them inside
    `enter_phase(RUNNING)`, so no caller outside `task_graph` ever holds one.

    This was `xfail(strict=True)` for about an hour and **XPASSed**, which is
    `docs/interfaces.md` §8.1's handshake working as a protocol rather than as a
    check: it went red the moment the other side landed, and named both sides
    while doing it. Nobody had to remember to come back. Kept as a plain
    assertion so a regression is still caught.
    """
    callers: list[str] = []
    for component in COMPONENTS:
        for source in (REPO / component).rglob("*.py"):
            tree = ast.parse(source.read_text(), filename=str(source))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_task"
                ):
                    callers.append(f"{source}:{node.lineno}")
    assert callers, "nothing gives a monitor a task to watch"


def test_body_paths_are_package_relative(package_root: Path) -> None:
    """`_common.schema.json`: *"Package-relative path to the readme.md"*.

    They were absolute for a while, filled from `config.package_root`. That was
    the **F-D3 workaround and it was correct when written** — `agent` had no
    `package_root` and `validator` did, so absolute was the only form both
    accepted. §4.16 then staged the package into the zone and absolute stopped
    being resolvable against anything: `Path(staged) / "/abs"` is `/abs`, so a
    staged body would never be reached. A reversal caused by a change elsewhere.

    **Now nothing in the package writes a body path at all** — the assets
    resolver derives them — so this pins the *resolver's* output rather than a
    fill's, and it is the guard that would catch it returning an absolute path.
    The two clauses are separate: `assets/`-prefixed says the convention found
    it; not-absolute and no-`..` says it is resolvable against the staged copy.
    """
    checked = 0
    for document in package.task_package(package_root).documents().documents:
        doc = document.doc
        bodies = [doc.get("body"), (doc.get("task") or {}).get("body")]
        for body in [b for b in bodies if b]:
            for key in ("readme", "entry"):
                declared = body.get(key)
                if declared is None:
                    continue
                checked += 1
                assert not declared.startswith("/"), f"{document.origin}: {key} is absolute"
                assert ".." not in Path(declared).parts, f"{document.origin}: {key} escapes"
                assert declared.startswith("assets/"), (
                    f"{document.origin}: {key} is {declared!r}, which is not under assets/ — "
                    f"either it was bound by hand or the resolver changed its base"
                )
    assert checked >= 8, checked


def test_no_body_is_bound_by_hand(package_root: Path) -> None:
    """*Every filename found by convention, nothing bound by hand.*

    An explicit binding is legal and reports a **non-fatal** `Problem` at load,
    so a package can drift into hand-binding without anything failing. This is
    the assertion that makes the demo's claim to be the best-practice reference
    checkable: no `body:` key exists in any source, and the loader raises no
    `explicit-binding` warning.

    Both halves, because either alone is weak: the text scan would miss a binding
    written some other way, and the warning list would be empty on a package that
    declared no bodies at all — which `test_body_paths_are_package_relative`'s
    `checked >= 8` is what rules out.
    """
    contents = package.task_package(package_root).documents()
    assert [p for p in contents.problems if p.keyword == "explicit-binding"] == []
    assert list(contents.problems) == [], contents.problems
    from ruamel.yaml import YAML

    def keys(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    # **The parsed tree, not the file text.** The sources explain in comments
    # that there is no `body:` key, and a text scan reports the explanation.
    for path in package_root.rglob("*.yaml"):
        assert "body" not in set(keys(YAML().load(path.read_text()))), path


def test_both_package_roots_are_supplied(tmp_path: Path, package_root: Path) -> None:
    """Relative paths need a resolver, and there are **two** of them.

    `Runner(package_root=)` reproduces what the absolute fill produced, and
    becomes the fallback when `agent` lands per-attempt resolution against the
    staged root. `PhaseRunner(package_root=)` is the one that would break first:
    `build_registry` constructs it as `PhaseRunner(level)` and
    `ScriptBodyRunner` joins `self._root / spec.body.entry`, defaulting to
    `Path.cwd()` — so a relative validator entry would resolve against wherever
    the demo happened to be started from.
    """
    from validator import PhaseRunner

    registry = cli_main._registry(
        package_root, cli_main.layout_for(tmp_path).create(), Stream(), resume=False, variables={}
    )
    assert Path(registry.get("runner").package_root) == package_root
    phase_runner = registry.get("phase_runner")
    assert isinstance(phase_runner, PhaseRunner)
    assert Path(phase_runner._package_root) == package_root
