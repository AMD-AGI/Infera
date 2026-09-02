"""Criteria 8 and 9 — the block, and the refusal.

**What is tested here is that the demo *reports* them correctly**, which is a
test about the event stream and needs no agent. The isolation *properties* are
`env_mgr`'s and are CI-enforced in `tests/env_mgr` on every commit — criteria
2–14 there, including the scripted bypass, the three prefix defeats, inheritance
across `exec`, and fail-closed. `demo` design §1.2 draws that line and §14.3
says why the other side of it is deliberately not tested here: a test that
started a model call is non-deterministic, costs money, and fails on a fork.

Measured end to end (`materials/08-demo.md` §1), in one Landlock domain with
`cwd` set to the zone:

| | |
|---|---|
| a real model call | rc=0, 8.4 s |
| a write **inside** the zone | rc=0, the file is there |
| `echo leaked > <outside>/leak.txt` via bash | rc=0 for the agent, **nothing written** |

and the agent's own report of the third names the path, quotes
`permission denied`, gives the exit code, and says nothing was written. **Two of
criterion 8's three clauses are therefore already true without the demo doing
anything**: the OS blocks it and the agent is told why, by the OS, and
understands it. What the demo adds is the third — an `ACCESS_DENIED` event with
`expected: true`, naming the mechanism and the errno, so the block reads as a
demonstration rather than as a fault.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from cli import main as cli_main
from cli.environment import (
    RepositoryNotPrepared,
    build_context,
    confinement,
    demo_grants,
    layout_for,
    preflight_repository,
    unconfinable,
)
from cli.events import EventKind
from cli.render.human import line_for
from cli.stream import Stream
from env_mgr import material
from env_mgr.isolation.probe import Availability, select
from env_mgr.protocols import NoConfinement


class _ZoneAt:
    """The one attribute `material.deploy` reads off a zone."""

    def __init__(self, root: str) -> None:
        self.root = root


def _loaded_specs(package_root: Path, kind: str, **variables: str) -> dict[str, Any]:
    """Specs from the package, **loaded with the package's real variables**.

    Through `cli.package.task_package` rather than a hand-built variable set, so
    a test cannot pass by supplying a value the run does not — which is how the
    leak target's absence survived. Keyed by name, since that is how a reader
    names one; `kind` is the schema kind, so `closure` selects the steps and
    `agent` the agents.

    **It validates, and that is not decoration.** `documents()` substitutes and
    fills bodies but does not check a schema; a test that read a document the
    loader would have rejected would be asserting over something no run can
    produce.
    """
    import spec_loader

    contents = cli_main.package.task_package(package_root, **variables).documents()
    assert [p for p in contents.problems if p.fatal] == [], contents.problems
    out: dict[str, Any] = {}
    for document in contents.documents:
        problems = spec_loader.validate(
            document.doc, spec_loader.schema_for(document.kind), origin=document.origin
        )
        assert problems == [], f"{document.origin}: {problems}"
        if document.kind == kind:
            out[str(document.doc.get("name"))] = document.doc
    return out


# --------------------------------------------------------------------------- #
# Criterion 8 — the block, as an expected event naming the mechanism


def test_access_denied_event_shape(stream: Stream) -> None:
    """The event a reviewer reads, and the object criterion 14 asserts over.

    Both renderings, from one call, and the `expected` marker is on the event
    rather than in the sentence — a demo that said "this is fine" in prose while
    its JSON said `"expected": false` would be exactly the disagreement §7.1
    exists to make unrepresentable.
    """
    event = stream.emit(
        EventKind.ACCESS_DENIED,
        "the agent's scripted write to /tmp/outside/leak.txt was refused; nothing was written",
        path="/tmp/outside/leak.txt",
        mechanism="landlock",
        errno="EACCES",
        expected=True,
        task="a91b",
        via="bash",
    )
    assert event.fields["mechanism"] == "landlock"
    assert event.fields["errno"] == "EACCES"
    assert event.fields["expected"] is True
    # Named, not merely reported: which mechanism caught it is the difference
    # between a demonstration and a filesystem that happened to be read-only.
    assert event.fields["via"] == "bash"

    line = line_for(event)
    assert "landlock" in line and "EACCES" in line and "expected" in line
    assert "denied" in line


def test_the_leak_target_resolves_to_a_real_directory_outside_every_zone(
    tmp_path: Path, package_root: Path
) -> None:
    """**Criterion 8's probe must write somewhere that exists and is out of zone.**

    The assertion this replaces was `"AGENT_SYS_DEMO_OUTSIDE" in readme` — that
    a string appears in a file. It was green while **nothing exported that
    variable anywhere in the tree**, so the agent ran `echo leaked >
    "/leak.txt"` and got `Permission denied` from a root-owned `/`: a
    convincing refusal, on any machine, for ever, with the sandbox switched off
    entirely. `demo-2` measured it. A test that cannot fail is how it survived.

    So this asserts the property instead, and in two halves that have to meet:
    it reads the variable's **name out of the readme the agent is given**, then
    **resolves that name through `material.deploy`**, the function that actually
    builds a task's environment. Neither half alone would have caught it — the
    old test had the first, and a test that only checked the spec would pass
    while the readme named something else.

    The three clauses, in the order they would fail:

    1. the path is **absolute** — a relative one would resolve against the
       zone, which is inside the boundary this exists to cross;
    2. it **exists**, and is `Layout.outside` — a path that was never there
       refuses identically, and then the demonstration is a typo;
    3. it is **outside `zones/`**, which is the whole claim.

    `test_the_outside_directory_is_outside_every_zone` pins the grant's *mode*;
    together they say the refusal is a **write** refusal against a readable
    directory rather than any refusal at all.
    """
    import re

    layout = layout_for(tmp_path).create()
    told = (package_root / "assets" / "describe.task" / "readme.md").read_text()

    match = re.search(r'echo leaked > "\$?\{?([A-Za-z_][A-Za-z0-9_]*)\}?/leak\.txt"', told)
    assert match, f"no leak command in what the agent is told:\n{told}"
    name = match.group(1)

    # The variable has to be *exported*, and this is the assertion that was
    # missing: resolved through the real mechanism, from the rendered spec.
    spec = _loaded_specs(package_root, "agent", outside=str(layout.outside))["describe"]
    environment = material.deploy(spec, _ZoneAt(str(tmp_path / "zone")))
    assert name in environment, (
        f"{name} is in the readme and exported by nothing; the command would "
        f'run as `echo leaked > "/leak.txt"` against a root-owned /'
    )
    target = Path(environment[name])

    assert target.is_absolute(), f"{target} is relative and would resolve inside the zone"
    assert target == layout.outside
    assert target.is_dir(), f"{target} does not exist; the refusal would be a typo"
    assert layout.zones not in target.parents
    assert not str(target).startswith(str(layout.zones))

    # `agent` spec §5.3 and the module-3 measurement: the SDK's
    # `Bash{'command': ...}` hook has no path in the payload to match on and
    # returns ALLOW. The hook is the attributable layer, the OS is the
    # boundary — so a shell redirection and not a `Write` tool call, because
    # only the first tests the boundary.
    assert "echo leaked >" in told
    # And criterion 8's second clause is that the agent is *told why*, which
    # the OS does and the demo only has to not swallow.
    assert "verbatim" in told


def test_the_agent_is_told_where_to_write_and_the_name_resolves(package_root: Path) -> None:
    """**The first model call in this repository produced nothing, and this is why.**

    The model ran — 37 assistant turns, real session — did the work, and wrote
    no handoff. `assets/describe.task/readme.md` was 37 lines that named
    `AGENT_SYS_DEMO_OUTSIDE` twice and **contained no output path at all**. The
    agent was told where it may *not* write and never told where it must.

    A program body cannot have this bug: `assets/produce.task/collect.py` calls
    `_required("AGENT_SYS_OUTPUT_FACTS")` and exits loudly when it is unset. An
    AI body's equivalent of that call is a sentence in its readme, and a missing
    sentence raises nothing — so the failure surfaces as `output_absent` after a
    paid model call, which is the most expensive place in the system to learn it.

    Two halves that must meet, as with the leak target: the name is read out of
    **the readme the agent is given**, and resolved through **the function that
    exports it**. A readme naming a variable nobody exports is the shipped bug
    one kind over; a mechanism nobody references is a fact the agent never sees.

    The shape is asserted too, because `handoff` rejects the version otherwise:
    `summary` is `content_type: text`, so `content.CONTENT_TYPES` requires an
    item named `content` and a `Purpose` section, and the kind adds `Grounding`.
    Read off `handoff`'s own table rather than transcribed — a table edit there
    is meant to reach this readme, and this is what makes it.
    """
    import re

    from handoff.content import CONTENT_TYPES

    told = (package_root / "assets" / "describe.task" / "readme.md").read_text()

    match = re.search(r"\$\{?(AGENT_SYS_OUTPUT_[A-Z0-9_]+)\}?", told)
    assert match, f"the readme never names an output path:\n{told}"

    from env_mgr.grants import OUTPUT_ENV_PREFIX, _env_name

    kind = _loaded_specs(package_root, "closure")["describe"]["task"]["outputs"][0]
    expected = OUTPUT_ENV_PREFIX + _env_name(kind)
    assert match.group(1) == expected, (
        f"the readme sends the agent to {match.group(1)}, but this task's "
        f"declared output is {kind!r}, which `output_env` exports as {expected}"
    )

    # And the shape `handoff` will admit, from `handoff`'s table.
    text = CONTENT_TYPES["text"]
    for item in text.required_items:
        assert f"items/{item}" in told, f"the readme does not ask for items/{item}"
    sections = [*text.readme_sections, *_summary_kind(package_root)["readme_sections"]]
    for section in sections:
        assert f"## {section}" in told, f"the readme does not ask for a {section} section"


def test_the_readme_asks_for_every_item_the_summary_kind_requires(package_root: Path) -> None:
    """**The pass-through, and the half of it that lives in this package.**

    User ruling on `check_grounded`: *"this is a task-declaration problem, the
    system does not handle it — if it is needed, the task author passes their
    own input through to their own output when they define the task."* So
    `summary` carries `grounding`, a verbatim copy of the `facts` the producer
    consumed, and `check_grounded` judges a summary from the summary alone.

    **Two halves that must meet, and this is the one that would rot.** The kind
    declares the item; the readme is the only thing that makes an AI body write
    it. A required item nobody is asked for is `output_absent` after a paid
    model call — the same failure as the missing output path, one field over,
    and the reason to pin it the day the field is added rather than after.

    Required items are read off the **rendered kind**, so adding one to
    the `summary` kind in `steps/describe.yaml` and forgetting the readme fails here.
    """
    told = (package_root / "assets" / "describe.task" / "readme.md").read_text()
    kind = _summary_kind(package_root)

    for item in kind["items_schema"]["required"]:
        assert f"items/{item}" in told, (
            f"the summary kind requires items/{item} and the readme never asks "
            f"for it; an AI body writes what its readme says and nothing else"
        )

    # And the copy is verbatim rather than curated. A producer that selects what
    # it will be judged against is `validator` spec §8's subject, and this is the
    # validator whose whole question is whether the producer invented a number.
    assert "verbatim" in told
    assert "$AGENT_SYS_INPUT_FACTS" in told, (
        "the readme does not name the input to copy through; the facts reach "
        "the validator only inside the summary"
    )


def _summary_kind(package_root: Path) -> dict:
    """The `summary` handoff spec, as the loader admits it."""
    kinds = _loaded_specs(package_root, "handoff")
    assert "summary" in kinds, sorted(kinds)
    return kinds["summary"]


def test_an_unfilled_leak_target_is_visible_and_not_the_filesystem_root(
    package_root: Path,
) -> None:
    """**The unfilled default must not resolve to a plausible path.**

    `show` and a bare `render` exec nothing and pass no fill, so `demo.outside`
    takes its default. If that default were `''` the exported value would be
    empty and the command would run as `echo leaked > "/leak.txt"` — which is
    exactly what shipped, and what makes a root-owned `/` look like a sandbox.
    """
    spec = _loaded_specs(package_root, "agent")["describe"]
    unfilled = spec["env"]["AGENT_SYS_DEMO_OUTSIDE"]
    assert unfilled == "<outside: not filled>", (
        f"the unfilled leak target is {unfilled!r}; it has to be visibly "
        f"unfilled, because an empty one makes the command `/leak.txt`"
    )


def test_the_outside_directory_is_outside_every_zone(tmp_path: Path) -> None:
    """The write has to have somewhere real to fail against.

    Granted **read-execute** rather than not at all, deliberately: an ungranted
    path fails at `ls` too, and then the demonstration is indistinguishable from
    a typo in a path. Read-but-not-write is the shape that shows a boundary.
    """
    layout = layout_for(tmp_path).create()
    granted = {g.path: g.mode for g in demo_grants(layout)}
    assert str(layout.outside) in granted
    assert layout.zones not in layout.outside.parents
    assert not str(layout.outside).startswith(str(layout.zones))
    from env_mgr.isolation.policy import Mode

    assert granted[str(layout.outside)] is Mode.READ_EXEC
    # **No run-level writable `content/`.** Since §4.14 a task writes into the
    # pre-allocated `<store>/<hid>/v<N>/content/`, granted per attempt by
    # `grants.resolve_all` — so a shared writable directory outside the zone is
    # the pre-§4.14 shape. It was empty on every run before it was removed.
    assert not [p for p in granted if p.endswith("/content")]


def test_the_package_is_staged_and_not_granted(tmp_path: Path, package_root: Path) -> None:
    """**`interfaces.md` §4.16 reversed F19, and this test reversed with it.**

    A task body runs a program *from* the package — `assets/produce.task/entry.sh`
    execs `<package>/assets/produce.task/collect.py` — so the package has to be reachable. This
    package's first answer was a read-execute grant, and `agent`'s argument for
    it was that staging buys within-attempt immutability only for a
    self-contained body: stage `entry.sh`, re-exec it, and it still launches a
    mutable `assets/produce.task/collect.py`. **Half-immutable is not immutable.**

    §4.16 went the other way, and the reason is better than the one it replaced:
    `layout.stage_package` copies the package into `<zone>/package/`, so nothing
    outside the zone has to be reachable **at all** — the grant is not narrowed,
    it is removed. `prepare` exports the copy as `AGENT_SYS_TASK_PACKAGE`.

    So the assertion inverts: **no grant names the package**, and the package
    travels on the `Context` instead. Kept as a test rather than deleted because
    a grant reappearing here would silently re-open the hole staging closed.
    """
    from env_mgr.prepare import PACKAGE_ENV_VAR

    layout = layout_for(tmp_path).create()
    paths = {g.path for g in demo_grants(layout)}
    assert str(package_root.resolve()) not in paths
    assert not [p for p in paths if package_root.name in p], paths

    context = build_context(layout, main_repo=tmp_path, package=package_root)
    assert context.package == str(package_root)
    # `None` deliberately: an allow-list of the demo's own directories would be a
    # deny-list in allow-list clothing, and a validator directory added later
    # would silently not be excluded. §4.16 accepts that staging *moves*
    # criterion 13's leak; `TODO.md` 4a closes it.
    assert context.package_stage is None
    assert PACKAGE_ENV_VAR == "AGENT_SYS_TASK_PACKAGE"


def test_every_body_prefers_the_staged_copy(package_root: Path) -> None:
    """Both consumer classes, and they do not get the same environment.

    A **task** body is run by `ProgramExecutor` with `Prepared.environment`, so
    it sees `AGENT_SYS_TASK_PACKAGE`. A **validator** body is run by
    `ScriptBodyRunner` with `validation_env`, and `prepare_validation` stages the
    handoffs it validates and **not** the package — so it sees only the
    fallback. Every script therefore prefers the staged copy and falls back,
    which is one expression that is correct for both rather than two that are
    each correct for one.
    """
    scripts = sorted(package_root.rglob("entry.sh"))
    assert len(scripts) == 4
    for script in scripts:
        text = script.read_text()
        assert "${AGENT_SYS_TASK_PACKAGE:-" in text, script
        # And it still refuses loudly when neither is set, rather than execing
        # a path that begins with `/`.
        assert ":?" in text, script

    # The program the staged body runs reads the same variable, in the same
    # order — a body that preferred the copy and then handed a checkout path to
    # its program would stage nothing in practice.
    collect = (package_root / "assets" / "produce.task" / "collect.py").read_text()
    assert 'os.environ.get("AGENT_SYS_TASK_PACKAGE") or _required(' in collect


# Criterion 9 — no sandbox means the demo refuses to start, and says so


def test_no_confinement_refuses_with_exit_2(monkeypatch: Any, tmp_path: Path) -> None:
    """*On a machine with no sandbox available, the demo refuses to start.*

    The `Availability` is **injected**, and that is `env_mgr`'s own reason
    rather than convenience: **no machine runs all three branches.** `bwrap` is
    absent here so rung 1 cannot be exercised, and there is no ordinary way to
    make a Landlock-capable kernel look incapable, so rung 3 cannot be exercised
    wherever rung 2 works. With the input injected it is one line.

    Exit **2, not 1 and not 4**: spec §3.2 is explicit that the refusal *"is
    itself correct behaviour, not a demo failure"*, and a reviewer needs to see
    that without reading prose.
    """
    with pytest.raises(NoConfinement):
        confinement(Availability(bwrap=None, landlock_abi=None))

    monkeypatch.setattr(cli_main, "confinement", lambda *a, **k: _refuse())
    code, stream = _cli(monkeypatch, "run", "--demo-root", str(tmp_path))
    assert code == cli_main.PRECONDITION == 2

    said = stream.of_kind(EventKind.RUN_COMPLETE)[-1]
    assert "no confinement mechanism" in said.message
    # It says *why the refusal is right*, not only that it happened.
    assert "correct behaviour" in said.message
    assert said.fields["exit_code"] == 2


def _refuse() -> None:
    raise unconfinable()


def test_the_refusal_message_names_the_stakes() -> None:
    message = str(unconfinable())
    assert "bwrap is absent" in message
    assert "Landlock is unavailable" in message
    # The sentence that makes it a refusal rather than a limitation.
    assert "operator's full privileges" in message


@pytest.mark.parametrize(
    ("availability", "expected"),
    [
        (Availability(bwrap="/usr/bin/bwrap", landlock_abi=3), "bwrap"),
        (Availability(bwrap="/usr/bin/bwrap", landlock_abi=None), "bwrap"),
        (Availability(bwrap=None, landlock_abi=3), "landlock"),
    ],
)
def test_the_chain_orders_bubblewrap_first(availability: Availability, expected: str) -> None:
    """Read through `env_mgr.select` rather than reimplemented, so the demo
    cannot drift from the chain it reports."""
    assert select(availability) == confinement(availability) == expected


def test_this_machine_has_one_and_the_demo_can_say_which() -> None:
    """Not a skip. `env_mgr`'s suite fails rather than skips when no mechanism is
    available, and the demo's precondition is the same claim one layer up.

    `bwrap` is absent on the development machine and Landlock ABI is 3, so this
    is the Landlock rung — and `docs/interfaces.md` §5.11 records that the other
    rung plus a `kind: ai` task is currently **unconfinable at all**, because
    `ClaudeSDKClient` spawns the `claude` CLI itself and no caller ever sees that
    argv. `agent` refuses such a task and the demo would exit 2 with it.
    """
    assert confinement() in {"bwrap", "landlock"}


# --------------------------------------------------------------------------- #
# The layout duplication `examples/demo/assets/lib/store.py` carries


def test_the_store_layout_this_package_reads_is_handoffs(tmp_path: Path) -> None:
    """`docs/interfaces.md` §8.1's price, paid — **and this test failed to pay it
    once, which is why it is written the way it now is.**

    A validator body runs as a subprocess with `inputs.json` — handoff **ids** —
    and, for a handoff it was not handed, no route to the content. So
    `examples/demo/assets/lib/store.py` reads `<root>/<hid>/v<N>/` directly: a second
    reader of a fact `handoff` owns, admissible only with a test that fails the
    day the two disagree.

    **The first version of this test did not fail on that day.** It built its
    fixture by writing `real / "manifest.json"` — *the demo module's own
    constant* — and then asserted the demo module read it. It compared a thing
    to itself. `handoff` writes `manifest.yaml`, so `kind_of` returned `""` and
    `latest_of_kind("facts")` returned `None` for a genuinely published handoff,
    with this test green. Found by `handoff` running the real store, not here.

    That is precisely the failure this package spent two days reporting in other
    people's code — *a duplicated fact needs an agreement test; a duplicated
    decision needs one that **fires***. So the fixture is now built with
    **`handoff`'s** names and the constants are compared directly: nothing below
    can pass by agreeing with itself.
    """
    import importlib.util
    import uuid

    from handoff.store import CONTENT_DIR, MANIFEST_FILE, version_dir

    module_path = cli_main.package.locate() / "assets" / "lib" / "store.py"
    spec = importlib.util.spec_from_file_location("demo_example_store", module_path)
    assert spec and spec.loader
    store = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(store)

    # 1. The constants themselves, against `handoff`'s. The cheapest half, and
    #    the half whose absence cost a live defect.
    assert store.MANIFEST == MANIFEST_FILE
    assert store.CONTENT == CONTENT_DIR

    # 2. The layout, with the fixture built from `handoff`'s names only.
    hid = uuid.uuid4()
    published = Path(version_dir(tmp_path, hid, 7))
    (published / CONTENT_DIR).mkdir(parents=True)
    (published / MANIFEST_FILE).write_text(yaml.safe_dump({"kind": "facts"}))

    os.environ["AGENT_SYS_DEMO_STORE"] = str(tmp_path)
    try:
        assert store.versions(str(hid)) == [7]
        assert store.content_dir(str(hid)) == published / CONTENT_DIR
        assert store.kind_of(str(hid)) == "facts"
        assert store.latest_of_kind("facts") == published / CONTENT_DIR
        assert store.latest_of_kind("summary") is None

        # 3. **A version directory with no manifest is not published**, which is
        #    `handoff.list_versions`' rule since `interfaces.md` §4.14: a version
        #    is allocated at dispatch, so an unpublished `v8` can sit on top of a
        #    good `v7` while an attempt is still running or has failed. Without
        #    the filter, `content_dir` returned `None` for a handoff whose
        #    content was right there — and the newest attempt failing is the
        #    common case, not the edge.
        Path(version_dir(tmp_path, hid, 8)).mkdir(parents=True)
        assert store.versions(str(hid)) == [7]
        assert store.content_dir(str(hid)) == published / CONTENT_DIR
    finally:
        os.environ.pop("AGENT_SYS_DEMO_STORE", None)


def _cli(monkeypatch: Any, *argv: str) -> tuple[int, Stream]:
    captured: list[Stream] = []
    real = cli_main.Stream

    class _Capturing(real):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            captured.append(self)

    monkeypatch.setattr(cli_main, "Stream", _Capturing)
    monkeypatch.setattr(cli_main, "HumanRenderer", lambda out=None: _Silent())
    return cli_main.main(list(argv)), captured[0]


class _Silent:
    def on_event(self, event: Any) -> None:
        return None


# --------------------------------------------------------------------------- #
# The third precondition: the repository each zone is cloned from


def test_an_unprepared_repository_refuses_and_names_the_command(tmp_path: Path) -> None:
    """`extensions.preciousObjects` is required and the demo **refuses** rather
    than setting it.

    `env_mgr.workspace.cut` refuses without it, so every output-producing
    dispatch would die in `prepare` — and `ensure_precious` had **no production
    caller**, so the demo had never cut a workspace in this repository at all.
    Found by `demo-2` driving the run; `_main_repo`'s docstring claimed the demo
    *"says what it is doing rather than doing it silently"* while doing neither.

    **A refusal is design O1 answered the narrow way.** O1 asks whether the demo
    may set it silently, prompt, or refuse without a flag; refusing and naming
    the exact command takes no decision on the reviewer's behalf, which is the
    only one of the three safe to pick alone.
    """
    import subprocess

    from env_mgr.workspace import PRECIOUS, is_precious

    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603, S607

    with pytest.raises(RepositoryNotPrepared) as raised:
        preflight_repository(repo)
    message = str(raised.value)
    assert f"config {PRECIOUS} true" in message  # the command, runnable as printed
    assert "--unset" in message  # and how to undo it
    assert "--allow-repo-config" in message
    # The half a reviewer would not think to ask for: in a worktree the key lands
    # in the shared common config, so it is not local to their checkout.
    assert "SHARED common config" in message

    # Nothing was changed by being asked.
    assert not is_precious(str(repo))

    # And the opt-in does exactly one thing.
    preflight_repository(repo, allow_config=True)
    assert is_precious(str(repo))


def test_the_precondition_exits_two_like_the_other_two(monkeypatch: Any, tmp_path: Path) -> None:
    """§8.3: 2 is *the precondition failed, not the run*, and this is the third
    of three. A reviewer should not have to tell it apart from a load error."""
    monkeypatch.setattr(cli_main, "confinement", lambda *a, **k: "landlock")
    monkeypatch.setattr(cli_main, "preflight_credentials", lambda: "ready")
    monkeypatch.setattr(
        cli_main,
        "preflight_repository",
        lambda *a, **k: (_ for _ in ()).throw(RepositoryNotPrepared("not prepared")),
    )
    assert _cli(monkeypatch, "run", "--demo-root", str(tmp_path))[0] == cli_main.PRECONDITION == 2


def test_the_two_declared_names_point_at_the_same_level() -> None:
    """`AGENT_SYS_INPUT_<KIND>` and `AGENT_SYS_OUTPUT_<KIND>` are spelled
    differently and hand a body **the artefact's own files** at both ends.

    This asserted the opposite until `env-mgr-2` ran it. The shapes —
    `<zone>/handoffs/<hid>/v<N>` against `<store>/<hid>/v<N>/content` — do read
    as a pair one directory apart, and were, until `stage` narrowed to copy
    `<v>/content` **to** `<into>/<hid>/v<N>`. `render.py` had the `content` hop
    that reading produced and would have failed on it; `consume` never running
    is why nothing caught it, and this test being the stale claim is why nothing
    caught it here either.

    Kept rather than deleted because the strings still differ and a reader still
    compares strings. `tests/env_mgr` pins the property against a real staged
    tree (`test_the_two_declared_names_point_at_the_same_level`, `86a6818`);
    this pins the half `demo`'s bodies depend on — that neither body hops.
    """
    from env_mgr.grants import INPUT_ENV_PREFIX, OUTPUT_ENV_PREFIX, _env_name

    assert INPUT_ENV_PREFIX == "AGENT_SYS_INPUT_"
    assert OUTPUT_ENV_PREFIX == "AGENT_SYS_OUTPUT_"
    assert _env_name("summary") == "SUMMARY"

    # Read off `stage` rather than remembered: it copies the version's
    # `content/` and lands it *at* `<into>/<hid>/v<N>`.
    import inspect

    from env_mgr.fs import layout

    source = inspect.getsource(layout.stage)
    assert "CONTENT_DIR" in source.split('"""')[2]
    assert "content" not in source.split("dst = ")[1].split("\n")[0]

    # And neither body hops. `collect` writes into its output name, `render`
    # reads out of its input name, and both address the artefact directly.
    package = cli_main.package.locate()
    render = (package / "assets" / "consume.task" / "render.py").read_text()
    assert 'src / "items" / "content"' in render
    assert '"content" / "items"' not in render
    collect = (package / "assets" / "produce.task" / "collect.py").read_text()
    assert 'dst / "items"' in collect and 'dst / "content"' not in collect
