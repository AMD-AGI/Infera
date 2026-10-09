"""W4 — `assets/` auto-discovery.

`refine.task_package.define.md` §2, and `docs/ui-stage.md` §4 W4. Four rules and
one prohibition:

1. a readme is `${name}` plus an optional `${type}` plus an optional literal
   `readme`, in any order, ending `.md`;
2. an entry is the same with `entry` and `.sh`;
3. a `${name}` / `${name}.${type}` / `${type}.${name}` **folder** groups one
   object's files;
4. the search is recursive and **a conflict crashes**;
5. an explicit binding is legal and **warns**.

The prohibition is `user_interface.ai.draft.md` §4.9's one survivor: derive the
*paths*, never the semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_loader import AssetIndex, SpecInconsistent, load_package

from .conftest import FakeRegistries, PackageBuilder

# --------------------------------------------------------------------------- #
# Rules 1 and 2 — the permutations


@pytest.mark.parametrize(
    "filename",
    [
        "collect.md",
        "collect.readme.md",
        "readme.collect.md",
        "collect.task.md",
        "task.collect.md",
        "collect.task.readme.md",
        "task.collect.readme.md",
        "readme.collect.task.md",
        "readme.task.collect.md",
        "collect.readme.task.md",
        "task.readme.collect.md",
    ],
)
def test_every_permutation_of_the_readme_tokens_is_found(tmp_path: Path, filename: str) -> None:
    """The user gave a list of examples ending in *"等等"* — "and so on".

    Generating the permutations rather than transcribing the list is the point:
    a transcription is a list somebody has to remember to extend, and the
    eleventh spelling is the one that gets forgotten. All eleven are here so that
    a change to the generator has to break something visible.
    """
    (tmp_path / filename).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / filename).write_text("x")

    got = AssetIndex(tmp_path).resolve("readme", name="collect", type_="task")

    assert got == Path("assets") / filename


@pytest.mark.parametrize(
    "filename", ["collect.sh", "collect.entry.sh", "entry.collect.sh", "collect.validator.entry.sh"]
)
def test_the_entry_permutations_follow_the_same_rule(tmp_path: Path, filename: str) -> None:
    """*"规则仿照以上"* — the rules follow the above, with `entry` and `.sh`."""
    (tmp_path / filename).write_text("x")

    got = AssetIndex(tmp_path).resolve("entry", name="collect", type_="validator")

    assert got == Path("assets") / filename


@pytest.mark.parametrize(
    "filename",
    [
        "readme.md",  # no name: it would be every object's readme at once
        "task.readme.md",  # type and the literal word, still no name
        "collect.txt",  # the extension is the one token never optional
        "collect.readme",  # likewise
        "collecting.md",  # a prefix is not a token
    ],
)
def test_what_is_not_a_match(tmp_path: Path, filename: str) -> None:
    """`name` is mandatory and the extension is mandatory. Everything else is
    optional, and "optional" is what makes the negative cases worth pinning: a
    matcher loose enough to accept `readme.md` at the root would give every
    object in the package the same body."""
    (tmp_path / filename).write_text("x")

    assert AssetIndex(tmp_path).resolve("readme", name="collect", type_="task") is None


def test_the_search_is_recursive(tmp_path: Path) -> None:
    """*"以上assets内递归检测"* — a package arranges `assets/` as it likes."""
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "collect.md").write_text("x")

    got = AssetIndex(tmp_path).resolve("readme", name="collect", type_="task")

    assert got == Path("assets/a/b/c/collect.md")


# --------------------------------------------------------------------------- #
# Rule 3 — folders scope a lookup


@pytest.mark.parametrize("folder", ["collect", "collect.task", "task.collect"])
def test_a_matching_folder_makes_the_name_optional_inside_it(tmp_path: Path, folder: str) -> None:
    """**The rule that had to be worked out, because no field takes a folder.**

    `task.body` and `validator.body` hold `readme`, `entry` and `materials`, and
    `materials`' own description says nothing reads it yet — so a folder could
    not be *bound* to anything. What it does instead is what `validator` spec
    §9.1 already means by "a validator is a folder": it groups one object's
    files, and inside it the name is implied.
    """
    (tmp_path / folder).mkdir()
    (tmp_path / folder / "readme.md").write_text("x")
    (tmp_path / folder / "entry.sh").write_text("x")

    index = AssetIndex(tmp_path)

    assert index.resolve("readme", name="collect", type_="task") == Path(
        f"assets/{folder}/readme.md"
    )
    assert index.resolve("entry", name="collect", type_="task") == Path(f"assets/{folder}/entry.sh")


def test_a_folder_that_matches_nothing_does_not_scope(tmp_path: Path) -> None:
    """A bare `readme.md` under an unrelated directory belongs to nobody."""
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "readme.md").write_text("x")

    assert AssetIndex(tmp_path).resolve("readme", name="collect", type_="task") is None


def test_a_folder_scopes_recursively(tmp_path: Path) -> None:
    """Nested under a matching folder, not only directly inside it.

    The user asked for recursion under `assets/` and said nothing about stopping
    at a folder boundary; stopping would make `check_facts.validator/logic/entry.sh`
    invisible for no reason a reader could reconstruct.
    """
    (tmp_path / "check_facts.validator" / "logic").mkdir(parents=True)
    (tmp_path / "check_facts.validator" / "logic" / "entry.sh").write_text("x")

    got = AssetIndex(tmp_path).resolve("entry", name="check_facts", type_="validator")

    assert got == Path("assets/check_facts.validator/logic/entry.sh")


# --------------------------------------------------------------------------- #
# Rule 4 — a conflict crashes


def test_two_matches_are_a_conflict_naming_both(tmp_path: Path) -> None:
    """*"不允许冲突，冲突直接崩溃"*.

    `SpecInconsistent` and not a new type: `registry.py` already answers *"two
    things claiming one name"* with it one layer up, and `docs/ui-stage.md` §4 W4
    says to match the existing collision policy rather than invent an error
    shape.
    """
    (tmp_path / "collect.md").write_text("x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "readme.collect.task.md").write_text("x")

    with pytest.raises(SpecInconsistent) as caught:
        AssetIndex(tmp_path).resolve("readme", name="collect", type_="task")

    message = str(caught.value)
    assert "collect.md" in message
    assert "readme.collect.task.md" in message


def test_a_conflict_becomes_a_problem_rather_than_taking_the_load_down(
    builder: PackageBuilder,
) -> None:
    """It crashes the *resolution*, not the run.

    "崩溃" is about not guessing, and the load-time equivalent of not guessing is
    a fatal `Problem` naming both files — which reaches the author through the
    same report as everything else instead of as a traceback that hides the other
    nine faults.
    """
    builder.asset("collect_trace.md", "x")
    builder.asset("deep/readme.collect_trace.task.md", "x")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: d
handoffs: []
validators: []
agent: {module: agent, name: tracer, kind: program, description: d}
task: {goal: g, version: "1", inputs: [], outputs: []}
""",
    )

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "inconsistent"]
    assert "collect_trace.md" in problem.message
    assert problem.fatal


def test_two_objects_of_different_kinds_may_share_a_name(builder: PackageBuilder) -> None:
    """`${type}` is what disambiguates them, and it must actually work.

    The registries are per kind, so `describe` may be both an agent and a task —
    `examples/demo` has exactly that pair. Without the type token their assets
    would collide and both would crash.
    """
    builder.asset("describe.task.md", "the task")
    builder.asset("describe.validator.md", "the check")

    index = AssetIndex(builder.root / "assets")

    assert index.resolve("readme", name="describe", type_="task") == Path("assets/describe.task.md")
    assert index.resolve("readme", name="describe", type_="validator") == Path(
        "assets/describe.validator.md"
    )


# --------------------------------------------------------------------------- #
# Rule 5 — an explicit binding warns


def test_an_explicit_binding_is_legal_and_warns(builder: PackageBuilder) -> None:
    """*"显式绑定在complie时会报warning"*.

    A **non-fatal** `Problem`, which is the mechanism `closure/check.py`'s check
    3 already uses for a report-severity finding: it reaches the same report and
    the same log, and `failed_names` correctly does not treat it as a failure. It
    is the second producer of `fatal=False` in the system and the first that is
    not about the handoff escape hatch.
    """
    builder.asset("collect_trace.md", "found by convention")
    builder.asset("by_hand.md", "bound by hand")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: d
handoffs: []
validators: []
agent: {module: agent, name: tracer, kind: program, description: d}
task:
  goal: g
  version: "1"
  inputs: []
  outputs: []
  body: {readme: assets/by_hand.md}
""",
    )

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    (warning,) = [p for p in result.problems if p.keyword == "explicit-binding"]
    assert not warning.fatal
    assert "collect_trace" in result.admitted
    assert registries.closures.get("collect_trace")["task"]["body"]["readme"] == "assets/by_hand.md"


def test_a_body_found_by_convention_does_not_warn(builder: PackageBuilder) -> None:
    """The best practice the demo is meant to show, and the silent case."""
    builder.asset("collect_trace.md", "x")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: d
handoffs: []
validators: []
agent: {module: agent, name: tracer, kind: program, description: d}
task: {goal: g, version: "1", inputs: [], outputs: []}
""",
    )

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    assert result.problems == (), result.problems
    task = registries.closures.get("collect_trace")["task"]
    assert task["body"] == {"readme": "assets/collect_trace.md"}


# --------------------------------------------------------------------------- #
# The prohibition — derive the paths, never the semantics


def test_an_entry_sh_fills_a_path_and_changes_nothing_else(builder: PackageBuilder) -> None:
    """`user_interface.ai.draft.md` §4.9's one survivor, from opa#6509.

    Finding an `entry.sh` fills `body.entry`. It does **not** reinterpret what the
    task is — it was always the presence of the file that said "programmatic",
    and the author has moved the declaration from a YAML key to a filename
    without changing who decides.

    The half that proves it: the existing named check that `entry` and a subgraph
    are mutually exclusive (`closure` spec §2.6) still runs over the filled
    document, so a non-leaf that acquires an `entry.sh` fails **loudly** rather
    than quietly becoming a leaf. That check is `closure`'s and runs at the
    composition root, so what is asserted here is the input it needs: the key is
    present and visible, not consumed.
    """
    builder.asset("collect_trace/readme.md", "x")
    builder.asset("collect_trace/entry.sh", "#!/bin/sh\n")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: d
handoffs: []
validators: []
agent: {module: agent, name: tracer, kind: program, description: d}
task: {goal: g, version: "1", inputs: [], outputs: []}
""",
    )

    registries = FakeRegistries()
    load_package(builder.package(), registries)

    task = registries.closures.get("collect_trace")["task"]
    assert task["body"]["entry"] == "assets/collect_trace/entry.sh"
    assert "kind" not in task, "nothing about the task's nature was rewritten"
    assert registries.agent_specs.get("tracer")["kind"] == "program"


def test_no_entry_sh_leaves_the_key_absent(builder: PackageBuilder) -> None:
    """An agent task: the same document, one fewer file, and no `entry`.

    `{}` versus a key holding `""` is the distinction `body_of`'s docstring was
    written about — an empty `entry` read as *no entry* is on record as the bug
    that shape causes.
    """
    builder.asset("collect_trace/readme.md", "x")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: d
handoffs: []
validators: []
agent: {module: agent, name: tracer, kind: program, description: d}
task: {goal: g, version: "1", inputs: [], outputs: []}
""",
    )

    registries = FakeRegistries()
    load_package(builder.package(), registries)

    assert registries.closures.get("collect_trace")["task"]["body"] == {
        "readme": "assets/collect_trace/readme.md"
    }


def test_an_agent_and_a_handoff_have_no_body_to_fill(builder: PackageBuilder) -> None:
    """**A gap, reported rather than filled.**

    The user's rule says a readme is found for *"某个命名的 agent/task/handoff/
    validator"*. Measured against the schemas, only `task.body` and
    `validator.body` exist: `agent` has `knowledge` / `rules` / `skills` and
    `handoff` has `readme_sections`, and none of those is a path. So for two of
    the four kinds there is nothing to fill, and inventing a field to fill would
    be `engineer_principle.md` §2's failure mode.

    Asserted so the gap is visible in the suite rather than only in a README.
    """
    builder.asset("tracer.md", "x")
    builder.asset("trace.md", "x")
    builder.write("a.yaml", "module: agent\nname: tracer\nkind: program\n")
    builder.write(
        "h.yaml",
        "module: handoff\nname: trace\ndescription: d\ncontent_type: text\nscope: fixed.required\n",
    )

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    assert result.problems == (), result.problems
    assert "body" not in registries.agent_specs.get("tracer")
    assert "body" not in registries.handoff_specs.get("trace")
