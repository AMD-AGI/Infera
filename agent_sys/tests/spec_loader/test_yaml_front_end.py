"""W3 — scan, parse, discriminate, expand, substitute, order-check.

`docs/ui-stage.md` §4 gives the pipeline as seven numbered steps. This file is
one section per step, in that order, so that a step with no test is visible as a
gap rather than assumed to be covered somewhere.

The parser's own behaviour — positions, YAML 1.2, duplicate keys — is measured in
`scratch/ui-yaml-2026-08/w3/probe_ruamel_semantics.py` and asserted here only
where the front end *depends* on it, which is the difference between testing our
code and testing `ruamel.yaml`'s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_loader import MODULE_KEY, load_package, report

from .conftest import HANDOFF, FakeRegistries, PackageBuilder

# --------------------------------------------------------------------------- #
# Step 2 — parse, with positions


def test_a_syntax_error_names_the_file_and_the_line(builder: PackageBuilder) -> None:
    """The whole reason main spec §7 adopted `ruamel.yaml` over PyYAML.

    `MarkedYAMLError.problem_mark` is 0-based and the front end adds one, so the
    number in the message is the number in the author's editor. Asserted rather
    than trusted: an off-by-one here is invisible in a passing test that only
    checks for "a line".
    """
    builder.write("broken.yaml", "module: handoff\nname: x\n  bad: indent\n")

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "parse"]
    assert problem.line == 3, "0-based mark not converted"
    assert problem.origin.endswith("broken.yaml")
    assert ":3" in report([problem])


def test_a_duplicate_key_is_rejected_with_a_position(builder: PackageBuilder) -> None:
    """The trap that went live when jsonnet's static rejection was deleted.

    Main spec §7 rev. 10 records that `validate.py`'s safety argument — *"jsonnet
    quotes every string and rejects a duplicate field statically"* — lost its
    premise, and hands the question to this package. The answer is that the
    parser closes it: PyYAML's `safe_load` silently keeps the last value, and
    `ruamel.yaml` raises with a mark.
    """
    builder.write("dup.yaml", "module: handoff\nname: first\nname: second\n")

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "parse"]
    assert "duplicate" in problem.message.lower()
    assert problem.line is not None


def test_yaml_one_point_two_leaves_no_and_yes_alone(builder: PackageBuilder) -> None:
    """The other half of §7's question, and the front end depends on the answer.

    Under YAML 1.1 — which is what PyYAML's `safe_load` implements — `NO` is
    `False`. A package author writing a country code, a `cost: on`, or a version
    `1.10` would get a bool or a rounded float with nothing to say so. The
    round-trip loader is 1.2 and they stay strings.

    This is asserted here and not left to the probe because it is a *property of
    the format users write*, and the day someone swaps the parser back for
    performance is the day it has to fail.
    """
    builder.asset("NO.md", "x")
    builder.write(
        "kinds.yaml",
        "module: handoff\nname: NO\ndescription: on\ncontent_type: text\nscope: fixed.required\n",
    )

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    assert result.problems == (), result.problems
    assert registries.handoff_specs.get("NO")["description"] == "on"


# --------------------------------------------------------------------------- #
# Step 3 — discriminate on `module:`


@pytest.mark.parametrize(
    ("module", "kind"),
    [("handoff", "handoff"), ("validator", "validator"), ("agent", "agent"), ("task", "closure")],
)
def test_the_module_key_decides_the_kind(builder: PackageBuilder, module: str, kind: str) -> None:
    """Four words in, five kinds out.

    **`task` produces a `closure`**, and that is the ruling rather than a
    mismatch: users write `task` and never `closure` (`closure` spec §2), and
    what comes out is the closure document with the task spec nested in it.
    """
    builder.write("anywhere.yaml", f"{MODULE_KEY}: {module}\nname: x\n")

    (document,) = [d for d in builder.package().documents().documents if d.doc.get("name") == "x"]

    assert document.kind == kind


def test_the_module_key_does_not_reach_the_schema(builder: PackageBuilder) -> None:
    """It is the discriminator, not a field.

    Every schema sets `additionalProperties: false`, so leaving `module:` in the
    document would make every object in the system fail its own schema. Stated as
    a test because the failure would be uniform and therefore easy to misread as
    a schema bug.
    """
    builder.asset("trace.md", "x")
    builder.write("kinds.yaml", f"module: handoff\nname: trace\n{HANDOFF}")

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    assert result.problems == (), result.problems
    assert MODULE_KEY not in registries.handoff_specs.get("trace")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("name: x\ndescription: d\n", "no module: key"),
        ("module: closure\nname: x\n", "not a module"),
        ("module: Validator\nname: x\n", "not a module"),
    ],
)
def test_an_unknown_or_absent_module_names_the_file_and_the_line(
    builder: PackageBuilder, source: str, expected: str
) -> None:
    """`module: closure` is in the list on purpose. It is the one wrong answer a
    reader of the *schemas* would give, because `closure` is a schema kind — and
    it is not a word a package author may type."""
    builder.write("thing.yaml", source)

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "module"]
    assert expected in problem.message
    assert problem.origin.endswith("thing.yaml")
    assert problem.line == 1


def test_a_list_entry_that_is_not_an_object_says_which_one(builder: PackageBuilder) -> None:
    builder.write("kinds.yaml", "- module: agent\n  name: a\n  kind: program\n- just a string\n")

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "shape"]
    assert problem.path == "$[1]"
    assert "str" in problem.message


# --------------------------------------------------------------------------- #
# Step 4 — inline definitions


def test_an_inline_definition_is_registered_and_replaced_by_its_name(
    builder: PackageBuilder,
) -> None:
    """`refine.task_package.define.md` §1.1.3 — an object written where it is used.

    Two things are checked and the second is the one that matters: the inline
    object is admitted under its own name, **and** the host now holds a string
    where it held a mapping. A hoist that registered without replacing would
    leave a document the schema rejects.
    """
    builder.asset("collect_trace.md", "x")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: run the collector
handoffs: []
validators: []
agent:
  module: agent
  name: tracer
  kind: program
  description: runs the collector
task:
  goal: collect a kernel trace
  version: "1"
  inputs: []
  outputs: []
""",
    )

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    assert result.problems == (), result.problems
    assert registries.agent_specs.get("tracer")["kind"] == "program"
    assert registries.closures.get("collect_trace")["agent"] == "tracer"


def test_an_inline_definition_is_emitted_before_its_host(builder: PackageBuilder) -> None:
    """Post-order, and it is what makes "defined before use" satisfiable.

    An inline definition is written *inside* the object that references it, so
    the author has no way to put it earlier. Emitting the child first is the only
    ordering under which the forward-reference rule does not contradict the
    inline-definition rule — two of the user's five requirements, which would
    otherwise disagree.
    """
    builder.asset("collect_trace.md", "x")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: run the collector
handoffs: []
validators: []
agent: {module: agent, name: tracer, kind: program, description: d}
task: {goal: g, version: "1", inputs: [], outputs: []}
""",
    )

    names = [d.doc.get("name") for d in builder.package().documents().documents]

    assert names.index("tracer") < names.index("collect_trace")


def test_an_inline_definition_carries_a_pointer_to_where_it_was_written(
    builder: PackageBuilder,
) -> None:
    """It has no file of its own, so its `origin` is the host's path plus a JSON
    pointer to the key it was written under."""
    builder.asset("collect_trace.md", "x")
    builder.write(
        "steps.yaml",
        """- module: task
  name: collect_trace
  description: d
  handoffs: []
  validators: []
  agent: {module: agent, name: tracer, kind: program, description: d}
  task: {goal: g, version: "1", inputs: [], outputs: []}
""",
    )

    inline = next(
        d for d in builder.package().documents().documents if d.doc.get("name") == "tracer"
    )

    assert inline.origin.endswith("steps.yaml#/0/agent")


def test_an_inline_object_with_no_name_is_left_for_the_schema(builder: PackageBuilder) -> None:
    """Deliberately not hoisted under a name this code invented.

    An object with nothing to register under reaches the enforcement point as a
    mapping where a string belongs, and is rejected there with the message that
    field already has. Making one up would put a spec in a registry under a name
    no author wrote.
    """
    builder.asset("collect_trace.md", "x")
    builder.write(
        "steps.yaml",
        """module: task
name: collect_trace
description: d
handoffs: []
validators: []
agent: {module: agent, kind: program, description: d}
task: {goal: g, version: "1", inputs: [], outputs: []}
""",
    )

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    assert registries.agent_specs.names() == []
    assert any(p.path == "$.agent" for p in result.problems), result.problems


# --------------------------------------------------------------------------- #
# Step 5 — variables


def test_a_variable_is_substituted_anywhere_in_the_tree(builder: PackageBuilder) -> None:
    builder.asset("trace.md", "x")
    builder.write(
        "kinds.yaml",
        "module: handoff\nname: trace\ndescription: from ${WHERE}\n"
        "content_type: text\nscope: fixed.required\n",
    )

    registries = FakeRegistries()
    result = load_package(builder.package(variables={"WHERE": "the collector"}), registries)

    assert result.problems == (), result.problems
    assert registries.handoff_specs.get("trace")["description"] == "from the collector"


def test_a_default_is_used_when_the_variable_is_absent(builder: PackageBuilder) -> None:
    """`${NAME:-default}` — the third of the three measured needs.

    Its jsonnet ancestor is the one construct every general spec used:
    `if std.objectHas(config, 'inputs') then config.inputs else ['any']`.
    """
    builder.asset("trace.md", "x")
    builder.write(
        "kinds.yaml",
        "module: handoff\nname: trace\ndescription: ${WHERE:-nowhere in particular}\n"
        "content_type: text\nscope: fixed.required\n",
    )

    registries = FakeRegistries()
    load_package(builder.package(), registries)

    assert registries.handoff_specs.get("trace")["description"] == "nowhere in particular"


def test_an_unsupplied_variable_with_no_default_is_a_fault(builder: PackageBuilder) -> None:
    """Not left literal, and the reason is on record in this tree.

    `${NOPE}/readme.md` passed through unchanged is a path that resolves to
    nothing later, in another module, with nothing to say why. `demo`'s own
    history has that bug: an unfilled value concatenated to `'' + "/leak.txt"`
    and produced a plausible absolute path that demonstrated nothing.
    """
    builder.write(
        "kinds.yaml",
        "module: handoff\nname: trace\ndescription: ${NOPE}\n"
        "content_type: text\nscope: fixed.required\n",
    )

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "variable"]
    assert "${NOPE}" in problem.message
    assert problem.path == "$.description"
    assert problem.line == 3


def test_a_value_that_would_break_the_document_does_not(builder: PackageBuilder) -> None:
    """Post-parse substitution, and this is the property it buys.

    Measured pre-parse over six ordinary values, four broke the document and one
    of the four was silent — `run #3` became `run`, because `#` starts a comment.
    Substituting into the parsed tree cannot do either: the value is already one
    scalar and stays one.
    """
    builder.asset("trace.md", "x")
    builder.write(
        "kinds.yaml",
        "module: handoff\nname: trace\ndescription: ${D}\n"
        "content_type: text\nscope: fixed.required\n",
    )

    registries = FakeRegistries()
    hostile = "a colon: a dash - a hash # and\na newline"
    result = load_package(builder.package(variables={"D": hostile}), registries)

    assert result.problems == (), result.problems
    assert registries.handoff_specs.get("trace")["description"] == hostile


def test_the_assets_token_is_supplied_and_is_package_relative(builder: PackageBuilder) -> None:
    """`${TASK_PACKAGE_ASSERT_DIR}` — the user's spelling, kept.

    **Package-relative and not absolute**, and that is a reversal already paid
    for once: `demo/README.md` F-D18 records body paths being made absolute and
    then becoming unresolvable when `interfaces.md` §4.16 moved the tree into the
    zone, because `Path(staged) / "/abs"` is `/abs`.
    """
    builder.asset("logic/x.md", "x")
    builder.write(
        "vals.yaml",
        """module: validator
name: shape
brief: b
dimension: completeness
strength: strong
inputs: [any]
body: {readme: "${TASK_PACKAGE_ASSERT_DIR}/logic/x.md"}
tags: {logic_source: external_static, cost: seconds}
""",
    )

    registries = FakeRegistries()
    load_package(builder.package(), registries)

    assert registries.validator_specs.get("shape")["body"]["readme"] == "assets/logic/x.md"


# --------------------------------------------------------------------------- #
# Step 6 — definition order


def test_a_forward_reference_within_a_file_is_an_error(builder: PackageBuilder) -> None:
    """`refine.task_package.define.md` §1.1.2 — *"不能前面引用后面的定义引用不到"*."""
    builder.asset("collect_trace.md", "x")
    builder.asset("trace.md", "x")
    builder.write(
        "steps.yaml",
        """- module: task
  name: collect_trace
  description: d
  handoffs: [trace]
  validators: []
  task: {goal: g, version: "1", inputs: [], outputs: [trace]}
  agent: {module: agent, name: tracer, kind: program, description: d}

- module: handoff
  name: trace
  description: a captured kernel trace
  content_type: text
  scope: fixed.required
""",
    )

    result = load_package(builder.package(), FakeRegistries())

    problems = [p for p in result.problems if p.keyword == "order"]
    assert problems, result.problems
    assert all("'trace'" in p.message for p in problems)
    assert {p.path for p in problems} == {"$.handoffs[0]", "$.task.outputs[0]"}


def test_the_same_reference_backwards_is_fine(builder: PackageBuilder) -> None:
    """The other order, and nothing else changed."""
    builder.asset("collect_trace.md", "x")
    builder.asset("trace.md", "x")
    builder.write(
        "steps.yaml",
        """- module: handoff
  name: trace
  description: a captured kernel trace
  content_type: text
  scope: fixed.required

- module: task
  name: collect_trace
  description: d
  handoffs: [trace]
  validators: []
  task: {goal: g, version: "1", inputs: [], outputs: [trace]}
  agent: {module: agent, name: tracer, kind: program, description: d}
""",
    )

    result = load_package(builder.package(), FakeRegistries())

    assert result.problems == (), result.problems


def test_a_name_defined_in_another_file_is_not_a_forward_reference(
    builder: PackageBuilder,
) -> None:
    """**The rule is within a file, and this is the measurement that says why.**

    `examples/demo` is the only real package in the tree and no total file order
    satisfies its own reference graph: sorted by path, `closures/produce`
    references `handoffs/facts` and `validators/check_facts`, both later;
    reversed, it references `agents/collect`, which is then last. Ordering by
    kind fails too, because a handoff names its validators and a validator names
    its input handoff kinds — a real 2-cycle
    (`examples/demo/handoffs/facts.jsonnet` and
    `validators/check_facts.jsonnet`).

    So a package-wide rule would be one no author could satisfy except by
    renaming files. Cross-file references are caught where they always were: the
    composition root's passes say a name does not resolve.
    """
    builder.asset("collect_trace.md", "x")
    builder.asset("trace.md", "x")
    builder.write(
        "a_first.yaml",
        """module: task
name: collect_trace
description: d
handoffs: [trace]
validators: []
task: {goal: g, version: "1", inputs: [], outputs: [trace]}
agent: {module: agent, name: tracer, kind: program, description: d}
""",
    )
    builder.write("z_later.yaml", f"module: handoff\nname: trace\n{HANDOFF}")

    result = load_package(builder.package(), FakeRegistries())

    assert [p for p in result.problems if p.keyword == "order"] == []


def test_a_validator_and_its_handoff_kind_can_share_a_file(builder: PackageBuilder) -> None:
    """The cycle, pinned so the rule cannot quietly grow back over `inputs`.

    A handoff names its validators and a validator names its input handoff
    kinds. **Both are references and only one may count**: with both in
    `_REFERENCE_KEYS`, neither order of these two objects is legal and the pair
    cannot share a file at all — which is a rule no package can satisfy, over a
    pair `examples/demo` already ships.

    So `inputs` is out, and what that buys is exactly one legal order. This
    asserts both halves, because asserting only the clean one would leave the
    rule free to grow back and still pass.
    """
    handoff = (
        "- module: handoff\n  name: trace\n  description: d\n"
        "  content_type: text\n  scope: fixed.required\n  validators: [check_shape]\n"
    )
    validator = (
        "- module: validator\n  name: check_shape\n  brief: b\n"
        "  dimension: completeness\n  strength: strong\n  inputs: [trace]\n"
        "  tags: {logic_source: external_static, cost: seconds}\n"
    )

    order_problems = {}
    for label, text in (
        ("validator first", validator + handoff),
        ("handoff first", handoff + validator),
    ):
        pkg = PackageBuilder(builder.root.parent / label.replace(" ", "_"))
        pkg.asset("trace.md", "x")
        pkg.write("pair.yaml", text)
        result = load_package(pkg.package(), FakeRegistries())
        order_problems[label] = [p for p in result.problems if p.keyword == "order"]

    assert order_problems["validator first"] == [], "no order would be legal"
    assert order_problems["handoff first"], (
        "`inputs` has been added back to the reference keys: if a validator's "
        "`inputs` also counted, this order would be the only legal one and the "
        "other would not be, so neither would."
    )


# --------------------------------------------------------------------------- #
# Step 7 — emit


def test_a_files_second_object_carries_a_pointer_and_the_first_of_a_lone_one_does_not(
    builder: PackageBuilder,
) -> None:
    """The two `origin` shapes, and why there are two.

    A file holding one object keeps the origin it had before rev. 10, so every
    message quoting it reads the same. A file holding a list gets `#/<index>`.
    There is no ambiguity between them: within one file the root is a mapping or
    a sequence, never both.
    """
    builder.write("one.yaml", "module: agent\nname: solo\nkind: program\n")
    builder.write(
        "many.yaml",
        "- module: agent\n  name: first\n  kind: program\n"
        "- module: agent\n  name: second\n  kind: program\n",
    )

    origins = {d.doc["name"]: d.origin for d in builder.package().documents().documents}

    assert origins["solo"].endswith("one.yaml")
    assert origins["first"].endswith("many.yaml#/0")
    assert origins["second"].endswith("many.yaml#/1")


def test_the_origin_a_registry_holds_is_the_one_a_problem_carries(
    builder: PackageBuilder,
) -> None:
    """`task_graph/bootstrap.py`'s `_names_for` bridges the two by **exact string
    equality**, and `SpecRegistry.origin_of`'s docstring records what a plausible
    but wrong origin cost the last time they drifted. With several objects per
    file there are now two places the string is built, so this pins that there is
    one.
    """
    builder.asset("trace.md", "x")
    builder.write(
        "many.yaml",
        "- module: handoff\n  name: trace\n"
        + "".join(f"  {line}\n" for line in HANDOFF.strip().splitlines())
        + "- module: agent\n  name: tracer\n  kind: program\n",
    )

    registries = FakeRegistries()
    load_package(builder.package(), registries)

    assert registries.handoff_specs.origin_of("trace").endswith("many.yaml#/0")
    assert registries.agent_specs.origin_of("tracer").endswith("many.yaml#/1")


def test_a_schema_fault_carries_the_document_s_position(builder: PackageBuilder) -> None:
    """In a file holding twenty objects, `$.scope` alone does not locate anything.

    The position is the **document's**, not the field's, and the limit is
    deliberate: joining a schema's `json_path` back onto a source position needs
    the parse tree, and `validate` cannot see one — that is the whole of what
    makes it path-free.
    """
    builder.write(
        "many.yaml",
        "- module: agent\n  name: a\n  kind: program\n"
        "- module: handoff\n  name: b\n  description: d\n"
        "  content_type: text\n  scope: not_a_scope\n",
    )

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.path == "$.scope"]
    assert problem.line == 4, "the second object starts on line 4"
    assert problem.origin.endswith("many.yaml#/1")


def test_a_binary_file_named_yaml_is_reported_not_raised(builder: PackageBuilder) -> None:
    """`UnicodeDecodeError` is not an `OSError`, and a scan that let it out would
    take the whole multi-package load down for one bad file."""
    (builder.root / "blob.yaml").write_bytes(b"\xff\xfe\x00binary")

    result = load_package(builder.package(), FakeRegistries())

    (problem,) = [p for p in result.problems if p.keyword == "unreadable"]
    assert problem.origin.endswith("blob.yaml")


def test_one_unreadable_file_does_not_hide_the_others(builder: PackageBuilder) -> None:
    """`load_package`'s first stated property, at the seam this time.

    The package reports its faults rather than raising, so a package with one
    broken file still delivers the rest — the alternative makes fixing a package
    an N-round trip.
    """
    builder.asset("trace.md", "x")
    builder.write("broken.yaml", "module: handoff\nname: x\n  bad: indent\n")
    builder.write("good.yaml", f"module: handoff\nname: trace\n{HANDOFF}")

    registries = FakeRegistries()
    result = load_package(builder.package(), registries)

    assert sorted(result.admitted) == ["main", "trace"]
    assert len([p for p in result.problems if p.keyword == "parse"]) == 1


def test_a_package_is_read_once_per_call_and_gives_the_same_answer_twice(
    builder: PackageBuilder,
) -> None:
    """Idempotence, which `docs/design.md` §9.2 asked of the deleted `render`.

    The parallel half of that test went with the thread pool — there is no
    ~23 ms VM construction to hide any more, and adding a pool to a `read_text`
    would be optimising something nobody has measured. What survives is the half
    that is a correctness property: two calls, one answer.
    """
    builder.asset("trace.md", "x")
    builder.write("kinds.yaml", f"module: handoff\nname: trace\n{HANDOFF}")
    pkg = builder.package()

    first = pkg.documents()
    second = pkg.documents()

    assert [(d.kind, d.origin, dict(d.doc)) for d in first.documents] == [
        (d.kind, d.origin, dict(d.doc)) for d in second.documents
    ]


def test_the_scan_does_not_leave_the_package(tmp_path: Path) -> None:
    """A sibling directory's YAML is not this package's.

    Kustomize's `LoadRestrictions` is the idea main spec §7 records as *"still
    wanted and not adopted"* — `spec_loader` design O3. This is the part of it
    that falls out for free from scanning under the root, and it is asserted so
    that a future absolute-path or symlink-following change has to break a test
    rather than a reader's assumption. **It does not close O3**: a symlink from
    inside the package still reaches out, which `test_package.py`'s
    cross-package test requires.
    """
    (tmp_path / "sibling").mkdir()
    (tmp_path / "sibling" / "other.yaml").write_text(
        "module: agent\nname: intruder\nkind: program\n"
    )
    builder = PackageBuilder(tmp_path / "pkg")

    names = [d.doc.get("name") for d in builder.package().documents().documents]

    assert "intruder" not in names
