"""Criteria 15 and 16 — the shipped set, and the reference workflow.

Criterion 16 doubles as the check that the vocabulary is sufficient (main spec
criterion 7): every row of spec §10's six-step optimisation loop must be
expressible, with its handoffs, its validators, and each validator's dimension
and strength resolving.

**The workflow's specs are built here rather than shipped.** Main spec §4.3: a
concrete workflow is a task package, outside this repository, and *"a
`collect_trace` handoff kind living in this repository would make every change to
that workflow a change to the system."* So the fixture is the package, and what
is asserted is that the vocabulary admits it.

The three that *are* shipped are the workflow-independent ones —
`validator/general_specs/`, main spec §4.5's documents *"this repository happens
to ship"*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spec_loader import MODULE_KEY, schema_for, validate
from spec_loader.yaml_source import read_yaml
from validator.protocols import Dimension, Strength
from validator.registry import ValidatorSpecRegistry
from validator.spec import check_body_resolves

ROOT = Path(__file__).resolve().parents[2]
GENERAL = ROOT / "validator/general_specs"


def load(path: Path) -> dict[str, Any]:
    """Read one shipped general spec.

    **This was `render()` and it called `_jsonnet.evaluate_file` directly.** Its
    docstring said *"`spec_loader.render` is the real path and it is wave 0's;
    this calls the library directly so that criterion 15 asserts the shipped
    files rather than a Python transcription of them. Swap the two lines when
    `spec_loader` lands."* Both halves of that have now resolved, in opposite
    directions: `spec_loader.render` no longer exists at all (main spec §7
    rev. 10 deleted jsonnet), and the intent — assert the **shipped files**, not
    a transcription — is why this still reads from disk rather than inlining
    three dicts.

    `spec_loader.yaml_source.read_yaml` and **not** PyYAML, and not
    `ruamel.yaml` called here. There must be exactly one reading of a spec
    document: measured, `ruamel` round-trip is YAML 1.2 and PyYAML's `safe_load`
    is 1.1, so `12:30` is the string on one side and the integer 750 on the
    other. A second parser in a test is how a document comes to mean two things.

    `load_package` is deliberately not used. These are documents, not a task
    package: main spec §4.5 puts them in *"their own directory, separate from any
    task package"*, and they have neither of §4.3's two mandatory names because
    they are not one.
    """
    doc, problems = read_yaml(path, origin=str(path.relative_to(ROOT)))
    assert not problems, f"{path.name}: {problems}"
    doc.pop(MODULE_KEY)  # the discriminator, not a field — every schema forbids it
    return doc


# --------------------------------------------------------------------------- #
# Criterion 15 — the shipped set


@pytest.fixture
def shipped() -> ValidatorSpecRegistry:
    registry = ValidatorSpecRegistry()
    for source in sorted(GENERAL.rglob("*.yaml")):
        record = load(source)
        registry.add(record["name"], record, origin=str(source.relative_to(ROOT)))
    return registry


def test_every_dimension_is_represented(shipped: ValidatorSpecRegistry) -> None:
    """Criterion 15. *"Three `strong` validators on one handoff mean little if all
    three check the schema."* Declaring the dimension is what stops a registry
    filling up with completeness checks while nobody notices that nothing checks
    trustworthiness."""
    assert shipped.dimensions_present() == set(Dimension)


def test_list_by_dimension_answers_the_gap_question(shipped: ValidatorSpecRegistry) -> None:
    """Criterion 15's second half — so *"nothing checks trustworthiness on this
    kind"* is answerable."""
    assert shipped.list_by_dimension(Dimension.TRUSTWORTHINESS) == ["production_grade"]
    assert shipped.list_by_dimension(Dimension.COMPLETENESS) == ["schema_conformance"]
    assert shipped.list_by_dimension(Dimension.USABILITY) == ["downstream_loads"]


def test_the_shipped_specs_pass_the_real_schema() -> None:
    """The whole pipeline over the files actually in the tree: read, then the
    one enforcement point.

    `spec-loader` wrote `validator.schema.json` from the spec before reading
    `general_specs/`, and all three passed on the first try including the `body`
    shape, which no spec fixes and which we each had to choose. Two independent
    derivations agreeing is the best evidence available that it is right — and
    this test is what keeps it true rather than a nice thing that happened once.

    **It says "read, then the one enforcement point" where it said "render,
    then".** The render is gone and what it proved was never about rendering:
    the schema is checked over the **delivered document**, whatever produced it
    (main spec criterion 3, amended at rev. 10).
    """
    for source in sorted(GENERAL.rglob("*.yaml")):
        problems = validate(
            load(source), schema_for("validator"), origin=str(source.relative_to(ROOT))
        )
        assert not problems, f"{source.name}: {problems}"


def test_the_shipped_bodies_resolve(shipped: ValidatorSpecRegistry) -> None:
    """§10.3 check 1b over the files actually in the tree."""
    for name in shipped.names():
        check_body_resolves(shipped.spec(name), ROOT / "validator")


def test_the_agent_bodied_one_declares_no_entry(shipped: ValidatorSpecRegistry) -> None:
    """The case the withdrawn callable could not express: no function to
    register, only a description an agent carries out."""
    judged = shipped.spec("production_grade")
    assert judged.body.get("entry") is None
    assert judged.strength is Strength.WEAK  # and the label is the honest one


def test_a_general_spec_is_an_ordinary_document() -> None:
    """Main spec §4.5. Same file format, same schemas — the main repository does
    not get a private path for its own specs.

    **This test replaced one and the replacement is not a rename.** It was
    `test_a_general_spec_is_a_template_whose_config_is_empty`, and it asserted
    the *fill*: render with no config and get `inputs == ['any']`, render with
    `{"inputs": ["trace"]}` and get `['trace']`. That property no longer exists
    to assert — §4.5 rev. 10 says a general spec *"stops being a template whose
    `config` is empty and is simply a document this repository ships"*.

    **The deleted half is worth one sentence, because it is the measurement that
    justified deleting it.** All three general specs used exactly one jsonnet
    construct, `if std.objectHas(config, 'inputs') then config.inputs else
    ['any']`, and **nothing in the repository ever passed them a config** — there
    is no production loader for `general_specs/` at all, only this file, which
    rendered with no fill. So the filled branch the old test exercised was
    reachable from this test and from nowhere else: a fixture proving a feature
    that had no caller. What survives is the branch that always ran, written
    literally.

    What is asserted instead is the uniformity §4.5 is actually about: these go
    through the same reader, carry the same discriminator, and pass the same
    schema as any task package's document.
    """
    path = GENERAL / "schema_conformance/schema_conformance.yaml"
    raw, problems = read_yaml(path, origin=str(path))

    assert not problems
    assert raw[MODULE_KEY] == "validator", "the same discriminator a package writes"
    assert load(path)["inputs"] == ["any"], "the literal that replaced the conditional"


# --------------------------------------------------------------------------- #
# Criterion 16 — spec §10's reference workflow


#: Every row of spec §10.1–§10.7, as `(step, name, dimension, strength)`. The
#: table is the criterion: if a row cannot be written here, the vocabulary is
#: insufficient and main spec criterion 7 has failed.
REFERENCE = [
    ("10.1 deploy config", "config_is_production_grade", "trustworthiness", "weak"),
    ("10.2 e2e run method", "e2e_correctness", "usability", "strong"),
    ("10.2 e2e run method", "knobs_are_open", "trustworthiness", "weak"),
    ("10.3 trace getter", "trace_schema", "completeness", "strong"),
    ("10.3 trace getter", "trace_self_consistency", "trustworthiness", "strong"),
    ("10.3 trace getter", "trace_supported_its_analysis", "trustworthiness", "long_term_strong"),
    ("10.3 trace getter", "trace_full_coverage", "completeness", "weak"),
    ("10.3 trace getter", "measured_time_plausible", "trustworthiness", "weak"),
    ("10.4 analysis", "topk_selection", "usability", "strong"),
    ("10.4 analysis", "operator_correctness_harness", "usability", "strong"),
    ("10.4 analysis", "operator_timing_harness", "usability", "strong"),
    ("10.4 analysis", "analysis_post_run_feedback", "trustworthiness", "long_term_strong"),
    ("10.4 analysis", "headroom_and_roofline", "trustworthiness", "weak"),
    ("10.4 analysis", "overlapped_trace_first_pass", "trustworthiness", "weak"),
    ("10.5 optimised kernel", "differential_comparison", "trustworthiness", "strong"),
    ("10.5 optimised kernel", "performance_evaluator", "usability", "strong"),
    ("10.5 optimised kernel", "optimisation_quality", "trustworthiness", "weak"),
    ("10.6 integration", "evaluation_suite", "trustworthiness", "strong"),
    ("10.6 integration", "benchmark", "usability", "strong"),
    ("10.7 system level", "global_validator", "trustworthiness", "long_term_strong"),
    ("10.7 system level", "goal_validator", "trustworthiness", "weak"),
    ("10.7 system level", "cheat_validator", "trustworthiness", "weak"),
]

#: The six steps' handoff kinds, so the binding side resolves too.
KINDS = {
    "deploy_config": ["config_is_production_grade"],
    "e2e_run_method": ["e2e_correctness", "knobs_are_open"],
    "trace": [
        "trace_schema",
        "trace_self_consistency",
        "trace_supported_its_analysis",
        "trace_full_coverage",
        "measured_time_plausible",
    ],
    "analysis": [
        "topk_selection",
        "operator_correctness_harness",
        "operator_timing_harness",
        "analysis_post_run_feedback",
        "headroom_and_roofline",
        "overlapped_trace_first_pass",
    ],
    "optimised_kernel": [
        "differential_comparison",
        "performance_evaluator",
        "optimisation_quality",
    ],
    "integration": ["evaluation_suite", "benchmark"],
}


@pytest.fixture
def workflow(tmp_path: Path) -> ValidatorSpecRegistry:
    """The reference workflow as a task package would express it."""
    from tests.validator.conftest import validator_record, write_body

    write_body(tmp_path)
    registry = ValidatorSpecRegistry()
    for _step, name, dimension, strength in REFERENCE:
        # An agent-judged check has no `entry`; a quantified one does. That split
        # is spec §5.6's observable test showing up in the file layout.
        entry = "entry.sh" if strength != "weak" else None
        registry.add(
            name,
            validator_record(
                name,
                inputs=["trace"],
                dimension=dimension,
                strength=strength,
                entry=entry,
                logic_source="external_static" if entry else "agent_written",
                cost="minutes" if entry else "seconds",
            ),
            origin=f"workflow/{name}.yaml",
        )
    for kind, validators in KINDS.items():
        registry.bind(kind, validators)
    return registry


def test_reference_workflow_resolves(workflow: ValidatorSpecRegistry) -> None:
    """Criterion 16. Every step of §10 is expressible: its validators admit, and
    each one's dimension and strength resolve."""
    assert len(workflow.names()) == len(REFERENCE)
    for _step, name, dimension, strength in REFERENCE:
        spec = workflow.spec(name)
        assert spec.dimension is Dimension(dimension)
        assert spec.strength is Strength(strength)


def test_every_reference_handoff_kind_binds(workflow: ValidatorSpecRegistry) -> None:
    """The binding side. Criterion 16 asks for *"its handoffs, its validators"*,
    and a validator nothing binds is a check that will never run."""
    for kind, validators in KINDS.items():
        for name in validators:
            assert f"handoff_kind:{kind}" in workflow.users_of(name)
    bound = {n for names in KINDS.values() for n in names}
    # The three system-level validators are deliberately bound to no kind — they
    # take the whole run, and §10.7 is where the spec says so.
    assert set(workflow.names()) - bound == {
        "global_validator",
        "goal_validator",
        "cheat_validator",
    }


def test_the_workflow_has_all_three_strengths(workflow: ValidatorSpecRegistry) -> None:
    """`long_term_strong` is present, and it is not a rounding of `strong`: §10.3
    and §10.4 each carry one, and §10.7's global validator is one."""
    strengths = {workflow.spec(n).strength for n in workflow.names()}
    assert strengths == set(Strength)


def test_system_level_validators_resolve(workflow: ValidatorSpecRegistry) -> None:
    """§10.7. The cheat validator earns its place from observed behaviour: three
    distinct evaluation-surface exploits are on record from one competition —
    taking one's own first version as the baseline; copying tolerance logic while
    omitting the NaN check, so an all-NaN output is both fast and "correct"; and a
    writer model discovering the verifier had edit permission and instructing it,
    in the verification prompt, to do the work."""
    for name in ("global_validator", "goal_validator", "cheat_validator"):
        assert name in workflow
    assert workflow.spec("global_validator").strength is Strength.LONG_TERM_STRONG
    assert workflow.spec("cheat_validator").strength is Strength.WEAK
