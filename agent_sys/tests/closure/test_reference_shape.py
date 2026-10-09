"""Criterion 10 — the six reference-workflow steps are each expressible as one
closure, and the set loads without error.

**This tests expressibility, not the real workflow, and the distinction is the
whole of the test.** The six reference steps have no artefact in this repository:
a concrete workflow's specs live in a task package outside it, and `demo` spec
§1.3 puts the reference workflow out of scope while the demo's own graph is three
tasks. So the fixture below is six closures named for the kickoff report's loop,
each carrying the handoffs, the phase validators and the body its step needs.

What it proves is that the schema and the seven checks admit the shape — which is
what the criterion says. It does not prove the real workflow is correct, and
nothing here should be read as claiming it does.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from closure.check import check_closures
from spec_loader import schema_for
from spec_loader.bundled import bundled_registry

from .conftest import NO_ESCAPE_HATCH, Regs, grant, make_closure

#: The kickoff report's loop. Each row is (step, inputs, outputs, phase validators).
STEPS = [
    ("prepare_e2e", [], ["deploy_config"], ["config_is_reachable"]),
    ("collect_trace", ["deploy_config"], ["trace"], ["config_is_reachable"]),
    ("analyse_trace", ["trace"], ["kernel_ir", "trace_summary"], ["summary_is_complete"]),
    ("optimise_kernel", ["kernel_ir"], ["kernel_patch"], []),
    ("integrate_patch", ["kernel_patch", "deploy_config"], ["build"], ["build_is_reproducible"]),
    ("verify_e2e", ["build", "trace_summary"], ["verdict"], ["summary_is_complete"]),
]

KINDS = sorted({k for _, i, o, _ in STEPS for k in (*i, *o)})
VALIDATORS = sorted({v for *_, vs in STEPS for v in vs}) + ["check_trace_shape"]
AGENTS = ["profiler", "analyst", "kernel_dev", "run_pytest"]


def _validator() -> Draft202012Validator:
    return Draft202012Validator(schema_for("closure"), registry=bundled_registry())


def six_closures() -> list[dict]:
    return [
        make_closure(
            name,
            inputs=inputs,
            outputs=outputs,
            validators=validators,
            agent=AGENTS[index % len(AGENTS)],
            body={"readme": f"{name}/readme.md", "entry": f"{name}/entry.sh"},
            repos=["sglang"] if name != "prepare_e2e" else [],
            monitor="pusher",
        )
        for index, (name, inputs, outputs, validators) in enumerate(STEPS)
    ]


@pytest.fixture
def workflow() -> Regs:
    regs = Regs().with_kinds(*KINDS, validators=["check_trace_shape"])
    regs.with_agents(*AGENTS).with_validators(*VALIDATORS)
    for doc in six_closures():
        regs.with_closure(doc)
    return regs


def test_six_step_shape_loads(workflow: Regs) -> None:
    """Each of the six is one closure, and the set loads with no problems."""
    assert len(workflow.closures.names()) == 6
    assert [list(_validator().iter_errors(doc)) for doc in six_closures()] == [[]] * 6
    assert check_closures(workflow, NO_ESCAPE_HATCH) == []


def test_a_shared_kind_reaches_every_step_that_uses_it(workflow: Regs) -> None:
    """The reverse indexes over the loaded set answer "what breaks if I change
    this", which is why they exist."""
    check_closures(workflow, NO_ESCAPE_HATCH)
    workflow.closures.freeze()

    # A producer is a user: "who touches this kind" is the question, and the
    # step that would have to change if the kind changed is the writer too.
    assert workflow.closures.closures_using_kind("deploy_config") == (
        "collect_trace",
        "integrate_patch",
        "prepare_e2e",
    )
    assert workflow.closures.closures_using_validator("config_is_reachable") == (
        "collect_trace",
        "prepare_e2e",
    )


def test_one_missing_grant_in_six_steps_is_found(workflow: Regs) -> None:
    """The failure criterion 5 exists to catch, in a set the right size for it to
    hide in."""
    broken = make_closure(
        "integrate_patch",
        inputs=["kernel_patch", "deploy_config"],
        outputs=["build"],
        grants=[grant("kernel_patch"), grant("deploy_config"), grant("build", "read")],
    )
    workflow.closures._specs["integrate_patch"] = broken

    (problem,) = [p for p in check_closures(workflow, NO_ESCAPE_HATCH) if p.keyword == "covers"]
    assert "'build'" in problem.message
    assert "grant no write" in problem.message
