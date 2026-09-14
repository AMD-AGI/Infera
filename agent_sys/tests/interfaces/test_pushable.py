"""`monitor.Pushable` and `agent.AgentBackend` are two records of one shape.

`monitor` may not import `agent` — `docs/interfaces.md` §4 and
`test_import_rules.py` — because the monitor needs `instruct` on a live agent
while the runner needs `report` from the monitor, and importing both ways is a
package cycle. The monitor breaks it by declaring the part it uses structurally.

That is the same duplication `test_stub_agreement.py` exists for, one package
apart instead of one file apart, and it is safe for the same reason: something
checks it. **A test may import both**, because tests are not under the import
rule.

**The runtime check now exists** — `test_a_constructed_backend_is_pushable`.
`claude-agent-sdk` was installed on 2026-08-29 and the premise that held this
file to static comparison expired the same day.

That premise had already been wrong once. The docstring first claimed neither
package had an implementation; `ClaudeSdkBackend` did exist. The second version
claimed it was **not constructible here**, which was true until the extra
landed. Both were true conclusions resting on expired premises, and neither was
noticed, because *the conclusion still passed*.

What stopped the third repetition was
`test_the_runtime_check_is_still_unavailable_for_the_reason_given`, which
asserted its own premise (`find_spec("claude_agent_sdk") is None`) and failed on
the day it stopped holding, with the fix written in the assertion message. **The
static comparison below is kept, not replaced.** It runs without the 376 MB
extra and catches a drift in the *declarations*; `isinstance` catches a drift in
what is actually built. They fail on different days.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

from agent.backends.program import ProgramExecutor
from agent.protocols import AgentBackend
from monitor.protocols import Pushable

ROOT = Path(__file__).resolve().parents[2]


def _members(pkg: str, cls: str) -> dict[str, str]:
    """Public members of one class in one `protocols.py`, as text.

    Read from the source rather than from the objects: a `Protocol`'s bare
    annotations do not survive into `__dict__` in a form worth comparing, and the
    signature text is what actually has to agree.
    """
    tree = ast.parse((ROOT / pkg / "protocols.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    out: dict[str, str] = {}
    for sub in node.body:
        if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
            ret = ast.unparse(sub.returns) if sub.returns else "None"
            out[sub.name] = f"({ast.unparse(sub.args)}) -> {ret}"
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            if not sub.target.id.startswith("_"):
                out[sub.target.id] = ast.unparse(sub.annotation)
    return out


def test_agent_backend_declares_everything_pushable_needs() -> None:
    """Every member of `Pushable` exists on `AgentBackend` or its base.

    `AgentBackend` extends `Executor`, so `status` is inherited rather than
    redeclared — hence the union.
    """
    pushable = _members("monitor", "Pushable")
    backend = _members("agent", "AgentBackend") | _members("agent", "Executor")

    missing = sorted(set(pushable) - set(backend))
    assert not missing, (
        f"monitor.Pushable requires {missing}, which agent.AgentBackend does not "
        f"declare. One of the two moved; the monitor's copy is the derived one."
    )


@pytest.mark.parametrize("name", sorted(_members("monitor", "Pushable")))
def test_signatures_agree(name: str) -> None:
    """And they agree on shape, not merely on the name existing.

    `query` is the one place a difference is deliberate and is asserted as such:
    `AgentBackend.query` returns `AgentHistory`, a name `monitor` may not import,
    so `Pushable` widens it to `Any`. Widening is safe in this direction — the
    monitor consumes the history and never constructs one.
    """
    pushable = _members("monitor", "Pushable")[name]
    backend = (_members("agent", "AgentBackend") | _members("agent", "Executor"))[name]

    if name in {"query", "status"}:
        assert "Any" in pushable, (
            f"{name} is the deliberately widened member; if it no longer says "
            f"Any, either the widening was removed or this test is stale"
        )
        return
    assert pushable == backend, f"Pushable.{name} is {pushable}; AgentBackend.{name} is {backend}"


def test_pushable_is_runtime_checkable() -> None:
    """So a real backend can be checked against it once one exists.

    The static tests above are what works today; this keeps the door open for the
    `isinstance` version without anyone having to remember the decorator.
    """
    assert getattr(Pushable, "_is_runtime_protocol", False)


def test_issubclass_cannot_answer_this_and_that_is_why_it_is_isinstance() -> None:
    """Why the runtime check below constructs a backend instead of testing a class.

    `status` is an **instance** attribute, set in `__init__` and therefore absent
    from `dir(ClaudeSdkBackend)`. A `runtime_checkable` Protocol with a non-method
    member refuses `issubclass` outright — it raises `TypeError` rather than
    answering — so a class-level check is not merely weaker here, it is
    unavailable. Asserted rather than narrated, so that a later reader who
    "simplifies" the construction away gets a red test.
    """
    from agent.backends.claude_sdk import ClaudeSdkBackend

    assert {"instruct", "query"} <= set(dir(ClaudeSdkBackend))
    assert "status" not in set(dir(ClaudeSdkBackend))

    with pytest.raises(TypeError):
        issubclass(ClaudeSdkBackend, Pushable)


@pytest.mark.skipif(
    importlib.util.find_spec("claude_agent_sdk") is None,
    reason="the `claude` extra is absent; the static comparison above is the check",
)
def test_a_constructed_backend_is_pushable() -> None:
    """**The runtime check the static comparison stood in for**, finally runnable.

    Constructs the real thing: no `config["client"]`, so `__init__` imports the
    extra and builds an actual `ClaudeSDKClient`. That is the point — a supplied
    double would make this a test of the double's shape, which is exactly the
    failure that let `session_ref` and `query()` ship broken (the old
    `FakeClient` declared a `session_id` and a `get_session_messages` the real
    client does not have).

    **Constructing connects to nothing.** `ClaudeSDKClient.__init__` stores
    options; `connect()` is what spawns the CLI, and nothing here calls it. So
    this needs no credential and no network.

    A structural `isinstance` against a `runtime_checkable` Protocol checks the
    *presence* of the members, not their signatures — which is why the static
    comparison above is kept rather than replaced. This one fails when the built
    object drifts; that one fails when the declarations drift.
    """
    from agent.backend import Assignment
    from agent.backends.claude_sdk import ClaudeSdkBackend

    backend = ClaudeSdkBackend("claude_sdk", {}, Assignment(goal="g"))

    assert isinstance(backend, Pushable)
    # The member `issubclass` choked on, present on the instance as promised.
    assert backend.status is not None


def test_pushable_stays_narrow() -> None:
    """It is the part of the backend the monitor uses, not a copy of it.

    A `Pushable` that grew to `AgentBackend`'s full surface would be the import
    the structural declaration exists to avoid, written out by hand. Three
    members is the whole of spec §7's push and §7.2's `answer`.
    """
    assert set(_members("monitor", "Pushable")) == {"status", "instruct", "query"}
    assert len(inspect.getmembers(AgentBackend)) > 0  # imported, not merely named


def test_a_program_body_is_not_pushable() -> None:
    """The other half of the guard, and **it fails at a distance**.

    `monitor.pusher.live_handle` answers *is there an agent to instruct* with
    `isinstance(executor, Pushable)`. That answer is only correct while the
    program body declares fewer than all three members — a task body is a
    process, and there is nobody on the other end to take an instruction.

    **What makes this worth pinning is that the flip needs two commits and
    neither looks like it is about the pusher.** `ProgramExecutor` has `status`
    already. Adding `instruct` alone leaves this `False`, so a reviewer sees no
    consequence; adding `query` afterwards "for symmetry" completes the set and
    `isinstance` silently becomes `True`. `_push` would then call an `instruct`
    that cannot work, which is the `AttributeError` of 2026-08-29 restored —
    and `demo`'s escalation, *"the executor is a program body: there is no
    agent to instruct"*, would become a traceback.

    **`status` is asserted present on purpose.** Without it this test would
    still pass if `ProgramExecutor` stopped satisfying `Pushable` for some
    unrelated reason, which is the expired-premise failure this file's
    docstring records twice already.

    If this fails: the executor gained a push surface. Either it really can be
    instructed — in which case `live_handle`'s third branch is now wrong and
    `monitor` must be told — or the members were added for symmetry and should
    not have been.
    """
    body = ProgramExecutor(config={"command": "true"})
    assert hasattr(body, "status"), "premise gone: the negative below proves nothing"
    assert not isinstance(body, Pushable), (
        "a program body now answers yes to Pushable, so live_handle will hand it "
        "to _push and _push will call instruct() on a process. See this test's "
        "docstring: tell monitor before completing the member set."
    )
