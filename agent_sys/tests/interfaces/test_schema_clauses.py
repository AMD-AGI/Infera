"""Clauses in `spec_loader`'s schemas that another package's correctness rests on.

A schema clause is authored by one package and depended on by others, and the
dependants cannot enforce it — they can only assume it. That asymmetry is the
same one `docs/interfaces.md` §8 prices for `Pushable` and `covers`, with the
wall in a different place: **a rule the producer writes and only the consumers
can be hurt by has to be enforced on the producer's side**, or nothing fails
when it is weakened.

These are not schema tests. `tests/spec_loader/` asserts what each schema means;
this file asserts the clauses whose *removal would be silent somewhere else*.
A clause belongs here when relaxing it leaves every affected package's own suite
green — which is exactly when a comment in one of them is not enough.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.validator_executor import ValidatorExecutor
from spec_loader import schema_for, validator_agent_of


def test_a_validator_agent_name_is_never_empty() -> None:
    """`validator.schema.json` `agent.minLength: 1`, and **`agent` depends on it silently**.

    `validator` spec §8.2 row 1 binds a validator to an agent whose declared
    environment it uses. Absent means *take the default*; a name that does not
    resolve is an error, because falling back would hand the author a working
    run in an environment they did not configure. **Absent and unresolvable are
    different questions**, and `minLength: 1` is what keeps `""` out of the
    first category.

    The two dependants are **not symmetric**, and the difference is the whole
    reason this test is in `tests/interfaces/` rather than in either of theirs.
    Measured, not relayed — both packages' own accounts of this had it wrong,
    including mine in the first version of this docstring:

    - **`agent/validator_executor.py` is the silent one.** `_agent_spec` is
      `spec.agent or self.agent_spec`, so `""` is falsy and takes the
      composition-root default with no error. Their own test states the
      dependency exactly: *"the `or` cannot reach this case: it falls back on a
      falsy value, and `minLength: 1` means a present name is never falsy."*
      **This is the hazard.** Relax the clause and an author writing
      `agent: ""` gets a validator that runs in an environment nobody chose,
      producing a verdict somebody trusts — with `tests/agent` still green,
      because the code is behaving exactly as designed on an input the schema
      was supposed to have made impossible.

    - **`validator/phase.py` is loud, and its own report said otherwise.**
      `_bound_environment` branches on `if name is None`, and `""` is not
      `None`, so an empty name falls through to the resolve and **raises
      `ValidatorInvalid`**. `validator` reported it as taking the quiet-`None`
      branch; it does not, and they have since pinned the raise in `4445c97`.
      Not re-asserted here — that behaviour is theirs and their pin is its one
      writer. The clause still matters there, as what keeps the *absent* and
      *unresolvable* answers from being reachable by one input, but it is not
      where silence would come from.

    So: one silent dependant is enough, and the count in the first version of
    this docstring was wrong in the direction that made the argument look
    stronger than it is. The precise claim is that **`agent` cannot notice this
    clause going**, and that is sufficient, because the clause lives in a third
    package and nothing in `tests/agent` reaches it.
    """
    agent = schema_for("validator")["properties"]["agent"]

    assert agent["minLength"] == 1, (
        'relaxing this makes `agent: ""` falsy-fall-back through '
        "`agent/validator_executor.py::_agent_spec`'s `or`, silently taking the "
        "composition-root default; `tests/agent` would stay green"
    )
    assert agent["type"] == "string"

    # This package's own reader: past the schema, `""` and absent are one input
    # with two intended meanings.
    assert validator_agent_of({"agent": ""}) is validator_agent_of({}) is None

    # **The premise, asserted rather than described.** The paragraph above used
    # to state `agent`'s behaviour and nothing checked it — a claim about
    # another package's code, in a file that package does not read, which is the
    # staleness this whole test exists to prevent. Driven directly instead:
    # `_agent_spec` is `spec.agent or self.agent_spec`, so an empty name is
    # **indistinguishable from an absent one** and silently takes the wiring's
    # default.
    #
    # This cannot live in `tests/agent`: `minLength: 1` makes the input
    # unreachable from inside that package, which is exactly why the clause
    # needs a test somewhere its removal is felt.
    wiring = SimpleNamespace(agent_spec="from_the_composition_root")
    fell_back = ValidatorExecutor._agent_spec(wiring, SimpleNamespace(agent=""))
    absent = ValidatorExecutor._agent_spec(wiring, SimpleNamespace(agent=None))

    assert fell_back == absent == "from_the_composition_root", (
        "an empty agent name must be indistinguishable from an absent one here "
        "— if it ever is not, this test's reason has changed and the docstring "
        "above is wrong"
    )
