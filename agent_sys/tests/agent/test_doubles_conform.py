"""**One rule: a double is checked against the real type in both directions.**

**Four green suites measured a fiction on 2026-08-29**, and they are one bug
seen from four sides:

| | the double | the real type | found by |
|---|---|---|---|
| **invented** | `FakeClient.session_id`, `FakeClient.get_session_messages` | has neither | driving the real SDK |
| **missing** | `StubEnvManager.prepare()` returned six fields | `Prepared` had gained a seventh, `agent_cli` | 11 red tests, after the fact |
| **absent** | `StubStore` had no `seal` at all | `FilesystemStore` does | the runner's broad `except` turned every `AttributeError` into a "refusal" — 174 green, the seal never running once |
| **contract** | `StubStore.seal` **raised** | `FilesystemStore.seal` **returns** the reason (`fd31a6c`) | `handoff` said so; **no test did**, because both objects had a `seal` and presence was all this file checked |

The fourth is the one that widened the rule. Presence is not a contract: a
double that raises where the real object returns keeps every test green while
production takes a branch no test has ever run. `seal` changed **two and a half
minutes** after the runner started calling it.

The first shipped two broken methods — `session_ref` returned `None` and
`query()` raised `AttributeError` on **every** real run — while 147 tests stayed
green, because the double answered every question the adapter asked. The second
was loud, but only because the runner happened to read the field
unconditionally; a `getattr` default would have made it silent too.

So neither direction is optional:

> **A double must declare nothing the real type lacks, and everything the
> production code reads.**

And, per `main`'s rule, **the check is shown to fail in both directions** —
`test_the_rule_catches_an_invented_member` and
`test_the_rule_catches_a_missing_member` run the checkers against deliberately
wrong doubles, so a checker that silently stopped checking is itself caught.

## Stand-ins and instruments — the distinction that makes the rule safe

**A double's members divide in two, and conflating them is how a conformance
rule does damage.** `monitor`'s wording, arrived at independently and adopted
here because it names the concept the first two drafts were missing:

| | | the real type |
|---|---|---|
| **stand-in** | production calls it | **has an opinion.** Match it — surface *and* contract |
| **instrument** | only the test reads it | **has none.** Leave it alone |

`FakeClient.queries`, `.responses`, `.connected` and `StubStore.sealed` are
instruments: bookkeeping that records what was called and claims nothing about
the vendor's API. `seal`, `prepare` and the client's six methods are stand-ins.

> **Conform on what production calls; leave alone what the test reads.**

**Both drafts of this file overreached, in opposite directions, within an hour
of each other** — and so did `monitor`'s. Mine compared whole public surfaces
and flagged a spy's bookkeeping as surplus; theirs was about to tighten a spy's
`**kw` into a `TypeError`, which would have replaced a *recorded* unexpected
call with a crash and broken four tests that assert `executions == 0`. **A
conformance rule that does not know what a double is for will delete the thing
that makes it useful.**

**Stand-ins are also where the whole class of bug lives** — all four rows above
are stand-ins — which is why the rule is worth having despite both overreaches.

So the rule is not *"the double equals the real type"* but:

- every member production reads must exist **on the real type** (or the code is
  calling something that is not there — `get_session_messages`), and
- every member production reads must exist **on the double** (or the test is
  measuring something the real object would refuse — `agent_cli`).

`Prepared` is the one case checked strictly in both directions, because it is a
`NamedTuple` with a closed field set and a stand-in has no reason to differ.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest

from env_mgr.protocols import Prepared
from tests.agent.conftest import StubEnvManager, StubStore
from tests.agent.test_claude_sdk import FakeClient

ROOT = Path(__file__).resolve().parents[2]

_HAS_SDK = importlib.util.find_spec("claude_agent_sdk") is not None


# --------------------------------------------------------------------------- #
# The two checkers. Plain functions, so the rule itself can be tested.


def invented(double: set[str], real: set[str]) -> set[str]:
    """Members the double declares and the real type does not have."""
    return double - real


def missing(reads: set[str], double: set[str]) -> set[str]:
    """Members production reads and the double does not provide."""
    return reads - double


def public(obj: Any) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


def attributes_read_on(expression: str, source: Path) -> set[str]:
    """Every `<expression>.<name>` in a module, by AST.

    Reads the source rather than exercising the object: a member only touched on
    an error path is still a member the double must have, and no test run
    reaches all of them. This is what would have caught
    `self._client.get_session_messages` on the day it was written.
    """
    found: set[str] = set()
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and ast.unparse(node.value) == expression:
            found.add(node.attr)
    return found


# --------------------------------------------------------------------------- #
# `Prepared` — the "missing" case, and it is a fixed tuple so both apply strictly


def test_the_prepared_double_carries_exactly_env_mgrs_fields() -> None:
    """**Both directions, strictly**, because `Prepared` is a `NamedTuple` with a
    closed field set — there is no legitimate reason for a stand-in to carry a
    different one.

    `agent_cli` was the seventh field. The runner reads it unconditionally
    (`runner.py:617`), the double built six, and eleven tests went red.
    """
    env = StubEnvManager(zone_root="/tmp/z")
    built = public(env.prepare(_task(), execution=1, agent_spec=None))
    real = set(Prepared._fields)

    assert not invented(built, real), f"the double declares {sorted(invented(built, real))}"
    assert not missing(real, built), (
        f"env_mgr.Prepared has {sorted(missing(real, built))} and this double does not. "
        f"The runner reads Prepared fields directly, so a dropped one is an "
        f"AttributeError in every test that prepares an environment."
    )


# --------------------------------------------------------------------------- #
# `ClaudeSDKClient` — the "invented" case, which shipped two broken methods


def test_the_client_double_invents_nothing_the_adapter_relies_on() -> None:
    """**The regression that cost the most**, pinned by name.

    `FakeClient` declared `session_id` and `get_session_messages`; the real
    `ClaudeSDKClient` has neither, so `session_ref` returned `None` and `query()`
    raised on every real run while this suite was green.

    **Scoped to what production reads, and the first draft of this test was not.**
    Comparing the double's whole public surface against the vendor's flagged
    `queries`, `responses`, `connected`, `interrupts` and `buffered` — a spy's
    **bookkeeping**, which records what was called and claims nothing about the
    vendor's API. Those are legitimate and the strict version would have deleted
    them. An invented member is only a lie when **production believes it**, which
    is what `test_every_client_member_the_adapter_calls_exists_on_the_real_client`
    checks against the real type; this one keeps the two historical names dead by
    name, since a helpful reader re-adding either would restore both bugs.
    """
    surface = public(FakeClient())

    assert "session_id" not in surface, (
        "the id lives on the messages, not on the client; a double that answers "
        "`client.session_id` makes `session_ref` look like it works"
    )
    assert "get_session_messages" not in surface, (
        "it is a module-level synchronous function; a double that answers it as a "
        "method makes `query()` look like it works"
    )


@pytest.mark.skipif(not _HAS_SDK, reason="the `claude` extra is absent")
def test_every_client_member_the_adapter_calls_exists_on_the_real_client() -> None:
    """The other half, and the one with teeth: the adapter may only call what is
    there. This fails on the day someone writes `self._client.<anything>` the
    vendor does not provide, rather than on the day a real backend is finally
    driven."""
    from claude_agent_sdk import ClaudeSDKClient

    called = attributes_read_on("self._client", ROOT / "agent" / "backends" / "claude_sdk.py")
    absent = missing(called, public(ClaudeSDKClient))

    assert not absent, (
        f"the adapter calls {sorted(absent)} on its client, and ClaudeSDKClient has "
        f"no such member. `get_session_messages` was exactly this: a module-level "
        f"function called as a method."
    )


def test_the_adapter_and_its_double_agree_on_what_the_client_is() -> None:
    """Runs without the extra, so CI keeps a check even where the SDK is absent.

    It cannot know what the vendor provides, but it can insist the double
    provides everything the adapter asks of it — which is the property that
    makes the rest of `test_claude_sdk.py` mean anything.
    """
    called = attributes_read_on("self._client", ROOT / "agent" / "backends" / "claude_sdk.py")
    absent = missing(called, public(FakeClient()))

    assert not absent, f"the adapter calls {sorted(absent)}; FakeClient does not provide it"


# --------------------------------------------------------------------------- #
# `HandoffStore` — the rule catching its third instance, hours after it was written


def test_the_store_double_provides_everything_agent_reads() -> None:
    """**This is the case the rule was written for, and it still got through.**

    The runner began calling `store.seal(...)` and `StubStore` had no `seal`.
    The runner catches broadly on purpose — a refusal is `handoff.Malformed`
    and `agent` may not import `handoff`, so there is no type to name — and it
    therefore turned every `AttributeError` into a "the seal refused". **169
    tests passed with the seal never running once.**

    A green suite proves least here, which is why this checks the surface
    statically instead of trusting a run.
    """
    reads = _store_reads()
    absent = missing(reads, public(StubStore()))

    assert not absent, (
        f"`agent` calls {sorted(absent)} on its store and StubStore does not "
        f"provide it. The runner's broad `except` turns that into a silent "
        f"refusal rather than an error."
    )


def test_everything_agent_reads_on_a_store_exists_on_the_real_one() -> None:
    """The other direction, against `handoff`'s own implementation."""
    from handoff.store import FilesystemStore

    absent = missing(_store_reads(), public(FilesystemStore))

    assert not absent, (
        f"`agent` calls {sorted(absent)} on its store; FilesystemStore has no such member"
    )


def test_the_store_double_agrees_on_seal_s_return_contract() -> None:
    """**Presence was not enough, and this is the instance that proved it.**

    `handoff` changed `seal` from *raises on an unpublishable artefact* to
    *returns the reason* (`fd31a6c`, two and a half minutes after the runner
    started calling it). `StubStore.seal` still raised. Both objects had a
    `seal`, so every presence check above passed — and the suite stayed
    **green** while the runner dropped every real refusal on the floor.

    That is the fourth instance of one bug today and the first where the drift
    was in the **contract** rather than the surface. A return annotation is the
    cheapest part of a contract to compare, and it is the part that moved.
    """
    from handoff.store import FilesystemStore

    real = inspect.signature(FilesystemStore.seal).return_annotation
    double = inspect.signature(StubStore.seal).return_annotation

    assert str(double) == str(real), (
        f"FilesystemStore.seal returns {real} and StubStore.seal returns {double}. "
        f"A double that raises where the real one returns keeps every test green "
        f"while production takes a branch no test has ever run."
    )


#: The members the store scan must always find. **A floor, and it is the whole
#: reason this constant exists.**
#:
#: `attributes_read_on` searches for a *name* — `store.<attr>` — so renaming the
#: local, or inlining `self.runner.component("handoff_store").seal(...)`, makes
#: it return the empty set. Every check built on it then passes: `missing(∅, …)`
#: is `∅`. **The guard against a drifted double would go quiet in exactly the way
#: the double did**, and nothing would be red.
#:
#: `demo`'s sentence, and it is about this: *a negative grep is
#: indistinguishable from a grep for the wrong name.* The client scan already
#: has its floor in `test_the_ast_reader_finds_a_member_no_test_run_touches`;
#: this is the store's, and it was missing.
_STORE_FLOOR = frozenset({"exists", "list_versions", "get_manifest", "copy_out", "seal"})


def _store_reads() -> set[str]:
    """Every `store.<name>` in the two modules that hold one.

    Refuses to return a set that has lost a member it has always found, so a
    rename silences the scan loudly instead of silently.
    """
    found: set[str] = set()
    for name in ("runner.py", "gate.py"):
        found |= attributes_read_on("store", ROOT / "agent" / name)
    assert _STORE_FLOOR <= found, (
        f"the store scan found {sorted(found)} and lost {sorted(_STORE_FLOOR - found)}. "
        f"Either `agent` genuinely stopped calling it — update this floor — or the "
        f"expression it is searched under was renamed, in which case every check "
        f"built on this scan is now passing on an empty set."
    )
    return found


# --------------------------------------------------------------------------- #
# The rule, shown to fail in both directions


def test_the_rule_catches_an_invented_member() -> None:
    """A checker that stopped checking would pass every test above."""
    assert invented({"connect", "session_id"}, {"connect"}) == {"session_id"}


def test_the_rule_catches_a_missing_member() -> None:
    assert missing({"zone", "agent_cli"}, {"zone"}) == {"agent_cli"}


@pytest.mark.skipif(not _HAS_SDK, reason="the `claude` extra is absent")
def test_the_rule_rejects_the_adapter_as_it_actually_was() -> None:
    """**Replayed against the real vendor type, not a synthetic set.**

    Before 2026-08-29 the adapter read `self._client.get_session_messages()` and
    `self._client.session_id`. Both are fed back in here and checked against the
    real `ClaudeSDKClient`, so this asserts the rule would have caught the two
    shipped bugs on the day they were written — which a pair of hand-made sets
    cannot show.
    """
    from claude_agent_sdk import ClaudeSDKClient

    as_it_was = attributes_read_on(
        "self._client", ROOT / "agent" / "backends" / "claude_sdk.py"
    ) | {"get_session_messages", "session_id"}

    assert missing(as_it_was, public(ClaudeSDKClient)) == {"get_session_messages", "session_id"}


def test_the_rule_rejects_the_prepared_double_as_it_actually_was() -> None:
    """The other direction, replayed against the real `Prepared`: the six fields
    the double built before `env_mgr` added a seventh, an eighth and a ninth.

    **Every addition is replayed, and the third is what settles the argument.**
    `agent_cli` and `staged_package` (`086c12e`) were fields this package asked
    for, so a reader could call them special. `permissions_enforced` (`ad730a2`)
    is a kill switch `agent` did not ask for and does not read — and the double
    needed it just the same. A stand-in for a closed `NamedTuple` does not drift
    once and does not drift only when you are involved; it drifts every time the
    far side grows.

    `env_mgr` named this one *before* it turned the suite red, which is the
    first time that has happened and is the outcome this test exists to make
    ordinary.
    """
    as_it_was = {"zone", "workspace", "policy", "confinement", "sync", "environment"}

    assert missing(set(Prepared._fields), as_it_was) == {
        "agent_cli",
        "staged_package",
        "permissions_enforced",
        "output_paths",
    }


def test_the_ast_reader_finds_a_member_no_test_run_touches() -> None:
    """The reason the scan is static. `interrupt` is only reached from
    `ClaudeSdkBackend.interrupt`, and a scan that only saw exercised code would
    miss any member on a path this suite does not run."""
    called = attributes_read_on("self._client", ROOT / "agent" / "backends" / "claude_sdk.py")

    assert {"connect", "disconnect", "query", "interrupt"} <= called


def _task() -> Any:
    class Task:
        id = "t1"

    return Task()
