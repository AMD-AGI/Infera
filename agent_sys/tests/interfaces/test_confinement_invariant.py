"""`agent`'s early return is safe only because of an `env_mgr` invariant.

`_apply_confinement` returns without handing `spawn` to the executor when
`prepared.confinement is None`:

```python
if prepared.confinement is None:
    return
```

**And a `ProgramExecutor` that never receives `spawn` runs unconfined** —
`backends/program.py:109` is `start = self._spawn or subprocess.Popen`. So if
`env_mgr` ever reported no confinement *while enforcing*, a program task would
start with the operator's privileges and nothing would say so.

**It cannot today, and the reason lives in the other package**
(`env_mgr/prepare.py:465-467`):

```python
conf = None
if enforcing:
    conf = _apply.confinement_for(select(av), av.landlock_abi)   # select raises NoConfinement
```

`select` raises rather than returning `None`, so `confinement is None` **implies**
`permissions_enforced is False`. That is a real guarantee and it is why the
early return is correct.

**But nothing checked it, and `agent` cannot see it.** A one-line change in
`env_mgr` — returning `None` where it now raises, or adding a fourth reason for
an absent confinement — would make a program task silently unconfined, with a
green suite on both sides. This is the file that fails instead.

**A test rather than a runtime refusal, deliberately.** Refusing in
`_apply_confinement` would be the stronger guard, and it is the wrong shape
here: the invariant is `env_mgr`'s to keep, `agent` asserting it at run time
would put the same rule in two places, and the fixtures that legitimately
report `confinement=None` while enforcing are stand-ins for exactly the
composition this invariant says cannot exist. The seam is what needs checking,
not the caller.
"""

from __future__ import annotations

import inspect

from env_mgr import prepare as prepare_mod


def test_no_confinement_implies_the_switch_is_off() -> None:
    """The invariant, read off `env_mgr`'s source.

    A behavioural test would need a machine with no isolation mechanism, and
    this one has bubblewrap absent but Landlock present — so the branch that
    matters is unreachable here and a green run would prove nothing. The source
    is what carries the guarantee, so the source is what is asserted.
    """
    source = inspect.getsource(prepare_mod.prepare)

    assert "conf = None" in source, "the shape this test is about has moved; re-read it"
    after = source.split("conf = None", 1)[1]
    head = after.strip().splitlines()[0].strip()

    assert head.startswith("if enforcing"), (
        f"`conf` is set to None and the next line is {head!r}. `agent`'s "
        f"`_apply_confinement` returns early on `confinement is None` without "
        f"handing `spawn` to the executor, and a ProgramExecutor without `spawn` "
        f"runs unconfined — so an absent confinement while enforcing would start "
        f"a program task with the operator's privileges, silently."
    )


def test_agent_still_returns_early_on_the_case_this_protects() -> None:
    """The other side of the same seam.

    If `agent` ever stops taking that early return — refusing instead, say —
    this file's premise changes and its reasoning should be re-read rather than
    left asserting something nobody depends on any more.
    """
    from agent import runner as runner_mod

    source = inspect.getsource(runner_mod._apply_confinement)

    assert "if prepared.confinement is None:" in source
    assert (
        "return" in source.split("if prepared.confinement is None:", 1)[1].strip().splitlines()[0]
    )


def test_a_program_executor_without_spawn_is_the_unconfined_case() -> None:
    """Why the invariant matters at all, pinned so the consequence is visible.

    `start = self._spawn or subprocess.Popen` is the line that turns "nobody
    called `accept_confinement`" into "ran with the operator's privileges".
    """
    from agent.backends import program

    source = inspect.getsource(program)

    assert "self._spawn or subprocess.Popen" in source, (
        "the fallback this file reasons about has changed shape; the invariant "
        "above may no longer be what keeps a program task confined"
    )
