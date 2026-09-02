# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The block a validator body is given — §8.2's GLOBAL row, which the CLI owns.

**These are unit tests over a mapping, and a unit test is not the evidence.**
The measurement is `scratch/single-real-task-2026-08/probe_validator_env/`, run
end to end against the real `claude` CLI, with the artefacts in
`scratch/single-real-task-2026-08/probe_out/` and the reasoning in
`scratch/single-real-task-2026-08/validator-env.md`. What it found:

| | INPUT phase | OUTPUT phase |
|---|---|---|
| §8.2 row | GLOBAL — this block | PRODUCER — `Prepared.environment` |
| `ANTHROPIC_API_KEY`, before | **ABSENT** | PRESENT |
| `claude -p 'reply with the single word OK'` | `Not logged in · Please run /login` | `OK` |

The output phase already worked, because `env_mgr/material.py:69` fills
`Prepared.environment` from `harness_env`. The input phase did not, because this
block is where it lands and this block did not carry it. The two rows disagreed
about whether a validation may reach a model, and only one had a reason.

`test_isolation_shown.py` is the file that owns *what the demo reports*; this
one owns *what the composition root decides*, which is `interfaces.md` §2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cli import main as cli_main
from cli.environment import CredentialsMissing, layout_for


@pytest.fixture()
def layout(tmp_path: Path) -> Any:
    return layout_for(tmp_path).create()


def _block(monkeypatch: Any, values: dict[str, str], layout: Any, root: Path) -> dict[str, str]:
    """`_validation_env` with the operator's harness block stubbed.

    Stubbed at `cli.main`'s own import site rather than at `env_mgr.harness`,
    because the import is inside the function: patching the module attribute
    would be patching something this call never reads.
    """
    monkeypatch.setattr("env_mgr.harness.harness_env", lambda: dict(values))
    return cli_main._validation_env(root, layout)


def test_the_operators_named_keys_reach_the_block(
    monkeypatch: Any, layout: Any, tmp_path: Path
) -> None:
    """A key the operator's settings file names is forwarded **by name**.

    Nothing in this repository ever holds the value: `harness_env` reads it out
    of the live environment, and the only thing written down anywhere — in a
    package, in a `--var`, in `args.json` — is the name. That is why this is the
    seam and `agent.env` is not.
    """
    named = {"ANTHROPIC_API_KEY": "x", "ANTHROPIC_BASE_URL": "y"}
    got = _block(monkeypatch, named, layout, tmp_path)
    assert got["ANTHROPIC_API_KEY"] == "x"
    assert got["ANTHROPIC_BASE_URL"] == "y"


def test_an_empty_harness_block_forwards_nothing(
    monkeypatch: Any, layout: Any, tmp_path: Path
) -> None:
    """**The non-vacuity control.**

    A machine with no Claude Code installed, or a settings file with no `env`
    block, gets the four keys and no more — so the test above is asserting a
    forward that happened rather than a set that was always there. `harness_env`
    returning `{}` is a legitimate configuration, not a degradation: non-AI
    tasks run perfectly well on such a machine.
    """
    got = _block(monkeypatch, {}, layout, tmp_path)
    assert set(got) == {
        "PATH",
        "AGENT_SYS_DEMO_PACKAGE",
        "AGENT_SYS_DEMO_STORE",
        "AGENT_SYS_DEMO_PYTHON",
    }


def test_the_runs_own_four_keys_outrank_the_operators_file(
    monkeypatch: Any, layout: Any, tmp_path: Path
) -> None:
    """Where is the package, where is the store, which interpreter — facts about
    *this run*, computed here, and an operator's settings file is not entitled to
    an opinion about them.

    The reverse precedence is what `material.deploy` reserves its own three keys
    for (`env_mgr/harness.py:31-43`), and `PATH` is already in that reserved set,
    so this only has to hold the line for the three `AGENT_SYS_DEMO_*`.
    """
    hostile = {
        "AGENT_SYS_DEMO_PACKAGE": "/elsewhere",
        "AGENT_SYS_DEMO_STORE": "/elsewhere",
        "AGENT_SYS_DEMO_PYTHON": "/elsewhere",
    }
    got = _block(monkeypatch, hostile, layout, tmp_path)
    assert got["AGENT_SYS_DEMO_PACKAGE"] == str(tmp_path)
    assert got["AGENT_SYS_DEMO_STORE"] == str(layout.handoffs)
    assert got["AGENT_SYS_DEMO_PYTHON"] != "/elsewhere"


def test_an_unreadable_settings_file_is_a_precondition_failure(
    monkeypatch: Any, layout: Any, tmp_path: Path
) -> None:
    """`harness_env` raises on a settings file that exists and will not parse.

    Left to escape it comes out of registry construction as an uncaught
    traceback. `CredentialsMissing` is the family `main()` already maps to the
    PRECONDITION exit code, and the classification is honest — the operator's
    harness configuration is what is broken.
    """

    def explode() -> dict[str, str]:
        raise ValueError("the harness settings file '/x/settings.json' will not parse")

    monkeypatch.setattr("env_mgr.harness.harness_env", explode)
    with pytest.raises(CredentialsMissing) as caught:
        cli_main._validation_env(tmp_path, layout)
    assert "settings.json" in str(caught.value)
