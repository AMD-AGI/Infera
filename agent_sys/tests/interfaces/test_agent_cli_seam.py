"""`env_mgr.Prepared.agent_cli` reaching `agent.Assignment.agent_cli`.

**The defect this replaces was a green test.** The backend used to read the CLI
out of `Assignment.environment` under the key `AGENT_SYS_CLAUDE_CLI`, a name
`env_mgr` never published, and the guard —
`test_the_cli_env_var_matches_env_mgrs_name` — asserted the literal against
itself:

    assert CLI_ENV_VAR == "AGENT_SYS_CLAUDE_CLI"

One side of a two-sided name, compared to a copy of itself. It could not fail,
and meanwhile `material.deploy` always sets `CLAUDE_CONFIG_DIR`, so every
prepared AI run took the refusal branch that fires when no CLI is reported.

The report is now a declared field on both sides, which is what makes it
checkable across the seam at all: this file imports both, which `interfaces.md`
§4.9 permits tests and forbids packages.
"""

from __future__ import annotations

import ast
from pathlib import Path

from agent.backend import Assignment
from env_mgr.protocols import Prepared

ROOT = Path(__file__).resolve().parents[2]


def test_both_sides_declare_the_field() -> None:
    """A field on each side, spelled the same. The old env-var name existed on
    one side only, which is exactly what nothing could see."""
    assert "agent_cli" in Prepared.__annotations__
    assert "agent_cli" in Assignment.model_fields


def test_the_runner_carries_it_across() -> None:
    """Declared on both sides and copied by nobody is the same silence.

    Static, because constructing a real `Prepared` needs a zone, a policy and a
    sync report; what has to hold is that the runner reads the field it was
    given rather than defaulting around it.
    """
    tree = ast.parse((ROOT / "agent" / "runner.py").read_text(encoding="utf-8"))
    carried = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "agent_cli"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "agent_cli"
    ]
    assert carried, (
        "agent/runner.py builds the Assignment without passing "
        "`agent_cli=prepared.agent_cli`, so the backend sees None on every "
        "prepared run and refuses"
    )
