"""o11y_core tests. Not collected by CI (agent_sys's suites do not include agent_sys/o11y); run with
`python -m pytest agent_sys/o11y/tests/core`."""

import core_paths  # noqa: F401  (puts the plugin's server dir on sys.path)
import pytest


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "fleet"
    monkeypatch.setenv("AGENT_SYS_O11Y_FLEET_DIR", str(r))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "sess-test-0001")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
    monkeypatch.delenv("AGENT_SYS_O11Y", raising=False)
    return r


@pytest.fixture
def ctx(root):
    from o11y_core import api

    return api.Context(root=root, session_id="sess-test-0001", role="main")
