"""Paths shared by the o11y_core tests (a module name unique across agent_sys/o11y/tests)."""

import sys
from pathlib import Path

O11Y = Path(__file__).resolve().parents[2]  # agent_sys/o11y
PLUGIN = O11Y / "plugins" / "agent-sys-o11y"
SERVER = PLUGIN / "server"
AGENT_SYS = O11Y.parent
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

# Ledger examples hand-converted from the real muse/glm52/dsv41 optimisation histories.
EXAMPLES = Path(__file__).resolve().parent / "fixtures" / "ledger_examples"
SLUG_VECTORS = O11Y / "protocol" / "test_vectors" / "slug.json"
