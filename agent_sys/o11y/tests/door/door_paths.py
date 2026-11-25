"""Paths shared by the door tests (a module name unique across agent_sys/o11y/tests)."""

import sys
from pathlib import Path

O11Y = Path(__file__).resolve().parents[2]  # agent_sys/o11y
PROTOCOL = O11Y / "protocol"
PROTOCOL_PY = PROTOCOL / "py"
DOOR = O11Y / "plugins" / "agent-sys-o11y-door"
SERVER = DOOR / "server"  # o11y_door
PY = DOOR / "bin" / "py"
AGENT_SYS = O11Y.parent
for p in (PROTOCOL_PY, SERVER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Ledger examples hand-converted from real optimisation histories (paths and ids neutralised).
EXAMPLES = Path(__file__).resolve().parent / "fixtures" / "ledger_examples"
SLUG_VECTORS = PROTOCOL / "test_vectors" / "slug.json"
