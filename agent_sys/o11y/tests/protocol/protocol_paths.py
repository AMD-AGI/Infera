"""Paths for the protocol tests (a module name unique across agent_sys/o11y/tests)."""

import sys
from pathlib import Path

O11Y = Path(__file__).resolve().parents[2]  # agent_sys/o11y
PROTOCOL = O11Y / "protocol"
AGENT_SYS = O11Y.parent
if str(PROTOCOL / "py") not in sys.path:
    sys.path.insert(0, str(PROTOCOL / "py"))
SLUG_VECTORS = PROTOCOL / "test_vectors" / "slug.json"
