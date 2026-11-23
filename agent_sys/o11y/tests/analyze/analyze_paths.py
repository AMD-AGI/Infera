"""Paths shared by the o11y_analyze tests (a module name unique across agent_sys/o11y/tests)."""

import sys
from pathlib import Path

O11Y = Path(__file__).resolve().parents[2]  # agent_sys/o11y
PLUGIN = O11Y / "plugins" / "agent-sys-o11y"
SERVER = PLUGIN / "server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

# Copies of real sources (dsv41 transcript slice, forge campaign, machine pool, OTel records); tests
# copy them again into tmp_path and never modify these.
FIX = Path(__file__).resolve().parent / "fixtures"
