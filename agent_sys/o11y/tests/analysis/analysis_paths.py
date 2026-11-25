"""Paths shared by the analysis tests (a module name unique across agent_sys/o11y/tests): the plugin's server
dir, the door's server dir and the protocol library go on sys.path, as o11y-run arranges."""

import sys
from pathlib import Path

O11Y = Path(__file__).resolve().parents[2]  # agent_sys/o11y
PLUGIN = O11Y / "plugins" / "agent-sys-o11y"
SERVER = PLUGIN / "server"
for p in (SERVER, PLUGIN / "door", PLUGIN / "protocol" / "py"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Copies of real sources (dsv41 transcript slice, forge campaign, machine pool, OTel records); tests
# copy them again into tmp_path and never modify these.
FIX = Path(__file__).resolve().parent / "fixtures"
