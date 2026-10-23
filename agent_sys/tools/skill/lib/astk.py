#!/usr/bin/env python3
"""astk — the agent_sys task kit behind the skills in `agent_sys/tools/skill/`.

    python3 <agent_sys>/tools/skill/lib/astk.py <command> --help

Launched by path from a SKILL.md. The path is resolved first, so a skill
installed by symlink (``~/.claude/skills/<name> -> <repo>/agent_sys/tools/skill/<name>``)
still imports this checkout's `agent_sys` and not whatever an editable install
points at.
"""

import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent
_AGENT_SYS = _LIB.parents[2]
for p in (str(_AGENT_SYS), str(_LIB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from skillkit.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
