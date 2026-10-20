"""skillkit — the shared toolkit behind `agent_sys/tools/skill/*/SKILL.md`.

Every module works on agent_sys's own formats through agent_sys's own code
(`spec_loader`, `handoff`), so a package or handoff this kit accepts is one the
runtime accepts, and a rule the runtime does not enforce is not invented here.
"""

from pathlib import Path

#: The `agent_sys/` directory of the checkout this file lives in.
AGENT_SYS = Path(__file__).resolve().parents[4]
