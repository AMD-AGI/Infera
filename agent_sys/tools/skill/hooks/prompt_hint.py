#!/usr/bin/env python3
"""UserPromptSubmit hook: point Claude at the infera-agent-sys skill when a
prompt is in its domain.

Installed (opt-in) by `install.sh --hook`. Reads the hook event JSON on stdin,
prints a `hookSpecificOutput.additionalContext` reminder when the entry skill
should engage, and nothing otherwise. The decision is `skillkit.settings.engaged`
— the same one `astk mode --prompt` makes — so explicit mode silences it unless
the prompt names agent_sys. It never blocks a prompt: every error exits 0 silently.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
        from skillkit import settings  # noqa: PLC0415

        prompt = str(json.load(sys.stdin).get("prompt") or "")
        if not settings.engaged(prompt):
            return 0
        hits = settings.matches(prompt) or ["agent_sys"]
    except Exception:  # noqa: BLE001 — a hint must never break a prompt
        return 0
    note = (
        "This request touches the infera / agent_sys domain (matched: "
        + ", ".join(hits[:6])
        + "). Consider the `infera-agent-sys` skill: it searches the shared knowledge repo for "
        "prior tasks and handoffs on this kind of work before you start. Skip it for a trivial question."
    )
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": note}}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
