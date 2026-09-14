#!/usr/bin/env python3
"""What `consume` would run. **Never reached in this demo.**

`consume`'s input is the `summary` handoff, which `check_grounded` seals
INVALID, so the task stays in `WAITING_HANDOFF` for the whole run. The file is
here because a step that never runs still has to be a real step — a demo whose
unreached node is a stub proves less than one whose unreached node would have
worked.

Imports nothing from `agent_sys`, for the reason `assets/produce.task/collect.py` gives.
"""

import json
import os
import sys
from pathlib import Path


def _required(name: str) -> str:
    """A named refusal instead of a bare `KeyError`.

    `AGENT_SYS_INPUT_<KIND>` is `env_mgr.grants.input_env`, landed after this
    package reported that outputs had a declared name and inputs did not and
    asked whether the asymmetry was deliberate. **It was not** — `prepare`
    already called `stage_handoffs`, which returns handoff id → staged path, and
    threw the mapping away.

    Kept loud for `output_env`'s reason: a kind naming two slots is exported for
    neither, and here that is *ordinary* rather than exotic — `validator` spec
    §4.1 makes many-to-many first class. A failure would then be correct
    behaviour and this message is the only thing that would say so.
    """
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. env_mgr.grants.input_env exports one per "
            f"staged input, keyed by kind — and exports NOTHING for a kind "
            f"naming two input slots, because an author who wrote two of a kind "
            f"cannot address either. If this task has two, that is why."
        )
    return value


def main() -> int:
    """**`consume` declares `outputs: []`, so this writes no handoff.**

    It renders into the zone — `ProgramExecutor` sets `cwd` there, and a task may
    write anywhere inside its own zone — rather than into a store path, because
    there is no output slot and therefore no `AGENT_SYS_OUTPUT_<KIND>` for it.
    That is `env_mgr.grants.output_env` behaving correctly, not a gap.

    **Inputs and outputs are spelled differently and point at the same level.**
    `AGENT_SYS_INPUT_<KIND>` is `<zone>/handoffs/<hid>/v<N>` and
    `AGENT_SYS_OUTPUT_<KIND>` is `<store>/<hid>/v<N>/content`, which reads like
    a pair one directory apart — and was, until `env_mgr` narrowed `stage` to
    copy `<v>/content` **to** `<into>/<hid>/v<N>`. Since then the staged
    directory *is* the content and there is no `content/` hop on the input side.

    This body had that hop and would have failed on it; `consume` never running
    is precisely why nothing caught it. `env-mgr-2` measured the current shape.
    The manifest is not staged at all now, so a body that needs one asks the
    store — `get_manifest` verifies a digest where a staged copy does not.

    Never reached in this demo — `consume`'s input never becomes valid.
    """
    src = Path(_required("AGENT_SYS_INPUT_SUMMARY"))
    summary = (src / "items" / "content").read_text(encoding="utf-8")
    report = Path.cwd() / "report.json"
    report.write_text(json.dumps({"summary": summary, "rendered": True}, indent=2), "utf-8")
    print(f"render: {len(summary)} characters -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
