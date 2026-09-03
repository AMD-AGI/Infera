#!/usr/bin/env python3
"""`envchk_echo_token` — an **in-process** tool. L3.

Module-level `TOOLS` is the whole declaration. `env_mgr` appends it to
`Prepared.tools`, and `agent/backends/claude_sdk.py:_tool_server` publishes the
list as one in-process MCP server called `env_mgr`, so the model calls
**`mcp__env_mgr__envchk_echo_token`**. There is no subprocess and no pipe: the
handler runs inside the session's own process.

That makes it the one capability of the seven whose `pid` is the agent's own
`pid`, and the readme beside this package's brief says so — a report claiming a
different pid for this section is reporting something else.

## The shape `TOOLS` must have

`claude_sdk.py` reads `defn.name`, `defn.description`, `defn.schema` and calls
`defn.call(**kwargs)`, and nothing else (`_adapt_tool`). It is **structural**,
not nominal: the backend does not import a `ToolDef` class to check against, so
a plain object carrying those four members is a tool.

`ToolDef` is defined in `env_mgr`, and this file must not import it. A component
is copied into a zone and executed there; `env_mgr` may not be importable from
inside one, and a capability that fails to load because of an import is
indistinguishable, from the model's side, from a capability that was never
installed. So the four members are supplied by a local dataclass, which is the
same structural contract with no dependency.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import pathlib
from typing import Any, Callable

#: ENVCHK_SALT: 3cccd425ac607b95d583870bbd987eb9
SALT = "3cccd425ac607b95d583870bbd987eb9"
LABEL = "tooldef"
LEVEL = "L3"


def token(varying: str) -> str:
    """`sha256(f"{salt}:{label}:{varying}")[:12]` — the shared derivation.

    **For this capability `varying` is the module's own absolute path, not the
    nonce**, and that is the whole of what run 2 changed. See `echo_token`.
    """
    digest = hashlib.sha256(f"{SALT}:{LABEL}:{varying}".encode()).hexdigest()[:12]
    return f"ENVCHK-{LABEL.upper()}-{digest}"


def echo_token() -> dict[str, Any]:
    """The tool's result. **Takes no input — not an argument, not the environment.**

    ## Why not `$ENVCHK_NONCE`, which every other capability uses

    Measured, run 2 (2026-09-03): this tool returned a token computed from an
    **empty** nonce. An in-process `ToolDef` runs in the **supervisor's**
    process, and `Prepared.environment` — which carries the agent spec's `env`
    block — is handed to the **CLI child**. The supervisor never sees it. The
    tool did not fail; it returned a well-formed token about nothing, and the
    agent quoted it correctly. **The tool lied to the agent.**

    So the route cannot offer *"the tool can see the run's environment"*, and no
    test written here changes that. What it can offer is this.

    ## What it uses instead: its own path

    `__file__`. **The one thing a Python module always knows is where it is**,
    and it is the only input in this package that travels through no channel
    measured to be broken or model-controlled — not the environment (broken for
    this route), not a tool argument (then the model chooses the input).

    `_tooldefs` imports the **placed copy** in `<zone>/config/tools/`, so the
    path carries this run's zone identifiers. **The freshness is in the path,
    not in the content**: placed copy, staged source and working tree are
    byte-identical, so `sha256` proves a file was read and digested correctly
    and cannot say *which*.

    That the token still moves run to run matters for a second reason: a token
    that is byte-identical across two runs is the signature of the empty-nonce
    bug, and a fix producing that signature would be indistinguishable from the
    defect it replaced.
    """
    here = os.path.abspath(__file__)
    return {
        "token": token(here),
        "path": here,
        "sha256": hashlib.sha256(pathlib.Path(here).read_bytes()).hexdigest(),
        "label": LABEL,
        "level": LEVEL,
        "pid": os.getpid(),
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class _ToolDef:
    """The four members `claude_sdk._adapt_tool` reads. Nothing more is a tool.

    Underscored because it is not an interface anyone should import: the
    interface is `TOOLS`, and this is how this file happens to satisfy it.

    **A plain class and not a `@dataclass`, and that is a measured decision.**
    `dataclasses._is_type` reaches for `sys.modules[cls.__module__].__dict__` to
    resolve a string annotation, and a module loaded by
    `importlib.util.module_from_spec` without being registered in `sys.modules`
    is not there — so the decorator raises
    `AttributeError: 'NoneType' object has no attribute '__dict__'` at import.
    Measured 2026-09-03 on CPython 3.13 while `check_capabilities_genuine` was
    importing this exact file. A capability whose module fails to import is
    indistinguishable, from the model's side, from one that was never installed,
    and `env_mgr` is free to load it any way it likes; four assignments in
    `__init__` depend on nothing.
    """

    def __init__(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        call: Callable[..., dict[str, Any]],
    ) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self.call = call


#: The declaration. A module-level list under this exact name is what `env_mgr`
#: looks for in a `*.tooldef.py`; the file's location and suffix are the rest of
#: it. Nothing names this file either — see `envchk_stdio.mcp.py` for the same
#: point about auto-registration.
TOOLS = [
    _ToolDef(
        name="envchk_echo_token",
        description=(
            "Return the in-process ToolDef capability token, with the pid and "
            "timestamp of the process that produced it. Takes no arguments. "
            "The pid is the agent session's own, because this tool runs in "
            "process rather than over a pipe."
        ),
        # `additionalProperties: False`, because the SDK validates arguments
        # against this schema *before* the handler runs
        # (`claude_sdk.py:_adapt_tool`), so a typo in a call is refused with a
        # message rather than silently ignored.
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        call=echo_token,
    )
]
