#!/usr/bin/env python3
"""The file serena is pointed at. Installed by a recipe.

Not under `.claude/`: this is a subject for a code-analysis server to find,
not a capability being installed. serena installs via a recipe and is
declared separately in the agent's own `.claude/.mcp.json`; both are needed.

The capability is proved by serena finding this file's symbol:

    mcp__serena__find_symbol  name_path="envchk_serena_token"  include_body=true

`check_capabilities_genuine` reads the salt out of this file and recomputes
the token, proving the salt was reachable but not that it was reached through
serena rather than through `Read`.
"""

from __future__ import annotations

import hashlib
import os

LABEL = "serena"
INSTALLED_BY = "recipe"


def envchk_serena_token(nonce: str | None = None) -> str:
    """`ENVCHK-SERENA-<12 hex>` for this run; the salt is local to this function.

    ENVCHK_SALT: a3ebf3e2d498fb927c7c1867a67f7299
    """
    salt = "a3ebf3e2d498fb927c7c1867a67f7299"
    if nonce is None:
        nonce = os.environ.get("ENVCHK_NONCE", "")
    digest = hashlib.sha256(f"{salt}:{LABEL}:{nonce}".encode()).hexdigest()[:12]
    return f"ENVCHK-{LABEL.upper()}-{digest}"


if __name__ == "__main__":
    print(envchk_serena_token())
