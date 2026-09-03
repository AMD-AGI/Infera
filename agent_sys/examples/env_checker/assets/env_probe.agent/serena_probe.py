#!/usr/bin/env python3
"""The file serena is pointed at. L1.

**Not under `.claude/`, and that is deliberate.** Everything in `.claude/` is a
capability being installed; this is a *subject* — a small Python file with one
distinctively named symbol in it, sitting in the agent's workspace so that a
code-analysis server has something to analyse. Putting it under `.claude/`
would offer it to the installer as a component, which it is not.

serena is L1: it is installed by an `env_mgr` recipe named in the agent spec's
`recipes:` key, it is third-party, and it is the one capability of the seven
whose artefact this repository does not author. So its token cannot be baked
into the tool that reports it, the way the other six are. It is baked in here
instead, and the capability is proved by serena being able to **find** it:

    mcp__serena__find_symbol  name_path="envchk_serena_token"  include_body=true

A tool that never activated a project, or that failed to install, returns
nothing for that symbol.

`check_capabilities_genuine` reads the salt out of *this* file, from the task
package, and recomputes the token — the same treatment the two markdown-borne
capabilities get, and with the same stated limit: it proves the salt was
reachable, not that it was reached through serena rather than through `Read`.

What raises the forgery cost, without closing that gap, is the **shape** of the
raw response, which the validator checks against a schema measured on this host
rather than remembered. Serena 1.28.1's `find_symbol` returns a JSON array of

    {"name_path", "kind", "relative_path", "body_location": {"start_line",
     "end_line"}, "body"}

so a forger must know that shape as well as the salt. It is a cost increase and
not a proof, and `check_capabilities_genuine`'s readme says so in those words.
"""

from __future__ import annotations

import hashlib
import os

LABEL = "serena"
LEVEL = "L1"


def envchk_serena_token(nonce: str | None = None) -> str:
    """`ENVCHK-SERENA-<12 hex>` for this run.

    The symbol name is long and unlike anything else in the tree on purpose:
    it is what `find_symbol` is asked for, and a name that also matched a dozen
    library functions would make a hit meaningless.

    **The salt is a local, inside this function, and that is not a style
    choice.** Measured 2026-09-03 against Serena 1.28.1: `find_symbol` with
    `include_body=true` returns the symbol's **body** and nothing above it, so a
    module-level `SALT` would be invisible in the response — the agent would
    call serena, get a correct answer, and still have no salt. Row 7 of
    `ACCEPTANCE.md` requires the salt to appear in the raw response, and this is
    what makes that satisfiable through serena rather than only through `Read`.

    ENVCHK_SALT: a3ebf3e2d498fb927c7c1867a67f7299
    """
    salt = "a3ebf3e2d498fb927c7c1867a67f7299"
    if nonce is None:
        nonce = os.environ.get("ENVCHK_NONCE", "")
    digest = hashlib.sha256(f"{salt}:{LABEL}:{nonce}".encode()).hexdigest()[:12]
    return f"ENVCHK-{LABEL.upper()}-{digest}"


if __name__ == "__main__":
    print(envchk_serena_token())
