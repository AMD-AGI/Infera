# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What the *operator's* harness configured, carried into a confined zone.

`material.CONFIG_DIR` relocates ``CLAUDE_CONFIG_DIR`` into the zone so that a
run's transcript does not change with the reviewer's dotfiles. **Measured, that
relocation is also what stops the agent authenticating**: with
``CLAUDE_CONFIG_DIR`` pointed at an empty directory the CLI answers
``Not logged in · Please run /login`` and exits, because the endpoint and the
credentials live in the ``env`` block of the settings file that was just
relocated away. Injecting that block as process environment restores it — the
same prompt then answers ``OK``
(`scratch/impl-2026-08/env_mgr/p7_relocated_config_loses_auth.py`).

So this module is the other half of the relocation: *if we move the config, we
carry what the config provided.*

**Values are never logged, formatted, or put in an exception message.** The
block routinely holds a subscription key. Every message here names **keys**.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

__all__ = ["harness_env", "settings_path"]

#: Never forwarded, whatever the block says.
#:
#: The first three are the zone's own: carrying the operator's configuration must
#: not undo the relocation that made carrying it necessary. ``PATH`` is here for
#: a different reason — `prepare` *derives* it from the granted set precisely so
#: that it can never name a directory the kernel will refuse, and a settings file
#: written for an unconfined machine names several.
#:
#: An agent spec's **declared** `env` may still set any of them: an author saying
#: so outranks a default, and that update is applied after this module. What is
#: excluded here is only the operator's ambient configuration, which is not a
#: statement about this run.
_RESERVED = ("CLAUDE_CONFIG_DIR", "CLAUDE_CODE_TMPDIR", "TMPDIR", "PATH")


def settings_path(environ: Mapping[str, str] | None = None) -> str:
    """The settings file **the supervisor** would have read, resolved its way.

    ``CLAUDE_CONFIG_DIR`` first, because an operator who set it means it; then
    ``~/.claude``. Resolved from the *supervisor's* environment, which is the
    only place the answer exists — by the time a task runs, that variable has
    already been rewritten to point into the zone.
    """
    env = os.environ if environ is None else environ
    root = env.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    return os.path.join(root, "settings.json")


def harness_env(
    path: str | None = None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The settings file's ``env`` block, valued from the live environment.

    Two sources and a precedence, because they disagree in a way that matters:
    the file is what the operator *wrote*, the process environment is what this
    supervisor is *actually running under*. When a key appears in both the live
    value wins — an operator who exported an override for this session meant it,
    and the file would silently undo it.

    **A key absent from the block is not forwarded**, even when it is set in the
    supervisor's environment. That is what keeps this an allow-list rather than
    a wholesale environment leak into the sandbox, and the allow-list is the
    operator's own configuration rather than a guess of ours.

    Returns ``{}`` when there is no settings file. That is a legitimate
    configuration — a machine with no Claude Code installed runs non-AI tasks
    perfectly well — and not a degradation to report.

    **A settings file that exists and does not parse raises.** It is an operator
    error, it is one character wide, and the alternative is the failure this
    whole module exists to remove: the agent reports ``Not logged in`` and names
    the wrong cause. The message carries the path and the parser's complaint;
    it never carries a value.
    """
    resolved = settings_path(environ) if path is None else path
    if not os.path.exists(resolved):
        return {}
    try:
        with open(resolved, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"the harness settings file {resolved!r} exists and could not be read: "
            f"{error}. Its 'env' block carries the endpoint and credentials a "
            f"confined agent needs, so continuing would start an agent that fails "
            f"to authenticate and blames itself"
        ) from error
    return _block(document, environ)


def _block(document: Any, environ: Mapping[str, str] | None) -> dict[str, str]:
    live = os.environ if environ is None else environ
    block = document.get("env") if isinstance(document, dict) else None
    if not isinstance(block, dict):
        return {}
    return {
        str(key): str(live.get(str(key), value))
        for key, value in block.items()
        if str(key) not in _RESERVED
    }
