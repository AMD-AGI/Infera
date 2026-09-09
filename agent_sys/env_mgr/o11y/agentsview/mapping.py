# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One AgentsView project per run.

**Why this cannot be solved by naming directories better.** AgentsView derives
a project from the session's **deepest** path segment, so the attempts of one
run arrive as several unrelated projects — measured on a real nested fixture,
four sessions of one run as `0_11e34171`, `0_f6daeb1b`, `task.main.b869ddf0_…`
and `task.solve_a.8c8fb4c1_…`. PR #156 made those strings readable; it did not
and cannot join them, because a nested child task fragments off from its parent
however prettily both are named. The only filesystem fix would be putting every
attempt of a run in one directory, which is what zone isolation exists to
prevent. One mapping over the run root collapses all four, at any depth.

**AgentsView is not modified.** This uses its published settings API, and every
property relied on below was measured against v0.42.0 rather than assumed —
see `design.md` §2, beside this file.

Like everything else under `o11y/`: every function returns a `Status` and
raises nothing.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from .agentsview import Status

__all__ = ["ensure_run_project", "name_for_run"]

log = logging.getLogger("env_mgr.o11y.agentsview.mapping")

#: The settings collection. `GET` lists mappings and names the machine; `POST`
#: creates one.
MAPPINGS_PATH = "/api/v1/settings/worktree-mappings"

#: The only layout that does what we need. The other legal value,
#: `repo_dot_worktrees`, matched **zero** sessions across thirteen prefix
#: shapes — including a genuine `<repo>/.worktrees/<name>` tree the archive had
#: correctly identified — and blanks `project` on write. Why it matches nothing
#: is unsettled and does not matter here: it is not this mechanism.
LAYOUT = "explicit"

#: Prefix on the project name, so a run is recognisable as one among whatever
#: else a user has in their panel. A dot, because dots round-trip verbatim.
NAME_PREFIX = "run."

#: The one transformation AgentsView applies to a project name: `-` becomes
#: `_`. Measured across `@ : / + # % .`, spaces, uppercase, leading digits and
#: non-ASCII — everything else round-trips, and there is no length cap. Applied
#: here so that the string we post is the string the panel shows, and a log
#: line cannot disagree with the UI.
_NORMALISE = str.maketrans({"-": "_"})

#: Long enough for a local daemon that has already answered a health check,
#: short enough that a wedged one cannot hold up a run.
TIMEOUT_S = 5.0


def name_for_run(run_root: Path) -> str:
    """The project name for a run, from its directory name alone.

    Nothing above the run root may change the label: two roots differing only
    in `--demo-root` must produce the same name, or the same run reads as two.
    """
    return NAME_PREFIX + run_root.name.translate(_NORMALISE)


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, method="GET")  # noqa: S310
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310
        return json.loads(r.read())


def _post_json(url: str, origin: str, body: dict[str, object]) -> None:
    # **`Origin` is mandatory on every mutating call.** Without it the answer is
    # a plain-text `403 Forbidden` rather than the JSON error shape, which reads
    # exactly like a missing endpoint. Measured.
    req = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Origin": origin},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S):  # noqa: S310
        return None


def ensure_run_project(url: str | None, run_root: Path, name: str | None = None) -> Status:
    """Give this run its own project on the panel. Never raises.

    Called once at run start, **before any session exists** — which is the whole
    reason there is no `apply`, `preview`, `reclassify` or token here. Measured:
    a mapping that exists before ingest is consulted at sync time and the
    session arrives already labelled.

    `url` is `None` when the panel did not start. That was already warned about
    once, so this says nothing: o11y reports a problem once or not at all.
    """
    if url is None:
        return Status(False, "no panel")
    try:
        machine = _machine(url)
        if machine is None:
            log.warning(
                "agentsview: could not read the panel's machine name from %s%s; "
                "this run will not get its own project.",
                url,
                MAPPINGS_PATH,
            )
            return Status(False, "no machine name")
        project = name or name_for_run(run_root)
        _post_json(
            url + MAPPINGS_PATH,
            origin=url,
            body={
                "machine": machine,
                "path_prefix": str(run_root),
                "project": project,
                "layout": LAYOUT,
                "enabled": True,
            },
        )
    except urllib.error.HTTPError as e:
        # **409 is the state we wanted.** `POST` is not idempotent — uniqueness
        # is `(machine, path_prefix)` — so a re-run of the same run id conflicts
        # with the row it created last time. Nothing to do and nothing to say.
        if e.code == 409:
            return Status(True, "already mapped")
        log.warning(
            "agentsview: could not give run %s its own project (HTTP %s); "
            "its sessions will appear under their directory names.",
            run_root.name,
            e.code,
        )
        return Status(False, f"http {e.code}")
    except Exception as e:  # noqa: BLE001 - see the module docstring
        log.warning(
            "agentsview: could not give run %s its own project (%s); "
            "its sessions will appear under their directory names.",
            run_root.name,
            e,
        )
        return Status(False, f"mapping failed: {e}")
    return Status(True, project)


def _machine(url: str) -> str | None:
    """The name the daemon will match against, read from the daemon.

    **Never assembled locally.** A mapping written with a `machine` the daemon
    does not recognise matches nothing and says nothing — and the recon that
    established all of this ran in a container whose hostname was not the
    host's, which is exactly how a hard-coded `socket.gethostname()` would have
    shipped broken.
    """
    listing = _get_json(url + MAPPINGS_PATH)
    if not isinstance(listing, dict):
        return None
    machine = listing.get("local_machine")
    return machine if isinstance(machine, str) and machine else None
