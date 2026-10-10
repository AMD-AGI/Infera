#!/usr/bin/env python3
"""The token scheme, and the register of the capabilities.

Each token is `sha256(f"{salt}:{label}:{nonce}")[:12]`. The derivation lives
here and inline in each artefact; each artefact carries only its own salt, and
this module carries none, so no single file yields all of them. Do not add a
central `SALTS` table here -- the validator obtains each salt the same way the
agent does, out of the artefact.

A token proves the capability's artefact was installed and reachable in the
zone. It does not prove the agent obtained it through the capability itself;
`check_capabilities_genuine` closes that gap for the process-backed
capabilities by running them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "BY_LABEL",
    "CAPABILITIES",
    "LABELS",
    "PROOF_KEYS",
    "REPORT_KEYS",
    "SALT_TAG",
    "server_of",
    "STATUSES",
    "Capability",
    "nonce_digest",
    "salt_of",
    "token",
]

#: The one marker a salt is found by. Exactly one occurrence per artefact is
#: required — zero means the artefact was edited into something this scheme
#: cannot check, and two means two salts, which is a silent ambiguity about
#: which one the capability actually uses.
SALT_TAG = re.compile(r"ENVCHK_SALT:\s*([0-9a-f]{32})")


class Capability(NamedTuple):
    """One of the six, and everything a validator needs to judge it.

    `artefact` is relative to `origin`'s root, never absolute, since a task
    package is staged into a zone.
    """

    #: The key under `capabilities` in the handoff's `items/text.json`, and the
    #: `label` half of every token derivation.
    label: str
    #: The section number in the brief. Kept as an explicit field, not derived
    #: from index, so a deleted section leaves a visible gap in the sequence.
    section: int
    #: Which of the two install routes put it in the zone: ``"copied"`` (the
    #: agent's own `assets/<name>.agent/.claude/` tree, copied into the zone
    #: config) or ``"recipe"`` (declared in a recipe and installed by
    #: `env_mgr`). Reports the install, not the declaration -- a capability can
    #: be installed by a recipe and separately declared in `.claude/.mcp.json`.
    installed_by: str
    #: Where `artefact` is rooted: ``"package"`` (relative to
    #: `$AGENT_SYS_TASK_PACKAGE`) or ``"zone_config"`` (relative to
    #: `<zone>/config/`, where a recipe placed it).
    origin: str
    #: Where the salt lives, relative to that root.
    artefact: str
    #: How `check_capabilities_genuine` re-derives the token independently.
    #: `mcp` starts the server and speaks the protocol to it; `import` imports
    #: the module and calls the handler; `file` reads the artefact for the
    #: hook's own output; `salt` recomputes from the salt.
    replay: str
    #: One line, for a failure message, so a reader does not need this file
    #: to understand what a token mismatch is about.
    what: str
    #: The full tool name (`mcp__<server>__<tool>`) the brief tells the agent
    #: to call, or `None` for capabilities not reached through MCP. Sourced
    #: from the brief's own prose, not the `.mcp.json` key; `server_of()`
    #: recovers the server half where a check needs it.
    surface: str | None


#: The register. **Order is the order a reader meets them in the brief**, and
#: each row carries its own `section` number rather than taking one from its
#: index — see `Capability.section`. Section **6** is absent, and its absence is
#: the record of a deleted capability rather than a gap to close by renumbering.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        label="skill",
        section=1,
        installed_by="copied",
        origin="package",
        artefact="assets/env_probe.agent/.claude/skills/envchk-probe/SKILL.md",
        replay="salt",
        what="a skill in the agent's own .claude/skills/",
        surface=None,
    ),
    Capability(
        label="hook",
        section=2,
        installed_by="copied",
        origin="package",
        artefact="assets/env_probe.agent/.claude/hooks/envchk_session_start.py",
        replay="file",
        what="a SessionStart hook declared in .claude/settings.json",
        surface=None,
    ),
    Capability(
        label="plugin",
        section=3,
        installed_by="copied",
        origin="package",
        artefact=(
            "assets/env_probe.agent/.claude/plugins/envchk-plugin"
            "/skills/envchk-plugin-skill/SKILL.md"
        ),
        replay="salt",
        what="a skill shipped inside a plugin installed from a local marketplace",
        surface=None,
    ),
    Capability(
        label="mcp_external",
        section=4,
        # Server file placed by the package-layer recipe, copied out of
        # `env_mgr/addons/envchk-baseline/`. Declared in the agent's own
        # `.mcp.json`.
        installed_by="recipe",
        origin="zone_config",
        artefact="servers/envchk_baseline_server.py",
        replay="mcp",
        what="an stdio MCP server whose file a recipe installs and whose entry the agent declares",
        surface="mcp__envchk_baseline__envchk_report",
    ),
    Capability(
        label="mcp_stdio",
        section=5,
        installed_by="copied",
        origin="package",
        artefact="assets/env_probe.agent/.claude/tools/envchk_stdio.mcp.py",
        replay="mcp",
        what="a bundled stdio MCP server auto-registered from .claude/tools/*.mcp.py",
        surface="mcp__envchk_stdio__envchk_report",
    ),
    # Section 6 (the in-process `ToolDef` route) is deleted, not moved; an
    # add-on now ships a server that runs on its own instead.
    Capability(
        label="serena",
        section=7,
        installed_by="recipe",
        origin="package",
        artefact="assets/env_probe.agent/serena_probe.py",
        replay="salt",
        what="the real serena, installed by an env_mgr recipe, reading a planted symbol",
        surface="mcp__serena__find_symbol",
    ),
)

LABELS: tuple[str, ...] = tuple(c.label for c in CAPABILITIES)
BY_LABEL: dict[str, Capability] = {c.label: c for c in CAPABILITIES}

#: The key each section's `proof` object must carry. `raw` is a tool's
#: unedited response; `record` is the hook's own output file, parsed;
#: `plugin_list` is `claude plugin list`'s stdout. `skill` has no key here --
#: it has no artefact beyond the token, and `how`/`min_how_chars` cover it.
PROOF_KEYS: dict[str, str] = {
    "hook": "record",
    "plugin": "plugin_list",
    "mcp_external": "raw",
    "mcp_stdio": "raw",
    "serena": "raw",
}

#: The two values `status` may take. `unavailable` is not a synonym for "I did
#: not try": `check_capabilities_genuine`'s `may_be_unavailable` arg names the
#: only capability it is admitted for, and only against a matching non-`ok`
#: entry in the run's own install report.
STATUSES: tuple[str, ...] = ("ok", "unavailable")

#: The top-level keys of `items/text.json`.
REPORT_KEYS: tuple[str, ...] = (
    "nonce_digest",
    "capabilities",
    "install_report",
    "install_report_source",
)


def token(salt: str, label: str, nonce: str) -> str:
    """`ENVCHK-<LABEL>-<12 hex>`.

    Twelve hex characters, short enough to read out of a report by eye.
    """
    digest = hashlib.sha256(f"{salt}:{label}:{nonce}".encode()).hexdigest()[:12]
    return f"ENVCHK-{label.upper()}-{digest}"


def nonce_digest(nonce: str) -> str:
    """What the handoff records instead of the nonce.

    The nonce itself is never written into an artefact; the digest lets the
    validator confirm the agent held the right value without publishing it.
    """
    return hashlib.sha256(f"nonce:{nonce}".encode()).hexdigest()[:12]


def salt_of(path: Path) -> tuple[str | None, str]:
    """The one salt in one artefact, and why not when it is `None`.

    Returns `(salt, reason)`. Zero tags and two-or-more tags are both faults,
    with different reasons.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path}: unreadable: {exc}"
    found = SALT_TAG.findall(text)
    if not found:
        return None, f"{path}: no `ENVCHK_SALT: <32 hex>` tag"
    if len(set(found)) > 1:
        return None, f"{path}: {len(set(found))} different ENVCHK_SALT tags: {sorted(set(found))}"
    return found[0], ""


def server_of(surface: str) -> str:
    """`mcp__<server>__<tool>` -> `<server>`.

    Relates the `.mcp.json` server vocabulary to the brief's tool vocabulary.
    """
    parts = surface.split("__")
    if len(parts) < 3 or parts[0] != "mcp":
        raise ValueError(f"{surface!r} is not mcp__<server>__<tool>")
    return parts[1]
