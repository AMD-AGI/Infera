#!/usr/bin/env python3
"""The token scheme, and the register of the six capabilities.

**This is the one place the derivation is written down as code.** The six
artefacts each carry their own copy of the arithmetic — they have to, because a
`SKILL.md` cannot import anything and an MCP server that imported this module
would have a dependency on the package that installs it — but each of them
carries only *its own* salt, and this module carries *no* salt at all.

That asymmetry is the whole design and it is worth stating before anything else
in this package is read:

| | where it lives | who can read it |
|---|---|---|
| the derivation | here, and inline in each artefact | everyone; it is not a secret |
| the per-capability **salt** | in that capability's artefact, once | whoever reached that artefact |
| the per-run **nonce** | `$ENVCHK_NONCE`, supplied at launch | the agent's environment, and the validator's |

A token is `sha256(f"{salt}:{label}:{nonce}")[:12]`, so producing one requires
both halves. A run whose `env_mgr` installed nothing has no salts, and an agent
that reports six tokens without them is reporting sha256 of something else.

**A central table of salts would destroy this** — one file read would yield all
six — so this module deliberately does not have one. The validator obtains
each salt the same way the agent does: out of the artefact, by the tag below.
Single source of truth, and it cannot drift, because there is only one copy.

**Adding `SALTS = {...}` to this module is a regression, not a tidy-up.** It is
the obvious next edit — a register of six capabilities that does not register
their salts looks incomplete — and it would reduce the whole scheme to a
formality: any agent, having installed nothing, could read one file and report
six correct tokens. If a caller needs a salt it reads the artefact, and if
that is inconvenient the inconvenience is the mechanism working.

## What the tokens do not prove

Written here rather than only in the validator's readme, because this is the
module a reader reaches first. Four of the six artefacts are files an agent
with `Read` can open, so a token proves **the capability was installed and its
artefact was reachable in this zone** — it does not prove the agent obtained it
through the capability. `check_capabilities_genuine` closes that gap for the
two capabilities that are processes, by running them itself, and states the
residual for the rest. Suspend, don't conclude: this module claims installation,
not honesty.
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

    `artefact` is relative to `origin`'s root, not to anything absolute: a task
    package is staged into a zone and an absolute path written here would point
    outside every grant (`env_mgr/prepare.py` step 6a).
    """

    #: The key under `capabilities` in the handoff's `items/text.json`, and the
    #: `label` half of every token derivation.
    label: str
    #: **The section number in the brief, written down rather than derived.**
    #: It used to be this tuple's index plus one. That convention cannot express
    #: a hole, and there is one: capability **6**, the in-process `ToolDef`, was
    #: deleted with the route it measured (`agent_sys/docs/spec.provisioning.md`
    #: §6), and renumbering serena from 7 to 6 would hide the deletion behind a
    #: tidy sequence. A reader counting six sections and seeing 1-5 and 7 asks
    #: the right question.
    section: int
    #: **Which of the two install routes put it in the zone** — the fact this
    #: whole package exists to measure. `spec.provisioning.md` has exactly two
    #: and this field carries their names:
    #:
    #: - ``"copied"`` — §3's copy route: the agent's own
    #:   `assets/<name>.agent/.claude/` tree, copied into the zone config.
    #: - ``"recipe"`` — §2/§4: declared in a recipe and installed by `env_mgr`.
    #:
    #: A section reporting the wrong one is a fault even when its token is
    #: right, because it means the report is describing another route.
    #:
    #: **This replaces `level: "L1"|"L2"|"L3"`, which is gone from this package
    #: entirely.** The levels named an install *hierarchy* that no longer
    #: exists — `spec.provisioning.md`'s opening row says it supersedes that
    #: vocabulary — and the middle rung, a declaration key reaching
    #: `env_mgr/addons/`, was deleted outright.
    #:
    #: **Installed, not declared, and for two capabilities the two differ.**
    #: `serena`'s binary and `mcp_external`'s server file are both installed by a
    #: recipe, and both are *declared* in the agent's own `.claude/.mcp.json`.
    #: This field reports the install; `ACCEPTANCE.md` rows 4 and 7 carry the
    #: other half. Run 1 shipped serena with the install and no declaration and
    #: every `mcp__serena__*` call failed, which is why the distinction is
    #: spelled out here instead of left to the word "delivered".
    installed_by: str
    #: Where `artefact` is rooted:
    #:
    #: - ``"package"`` — relative to `$AGENT_SYS_TASK_PACKAGE`, the staged copy.
    #: - ``"zone_config"`` — relative to `<zone>/config/`, where a recipe placed
    #:   it. Used by `mcp_external`, whose server file this repository ships
    #:   under `env_mgr/addons/` and whose only copy inside the run is the placed
    #:   one.
    origin: str
    #: Where the salt lives, relative to that root.
    artefact: str
    #: How `check_capabilities_genuine` re-derives the token independently.
    #: `mcp` starts the server and speaks the protocol to it; `import` imports
    #: the module and calls the handler; `file` reads the artefact for the
    #: hook's own output; `salt` recomputes from the salt and can do no better,
    #: because the artefact is prose rather than a program.
    replay: str
    #: One line, for a failure message. A validator that says
    #: `capabilities.plugin: token mismatch` and nothing else sends a reader to
    #: this file; saying it in the message saves the trip.
    what: str
    #: The **full name the brief tells the agent to call**, or `None` for the
    #: three capabilities not reached through MCP at all — sections 1, 2 and 3.
    #:
    #: **Written as the brief's name, not as the `.mcp.json` key, and the
    #: provenance is the point.** The same string lives in
    #: `assets/probe_env.task/readme.md` as prose — `mcp__envchk_baseline__…`,
    #: `mcp__envchk_stdio__…`, `mcp__serena__find_symbol` — and until this field
    #: existed nothing could compare that prose to the `.mcp.json` data. Sourcing
    #: the row from the brief makes a **brief-versus-declaration** disagreement
    #: visible too, which a row copied from the `.mcp.json` could never see.
    #:
    #: **The full name and not the bare server key**, because the tool half is
    #: where the mistake actually happens: the brief itself warns that the
    #: in-process tool is `mcp__env_mgr__…` — *"`env_mgr`, not `envchk`, and
    #: getting that wrong is the most common way this section fails"*. A bare
    #: `env_mgr` would drop exactly the half that gets typed wrong. The server
    #: half is recovered with `server_of()` for the checks that need it.
    #:
    #: `label` is the *report* key (`mcp_external`); this is the *registered*
    #: name (`envchk_baseline`), and nothing else in the tree relates the two.
    #: Run 1 is why the field exists: serena installed cleanly, no artefact
    #: declared a server called `serena`, every `mcp__serena__*` call returned
    #: `No such tool available`, and **nothing could ask whether the name had
    #: ever been registered** because nothing recorded what the capability
    #: expected to be called.
    #:
    #: **This is a second statement of a name that also lives in a `.mcp.json`,
    #: and the two can drift.** That is deliberate and is the trade this package
    #: makes everywhere: drift surfaces as a **failed check** naming both sides,
    #: not as a capability that quietly is not there. Do not "remove the
    #: duplication" — removing it removes the only cross-check.
    #:
    #: **Every surviving surface names a server somebody had to declare.** The
    #: one that did not was `mcp__env_mgr__envchk_echo_token` — `env_mgr` was a
    #: *constant* server, present the moment any `ToolDef` was, so a runtime
    #: check asserting only its server half could not fail. That capability is
    #: gone (section 6), and the argument it forced is kept: compare the tool
    #: half, which is the half that gets typed wrong.
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
        # The server FILE is placed by the package-layer recipe
        # (`assets/main.env_recipe.yaml`), which copies it out of
        # `env_mgr/addons/envchk-baseline/`. Its `.mcp.json` entry is the
        # agent's own, which is what section 5 now shares with it.
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
    # **Section 6 was `tooldef` and is deleted, not moved.** It measured the
    # in-process `ToolDef` route — a `.claude/tools/*.tooldef.py` imported into
    # the supervisor and published as `mcp__env_mgr__envchk_echo_token` — and
    # `spec.provisioning.md` §6 deleted that route for component-supplied tools:
    # third-party code executing in the process that supervises every agent, with
    # its memory, file descriptors and credentials, and no boundary to fail
    # closed. Nothing replaces it here, because an add-on now ships a server that
    # runs on its own, which is what sections 4 and 5 already measure.
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

#: The key each section's `proof` object must carry, and the reason there is a
#: table rather than one universal key: what counts as evidence differs by
#: capability, and a single `proof: "..."` string would flatten the difference
#: away. `raw` is a tool's unedited response; `record` is the hook's own output
#: file, parsed; `plugin_list` is `claude plugin list`'s stdout.
#:
#: Two sections are absent from this table on purpose. `skill` has no artefact
#: beyond the token — a skill invocation leaves nothing but its answer — so
#: requiring a key would be requiring the agent to invent one, and an invented
#: proof field is worse than none. It is the brief's `how` that carries that
#: section, and `min_how_chars` is what makes `how` non-trivial.
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

    Twelve hex characters — 48 bits. Enough that a guess is not a strategy, and
    short enough to read out of a report and compare by eye when a validator
    says two of them differ.
    """
    digest = hashlib.sha256(f"{salt}:{label}:{nonce}".encode()).hexdigest()[:12]
    return f"ENVCHK-{label.upper()}-{digest}"


def nonce_digest(nonce: str) -> str:
    """What the handoff records instead of the nonce.

    **The nonce itself is never written into an artefact.** It is a per-run
    secret in the sense that matters here — a handoff carrying it would let the
    next run's report be computed from the previous run's deliverable — and a
    digest lets the validator confirm the agent's environment held the right
    value without publishing it.
    """
    return hashlib.sha256(f"nonce:{nonce}".encode()).hexdigest()[:12]


def salt_of(path: Path) -> tuple[str | None, str]:
    """The one salt in one artefact, and why not when it is `None`.

    Returns `(salt, reason)`. Zero and two are both faults and get different
    reasons: an artefact with no tag cannot be checked at all, and one with two
    is ambiguous about which salt its capability uses.
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

    The declaration routes name **servers**; the brief names **tools**. This is
    the one place the two vocabularies are related, so a check comparing an
    expectation against a `.mcp.json` key and a check comparing it against the
    brief's prose cannot disagree about what the server half is.
    """
    parts = surface.split("__")
    if len(parts) < 3 or parts[0] != "mcp":
        raise ValueError(f"{surface!r} is not mcp__<server>__<tool>")
    return parts[1]
