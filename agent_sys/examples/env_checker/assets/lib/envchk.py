#!/usr/bin/env python3
"""The token scheme, and the register of the seven capabilities.

**This is the one place the derivation is written down as code.** The seven
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
that reports seven tokens without them is reporting sha256 of something else.

**A central table of salts would destroy this** — one file read would yield all
seven — so this module deliberately does not have one. The validator obtains
each salt the same way the agent does: out of the artefact, by the tag below.
Single source of truth, and it cannot drift, because there is only one copy.

**Adding `SALTS = {...}` to this module is a regression, not a tidy-up.** It is
the obvious next edit — a register of seven capabilities that does not register
their salts looks incomplete — and it would reduce the whole scheme to a
formality: any agent, having installed nothing, could read one file and report
seven correct tokens. If a caller needs a salt it reads the artefact, and if
that is inconvenient the inconvenience is the mechanism working.

## What the tokens do not prove

Written here rather than only in the validator's readme, because this is the
module a reader reaches first. Four of the seven artefacts are files an agent
with `Read` can open, so a token proves **the capability was installed and its
artefact was reachable in this zone** — it does not prove the agent obtained it
through the capability. `check_capabilities_genuine` closes that gap for the
three capabilities that are processes, by running them itself, and states the
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
    """One of the seven, and everything a validator needs to judge it.

    `artefact` is relative to `origin`'s root, not to anything absolute: a task
    package is staged into a zone and an absolute path written here would point
    outside every grant (`env_mgr/prepare.py` step 6a).
    """

    #: The key under `capabilities` in the handoff's `items/text.json`, and the
    #: `label` half of every token derivation.
    label: str
    #: Which install level **installed** it — the fact this whole package exists
    #: to measure. A section reporting the wrong level is a fault even when its
    #: token is right, because it means the report is describing another route.
    #:
    #: **Installed, not declared, and for `serena` the two differ.** Its binary
    #: comes from L1 (`recipes: [serena]`) and its MCP registration from L2
    #: (`agent_plugins: [serena]`), so this field reads `L1` and `ACCEPTANCE.md`
    #: row 7 carries the other half. Run 1 shipped with the install and no
    #: declaration and every `mcp__serena__*` call failed, which is why the
    #: distinction is spelled out here instead of left to the word "delivered".
    level: str
    #: `package` — relative to `$AGENT_SYS_TASK_PACKAGE`; or
    #: `component:<name>` — relative to that component's directory in the
    #: repository's `agent_sys/env_mgr/addons/` registry.
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
    #: three capabilities not reached through MCP at all.
    #:
    #: **Written as the brief's name, not as the component's `.mcp.json` key,
    #: and the provenance is the point.** The same string lives in
    #: `assets/probe_env.task/readme.md` as prose — `mcp__envchk_baseline__…`,
    #: `mcp__envchk_stdio__…`, `mcp__env_mgr__envchk_echo_token`,
    #: `mcp__serena__find_symbol` — and until this field existed nothing could
    #: compare that prose to the `.mcp.json` data. Sourcing the row from the
    #: brief makes a **brief-versus-component** disagreement visible too, which
    #: a row copied from the component could never see.
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
    #: `env_mgr` is a **constant** server: it exists the moment any `ToolDef`
    #: does. Statically it is still worth checking — a `*.tooldef.py` declaring
    #: no `TOOLS` publishes nothing — but a *runtime* check asserting only
    #: `env_mgr` would be a check that cannot fail, which is why the runtime
    #: comparison must use the tool half this field preserves.
    surface: str | None


#: The register. **Order is the order a reader meets them in the brief**, and
#: the brief's section numbers are this tuple's indices plus one — a reordering
#: here is a reordering there.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        label="skill",
        level="L3",
        origin="package",
        artefact="assets/env_probe.agent/.claude/skills/envchk-probe/SKILL.md",
        replay="salt",
        what="a skill in the agent's own .claude/skills/",
        surface=None,
    ),
    Capability(
        label="hook",
        level="L3",
        origin="package",
        artefact="assets/env_probe.agent/.claude/hooks/envchk_session_start.py",
        replay="file",
        what="a SessionStart hook declared in .claude/settings.json",
        surface=None,
    ),
    Capability(
        label="plugin",
        level="L3",
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
        level="L2",
        origin="component:envchk-baseline",
        artefact=".claude/servers/envchk_baseline_server.py",
        replay="mcp",
        what="an external MCP server declared in a component's .claude/.mcp.json",
        surface="mcp__envchk_baseline__envchk_report",
    ),
    Capability(
        label="mcp_stdio",
        level="L3",
        origin="package",
        artefact="assets/env_probe.agent/.claude/tools/envchk_stdio.mcp.py",
        replay="mcp",
        what="a bundled stdio MCP server auto-registered from .claude/tools/*.mcp.py",
        surface="mcp__envchk_stdio__envchk_report",
    ),
    Capability(
        label="tooldef",
        level="L3",
        origin="package",
        artefact="assets/env_probe.agent/.claude/tools/envchk_inproc.tooldef.py",
        replay="import",
        what="an in-process ToolDef published as mcp__env_mgr__envchk_echo_token",
        surface="mcp__env_mgr__envchk_echo_token",
    ),
    Capability(
        label="serena",
        level="L1",
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
    "tooldef": "raw",
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
