#!/usr/bin/env python3
"""`check_capabilities_genuine` — trustworthiness, **strong**.

Every token in the report is the token that capability actually produces. Two
of the six are re-derived by **running the capability**: both MCP servers are
started here and spoken to over the protocol. The other four are recomputed from
the salt in the installed artefact.

**It used to be three of seven.** The third was the in-process `ToolDef`, whose
module this body imported and whose handler it called; the route was deleted
(`agent_sys/docs/spec.provisioning.md` §6) and the capability with it. What that
costs is stated in the readme's *What it cannot catch* rather than left for a
reader to notice from the arithmetic.

The readme beside this file states, at length and by capability, exactly what
each of those two treatments proves and what it does not. Read it before
quoting this validator's PASS as evidence of anything.

**No credential is read, echoed or written here, and neither is the nonce.**
`ENVCHK_NONCE` is compared and hashed; its value never reaches a file, a
message or a subprocess argument. The two servers get it in their environment
because that is where they read it from, which is the same place this body got
it.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import envchk  # noqa: E402 — the path insert above is what makes it importable
import zone  # noqa: E402

#: The MCP protocol version this body asks for. A server is free to answer with
#: its own; both servers in this package echo whatever they are given.
PROTOCOL = "2025-06-18"

#: The one tool both MCP servers expose.
MCP_TOOL = "envchk_report"


# ------------------------------------------------------------------ roots


def package_root() -> Path:
    """The staged task package. Both rows of `validator` spec §8.2, in order."""
    for name in ("AGENT_SYS_TASK_PACKAGE", "AGENT_SYS_DEMO_PACKAGE"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    raise SystemExit("check_capabilities_genuine: neither package variable is set")


def zone_config(package: Path) -> Path:
    """`<zone>/config/` — where a recipe places what it installs for the session.

    `<staged package>/../config`. Measured against run 2's tree:
    `fs/layout.PACKAGE` and `material.CONFIG_DIR` are siblings inside the zone,
    so the zone root is the staged package's parent.

    **`$AGENT_SYS_TASK_PACKAGE` and not `$AGENT_SYS_MY_ZONE`**, deliberately:
    `entry.sh` already refuses to start without the former, so a body that is
    running has it *by the fact of running*. Whether a validation zone carries
    `AGENT_SYS_MY_ZONE` is still open and nothing here needs the answer.

    **This function is what replaced a search up the filesystem.** Capability 4's
    artefact used to be read out of `agent_sys/env_mgr/addons/`, found by
    `$AGENT_SYS_ADDONS_ROOT` or, failing that, by walking up from the staged
    package looking for a repository layout — a guess that can find the wrong
    checkout on a machine with two. Neither is needed now: the server file is
    **copied into the zone** by the package-layer recipe, so the only copy this
    run has is one directory away and no environment variable names anything
    outside the zone. That variable no longer exists
    (`agent_sys/docs/spec.provisioning.md` §4).
    """
    return package.parent / "config"


def artefact_of(capability: envchk.Capability, package: Path) -> Path:
    """The file carrying this capability's salt.

    Two roots and no failure case, which is the change: every artefact is now
    inside the run, either in the staged package or in the zone's config
    directory. A missing file is caught where it is read — `replay_mcp` and
    `envchk.salt_of` both name the path — rather than by a root that could not be
    located at all.
    """
    if capability.origin == "zone_config":
        return zone_config(package) / capability.artefact
    return package / capability.artefact


# ------------------------------------------------------------------ replays


def replay_mcp(server: Path, nonce: str, timeout: float) -> tuple[dict | None, str]:
    """Start the server, speak the protocol, return its tool result.

    Three frames on stdin and the pipe closed, rather than an interactive
    exchange: both servers here are stateless and read stdin to EOF, so writing
    everything and reading everything is the whole conversation and it cannot
    deadlock on a server that answers out of order.

    `env` is this body's own with `ENVCHK_NONCE` set explicitly — inheriting it
    would be relying on it already being there, and the point of this call is to
    know what the server computes for *this* nonce.
    """
    if not server.is_file():
        return None, f"{server}: not installed"
    frames = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "check_capabilities_genuine", "version": "1.0.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": MCP_TOOL, "arguments": {}},
        },
    ]
    stdin = "".join(json.dumps(frame) + "\n" for frame in frames)
    environment = dict(os.environ, ENVCHK_NONCE=nonce)
    try:
        completed = subprocess.run(  # noqa: S603 — argv form, no shell
            [sys.executable, str(server)],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"{server.name}: no answer inside {timeout:.0f}s"
    except OSError as exc:
        return None, f"{server.name}: could not start: {exc}"

    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") != 2:
            continue
        if "error" in message:
            return None, f"{server.name}: tools/call returned {message['error']}"
        blocks = message.get("result", {}).get("content", [])
        for block in blocks:
            if block.get("type") == "text":
                try:
                    return json.loads(block["text"]), ""
                except json.JSONDecodeError as exc:
                    return None, f"{server.name}: tool result is not JSON: {exc}"
        return None, f"{server.name}: tools/call returned no text content"
    tail = (completed.stderr or "").strip().splitlines()[-3:]
    return None, f"{server.name}: no response to tools/call (rc {completed.returncode}); {tail}"


def replay_salt(artefact: Path, label: str, nonce: str) -> tuple[dict | None, str]:
    """Recompute from the salt in the artefact. The weakest of the three.

    It establishes that the artefact carrying this salt is installed where the
    capability puts it. It cannot establish that the agent reached it through
    the capability — the readme says so per capability rather than once, because
    the residual differs.
    """
    salt, why = envchk.salt_of(artefact)
    if salt is None:
        return None, why
    return {"token": envchk.token(salt, label, nonce)}, ""


# ------------------------------------------------------------------ the checks


def iso(value: object) -> bool:
    """Whether `value` parses as an ISO-8601 timestamp."""
    if not isinstance(value, str):
        return False
    try:
        datetime.datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def check_liveness(where: str, produced: dict) -> list[str]:
    """`pid` and `at`, on the capabilities that are processes.

    Checked for **shape and nothing else**. There is no run window available to
    this body — it does not know when the session started — so a freshness rule
    here would be a number with no basis. Saying that is better than enforcing
    a guess: see the readme's *What it cannot catch*.
    """
    faults = []
    if not isinstance(produced.get("pid"), int):
        faults.append(f"{where}: replay returned no integer pid")
    if not iso(produced.get("at")):
        faults.append(f"{where}: replay returned no ISO-8601 `at`")
    return faults


def check_hook_payload(where: str, proof: object) -> list[str]:
    """The hook's own output file, as the agent copied it in.

    `payload.session_id` and `payload.hook_event_name` are the two fields only
    the harness supplies. An agent that ran the hook script by hand gets neither
    — the script reads them off stdin — so their absence is the difference
    between *the hook fired* and *the file exists*.
    """
    if not isinstance(proof, dict) or not isinstance(proof.get("record"), dict):
        return [f"{where}.proof.record: missing or not an object"]
    record = proof["record"]
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return [f"{where}.proof.record.payload: missing — the hook was not invoked by the harness"]
    faults = []
    if not payload.get("session_id"):
        faults.append(
            f"{where}.proof.record.payload.session_id: absent. The hook writes "
            f"whatever Claude Code hands it on stdin, so an empty payload means "
            f"the script ran without the harness invoking it"
        )
    event = payload.get("hook_event_name")
    if event != "SessionStart":
        faults.append(f"{where}.proof.record.payload.hook_event_name: {event!r}, needs 'SessionStart'")
    return faults


#: The keys Serena 1.28.1's `find_symbol` puts on each hit. **Measured on this
#: host on 2026-09-03**, by driving the binary probe D installed and calling the
#: tool for real — not remembered, and not read off documentation. The observed
#: response was a JSON array of
#: `{name_path, kind, relative_path, body_location: {start_line, end_line}, body}`.
SERENA_HIT_KEYS = ("name_path", "kind", "relative_path", "body_location", "body")

#: The symbol and the file row 7 plants. A hit naming anything else is a hit for
#: a different question.
SERENA_SYMBOL = "envchk_serena_token"
SERENA_FILE = "serena_probe.py"


def serena_hits(raw: object) -> list[dict] | None:
    """`proof.raw` as a list of `find_symbol` hits, or `None`.

    The tool's result arrives as `content[0].text` holding a JSON **string** of
    the array, so an agent may reasonably paste either the string or the parsed
    array. Both are accepted; anything else is not a `find_symbol` response.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    return [hit for hit in raw if isinstance(hit, dict)]


def check_serena_proof(where: str, proof: object, salt: str) -> list[str]:
    """Row 7's raw response: the measured shape, and the salt inside the body.

    **This is a forgery-cost increase and not a proof, and the distinction is
    the whole reason it is written here rather than sold as a fix.** The salt is
    still in a file an agent can `Read`; what the shape adds is that a forger
    must also know what Serena 1.28.1 answers with. A model that has seen
    serena's schema can still fabricate this.

    **The salt and not the token.** Measured: `find_symbol` with
    `include_body=true` returns the symbol's body and nothing above it. The
    token is not in the response — it is derived from the salt, which
    `serena_probe.py` keeps *inside* the function for exactly this reason.
    Requiring the token here would fail every honest run.
    """
    if not isinstance(proof, dict):
        return [f"{where}.proof: missing or not an object"]
    hits = serena_hits(proof.get("raw"))
    if hits is None:
        return [
            f"{where}.proof.raw: not a find_symbol response. Expected the "
            f"tool's array of hits, or the JSON string of it, unedited"
        ]
    for hit in hits:
        if hit.get("name_path") != SERENA_SYMBOL:
            continue
        faults = []
        missing = [key for key in SERENA_HIT_KEYS if key not in hit]
        if missing:
            faults.append(
                f"{where}.proof.raw: the hit for {SERENA_SYMBOL} lacks {missing}. "
                f"Serena 1.28.1 supplies all of {list(SERENA_HIT_KEYS)} — measured "
                f"2026-09-03. If serena's response schema has changed, this is a "
                f"validator update and not a capability failure"
            )
        if not str(hit.get("relative_path", "")).endswith(SERENA_FILE):
            faults.append(
                f"{where}.proof.raw: relative_path is {hit.get('relative_path')!r}, "
                f"needs to name {SERENA_FILE}"
            )
        location = hit.get("body_location")
        if not isinstance(location, dict) or not all(
            isinstance(location.get(edge), int) for edge in ("start_line", "end_line")
        ):
            faults.append(f"{where}.proof.raw: body_location lacks integer start_line/end_line")
        if salt not in str(hit.get("body", "")):
            faults.append(
                f"{where}.proof.raw: the returned body does not contain the salt. "
                f"The salt is a local inside {SERENA_SYMBOL} precisely so that "
                f"find_symbol's body carries it"
            )
        return faults
    found = sorted({str(hit.get("name_path")) for hit in hits})
    return [f"{where}.proof.raw: no hit for {SERENA_SYMBOL} (found: {found})"]


def check_raw_token(where: str, proof: object, token: str) -> list[str]:
    """The tool response the agent pasted in must carry the token it reported.

    A section whose `token` and whose `proof.raw.token` disagree is a section
    assembled from two different sources, and that is worth catching separately
    from a token that is simply wrong: the message differs and so does the fix.
    """
    if not isinstance(proof, dict):
        return [f"{where}.proof: missing or not an object"]
    raw = proof.get("raw")
    if not isinstance(raw, dict):
        return [f"{where}.proof.raw: missing"]
    if raw.get("token") != token:
        return [f"{where}.proof.raw.token: {raw.get('token')!r} but the section reports {token!r}"]
    return []


def serena_excused(installs: object) -> tuple[bool, str]:
    """Whether the run's own install report says serena failed.

    **The partition is `warn`/`fail`, and the owner is named because the
    vocabulary is not ours.** `env_mgr.outcome.LEVELS` is exactly
    `("ok", "info", "warn", "fail")` — there is no `refused` — and nothing
    revalidates a level downstream, so two consumers of the same file can
    silently disagree about where the line falls. That happened on run 1: this
    body read `warn`/`fail` as failure while a second reader read `!= "ok"`, so
    the benign `info | recipe serena.yaml: OK` line counted as evidence of a
    failed install and would have let an `unavailable` through against a clean
    report. **`info` is not a failure.** Stating the owner and the chosen
    partition is what stops this from being a second vocabulary that merely
    looks like `env_mgr`'s.

    **Deliberately tolerant about the entry's shape and strict about its
    content.** `env_mgr`'s `Outcome` is a level, a message and an extra mapping,
    but the report reaches here through the agent's JSON and this body does not
    own that schema. So an entry counts when its serialised form mentions serena
    and it does not look like a success — which cannot be satisfied by an entry
    that reports serena installing cleanly, and cannot be satisfied at all by an
    install report that never mentions serena.
    """
    if not isinstance(installs, list):
        return False, "install_report is not a list"
    for entry in installs:
        text = json.dumps(entry, default=str).lower()
        if "serena" not in text:
            continue
        level = entry.get("level") if isinstance(entry, dict) else None
        if isinstance(level, str) and level.lower() in ("ok", "info"):
            continue
        if isinstance(level, str) or any(word in text for word in ("warn", "refus", "fail", "error")):
            return True, ""
    return False, (
        "capabilities.serena is 'unavailable' and install_report carries no "
        "non-ok outcome naming serena. The exemption is the install report's to "
        "grant, not the agent's to claim"
    )


def check_capability(
    capability: envchk.Capability,
    section: dict,
    nonce: str,
    package: Path,
    installs: object,
    excused: list[str],
    timeout: float,
) -> list[str]:
    """One capability, end to end."""
    where = f"capabilities.{capability.label}"
    status = section.get("status")

    if status != "ok":
        if capability.label not in excused:
            return [
                f"{where}.status: {status!r}. Only {excused} may be reported "
                f"other than 'ok'; everything else is a capability that must work"
            ]
        ok, why = serena_excused(installs)
        return [] if ok else [why]

    token = section.get("token")
    if not isinstance(token, str):
        return [f"{where}.token: missing — check_env_report_shape reports the shape"]

    artefact = artefact_of(capability, package)

    if capability.replay == "mcp":
        produced, why = replay_mcp(artefact, nonce, timeout)
    else:
        produced, why = replay_salt(artefact, capability.label, nonce)

    if produced is None:
        return [f"{where}: {why}"]

    faults: list[str] = []
    if produced.get("token") != token:
        faults.append(
            f"{where}.token: the report says {token!r} and {capability.what} "
            f"produces {produced.get('token')!r} for this run's nonce"
        )
    if capability.replay == "mcp":
        faults += check_liveness(where, produced)

    proof = section.get("proof")
    # **Row 6b was here and went with row 6.** It asserted that the path the
    # agent reported for the in-process `ToolDef` was the copy placed in this
    # run's zone rather than the component source — the one check in this
    # repository that could see `env_mgr`'s isolation property break for every
    # package that ever shipped a tooldef. The route is deleted
    # (`agent_sys/docs/spec.provisioning.md` §6), so the property has no subject;
    # the readme's *What it cannot catch* records that this validator no longer
    # observes where a placed file was loaded from, because nothing here is
    # loaded any more.
    if capability.label == "serena":
        # The salt again, from the artefact this body already read once for the
        # replay. Read rather than threaded through `replay_salt`'s return: that
        # function's contract is "what the capability produces", and the salt is
        # an input to it, not a result.
        salt, why = envchk.salt_of(artefact)
        faults += [why] if salt is None else check_serena_proof(where, proof, salt)
    elif capability.label == "hook":
        faults += check_hook_payload(where, proof)
        if isinstance(proof, dict) and isinstance(proof.get("record"), dict):
            recorded = proof["record"].get("token")
            if recorded != token:
                faults.append(
                    f"{where}.proof.record.token: {recorded!r} but the section reports {token!r}"
                )
    elif envchk.PROOF_KEYS.get(capability.label) == "raw":
        faults += check_raw_token(where, proof, token)


    return faults


def main() -> int:
    parameters = zone.args()
    timeout = float(parameters.get("replay_timeout_seconds", 30))
    excused = list(parameters.get("may_be_unavailable") or [])

    nonce = os.environ.get("ENVCHK_NONCE")
    if not nonce:
        # **Named, not worked around.** Recomputing from an empty string would
        # produce six mismatches and send a reader looking at the agent.
        print("check_capabilities_genuine: FAIL: $ENVCHK_NONCE is not set in this body's environment")
        zone.write_verdict(dict.fromkeys(zone.inputs(), False))
        return 0

    package = package_root()

    results: dict[str, bool] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            results[hid] = False
            print(f"check_capabilities_genuine: {hid}: no staged content")
            continue
        payload, why = zone.report(content)
        if payload is None:
            results[hid] = False
            print(f"check_capabilities_genuine: {hid}: FAIL: {why}")
            continue

        faults: list[str] = []
        if payload.get("nonce_digest") != envchk.nonce_digest(nonce):
            faults.append(
                "nonce_digest: does not match this run's $ENVCHK_NONCE. The "
                "report was produced against a different nonce, so none of its "
                "tokens is about this run"
            )
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, dict):
            faults.append("capabilities: missing — check_env_report_shape reports the shape")
        else:
            for capability in envchk.CAPABILITIES:
                section = capabilities.get(capability.label)
                if not isinstance(section, dict):
                    faults.append(f"capabilities.{capability.label}: missing")
                    continue
                faults += check_capability(
                    capability,
                    section,
                    nonce,
                    package,
                    payload.get("install_report"),
                    excused,
                    timeout,
                )
        results[hid] = not faults
        for fault in faults:
            print(f"check_capabilities_genuine: {hid}: FAIL: {fault}")
    zone.write_verdict(results)
    print(f"check_capabilities_genuine: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
