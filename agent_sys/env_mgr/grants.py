# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Resolving a `Grant` into real locations. Design §6.

`closure` design D2 handed this step over: a grant references a handoff **kind
name**, `Permissions` never holds a `HandoffId`, and where a runtime component
needs the instance it resolves the name at that point — which is here, at the
moment the zone is built.

The mapping is already on the runtime object (``Handoff.type`` carries the kind
name), so resolution needs no manifest read and no store access.

`task_graph`'s `Grant` and `Access` are read **structurally**. That is not
laziness about types: the two packages deliberately do not import each other
(`task_graph` carries `Permissions` and never interprets it; this module
interprets it and never carries it), and reading ``grant.access`` by value keeps
the edge one-way at run time as well as on paper.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from env_mgr.fs.layout import CONTENT_DIR as _CONTENT_DIR
from env_mgr.fs.layout import handoff_version_dir
from env_mgr.fs.path import canonical_here, canonical_syntax
from env_mgr.protocols import Granted, Mode, UnresolvedGrant

__all__ = [
    "INPUT_ENV_PREFIX",
    "OUTPUT_ENV_PREFIX",
    "input_env",
    "mode_for",
    "output_env",
    "output_paths",
    "resolve",
    "resolve_all",
]

_WRITE = "write"


def mode_for(access: Any) -> Mode:
    """`task_graph.Access` → `Mode`. **The seam between two vocabularies.**

    `Access` answers *what did the package author declare* — read or write.
    `Mode` answers *what rights does the kernel get* — combinable, and
    `READ_EXEC` has no declaration-side meaning at all. Rev. 1 of the design
    called both `Access` and mixed them in one `Policy`; making the mapping a
    named function is what stops that being a name collision that happens to
    type-check because neither is annotated.
    """
    value = getattr(access, "value", access)
    return Mode.READ_WRITE if str(value).lower() == _WRITE else Mode.READ_EXEC


def _versions(task: Any, execution: Any) -> dict[Any, int]:
    versions = dict(execution.input_versions)
    versions.update(execution.output_versions)
    return versions


def _slots(task: Any) -> tuple[Any, ...]:
    return tuple(task.inputs) + tuple(task.outputs)


def resolve(
    grant: Any,
    task: Any,
    execution: Any,
    handoffs: Mapping[Any, Any],
    store_root: str,
) -> tuple[Granted, ...]:
    """One grant → the locations it names, for **this attempt**.

    A kind-named grant resolves **inside** ``<store_root>/<hid>/v<N>/`` for
    every instance of that kind this attempt has — see `_version_paths` for
    which subdirectories and why it is not the version directory itself. ``N``
    lives on the `Execution`, so a retry gets a different granted set.

    **Two kinds of "nothing", and only one is an error.** The discriminator is
    `Task.kinds` — whether this task *declares* a slot of that kind:

    | | |
    |---|---|
    | the task declares the kind and it does not resolve | **raises** — the forgotten `Handoff.type` defect, silent before |
    | the task declares no such kind | returns `()` — a grant inherited for a sibling's kind, and `interfaces.md` §4.16 makes permissions **wide** |

    The raise is why rev. 1's silent empty granted set is gone. The no-op is why
    inheritance works: grants come wholesale from a root, so a subgraph member
    routinely carries grants for kinds its siblings produce, and **a wide model
    that raises on its own width cannot be used with inheritance**.
    """
    mode = mode_for(grant.access)
    kind = grant.kind
    if not kind:
        return (_literal(grant, mode),)

    versions = _versions(task, execution)
    out: list[Granted] = []
    matched: list[Any] = []  # this kind, but no version pinned on this attempt
    for hid in _slots(task):
        handoff = handoffs.get(hid)
        if handoff is None or handoff.type != kind:
            continue
        matched.append(hid)
        version = versions.get(hid)
        if version is None:
            continue
        out.extend(_version_paths(handoff_version_dir(store_root, hid, version), mode))
    if not out:
        if not _participates(task, kind):
            # **A grant for a kind this task has no part in is a no-op, not an
            # error**, and that follows from `interfaces.md` §4.16's rescoping:
            # permission management is *wide*, and its only job is to stop
            # agents cross-contaminating. Grants are inherited wholesale from a
            # root, so a subgraph member routinely carries grants for kinds its
            # siblings produce — measured by `demo`, two of three subtasks.
            #
            # **A wide model that raises on its own width cannot be used with
            # inheritance.** And a permission for something absent grants
            # nothing, so it is not a cross-contamination risk either.
            return ()
        raise UnresolvedGrant(_why(kind, matched, task, execution, handoffs))
    return tuple(out)


#: The artefact. Re-exported from `fs/layout.py`, which is where it is now
#: defined: `stage` needs it and `layout` sits below this module, so keeping the
#: literal here would have meant two spellings of it *inside* `env_mgr` — which
#: is the duplication `interfaces.md` §8.1 tolerates across a package boundary
#: and not within one. The name is unchanged, so
#: `test_the_store_layout_names_match` still pins it against `handoff`'s.
CONTENT_DIR = _CONTENT_DIR

#: The producing agent's `done_by_self_check` claim. **Named here**, because the
#: user ruled the destination and left the name to this module.
#:
#: Spelled again in `handoff/store.py::CLAIM_DIR`, which is where `allocate`
#: creates it — §8.1's forced duplication, the same as `CONTENT_DIR`.
#:
#: A directory rather than a file, so a second claim needs no second ruling, and
#: a *sibling* of `content/` rather than a child. Measured
#: (`scratch/impl-2026-08/env_mgr/p9_a_sibling_claim_dir_survives_seal.py`, and
#: `handoff`'s `probe_claim_location.py` from the other side): a sibling survives
#: `seal` and leaves the digest byte-identical at ``718d7aeb31a76c32…``, where
#: putting it *inside* `content/` moves the digest — the claim would become part
#: of the artefact's identity, so the same artefact claimed differently would be
#: a different artefact.
CLAIM_DIR = "claim"


def _version_paths(version_dir: str, mode: Mode) -> tuple[Granted, ...]:
    """One output slot → **two** granted paths. The user's ruling.

    Not ``v<N>/`` itself, and this is the correction that makes §4.14 safe:
    under it the **manifest is the seal**, so an agent granted the version
    directory could write `manifest.yaml` and publish its own unsealed version.
    `validation.yaml` is out of reach for the same reason.

    | | |
    |---|---|
    | ``v<N>/content/`` | the artefact. Created by `handoff.allocate`, because a granted path must exist when the ruleset is built |
    | ``v<N>/claim/`` | the producing agent's self-check claim, **write only** |

    **A read grant gets `content/` alone.** A consumer has nothing to claim, and
    an input's claim is the producer's rather than the reader's.

    *Whether a body needs to read its input's `manifest.yaml`* was raised here
    as open and **`handoff` has closed it: no.** Verified from this side rather
    than relayed — `agent/gate.py:91` is the only `get_manifest` caller outside
    `handoff` and its own tests, and `runner.py:574` reaches it **after the
    executor returns**, in the supervisor's process rather than inside the
    confinement. And by design: spec §6.3 has a consumer work on a copy and
    `copy_out` verifies the digest *before returning*, so integrity arrives as
    content the body can trust rather than a manifest it must check. A body that
    verified its own input would reimplement `copy_out`'s one job in the one
    place an agent could skip it.

    It reopens only for a consumer that needs **provenance**, and `handoff`'s
    answer to that is an operation on the store rather than a wider grant.

    **Both are created by `handoff.allocate`, and this function creates
    nothing.** §4.18 ruled it: *the allocator creates every directory it expects
    to be granted*. A granted path that does not exist is either a
    `FileNotFoundError` that kills every output dispatch — `landlock.py:198`
    opens every granted path and `Granted.optional` defaults `False` — or, if
    made optional, a rule dropped silently. **And the agent cannot create it
    either**, since `mkdir` inside `v<N>/` needs write on `v<N>/`, which is
    exactly what the narrowing removed.

    This briefly did create `claim/` itself, because the user left the *name*
    here and `handoff` could not act until it had one. `handoff/store.py:334`
    now does it in `allocate`, so the bridge is gone: **a resolver with a side
    effect is a resolver a test cannot call twice**, and `allocate`'s `os.mkdir`
    is not `exist_ok`.
    """
    content = os.path.join(version_dir, CONTENT_DIR)
    if not (mode & Mode.READ_WRITE):
        return (Granted(content, mode),)
    return (Granted(content, mode), Granted(os.path.join(version_dir, CLAIM_DIR), mode))


def _participates(task: Any, kind: str) -> bool:
    """Does this task have any slot **declared** as `kind`?

    `Task.kinds` is the declaration — slot to kind name — and it is what
    separates *"a grant inherited for a sibling's kind"* from *"a grant for my
    own kind that failed to resolve"*. Only the second is a defect.

    **An empty `kinds` cannot distinguish them**, so it answers `True` and the
    raise stands. That is deliberate: an unfilled `kinds` is precisely the
    forgotten-`Handoff.type` bug the raise was added to catch, and silently
    skipping there would restore the empty-granted-set failure it replaced.
    """
    kinds = task.kinds
    return not kinds or kind in set(kinds.values())


def _why(
    kind: str, matched: list[Any], task: Any, execution: Any, handoffs: Mapping[Any, Any]
) -> str:
    """Say **which** of the two conditions was unmet.

    Both branches used to fall into one raise with one message, so a caller who
    fixed the first and re-ran got a byte-identical error and had to write a
    probe to discover their fix had worked. One read instead of one run, and the
    information was already in hand.

    The same rule as everywhere else here: *an error about something absent
    should name where it looked.*
    """
    where = f"task {task.id} attempt {execution.attempt}"
    if matched:
        return (
            f"grant for kind {kind!r} on {where}: {len(matched)} slot(s) of that kind "
            f"exist and none has a version on this attempt — "
            f"{[str(h) for h in matched]}. A version is pinned for an input at "
            f"dispatch and for an output only when the attempt closes, so an output "
            f"grant cannot resolve before the output is written."
        )
    # `.get` may miss, and a missing slot is a different fact from an unset
    # kind — so it is spelled out rather than folded into a getattr default.
    # A third cause, added because `demo` ruled the other two out with a probe
    # before they could see what was happening: the message named two causes and
    # the real situation was neither. It is now impossible to reach this branch
    # with a kind the task does not declare — that returns `()` — so what is left
    # is a declared kind whose slot did not resolve.
    known = {
        str(h): (handoffs[h].type if h in handoffs else "<no such handoff>") for h in _slots(task)
    }
    return (
        f"grant for kind {kind!r} on {where}: no slot has that kind. "
        f"Slot kinds are {known}. An empty or missing kind means `Task.kinds` did "
        f"not reach `Handoff.type`, and a `Context.handoffs` that is a snapshot "
        f"rather than a live view is empty for every handoff declared after it."
    )


def _literal(grant: Any, mode: Mode) -> Granted:
    """A grant naming a path rather than a kind.

    Design D4: a grant path must already **be** its own canonical form. Exact
    string equality (the covering relation) and realpath disagree on every form
    tried — ``zone`` versus ``zone/``, ``zone/.``, a symlink, ``zone/../out`` —
    and always in the direction `closure` §6.3 forbids, with ``covers()`` saying
    "not covered" while this module would grant. Requiring canonical form makes
    the two agree by construction rather than by care.
    """
    path = grant.path
    if not canonical_syntax(path):
        raise UnresolvedGrant(
            f"grant path {path!r} is not in canonical form: it must be absolute, "
            f"with no '.' or '..' segment, no trailing or repeated separator, no "
            f"NUL and no wildcard"
        )
    if os.path.exists(path) and not canonical_here(path):
        raise UnresolvedGrant(
            f"grant path {path!r} is not its own realpath; granting the canonical "
            f"form would grant something the covering relation believes it did not"
        )
    return Granted(path, mode)


def resolve_all(
    task: Any, execution: Any, ctx: Any, *, enforce: bool = True
) -> tuple[Granted, ...]:
    """Every grant on ``task.permissions``, flattened. `prepare`'s only caller.

    The wrapper exists because `resolve` needs a handoff mapping and a store
    root that `Context` carries and a single `Grant` does not know about.

    `enforce=False` is the kill switch's step 2 — **resolve best-effort, never
    raise.** A run that asked for no permission management must not be stopped
    by a permission that failed to resolve, and the granted set is not enforcing
    anything in that mode anyway. What it still feeds is `executable_path`, so
    resolving as much as possible is better than resolving nothing.

    The switch itself is read in `prepare`, once, and arrives here as an
    argument: a switch with three readers is three switches.
    """
    out: list[Granted] = []
    for grant in task.permissions.grants:
        try:
            out.extend(resolve(grant, task, execution, ctx.handoffs, ctx.store_root))
        except UnresolvedGrant:
            if enforce:
                raise
    return tuple(out)


#: ``AGENT_SYS_OUTPUT_<KIND>`` — where this attempt's output of that kind goes.
#:
#: `demo`'s F-D17, and it is `AGENT_SYS_TASK_PACKAGE`'s argument one slot over:
#: a path known only at prepare time that the body cannot compute and must have.
#: The granted output directories exist and are granted, and until this they
#: lived only in `prepared.policy.granted`, which no body ever sees.
#:
#: **Keyed by kind because that is the author's only handle.** A closure
#: declares ``outputs: ['facts']`` — a list of kind names, not slots — so the
#: kind name is the one identifier a body can be written against. Keying by
#: `HandoffId` would name it by a uuid minted at submit, which no author can
#: write down.
OUTPUT_ENV_PREFIX = "AGENT_SYS_OUTPUT_"

#: ``AGENT_SYS_INPUT_<KIND>`` — the **staged copy** of that input, in the zone.
#: `output_env`'s mirror; see `input_env` for why the asymmetry was an oversight
#: rather than a decision.
INPUT_ENV_PREFIX = "AGENT_SYS_INPUT_"


def output_env(task: Any, execution: Any, store_root: str) -> dict[str, str]:
    """The declared name for each output's `content/`, for the body that writes it.

    Only outputs with a version pinned on **this** attempt, which since §4.14 is
    all of them at dispatch. The value is the `content/` subdirectory rather
    than ``v<N>/``, so it agrees with `_version_paths` by construction: an
    exported path the policy does not grant would be the evaporating allow-list
    one level up, and a body would fail on a path we told it to use.

    **A kind naming two output slots is exported for neither, and that is a
    hole rather than a resolution.** `env_mgr` could discriminate them by
    `HandoffId`, and it would be inventing a naming scheme **no author can write
    against**: the declaration is a list of kind names, so an author with
    ``outputs: ['facts', 'facts']`` has no way to address one of them either.
    Choosing one silently is the failure `demo` just hit from the other side —
    an output never written, surfacing as `output_absent` with no cause. Nothing
    is exported, the body's own refusal fires, and the gap is named here and
    reported rather than papered over.

    Two kinds that collide only after being made into a variable name — say
    ``my-facts`` and ``my_facts`` — are dropped by the same rule and for the
    same reason.
    """
    resolved = output_paths(task, execution, store_root)
    rows = [(hid, task.kinds.get(hid), path) for hid, path in resolved.items()]
    return _by_unique_kind(OUTPUT_ENV_PREFIX, rows)


def output_paths(task: Any, execution: Any, store_root: str) -> dict[Any, str]:
    """Every output slot with a version pinned → its ``content/`` directory.

    **Keyed by `HandoffId`, and that is what makes it different from
    `output_env` rather than a duplicate of it.** `agent` asked for it under a
    ruling that a runner must state each declared output, its kind, **and its
    resolved path**, in the conversation — because an environment variable
    cannot instruct a model and a readme cannot name a per-attempt path.

    **It closes the hole `output_env` names and declines to close.** A kind
    naming two output slots is exported for neither, because an author who wrote
    ``outputs: ['facts', 'facts']`` cannot address either one *by name*. A slot
    id has no such collision: it addresses slots, not names. So the **agent** can
    be told about both, which is exactly the case the ruling's constraint is
    about — an agent told about two of three outputs writes two and finishes
    successfully. A shell **body** still has only the variable, so the hole is
    closed for the reader that can use an id and stays open for the one that
    cannot; that is the honest state and not a full fix.

    **A slot with no version pinned is absent**, not present-and-empty. `agent`
    enumerates `task.outputs` itself and renders the difference as *"no resolved
    path"* rather than skipping it — so absence here is unambiguous, and each
    side reads only what it owns.
    """
    versions = dict(execution.output_versions)
    out: dict[Any, str] = {}
    for hid in tuple(task.outputs):
        version = versions.get(hid)
        if version is None:
            continue
        out[hid] = os.path.join(handoff_version_dir(store_root, hid, version), CONTENT_DIR)
    return out


def input_env(task: Any, staged: Mapping[Any, str]) -> dict[str, str]:
    """The declared name for each **staged input**, for the body that reads it.

    `demo` reported the asymmetry — outputs had a declared name and inputs did
    not — and asked whether it was deliberate, since an output is a path a body
    must *create* into while an input might reasonably be handed differently.
    **It was not deliberate.** It is the same gap one slot over, and the proof
    was already in the code: `prepare` called `stage_handoffs`, which returns
    handoff id → staged path, and **threw the mapping away**. That is
    `engineer_principle.md` §4.4's named smell in the direction that hurts —
    discarding an association the module had in hand, leaving the only way to
    find a staged input to be parsing this module's directory layout, which
    `examples/demo/logic/store.py` had already become the second reader of.

    The value is the **staged copy in the zone**, not the store path, because
    spec §6.3 rule 2 is that an agent works on a copy — the store path is a
    place a body must not read from.

    Same kind-keying and the same collision rule as `output_env`, and the
    collision is *more* likely here: `validator` spec §4.1 makes many-to-many
    first class, so a body taking several inputs of one kind is ordinary rather
    than exotic. It is still exported for neither, because an author who wrote
    two of a kind cannot address either one by name.

    **This and `output_env` point at the same level**, and it is worth one line
    because the two path *shapes* do not look like it:
    ``<zone>/handoffs/<hid>/v<N>`` against ``<store>/<hid>/v<N>/content``. Since
    `stage` narrowed, it copies ``<v>/content`` **to** ``<into>/<hid>/v<N>``, so
    a body finds the artefact's own files directly at the end of either name and
    there is no ``content/`` hop on the input side. `demo` read the shapes and
    reported them as one directory apart, which they were before the narrowing;
    `test_the_two_declared_names_point_at_the_same_level` pins the property
    rather than either spelling.
    """
    return _by_unique_kind(
        INPUT_ENV_PREFIX, ((hid, task.kinds.get(hid), path) for hid, path in staged.items())
    )


def _by_unique_kind(prefix: str, rows: Any) -> dict[str, str]:
    """kind → path, dropping any name two rows would claim.

    **Counted rather than trusted.** `_env_name` is lossy on purpose and its
    inverse is never taken, so injectivity is checked by collecting and
    measuring instead of assumed from the mapping.
    """
    by_name: dict[str, list[str]] = {}
    for _hid, kind, path in rows:
        if not kind or not path:
            continue
        by_name.setdefault(prefix + _env_name(kind), []).append(path)
    return {name: paths[0] for name, paths in by_name.items() if len(paths) == 1}


def _env_name(kind: str) -> str:
    """A kind name as an environment variable suffix. Uppercased, non-alphanumerics to ``_``.

    Lossy on purpose — the inverse is never taken. What matters is that two
    distinct kinds mapping to one name are **detected**, which `_by_unique_kind`
    does by counting rather than by trusting this to be injective.
    """
    return "".join(c if c.isalnum() else "_" for c in kind).upper()
