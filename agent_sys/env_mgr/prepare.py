# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Preparing an environment. Design §11. **The order is the design.**

Step 7 is last and that is load-bearing twice over. Everything before it writes
outside the zone by design — creating the zone, cutting the workspace, staging
handoffs — and none of it is possible afterwards. And the supervisor and every
process that already exists are outside the resulting Landlock domain, so the
kernel's ptrace hook protects them from the agent: the supervisor's environment,
which on a real machine holds API keys, is protected by **ordering**, not by any
filesystem rule.

**Why so much of this file is annotated `Any`, stated rather than imitated.**
This module names types from **its own package** and leaves every type belonging
to another package as `Any`, because `env_mgr` does not import `task_graph`,
`agent` or `validator` and an import edge is permanent where a structural read
is not (`grants.py` carries the long form of the argument). So:

| `Any` here | belongs to |
|---|---|
| `task`, `execution` | `task_graph` — `Task`, `Execution` |
| `output_paths`' keys | `task_graph` — `HandoffId`; `protocols.py` may name it, this module may not |
| `agent_spec` | `agent` |
| `phase` | `validator` — `PhaseKind`, read for its value |
| `workspace` | nobody: `Any` in the declaration too, so not a divergence |
| `**popen_kwargs` | genuinely open |

**An `Any` that is not on that list is a defect, not a convention.** `zone` was
one — `env_mgr.fs.zone.Zone` is intra-package and nameable, and it was `Any`
only because it sat beside `output_paths`, which must be. Ruled and narrowed.
Anything else here that names an intra-package type as `Any` should be narrowed
the same way rather than read as following a rule.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, NamedTuple

from env_mgr import grants, material, paths
from env_mgr import sync as _sync
from env_mgr import workspace as _workspace
from env_mgr.fs import layout
from env_mgr.fs.path import contained, resolve_strict
from env_mgr.fs.zone import Zone
from env_mgr.isolation import apply as _apply
from env_mgr.isolation import bwrap as _bwrap
from env_mgr.isolation import landlock
from env_mgr.isolation.policy import (
    DEFAULT_SYSTEM_SET,
    Granted,
    Mode,
    Policy,
    agent_cli_grants,
    component_grants,
    executable_path,
)
from env_mgr.isolation.probe import Availability, probe, select
from env_mgr.protocols import Confinement, Context, NoConfinement, PrepareRefused, SyncReport
from env_mgr.remote import tools as _tools
from env_mgr.sync import Direction

__all__ = [
    "NO_PERMISSIONS_ENV_VAR",
    "PACKAGE_ENV_VAR",
    "EnvManager",
    "Prepared",
    "ValidationZone",
    "place_zone",
    "prepare",
    "permissions_enforced",
    "prepare_validation",
]

#: **The switch, and it is on by default.** Truthy — or *absent* — and this run
#: performs no permission management at all. Spell it `0` to enforce.
#:
#: Ruled by the user for a demo bring-up, and **re-ruled 2026-08-30 to make off
#: the default** rather than something an operator opts into (`interfaces.md`
#: §4.22f). What it turns off is **confinement and grant enforcement** and
#: nothing else: staging, the agent spec's material, the workspace,
#: `staged_package` and the whole of `environment` are *materialisation* — they
#: make a file appear where a task needs it — and they are unchanged. If
#: disabling something stops a task finding its own inputs, the line has been
#: crossed.
NO_PERMISSIONS_ENV_VAR = "AGENT_SYS_NO_PERMISSIONS"

#: What an **unset** variable reads as. This one constant is the default, and
#: flipping it is the whole of the 2026-08-30 ruling.
_UNSET_READS_AS = "1"

#: Falsy spellings, so `AGENT_SYS_NO_PERMISSIONS=0` does not read as "on".
#: An operator who writes `0` means off, and a switch that treats every
#: non-empty string as *set* would enforce nothing for exactly the person trying
#: to say "no thanks". **Since the default flipped these are the only way back to
#: enforcement**, which is why the set is generous rather than just `"0"`.
_FALSE = frozenset({"", "0", "false", "no", "off"})


def permissions_enforced(environ: Mapping[str, str] | None = None) -> bool:
    """**The only place the switch is read.** A switch with three readers is three.

    `interfaces.md` §4.16's neighbourhood, and the reason it is a function rather
    than a constant: a module-level read is taken at import, so a test that sets
    the variable would be answered by whatever the environment held when the
    first import happened.

    **Unset means not enforced**, since 2026-08-30. Enforcement is still fully
    available and is reached by spelling the switch off —
    ``AGENT_SYS_NO_PERMISSIONS=0`` (or ``false`` / ``no`` / ``off`` / empty).

    **Why the default moved rather than the variable.** A companion opt-in
    (``AGENT_SYS_ENFORCE_PERMISSIONS=1``) would be two variables that can
    disagree, and two variables need a precedence rule that a reader has to
    look up — against §4.22's own *a switch with three readers is three
    switches*. A rename would break every operator command line, the exported
    `NO_PERMISSIONS_ENV_VAR`, three documents and a structural test, and buy no
    behaviour. So: same variable, same spellings, same single reader, one
    changed default. The cost is a double negative — you set *no permissions* to
    *0* to get permissions — and it is stated here because it is the one
    surprising thing about this shape.
    """
    env = os.environ if environ is None else environ
    return env.get(NO_PERMISSIONS_ENV_VAR, _UNSET_READS_AS).strip().lower() in _FALSE


#: Where the staged task package went, for whoever launches the body.
#:
#: **Re-exported, not defined.** It moved to `paths.py` when it stopped being the
#: only zone-path variable and became one member of a family; defining it in both
#: places would give one fact two writers (§1). The name is imported from here by
#: `tests/cli/test_isolation_shown.py`, so this line is the compatibility half
#: and is not decoration.
PACKAGE_ENV_VAR = paths.PACKAGE_ENV_VAR

#: A run with no remote mapping has nothing to sync and no conflict to find.
_NO_SYNC = SyncReport(sent=0, received=0, conflicts=())

#: An empty mapping that cannot be written through. A `dict` default on a
#: NamedTuple field is shared by every instance, which is one edit away from one
#: task's environment leaking into another's.
_NO_ENV: Mapping[str, str] = MappingProxyType({})

#: The same guarantee for `output_paths`, whose keys are `HandoffId` rather than
#: `str`. Reusing `_NO_ENV` was behaviourally identical and wrong twice: it
#: annotated a slot-keyed mapping as `Mapping[str, str]`, and it named a field
#: that is not an environment *no environment*. Reported by `env-mgr`.
_NO_PATHS: Mapping[Any, str] = MappingProxyType({})

#: And again for `mcp_servers`, for the reason `_NO_PATHS` is not `_NO_ENV`: the
#: values are the SDK's server declarations, not strings, and a shared name for
#: three differently-typed empties is a mis-annotation waiting for whoever reads
#: the third one.
_NO_MCP: Mapping[str, Any] = MappingProxyType({})


class Prepared(NamedTuple):
    """What the runner is handed.

    The first five fields are `protocols.Prepared`'s, in order, so positional
    unpacking against the frozen declaration keeps working. The sixth and the
    method are additions the caller needs and the declaration does not yet have
    — reported to `main`, and see this package's README.

    `confinement` is here because the chain's degradation must not be invisible:
    bubblewrap isolates network and PID, Landlock below ABI 4 isolates neither.
    """

    #: `Zone`, not `Any`. `env-mgr` found this beside `output_paths` in an
    #: exemption list and declined the pairing: `output_paths` **must** say
    #: `Mapping[Any, str]`, because naming `HandoffId` would import `task_graph`
    #: into this module across the one-way seam. `Zone` is
    #: `env_mgr.fs.zone.Zone` — intra-package, no seam, nothing forced. It was
    #: `Any` because it sat next to something that had to be.
    zone: Zone
    #: `Any` on **both** sides, so not a divergence: `workspace.cut` returns
    #: whatever the worktree helper hands back and neither declaration narrows it.
    workspace: Any
    policy: Policy
    confinement: Confinement | None
    sync: SyncReport
    environment: Mapping[str, str] = _NO_ENV
    agent_cli: str | None = None
    #: **False when this run performed no permission management at all.**
    #:
    #: `confinement is None` already means *unconfined* and would now mean it for
    #: two different reasons — the machine had no mechanism, or somebody set
    #: `AGENT_SYS_NO_PERMISSIONS`. That is one reason too many, and
    #: `interfaces.md` §4.17a is the rule it breaks: **a fact a reader has to
    #: infer is a fact a reader can miss.** So the switch is *stated* rather than
    #: recoverable, and `demo` can print it and `agent` can read it without
    #: either of them learning the variable's name.
    #:
    #: **The default is `False` since 2026-08-30, and it follows the switch's.**
    #: `prepare` always passes this explicitly, so the default only answers a
    #: hand-built `Prepared` — and *"claim the ordinary case"* is the rule that
    #: chose `True` originally (§4.22a). The ordinary case is now unenforced, so
    #: the same rule now says `False`. It is not merely cosmetic: a `True` here
    #: propagates to `Assignment.permissions_enforced` and leaves the Claude
    #: SDK's own approval layer in ask-mode with no approval channel, which is
    #: **more** restrictive than enforcement, not less (`agent/backend.py`).
    permissions_enforced: bool = False
    #: Every output slot with a version pinned → its `content/` directory.
    #:
    #: **Keyed by `HandoffId`, which is the point.** `agent` must state each
    #: declared output, its kind and its resolved path *in the conversation* —
    #: an environment variable cannot instruct a model. `AGENT_SYS_OUTPUT_<KIND>`
    #: cannot serve: reading it from `agent` means copying this module's prefix
    #: and kind-keying across a boundary neither side checks, and a kind naming
    #: two slots is exported for neither.
    #:
    #: A slot with no pinned version is **absent** rather than present-and-empty.
    #: `agent` enumerates `task.outputs` itself and renders the difference as
    #: *"no resolved path"*, which the ruling requires — an agent told about two
    #: of three outputs writes two and finishes successfully.
    output_paths: Mapping[Any, str] = _NO_PATHS
    #: Where the task package was staged, or `None` when none was configured.
    #:
    #: **`agent_cli`'s precedent, and `agent` asked for it with the measurement**
    #: — `exit 2: /bin/sh: 0: cannot open .../bodies/produce/entry.sh: Permission
    #: denied`. The same value is in `environment[PACKAGE_ENV_VAR]` and that is
    #: right for a *body*, which reads a variable out of its own process
    #: environment where the name **is** the interface. It is wrong for
    #: `agent.Runner`, which is building an argv and would be reaching into a
    #: mapping for a key spelled by a name copied across a boundary neither side
    #: checks.
    #:
    #: **Named `staged_package`, not `package`.** `Context.package` is the
    #: original checkout and this is the copy — same type, different path, and a
    #: shared name would make substituting one for the other silent. That is the
    #: rename-on-incompatible-change rule applied to meaning rather than type.
    staged_package: str | None = None
    #: This attempt's far-side tool surface — `remote.tools.ToolDef`s, or `()`.
    #: Declared in `protocols.Prepared` too; this is the implementation half and
    #: the two are compared by `tests/interfaces/`.
    #:
    #: **Since per-agent components, it also carries theirs.** A component's
    #: `tools/*.tooldef.py` publishes the same `ToolDef` shape, and the backend
    #: adapts every element of this tuple into one in-process MCP server — so
    #: there is nothing to key them by and no consumer that needs to tell a
    #: remote tool from a component's.
    tools: tuple[Any, ...] = ()
    #: **External MCP servers for this attempt**, keyed by the name the model
    #: addresses them under, as the SDK's `mcp_servers` option spells them.
    #:
    #: Separate from `tools` because they are a different *kind* of thing and
    #: not a different source of the same thing: a `ToolDef` is a Python object
    #: this process calls, and one of these is a declaration of a **process to
    #: start** — `{"type": "stdio", "command": …}` — that the harness starts and
    #: this one never sees. Merging them would mean inventing a discriminator to
    #: split them again at the backend.
    #:
    #: Typed `Mapping[str, Any]` for `tools`' reason: the values are the SDK's
    #: vocabulary, and `agent` may not import this package to learn it.
    mcp_servers: Mapping[str, Any] = _NO_MCP

    def spawn(self, argv: Sequence[str], **popen_kwargs: Any) -> subprocess.Popen:
        """Start `argv` **confined**, and hand back the process. One verb.

        `wrap_argv`'s shape does not carry over to Landlock and this is why:
        bubblewrap *is* the exec, so its confinement crosses the fork/exec
        boundary **as data** in a command line. Landlock is a syscall against a
        live thread, so there is nothing to put in an argv — it has to be
        executed in the child, after fork, before exec.

        Three cases, and the caller branches on none of them:

        | mechanism | |
        |---|---|
        | bwrap | the argv carries the policy; an ordinary `Popen` |
        | landlock, this process already confined | plain `Popen` — the child inherits the domain |
        | landlock, this process not confined | the ruleset is built **here** and applied in the child |

        **The child's whole job is two syscalls, and that is deliberate.**
        Forking a threaded process gives the child locks held by threads that do
        not exist in it, which is the documented reason `preexec_fn` is unsafe —
        and `agent.Runner` is threaded by construction. A ruleset fd survives
        fork, so building it in the parent leaves the child with `prctl` and
        `restrict_self` and nothing that allocates.

        Measured under four deliberately contending threads, 150 rounds each
        (`scratch/impl-2026-08/env_mgr/p4_fork_from_a_threaded_parent.py`): no
        hangs in this arrangement, and none in the two neighbouring ones either.
        **That is evidence and not proof** — a fork deadlock is probabilistic
        and CPython warns about the pattern on principle. What the measurement
        does establish is that the post-fork footprint is as small as it can be
        made.

        The same probe found the cost: building the ruleset in a *threaded*
        parent is GIL-bound and about 15x slower than building it in a
        single-threaded child. One spawn per task, so it is a real cost and a
        payable one.
        """
        if not self.permissions_enforced:
            # The kill switch. Without this the next line calls `select(probe())`
            # and raises `NoConfinement` on a machine with no mechanism — a live
            # exception on the path of a run that asked for no permission
            # management at all, which would make the switch fail closed in the
            # one mode whose whole point is not to.
            return subprocess.Popen(list(argv), **popen_kwargs)
        mechanism = (
            self.confinement.mechanism if self.confinement else select(probe())
        )  # raises NoConfinement; the task does not start
        if mechanism == "bwrap":
            return subprocess.Popen(self.wrap_argv(argv), **popen_kwargs)

        # Always applied here, because since the split `prepare` never does. If
        # this process happens to be confined already the child simply gets a
        # second identical layer, which costs one of sixteen and changes
        # nothing: layers intersect.
        ruleset = landlock.build(self.policy)
        try:
            return subprocess.Popen(
                list(argv), preexec_fn=lambda: landlock.restrict(ruleset), **popen_kwargs
            )
        except OSError as e:
            raise self._blame(e, argv) from e
        finally:
            # `restrict` closed the child's copy; this is the parent's.
            with contextlib.suppress(OSError):
                os.close(ruleset.fd)

    def _blame(self, error: OSError, argv: Sequence[str]) -> OSError:
        """Say *sandbox* when it was the sandbox, and stay quiet when it was not.

        This module's characteristic failure is that **the symptom names the
        wrong cause** — a tool reports itself broken because a path it merely
        probed was ungranted. `subprocess` does the same thing to us: an exec
        the policy did not permit surfaces in the **parent** as
        ``PermissionError: /home/…/bin/python3``, blaming the interpreter. That
        is M3, it cost an afternoon when it was first measured, and it reappeared
        unprompted in this package's own tests the first time `spawn` ran a real
        child.

        So the message is corrected **only when the executable is genuinely
        ungranted** — checked, not guessed. Attributing every exec failure to
        the sandbox would be the same defect pointing the other way.
        """
        exe = argv[0] if argv else ""
        roots = [g.path for g in self.policy.granted]
        if not exe or any(contained(exe, root) for root in roots):
            return error
        return PermissionError(
            error.errno,
            f"{exe} is not in this task's granted set, so the sandbox refused to "
            f"execute it — this is the policy, not the interpreter. Every ordinary "
            f"Python install (conda, pyenv, uv, venv) lives under $HOME, which the "
            f"default set deliberately excludes; whatever builds the Context must "
            f"pass interpreter_grants(). Granted roots: {sorted(roots)}",
            exe,
        )

    def wrap_argv(self, argv: Sequence[str]) -> list[str]:
        """The executor's command line, confined. **Ask, do not assemble.**

        The runner has no way to build this for itself: a bubblewrap argv needs
        the policy *and* the binary, and `Availability` is not a type `agent`
        may import. Handing over the raw material and letting the caller compute
        would be this module publishing its internals so somebody else can do
        its job — so it hands over the answer instead.

        On Landlock the process is **already** confined when `prepare` returned
        and the argv is unchanged. On bubblewrap nothing has been applied yet,
        because bubblewrap *is* the exec, and this is where the policy becomes
        real.

        Raises `NoConfinement` when there is nothing to wrap with, and the
        caller does not catch it. That includes the binary having vanished
        between prepare and exec — resolved here rather than remembered from
        probe time, which is the same rule as canonicalising per check.
        """
        if not self.permissions_enforced:
            return list(argv)  # nothing to wrap with, and nothing was meant to be
        if self.confinement is None:
            raise NoConfinement("nothing was confined; there is no executor to start")
        if self.confinement.mechanism == "landlock":
            return list(argv)
        binary = shutil.which("bwrap")
        if binary is None:
            raise NoConfinement(
                "bubblewrap was selected at prepare time and is not on PATH now; "
                "the task does not start"
            )
        return _bwrap.argv(self.policy, bwrap=binary, command=tuple(argv))


def prepare(
    task: Any,
    execution: Any,
    ctx: Context,
    agent_spec: Any = None,
    *,
    availability: Availability | None = None,
) -> Prepared:
    """The composition, in the order §11.1 fixes.

    `agent_spec` carries the four keys — ``env``, ``rules``, ``hooks``,
    ``skills`` — whose declared consumer is this module and which design rev. 4
    added. **It has a default**, so the frozen two-argument call in
    `interfaces.md` §4.6 and `protocols.py` still type-checks and still works;
    the README reports that seam.

    **Nothing here confines anything.** Step 7 checks that a mechanism exists
    and refuses if it does not; `Prepared.spawn` applies it in the child. That
    split is what lets a threaded runner call this at all — confining a thread
    that must write the store afterwards cripples it irreversibly.

    `availability` is injectable for the same reason `select` takes one: no
    machine can run all three branches of the chain, so the branches are unit
    tests and one end-to-end runs against whatever the machine has.
    """
    zone = layout.create(task, execution, ctx.domains)  # 1

    # Read **once**, here, and passed down as an argument everywhere it is
    # needed. Nothing below this line and nothing in another package looks at
    # the environment for it.
    enforcing = permissions_enforced()

    policy = Policy(tuple(DEFAULT_SYSTEM_SET)).with_(
        Granted(zone.root, Mode.READ_WRITE),
        *grants.resolve_all(task, execution, ctx, enforce=enforcing),  # 2
        *ctx.interpreter_grants,  # 3
        *agent_cli_grants(ctx.agent_cli),
        # **Conditional, and paired with an export.** `agent_assets` emits
        # `AGENT_SYS_COMPONENTS_ROOT` under the same test, so `paths.py`'s rule —
        # exported and granted agree by construction — holds without either side
        # checking the other. Read-only: a component is read from here and copied
        # into the zone before anything runs it.
        *component_grants(agent_spec),
    )

    # 4. **No `repos` is passed, and that is a gap rather than a decision.**
    # Rev. 1 read `getattr(task, "repos", ())`, which looks like a field access
    # and is not: the real `Task` has no such field and never did. Design
    # §7.1.1 says a declared `repos` entry comes from the **task spec**, through
    # `task.closure` — and this module does not read task specs. So the read
    # always yielded `()` against a real task and only appeared to work because
    # a stub in this package's own tests had invented the field.
    #
    # `repo_locations` went the same way and is its twin: `Context` has no such
    # field either, so `getattr(ctx, "repo_locations", {})` took its fallback
    # every single time. **Which repos** and **where they are** are two halves
    # of one missing route, and neither half existed.
    #
    # `workspace.cut` keeps both parameters, tested, for whoever can supply
    # them. The route is the one `agent_spec` needed and the task body still
    # needs; inventing a fourth here would be guessing at the answer.
    ws = _workspace.cut(  # 4
        ctx.main_repo,
        zone,
        branch=f"task/{task.id}/attempt-{execution.attempt}",
    )

    report = _NO_SYNC  # 5
    if ctx.mapping:
        report = _sync.sync(
            zone,
            dict(ctx.mapping),
            direction=Direction.LOCAL_TO_REMOTE,
            # Keyed by the same `local_root` as the mapping. Absent means both
            # ends are local, which is the pre-R1 shape and still the default.
            transports=getattr(ctx, "transports", None),
        )
    if report.conflicts:
        raise PrepareRefused(
            f"sync found {len(report.conflicts)} path(s) changed on both sides: "
            f"{list(report.conflicts)}. Detection is ours because no rsync flag "
            f"reports it, and refusing converts silent data loss into a stopped task."
        )

    # **The switch does NOT widen this, and that reverses one of its three ruled
    # rows.** Applying the ruling's own line rather than overriding it:
    # *materialisation is not permission management, and if you find yourself
    # disabling something that makes a file appear where a task needs it, you
    # have crossed the line.*
    #
    # Measured (`task_graph`'s `probe_narrow_staging.py`, and p12 here): widening
    # moves every staged input **down one level** — the artefact's files land at
    # `<materials>/<hid>/v<N>/content/…` instead of `<materials>/<hid>/v<N>/…`.
    # `examples/demo/bin/render.py:67` reads the narrow shape, so the switch
    # would break a body **by moving its input**, and it would present as a body
    # reading one level short rather than as a switch.
    #
    # And widening buys no permission property in this mode: with nothing
    # confined a body can read the store directly (`p11`'s unconfined control
    # succeeds on all four reads), so narrowing the *copy* denies it nothing it
    # could not already reach. What is left is a path convention.
    #
    # `stage(narrow=)` stays, tested, for whoever wants the wide shape.
    staged_inputs = layout.stage_handoffs(task, execution, zone, ctx)  # 6
    # `PATH` is derived from the policy, never chosen: it cannot then name a
    # directory the kernel will refuse. A declared `env` may still override it,
    # because an author saying so outranks a default — but an override naming an
    # ungranted directory is unreachable, and nothing here can make it otherwise.
    environment = {"PATH": executable_path(policy)}

    # **Resolved once, read twice.** `Prepared.agent_cli` reports it and
    # `material.deploy` runs plugin installs with it, and those two must be the
    # same binary or the run installs into one build and talks to another. Two
    # `resolve_strict` calls would be one fact with two writers.
    agent_cli = resolve_strict(ctx.agent_cli) if ctx.agent_cli else None

    # 6a. **The task package: a copy in the zone, not a grant on the root.**
    # `interfaces.md` §4.16, F19's third position. It sits beside handoff
    # staging because it is the same act — putting what the executor needs where
    # the executor can reach it — and before confinement for the same reason.
    #
    # **Exported rather than merely returned, and that is the whole seam.** A
    # package-relative body path resolved against the *original* root now points
    # outside every grant, so whoever launches the body has to be told where the
    # copy went. An environment variable is the one channel that reaches both
    # consumers — `agent.Runner`'s `package_root` and a body that reads its own
    # package out of the environment — without either of them importing this.
    staged = layout.stage_package(ctx.package, zone, ctx.package_stage)  # 6a
    # **The zone's own directories, by name.** This was one hand-written line
    # exporting `PACKAGE_ENV_VAR`, and the user's path-variable requirement is
    # that same act generalised: the package is one of the zone's directories
    # and had a name, while `workspace/`, `playground/`, `handoffs/` and `logs/`
    # had none. So this is a sixth *call site* and not a sixth kind of source —
    # see `paths.py`, which also records why the four `*_root` names the
    # requirement asks for are **not** here (measured `EACCES`, all four).
    #
    # After `cut` and `stage_package`, because it exports only directories that
    # exist; before `material.deploy`, because a declared `env` outranks us.
    # **`far_roots`, not `mapping` — the same correction `_remote_tools` already
    # carries, forty lines below, and this call site was left behind.**
    # `ctx.mapping` is weak-only because it is `sync`'s input and strength
    # answers *must bytes be copied*. A **strong** mapping still has a far side
    # and its `remote_root` is not in `ctx.mapping` at all, so this returned
    # `None` and every `AGENT_SYS_*_REMOTE` variable was omitted — while
    # `_remote_tools`, reading `far_roots`, handed the same agent
    # `env_remote_run`/`push`/`pull`. Tools pointed at a far side, and no
    # variable saying where it is. That is the configuration the accepted
    # remote run used (`scratch/remote-mode-2026-09/acceptance.md`), so this was
    # live rather than latent.
    environment.update(
        paths.zone_env(
            zone,
            staged_package=staged,
            remote_zone_root=_sync.remote_root(zone, _far_side(ctx)),
        )
    )
    environment.update(grants.output_env(task, execution, ctx.store_root))  # 6a'
    # The mirror, and `stage_handoffs`' mapping was being **discarded** here —
    # the association this module had in hand, thrown away, leaving a body to
    # find its own staged input by parsing our directory layout.
    environment.update(grants.input_env(task, staged_inputs))

    # 6b. **Now four destinations, not one.** `deploy` used to return an
    # environment mapping and nothing else could reach a backend from here;
    # per-agent components produce MCP servers and in-process tools, which are
    # not strings and never could be. `staged` and `ws` are passed because
    # components resolve against the **staged** copy (the original checkout is
    # outside every grant) and the asset directory is copied into the workspace.
    deployed = None
    if agent_spec is not None:
        # **`environment` and `agent_cli` are passed because the installs are
        # subprocesses, and a subprocess needs both.**
        #
        # `environment` carries the policy-derived `PATH`. Measured 2026-09-03:
        # without it the child gets **no `PATH` at all** — `material.deploy`
        # builds its mapping from scratch and `harness._RESERVED` excludes
        # `PATH` — so `sh -c "uv --version"` answers `uv: not found` (rc 127)
        # and every recipe that needs a toolchain fails for a reason that names
        # the toolchain rather than the cause.
        #
        # `agent_cli` is the **pinned** CLI, resolved once here and used twice:
        # a plugin install must run the same binary the session will
        # (`interfaces.md` §4.11, and `claude_sdk._prepared_cli`'s refusal is
        # the same rule on the other side of the seam).
        deployed = material.deploy(
            agent_spec, zone, staged, ws.path, base_env=environment, agent_cli=agent_cli
        )
        environment.update(deployed.environment)

    # 7 -- LAST, and since the split it **checks** rather than applies.
    #
    # `select` raises `NoConfinement` when no mechanism exists, so *no
    # isolation, no start* still holds — and now refuses **before** the
    # workspace is cut rather than after, which is strictly better.
    #
    # The syscall itself happens in `spawn`, in the child. §11.1's
    # "confinement last" existed so that the supervisor and every prior process
    # stay outside the domain; moving it into the child achieves that **by
    # construction rather than by ordering**, which is stronger than the
    # sequence that expressed it.
    #
    # **With the switch off, step 7 is not attempted.** Not attempted and
    # discarded — `select` raises `NoConfinement` when no mechanism exists, and
    # computing it only to throw it away would put a live exception on the path
    # of a run that asked for no permission management.
    conf = None
    if enforcing:
        av = availability or probe()
        conf = _apply.confinement_for(select(av), av.landlock_abi)
    return Prepared(
        permissions_enforced=enforcing,
        output_paths=MappingProxyType(grants.output_paths(task, execution, ctx.store_root)),
        agent_cli=agent_cli,
        staged_package=staged,
        zone=zone,
        workspace=ws,
        policy=policy,
        confinement=conf,
        sync=report,
        environment=MappingProxyType(environment),
        # **The remote surface and the agent's own components, in one tuple.**
        # `Assignment.tools` is a flat list of `ToolDef`s and the backend adapts
        # them all into one in-process MCP server, so there is nothing to key
        # them by and nothing that needs to tell them apart. Remote first
        # because it is `env_mgr`'s own and a component's name collision with it
        # should be visible as the component's, not the reverse.
        tools=(*_remote_tools(zone, ctx), *(deployed.tools if deployed else ())),
        mcp_servers=MappingProxyType(dict(deployed.mcp_servers) if deployed else {}),
    )


def _far_side(ctx: Context) -> dict[str, str]:
    """Every far-side root this context knows, from **both** fields that carry one.

    `Context` has two: `mapping` is weak-only because it is `sync`'s input, and
    `far_roots` covers every mapping because a **strong** mapping still has a far
    side. `far_roots` is therefore a superset in any context `cli/main.py` builds
    — it sets both — and the union is only ever needed for a context that sets
    one and not the other, which every direct construction in the tests does.

    **It exists because two call sites resolved this differently and one was
    wrong.** `zone_env(remote_zone_root=…)` read `mapping` while `_remote_tools`
    read `far_roots`, so a strong mapping produced remote *tools* and no remote
    *variables*. Swapping the one for the other merely moves the hole to the
    opposite configuration — a weak-only context then loses the variables — which
    is why this is one function and not a second edit.

    `far_roots` wins a collision. The two disagree only when one `local_root` is
    declared twice with different strengths, and there the weak value belongs to
    `sync` alone; what an agent is told about its own far side is the tools'
    question, and the tools read `far_roots`.
    """
    return {**dict(ctx.mapping or {}), **dict(getattr(ctx, "far_roots", None) or {})}


def _remote_tools(zone: Zone, ctx: Context) -> tuple[Any, ...]:
    """Spec §5.5's tool surface for this zone's far side, or `()`.

    Built here because `remote.tools.tools` needs three things and this is the
    only place holding all three: the connection, the zone, and **the zone's
    far-side root** — which comes from the configuration and is why `tools` takes
    it as a parameter rather than computing it.

    **`far_roots`, not `mapping`.** `ctx.mapping` is weak-only, because it is
    `sync`'s input and strength answers *must bytes be copied*. A **strong**
    mapping still has a far side and its `remote_root` is not in `ctx.mapping`
    at all — so resolving against it would have given a strong mapping tools
    pointed at nothing, which is the configuration R1b uses.

    `()` when nothing maps this zone, which is every configuration with no meta
    file. A task with no far side gets no remote tools, and that absence is what
    the agent sees: no tool, rather than a tool that fails.
    """
    transports = dict(getattr(ctx, "transports", None) or {})
    found = _sync.match(zone, _far_side(ctx))
    if found is None:
        return ()
    key, far = found
    conn = transports.get(key)
    if conn is None:
        return ()
    return _tools.tools(conn, zone, far)


def place_zone(task: Any, execution: Any, ctx: Context) -> Zone:
    """Create this attempt's zone and **nothing else**. Returns the `Zone`.

    `prepare`'s first step, on its own. A non-leaf task never executes — the
    scheduler runs its main phase by unfolding — so it reaches no code path that
    calls `prepare`, and therefore never gets a zone. But a subtask's storage
    nests **inside its parent's** (criterion 2), so a parent with no zone is a
    parent whose children cannot be placed at all: no nested graph can run.

    Doing it with `prepare` was the obvious repair and is wrong three ways, two
    of them measured. It would cut a workspace and stage handoffs for a task
    that never executes; it would end in `apply()`, confining a thread that is
    about to be handed back and re-entered for output validation; and `apply()`
    would refuse anyway, because an attempt thread plus the main thread is
    already two.

    So this is the first step and none of the rest. It confines nothing, cuts
    nothing, stages nothing, and syncs nothing.

    **It does not decide who calls it.** The scheduler at `unfold` and the
    attempt before it releases its thread are both plausible, they need the same
    verb, and the choice is not this module's. What *is* this module's is that
    the parent's `Execution` has to come from somewhere: `Task.parent` is a
    `TaskId`, so a zone that does not exist has no discoverable attempt number,
    and creating one at attempt 0 would be wrong the moment a parent is retried.
    """
    return layout.create(task, execution, ctx.domains)


class ValidationZone(NamedTuple):
    """Where a validation's materials go, and what was put there.

    `root` is a **sibling** of the producing task's zone, never a descendant.
    That is design D5 and criterion 13 is untrue without it: anything under the
    producing task's directory is inside its subtree, and permissions cover a
    task's own subtree recursively.

    `materials` are **copies**, staged out of the store — spec §6.3 rule 2, so a
    validation cannot edit what it is validating, and so a body handed handoff
    ids has somewhere to read them from. It maps **handoff id → staged path**,
    because a validator taking more than one input must know which copy is
    which, and recovering that by parsing the path would make this module's
    directory shape a contract another package quotes.
    """

    root: str
    phase: str
    materials: Mapping[Any, str]


def prepare_validation(task: Any, execution: Any, phase: Any, ctx: Context) -> ValidationZone:
    """Place a validation's zone and stage what it validates.

    `phase` is read for its value — ``input_validation`` or
    ``output_validation`` — the same structural read this module already uses
    for `task_graph.Access`, and for the same reason: the two packages do not
    import each other.

    Which slots are staged follows from the phase: an input validation checks
    what the task was given, an output validation what it produced. The versions
    come off the `Execution`, so a retry validates that attempt's artefacts and
    not the previous one's.

    **It does not confine anything**, and that is deliberate rather than
    forgotten — see the note on `EnvManager.prepare_validation`.
    """
    kind = str(getattr(phase, "value", phase))
    root = layout.validation_zone(task, kind, ctx.domains)
    if kind.startswith("input"):
        slots, versions = task.inputs, execution.input_versions
    else:
        slots, versions = task.outputs, execution.output_versions
    materials = layout.stage(slots, versions, os.path.join(root, "materials"), ctx.store_root)
    return ValidationZone(root=root, phase=kind, materials=materials)


class EnvManager:
    """The registered component, ``env_mgr``.

    A thin object over two functions, and it exists for one reason: `Context` is
    composition-time configuration — the domains, the store root, the main
    repository, the sync mapping, the tier — and threading it through a caller
    would make that caller carry configuration it has no opinion about. The
    object binds the context once at the root; callers pass only what varies.

    **The "one method, and it stays one" rule is amended here rather than
    reinterpreted, and this is the amendment.** The rule's stated hazard was
    *"a second is how the runner would start making environment decisions"* —
    the runner, accreting. `prepare_validation` is not that: it is a different
    caller asking the layout owner a question only the layout owner can answer,
    and it was ruled in after two modules were found to be answering *where does
    a validation go*.

    What decides it is not the hazard's wording but what this object **is**: a
    `Context` bound once. A validation zone needs `ctx.domains` and
    `ctx.store_root` — the *same* bound context — so a separate component would
    bind one configuration twice, and one fact with two writers is the thing the
    rule was protecting against in the first place.

    The guard survives the amendment: `test_env_manager_exposes_exactly_these`
    pins the **set**, so a third method still fails a test and still needs a
    decision. That is what the original guard was for.
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    def prepare(self, task: Any, execution: Any, agent_spec: Any = None) -> Prepared:
        """Raises `NoConfinement`, `PrepareRefused` or `UnresolvedGrant`.

        **The caller catches none of them.** Criterion 14 is *no isolation, no
        start*, and it is only a rule if nothing anywhere converts the refusal
        into a warning.

        **This confines nothing.** Step 7 checks that a mechanism exists and
        refuses if it does not; `Prepared.spawn` applies it in the child. The
        split is what lets a threaded runner call this at all — a thread that
        confines itself can no longer write the store, irreversibly.
        """
        return prepare(task, execution, self._ctx, agent_spec)

    def prepare_validation(self, task: Any, execution: Any, phase: Any) -> ValidationZone:
        """Place a validation's zone as a **sibling** of the producing task's,
        and stage copies of what it validates into it.

        Resolved by name, never imported: `validator` may not import this
        package, and an import edge is permanent where a name lookup is not.

        **It confines nothing, and that is a boundary rather than an omission.**
        `prepare` applies Landlock to *its own process* because the executor is
        that process's child; a phase runner calling this is the supervisor, and
        confining it would confine the supervisor. Who applies a policy to a
        validation *body* is a third question that this ruling did not settle
        and that this method does not quietly answer.
        """
        return prepare_validation(task, execution, phase, self._ctx)

    def place_zone(self, task: Any, execution: Any) -> Zone:
        """Create this attempt's zone and nothing else.

        For a task that will never execute — a non-leaf, whose children nest
        inside its zone and cannot be placed until it exists. Confines nothing,
        cuts no workspace, stages nothing.
        """
        return place_zone(task, execution, self._ctx)
