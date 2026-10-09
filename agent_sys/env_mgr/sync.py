# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The one-shot sync job. Design §9.

Not a reconciliation loop: once, at task start, scoped to this task's subtree
and never the root. ``rsync`` is what spec §5.2 names and what §9.1 needs — a
one-shot copy, not a reconciler — and §9.3 adds the one thing it cannot do.
"""

from __future__ import annotations

import filecmp
import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from enum import Enum

from env_mgr.fs.domain import DomainKind, subdir_for
from env_mgr.fs.zone import Zone
from env_mgr.protocols import PrepareRefused, SyncReport
from env_mgr.remote.connection import LocalConnection, SyncTransport, rsync_stat

__all__ = [
    "Direction",
    "PLAYGROUND",
    "check_delete_scope",
    "match",
    "conflicts",
    "remote_root",
    "sync",
]

PLAYGROUND = subdir_for(DomainKind.PLAYGROUND)


class Direction(str, Enum):
    """Required, never defaulted.

    Measured: ``rsync -a --delete`` makes two trees equal by **destroying**
    everything the destination had that the source did not, and there is no
    symmetric mode. Spec §5.3's "local and remote are made identical" is
    therefore implemented as "the destination is made identical to the source,
    and the caller names which is which".
    """

    LOCAL_TO_REMOTE = "local_to_remote"
    REMOTE_TO_LOCAL = "remote_to_local"


def _under(path: str, root: str) -> bool:
    """Is `path` at or below `root`, **by path component**?

    Not `startswith`: `/data/yihou2` starts with `/data/yihou` and is somebody
    else's directory. `normpath` first, so a mapping written with `..` in it
    cannot climb out of a root it appears to be under.
    """
    path, root = os.path.normpath(path), os.path.normpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def check_delete_scope(mapping: Mapping[str, str], deletable_roots: Sequence[str]) -> None:
    """Refuse a mapping whose far side is not declared destroyable. **Fail closed.**

    `sync` runs ``rsync --delete``, and where it points is decided by a **meta
    file somebody edits** — a value supplied from outside deciding what gets
    destroyed, on a machine nobody in the session is watching. That is the shape
    of the 2026-08-31 accident, and the operator's `rm` hook does not reach it:
    that hook intercepts a shell `rm`, and this deletion happens inside `rsync`,
    invoked from Python, which never goes near a shell.

    So a remote root must appear in an **allow-list the configuration states
    explicitly**, and the default is empty — a meta file that declares a weak
    mapping and no `deletable_roots` is refused, naming the root. A deny-list of
    `/`, `/usr`, `/home` was the alternative and is a deny-list in allow-list
    clothing: the next dangerous root is the one nobody thought to add.

    **Called at composition, when the configuration is read**, so a bad meta file
    fails before anything runs rather than at the first copy. A caller that
    builds a `Context` by hand — every test does — bypasses it, which is the
    honest limit: this guards the configuration route, and the configuration
    route is where the danger comes from.

    **Not the only thing standing there, and this must survive whoever fixes
    open question 4.** `_conflicts_across` already refuses when the far-side path
    exists, so `--delete` can today only run against a directory `sync` itself
    has just created. Answering question 4 will relax that refusal, and at that
    moment this allow-list becomes the sole guard. It is defence in depth now and
    the only defence later.
    """
    roots = [r for r in deletable_roots if r]
    for local_root, far_root in sorted(mapping.items()):
        if any(_under(far_root, root) for root in roots):
            continue
        raise PrepareRefused(
            f"the mapping {local_root!r} -> {far_root!r} would let `rsync --delete` "
            f"write outside every declared deletable root {sorted(roots)}. `sync` "
            f"deletes whatever the destination has that the source does not, and "
            f"here the destination is aimed by a configuration file. Add the root "
            f"to `deletable_roots` in the meta file if it is genuinely yours to "
            f"destroy, or point the mapping somewhere that is."
        )


def match(zone: Zone, mapping: Mapping[str, str]) -> tuple[str, str] | None:
    """``(local_root_key, far_path)`` for the mapping covering this zone.

    **One walk, and the key comes back with the path.** `remote_root` wants only
    the path; `_ends` additionally needs the key, because `Context.transports`
    is keyed by the same `local_root` and looking it up with a second walk would
    give the prefix rule two writers — the hazard `remote_root`'s own docstring
    was written to close.
    """
    for local_root, far_root in mapping.items():
        prefix = local_root.rstrip(os.sep)
        if zone.root == prefix or zone.root.startswith(prefix + os.sep):
            rel = os.path.relpath(zone.root, prefix)
            return local_root, (os.path.join(far_root, rel) if rel != os.curdir else far_root)
    return None


def remote_root(zone: Zone, mapping: Mapping[str, str]) -> str | None:
    """This zone's counterpart on the far side, or `None` if nothing maps it.

    **The mapping is this module's, so the walk over it is too.** It was inlined
    in `_ends`, which is the shape a caller needs for an rsync — a source and a
    destination ordered by direction. `paths.zone_env` needs the other shape:
    *"where does this zone live over there"*, with no direction and no copy. A
    second walk in that module would have been `engineer_principle.md` §3's
    symptom exactly — an outsider taking `Context.mapping` and computing with it
    — and would have given one fact two writers, so the next mapping rule
    (a trailing separator, a nested mapping, a longest-prefix tie) would land in
    one of them.

    **`None` rather than a raise**, because the two callers want different
    things from a miss. `_ends` cannot proceed and says so; an environment
    variable for a side that is not configured is simply *absent*, which is this
    module's established shape for an unresolvable path — `grants.output_paths`
    omits a slot with no pinned version rather than presenting it as empty.
    """
    found = match(zone, mapping)
    return found[1] if found is not None else None


def _ends(zone: Zone, mapping: dict[str, str], direction: Direction) -> tuple[str, str, str]:
    """This zone's two sides, and the mapping key that produced them.

    Both reasons the spec gives were measured, and only one holds at the size
    measured: 20 tasks × 50 files is 108 ms for the whole root against 53 ms for
    one task, because rsync's fixed startup dominates. **The argument that holds
    here is correctness** — not touching another task's material.
    """
    found = match(zone, mapping)
    if found is None:
        raise KeyError(
            f"no mapping covers zone {zone.root!r} (have {sorted(mapping)}); "
            f"sync is per task and needs the task's own mapping"
        )
    key, remote = found
    if direction is Direction.LOCAL_TO_REMOTE:
        return zone.root, remote, key
    return remote, zone.root, key


def conflicts(src: str, dst: str) -> tuple[str, ...]:
    """Paths that exist on **both** sides and differ, before anything is written.

    ``rsync`` cannot report this and no flag makes it: with both sides edited,
    ``-a`` and ``--checksum`` silently discard one and ``--update`` guesses by
    mtime. Detection is a pre-pass, and it is what converts silent data loss
    into a stopped task. That is not conflict *resolution*, which stays on the
    roadmap — it is the difference the open question is actually about.
    """
    if not (os.path.isdir(src) and os.path.isdir(dst)):
        return ()
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        if rel.split(os.sep)[0] == PLAYGROUND:
            dirnames[:] = []
            continue
        for name in filenames:
            a = os.path.join(dirpath, name)
            b = os.path.join(dst, os.path.relpath(a, src))
            if os.path.exists(b) and not filecmp.cmp(a, b, shallow=False):
                found.append(os.path.relpath(a, src))
    return tuple(sorted(found))


def sync(
    zone: Zone,
    mapping: dict[str, str],
    *,
    direction: Direction,
    transports: Mapping[str, SyncTransport] | None = None,
) -> SyncReport:
    """Once, at task start. Excludes the playground.

    ``--exclude playground/`` omits the contents but still creates the directory
    on the far side, empty — which is consistent with the remote having its own
    playground, and is what criterion 16's assertion must actually say.

    **`transports` is keyed by the same `local_root` as `mapping`**, and a key
    with no entry means both ends are on this machine — which is every
    configuration before R1 and stays the default. The flags below never move
    into the transport: `--delete` *is* spec §5.3's "made identical" and the
    exclude *is* criterion 16, and `Connection.push` has neither.
    """
    src, dst, key = _ends(zone, mapping, direction)
    conn: SyncTransport = (transports or {}).get(key) or LocalConnection()
    rsh, prefix = conn.rsync_spec()
    # Only the *far* end takes the prefix. Which end that is follows from the
    # direction, and getting it backwards would address the wrong machine.
    far_is_dst = direction is Direction.LOCAL_TO_REMOTE
    # **Whether the pre-pass can read a side is about the *host*, not about the
    # direction**, and `remote_dst=far_is_dst and bool(prefix)` conflated them.
    # With `REMOTE_TO_LOCAL` over an `Ssh` transport, `far_is_dst` is `False`, so
    # `_conflicts_across` took the local branch and ran `filecmp` with `src` —
    # a path on the *other machine* — read here. It either found nothing (silent
    # pass) or found a same-named local tree and compared the wrong one, and
    # `rsync -a --delete` then overwrote the local zone with the single guard
    # against both-sides-changed data loss never having run. `_conflicts_across`
    # promises "never a silent skip"; this is what made that untrue.
    #
    # `bool(prefix)` alone is the honest question: a prefix means one end is on
    # another host, and `_conflicts_across` cannot `filecmp` across that boundary
    # in either direction. No production caller uses `REMOTE_TO_LOCAL` today, so
    # this shipped untested rather than shipped broken.
    found = _conflicts_across(conn, src, dst, far_is_dst=far_is_dst, remote=bool(prefix))

    playground_dst = os.path.join(dst, PLAYGROUND)
    if far_is_dst and prefix:
        # `playground_dst` was created here **and** after the copy, and only the
        # second one does anything. `dst` itself stays: rsync creates the final
        # missing component of a destination and not its parents — measured, a
        # two-level-deep dst gives `mkdir failed: No such file or directory`.
        _checked(conn.run(["mkdir", "-p", dst]), f"mkdir -p {dst}")
    else:
        os.makedirs(dst, exist_ok=True)

    rsync = shutil.which("rsync")
    if rsync is None:
        raise RuntimeError("rsync is not installed; the one-shot sync has no mechanism")
    cmd = [rsync, "-a", "--delete", "--stats", f"--exclude={PLAYGROUND}/**"]
    if rsh:
        # **`shlex.join`, not `" ".join`.** rsync word-splits the `-e` value, so
        # `Ssh(host, options=("-o", "ProxyCommand=ssh -W %h:%p bastion"))` sent
        # `-o ProxyCommand=ssh` plus three stray arguments. Same class for any
        # option containing a space. This is the argv-boundary defect `Ssh.run`
        # was fixed for, reappearing one layer out because the fix was applied
        # to the transport and this string is built here.
        cmd += ["-e", shlex.join(rsh)]
    src_arg = src.rstrip(os.sep) + os.sep
    dst_arg = dst.rstrip(os.sep) + os.sep
    if far_is_dst:
        dst_arg = prefix + dst_arg
    else:
        src_arg = prefix + src_arg
    proc = subprocess.run([*cmd, src_arg, dst_arg], capture_output=True, text=True, check=True)

    if far_is_dst and prefix:
        # **Criterion 16's empty far-side playground, and this comment used to
        # describe a case production never reaches.** It said rsync "does not
        # create the directory either" and that `--delete` "removes it". Measured,
        # rsync 3.2.7, `--delete --exclude='playground/**'`:
        #
        #   sender HAS playground (contents excluded)  -> rsync CREATES it empty
        #   sender has none, receiver's is empty       -> `--delete` REMOVES it
        #   sender has none, receiver's is non-empty   -> kept, rc still 0
        #
        # `layout.create` makes every `_subdirs(domains)` entry and `_subdirs`
        # falls back to `tuple(DomainKind)` when the registry declares no kinds,
        # so a zone normally **has** a playground and the first row is what runs:
        # rsync creates it and nothing removes it. The second row is why this call
        # stays — a registry that declares kinds without PLAYGROUND reaches it,
        # and then this is the only thing that puts the directory there.
        _checked(conn.run(["mkdir", "-p", playground_dst]), f"mkdir -p {playground_dst}")
    else:
        os.makedirs(playground_dst, exist_ok=True)
    return SyncReport(
        sent=_stat(proc.stdout, "Number of regular files transferred"),
        received=_stat(proc.stdout, "Number of deleted files"),
        conflicts=found,
    )


def _checked(proc: subprocess.CompletedProcess[str], what: str) -> None:
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed on the far side: rc={proc.returncode} {proc.stderr!r}")


def _conflicts_across(
    conn: SyncTransport, src: str, dst: str, *, far_is_dst: bool, remote: bool
) -> tuple[str, ...]:
    """The pre-pass, or a refusal — **never a silent skip**.

    `conflicts` is `filecmp` over two local trees. Across a host boundary there
    is no local read of the far side, so it cannot run, and open question 4 —
    what replaces it — is unanswered. What is *not* in question is what to do
    meanwhile: `PrepareRefused` exists because rsync cannot report both-sides-
    changed, so a pre-pass that cannot run is a refusal and not a pass. Proceeding
    as though it had checked would convert the one guard against silent data loss
    into a comment.

    **The exception is exact rather than convenient: a destination that does not
    exist yet has nothing to conflict with.** `conflicts` already returns `()`
    when either side is not a directory, and this is that same rule asked over
    the wire. It is also the ordinary case — a zone is named per attempt, so the
    far side is fresh almost every time — which is why the refusal below has to
    be tested deliberately rather than waited for.
    """
    if not remote:
        return conflicts(src, dst)
    # **Ask the side `dst` is actually on.** `remote` says a host boundary is
    # crossed, so `filecmp` cannot run — that much is symmetric. *Where the
    # destination lives* is not, and conflating the two is how this went wrong
    # twice. It first read `far_is_dst and bool(prefix)`, which sent
    # `REMOTE_TO_LOCAL` down the local branch and ran `filecmp` against a
    # far-side `src` read here. Replacing that with `bool(prefix)` alone fixed
    # the branch and broke the probe: measured, the far side was asked
    # `test -e /…/local/task.a` about a path that only exists on this machine,
    # and answered "no" — a silent pass, in the same reassuring direction.
    exists = conn.run(["test", "-e", dst]).returncode == 0 if far_is_dst else os.path.exists(dst)
    if not exists:
        return ()
    raise PrepareRefused(
        f"the far side already has {dst!r} and the conflict pre-pass cannot read "
        f"it: `conflicts` is filecmp over two local trees and there is no "
        f"cross-host equivalent yet. Refusing rather than copying unchecked — "
        f"rsync cannot report that both sides changed, which is the whole reason "
        f"this pre-pass exists. Remove the far-side path, or sync a fresh zone."
    )


#: **One reader, and there were two.** This module parsed the integer after the
#: colon; `remote/connection.py:_rsync` counted matching lines and so reported 0
#: or 1 for every copy. Both read the same line of `rsync --stats` output, one of
#: them wrongly, and the wrong one is what an agent sees through
#: `env_remote_push`/`env_remote_pull`. The body moved to `connection` because
#: this module already imports it and the other direction is a cycle; the name
#: stays here because every call site in this file uses it.
_stat = rsync_stat
