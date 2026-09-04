# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The nested layout, and where a validation goes. Design §8.

Spec §5.1, unchanged::

    <root>/task.<uuid>.<version>.<hash>/
      ├── handoffs/     ├── workspace/    ├── playground/    ├── logs/
      └── task.<child-uuid>.<version>.<hash>/     ← a subtask, nested

The nesting is what makes containment answer both *"is this path in the zone"*
and *"may this task reach that path"*, because permissions cover the task's own
subtree recursively and the subtree **is** the nesting.
"""

from __future__ import annotations

import filecmp
import os
import shutil
from collections.abc import Sequence
from typing import Any

from env_mgr.fs.domain import DomainKind, DomainRegistry, subdir_for
from env_mgr.fs.path import resolve_strict
from env_mgr.fs.zone import Zone, validation_dirname, zone_dirname

__all__ = [
    "LOGS",
    "copy_out",
    "create",
    "find_zone_dir",
    "handoff_version_dir",
    "stage",
    "stage_handoffs",
    "stage_package",
    "validation_zone",
]

#: Where a staged task package lands inside the zone. A fixed name rather than a
#: configured one: whoever launches the body is told the absolute path, so the
#: name is this module's and nobody else quotes it.
PACKAGE = "package"

#: Not a domain: created with the zone and granted read-write like anything else
#: in it, which is what spec §6.1's last row asks for. Design §8.2.
LOGS = "logs"

_ZONE_PREFIX = "task."


def _subdirs(domains: DomainRegistry) -> tuple[str, ...]:
    """The per-zone directories. Kind decides layout, and only that (§8.2)."""
    kinds = domains.kinds() or tuple(DomainKind)
    return tuple(subdir_for(k) for k in kinds) + (LOGS,)


def find_zone_dir(base: str, task_id: Any) -> str | None:
    """The directory of `task_id`'s most recent attempt, anywhere under `base`.

    A task's uuid is unique, so the ``task.<uuid>.`` prefix identifies it
    wherever the tree happens to have put it — which is what lets a subtask be
    placed under its parent without the parent's `Task` object being in hand.
    """
    prefix = f"{_ZONE_PREFIX}{task_id}."
    best: tuple[int, str] | None = None
    for dirpath, dirnames, _ in os.walk(base):
        for name in dirnames:
            if not name.startswith(prefix):
                continue
            parts = name.split(".")
            try:
                attempt = int(parts[-2])
            except (IndexError, ValueError):
                continue
            if best is None or attempt > best[0]:
                best = (attempt, os.path.join(dirpath, name))
    return best[1] if best else None


def create(task: Any, execution: Any, domains: DomainRegistry) -> Zone:
    """Create (or reload) this **attempt's** zone and return it.

    A subtask's storage is nested inside its parent's — criterion 2. The
    playground is *reloaded, not recreated* when the zone already exists, which
    is criterion 17 and spec §6.2's "it survives a resume".
    """
    base = domains.storage_root()
    parent_id = task.parent
    if parent_id is not None:
        parent_dir = find_zone_dir(base, parent_id)
        if parent_dir is None:
            raise ValueError(
                f"task {task.id} declares parent {parent_id}, which has no zone under {base}"
            )
        base = parent_dir
    root = os.path.join(base, zone_dirname(task.id, execution.attempt))
    for sub in _subdirs(domains):
        # exist_ok: a resume finds its own playground and keeps the contents.
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    resolved = resolve_strict(root)
    if resolved is None:  # pragma: no cover - we just created it
        raise ValueError(f"zone root {root!r} does not resolve")
    return Zone(task_id=task.id, attempt=execution.attempt, root=resolved)


def validation_zone(task: Any, phase: str, domains: DomainRegistry) -> str:
    """A validation's materials, as a **sibling** of the producing task's zone.

    Design D5. Criterion 13 says containment resolves the producer/validator
    separation, and it is untrue without this: spec §5.1's layout is five things
    and none of them is a validation, and the one placement it suggests is
    inside the producing subtree, which is reachable.
    """
    base = domains.storage_root()
    zone_dir = find_zone_dir(base, task.id)
    parent = os.path.dirname(zone_dir) if zone_dir else base
    root = os.path.join(parent, validation_dirname(task.id, phase))
    os.makedirs(root, exist_ok=True)
    resolved = resolve_strict(root)
    if resolved is None:  # pragma: no cover - we just created it
        raise ValueError(f"validation root {root!r} does not resolve")
    return resolved


#: The artefact, inside a version directory. `handoff/store.py::CONTENT_DIR`,
#: spelled again rather than imported: `env_mgr` does not import `handoff`, and
#: an import edge is permanent where a duplicated constant is one grep.
#: `interfaces.md` §8.1's forced duplication;
#: `tests/interfaces/test_handoff_layout.py` is the price and pins the spelling
#: against `handoff`'s.
#:
#: **Defined here rather than in `grants.py`**, which is where it started, because
#: `stage` needs it and `layout` sits *below* `grants` in the import graph — the
#: alternative was a second literal ``"content"`` in this file, which is the
#: duplication §8.1 tolerates across a package boundary and does not tolerate
#: within one. `grants.CONTENT_DIR` re-exports it, so the name the interfaces
#: test pins is unchanged.
CONTENT_DIR = "content"


def handoff_version_dir(store_root: str, handoff_id: Any, version: int) -> str:
    """``<store_root>/<hid>/v<N>/`` — `handoff` design §6.2's layout.

    This module grants access to that directory and computes nothing about its
    contents (design §1.2).
    """
    return os.path.join(store_root, str(handoff_id), f"v{version}")


def stage(
    slots: Any, versions: Any, into: str, store_root: str, *, narrow: bool = True
) -> dict[Any, str]:
    """Copy each slot's version out of the store and under `into`.

    Returns **handoff id → staged path**, not a bare list of paths.

    The association is a fact this function has in hand and would otherwise
    throw away, leaving a caller to recover it by parsing
    ``<into>/<hid>/v<N>`` — which is `engineer_principle.md` §4.4's second named
    smell verbatim: *computing a key to recover an order that the module handing
    you the data already knew*. It would also make this module's directory shape
    a contract another package quotes, which is the leaked knowledge the seam
    exists to prevent.

    It matters as soon as a validator takes more than one input, which
    `validator` spec §4.1 makes first-class rather than an edge — *"the binding
    is many-to-many"* — and criterion 4 requires a `dict[HandoffId, bool]`
    verdict, so a body must know which copy is which.

    A mapping rather than pairs, for two reasons beyond convenience: one slot
    has exactly one staged version per attempt, so the key is unique **by
    construction**; and a slot that staged nothing is then *visibly* absent
    rather than silently shortening a list.

    Spec §6.3 rule 2 — an agent works on a copy, never on the stored artefact —
    and it is what makes a re-run comparable to the run before it. The layout
    decides where a staged artefact lands, so a second module doing this would
    be a second answer to *where does this go*.

    **What is copied is the version's ``content/``, not the version directory.**
    This copied the whole of ``v<N>/`` until `handoff` measured what that put in
    front of a consumer: ``manifest.yaml``, ``validation.yaml``, and the
    producer's own claim of completeness under ``claim/``. `handoff.copy_out`
    already answers this question — *"copy this version's ``content/`` into
    ``dst``"* — so the wide copy was **a second answer to a question that was
    already answered**, and the party it misled was the independent validator.
    `validator` spec §8 is a table of *"the producer cannot"*; the producer
    asserting its own completeness to the agent judging completeness
    (`validator` spec:676, the `weak` goal validator) is that table's subject
    even though no row names it.

    Two facts made the change safe rather than merely tidy, both first-hand:
    nothing in any consuming package reads a staged sibling, and narrowing broke
    nothing across five suites with a control proving the narrowing was live
    (`scratch/impl-2026-08/env_mgr/p9_what_narrowing_stage_would_cost.py`).
    `validator` confirmed against their own code that their prior-verdict path
    goes to `store.read_verdicts` and never to the staged tree.

    **The route it leaves is checked, but it is not open to a body**, and that
    distinction was measured after the change rather than before it
    (`scratch/impl-2026-08/env_mgr/p11_can_a_body_reach_the_store_root.py`).
    `handoff.get_manifest` verifies a digest where a staged copy does not, so
    *in-process* code — `validator`'s own Python, `agent/gate.py` — is strictly
    better served by the store. A **confined body** is not: granted its zone and
    its inputs' `content/` and nothing naming the store root, it gets `EACCES`
    on the store root and on any manifest under it, against an unconfined
    control where all four reads succeed.

    So for a body both routes are now closed, and the second was already closed
    before this change — what narrowing removed is the *accidental* one, a
    manifest that happened to ride along in a directory copy. Anything relying
    on it was relying on the accident. `examples/demo/logic/store.py` reads
    ``AGENT_SYS_DEMO_STORE`` and walks to a manifest as its F-D5 fallback for
    *"a validator that must reach a handoff it was not handed"*; that fallback
    does not work under confinement and did not before. Naming it here because
    it is the kind of thing that fails only in front of a user.

    **The shape matches `handoff.copy_out` exactly**, and deliberately: that is
    ``copytree(<v>/content, dst)``, so the artefact's own files land *at* the
    mapped path rather than under a ``content/`` level inside it. Preserving
    ``content/`` here would have kept a third answer alive and made every body's
    relative paths quote `handoff`'s directory vocabulary. `validator` records
    the mapped path verbatim in `materials.json`, so a body sees
    ``<materials>/<hid>/v<N>/<the artefact's own files>``.

    A version whose ``content/`` is absent is skipped like a missing version
    directory, rather than falling back to copying the whole of ``v<N>/``: a
    fallback here would reinstate the wide copy exactly when the layout is
    unexpected, which is when it is least safe to guess.
    **`narrow=False` is the kill switch's third row.** With permission management
    off (`AGENT_SYS_NO_PERMISSIONS`), the whole version directory is copied —
    what this did before the ruling. It belongs to the switch because the
    narrowing's *purpose* is deciding what a consumer may see; the copy itself is
    materialisation and happens either way. The switch is read in `prepare`,
    once, and arrives here as an argument.
    """
    versions = dict(versions or {})
    staged: dict[Any, str] = {}
    for hid in slots or ():
        version = versions.get(hid)
        if version is None:
            continue
        version_dir = handoff_version_dir(store_root, hid, version)
        src = os.path.join(version_dir, CONTENT_DIR) if narrow else version_dir
        if not os.path.isdir(src):
            continue
        dst = os.path.join(into, str(hid), f"v{version}")
        copy_out(src, dst)
        staged[hid] = dst
    return staged


def stage_handoffs(
    task: Any, execution: Any, zone: Zone, ctx: Any, *, narrow: bool = True
) -> dict[Any, str]:
    """Copy this attempt's declared inputs into the zone. Spec §8, step 6."""
    return stage(
        task.inputs,
        execution.input_versions,
        os.path.join(zone.root, subdir_for(DomainKind.HANDOFF_STORAGE)),
        ctx.store_root,
        narrow=narrow,
    )


def stage_package(
    package_root: str | None, zone: Zone, include: Sequence[str] | None = None
) -> str | None:
    """A **copy** of what the task needs, inside the zone. `interfaces.md` §4.16.

    F19's third position, and the two before it are why this is a copy rather
    than a grant. *Stage* was argued and reversed on a measurement — a body is a
    launcher, ``bodies/produce/entry.sh`` runs ``python3 <package>/bin/collect.py``,
    so staging the entry alone still launches a file the package holds. What
    reversed it back is not a new measurement but a **scoping correction**:
    permission management here is wide, its only job is to stop several agents
    cross-contaminating, and the criterion the package grant was protecting
    (13, anti-gaming) is not that.

    `include` is an **allow-list of package-relative paths**, and the list shape
    is the whole point rather than an ergonomic choice. Criterion 14 holds
    because a zone nobody anticipated is *absent from a list*; a deny-list would
    give exactly the anticipated-only guarantee §4.5 rejects, and a validator
    directory added next month would not be on it. So when `TODO.md` 4a lands and
    a task's executable set becomes nameable, criterion 13 closes here by the
    same construction that already closes 14.

    ``include=None`` stages the whole package. **That is today's honest state,
    not a convenience default**: without 4a nothing can name the executable set,
    so the copy carries ``validators/`` too and criterion 13's second route is
    *moved* rather than closed. §4.16 states and accepts this.

    **The accepted cost, stated rather than discovered.** The copy lands in the
    zone, which the agent can write, so *a task may not modify the package it was
    loaded from* stops being kernel-enforced — an agent can edit its own body
    mid-attempt. Under the scope above that is explicitly not ours to prevent.

    Returns the staged root, or ``None`` when there is no package. **The caller
    must tell the body where it went**; `prepare` does that through
    `Prepared.environment`, because a package-relative path resolved against the
    *original* root now points outside every grant.
    """
    if package_root is None:
        return None
    resolved = resolve_strict(package_root)
    if resolved is None:
        raise ValueError(f"task package {package_root!r} does not resolve")
    into = os.path.join(zone.root, PACKAGE)
    if include is None:
        copy_out(resolved, into)
        return into
    for relative in include:
        src = os.path.join(resolved, relative)
        if not os.path.exists(src):
            # Principle 3, and the same rule as every other declared path here: a
            # named entry that is absent is an error, not an empty stage. The
            # alternative is a body that launches nothing and blames itself.
            raise ValueError(
                f"task package {resolved!r} declares {relative!r} in its staged set "
                f"and it does not exist"
            )
        copy_out(src, os.path.join(into, relative))
    return into


def copy_out(src: str, dst: str, *, dereference: bool = False) -> str:
    """Copy a stored artefact to `dst`. Spec §6.3 rule 2: an agent works on a copy.

    `handoff`'s own ``copy_out(hid, version, dst)`` has no default for `dst`
    because an agent handed the store's own path edits the store in place. The
    same reasoning applies one level down: this never returns the source.

    **`dereference` exists because the default is asymmetric with depth**, which
    is a property of `shutil` rather than a decision anyone here made. Measured
    2026-09-03: a symlink passed as `src` goes through `copy2` and arrives as a
    **real file with the target's content**, while a symlink *nested inside* a
    directory `src` goes through ``copytree(symlinks=True)`` and arrives **still
    a symlink**. Same input, two results, decided by how deep it sits.

    The default is unchanged — every existing caller keeps the behaviour it was
    written against. `dereference=True` makes the two agree by resolving links
    at both depths, and `agent_assets._place_tree` is the caller that needs it:
    a preserved link pointing outside the zone measurably **fails `contained`**,
    so the confined session cannot follow it, and *copy into the zone, do not
    reference out of it* is that module's ruling.
    """
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("refusing to copy a stored artefact onto itself")
    os.makedirs(os.path.dirname(dst) or os.curdir, exist_ok=True)
    if os.path.isdir(src) and not (dereference and os.path.islink(src)):
        shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=not dereference)
    else:
        # `copy2` follows a symlink, so this is already the dereferencing branch
        # for a plain file; a symlinked *directory* under `dereference` lands
        # here too and `copytree` above resolves it on the recursive call.
        if dereference and os.path.islink(src) and os.path.isdir(src):
            shutil.copytree(os.path.realpath(src), dst, dirs_exist_ok=True, symlinks=False)
        else:
            shutil.copy2(src, dst)
    return dst


def trees_identical(a: str, b: str) -> bool:
    """Byte-for-byte, recursively. Criterion 19's "still verifies", without a
    digest: `handoff` owns digests (design §1.2) and this module owns copies."""
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(trees_identical(os.path.join(a, d), os.path.join(b, d)) for d in cmp.common_dirs)
