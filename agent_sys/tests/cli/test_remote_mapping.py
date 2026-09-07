# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Does a **run** carry a local↔remote mapping? Not: does `Meta` parse one.

`Context.mapping` was `{}` in every production run this repository had ever
made, so `prepare.py`'s `if ctx.mapping:` was never true, and `sync.sync`,
`sync.remote_root` and the `_REMOTE` half of `paths.zone_env` had no production
caller. Every one of them was unit-tested and green. That is the shape of defect
this directory has now found four times — a mechanism wired to nothing, with
passing tests of the mechanism — so the assertion here is deliberately made
against what `main()` calls rather than against `build_context`, which any
caller could decline to pass a mapping to exactly as `main()` used to.

`_registry` is private and imported anyway, on purpose: a public seam invented so
that a test can avoid a private one would move the untested gap rather than close
it. It makes no model call, needs no credentials and no sandbox, which is the
property `conftest.py` states for this directory.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from cli.environment import layout_for
from cli.main import _registry
from cli.stream import Stream


@pytest.fixture()
def isolated_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """No meta file is visible unless a test writes one.

    `$XDG_CONFIG_HOME` is pointed at an empty directory rather than left alone:
    the resolution order ends at `~/.config/env_mgr/meta.json`, so a developer
    who happens to have one would otherwise decide whether these tests pass.
    """
    monkeypatch.delenv("ENV_MGR_META", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # `_registry` calls `install_excepthook`, which claims `threading.excepthook`
    # for the whole interpreter — correctly, since in production it *is* the
    # process owner. Under pytest it would outlive the test and swallow another
    # one's thread failures, so it is restored on teardown.
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    return tmp_path


def _mapping_of(root: Path, at: Path) -> Any:
    """The `Context` the run's `EnvManager` is bound to, via the registry."""
    layout = layout_for(at).create()
    registry = _registry(root, layout, Stream(), resume=False, variables={})
    return registry.get("env_mgr")._ctx.mapping


def test_a_run_with_no_meta_file_maps_nothing(package_root: Path, isolated_meta: Path) -> None:
    """**The control, and it is the configuration everything ships with.**

    Without it the positive test below measures nothing: a mapping that appeared
    whatever the configuration said would be indistinguishable from one that was
    read. This is also the assertion that the change is inert by default — the
    whole suite and every existing demo run are this case.
    """
    assert _mapping_of(package_root, isolated_meta / "run") == {}


def test_a_run_reads_its_weak_mappings_out_of_the_meta_file(
    package_root: Path, isolated_meta: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`$ENV_MGR_META` → `meta.configured_path` → `load` → `mapping_roots()` →
    `build_context(mapping=)` → `Context.mapping`, through `main()`'s own call.

    The **strong** mapping in the same file is the second control: it is the same
    bytes on both sides and has nothing to copy, so `mapping_roots()` must drop
    it. Were the filter not running, this test would still be green on the first
    assertion alone.
    """
    meta_path = isolated_meta / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "domains": [],
                "mappings": [
                    {
                        "local_root": "/var/tmp/example/state",
                        "remote_root": "/var/tmp/example/mirror",
                        "strength": "weak",
                        "transport": "ssh",
                        "target": "",
                    },
                    {
                        "local_root": "/var/tmp/example/shared",
                        "remote_root": "/var/tmp/example/shared",
                        "strength": "strong",
                        "transport": "ssh",
                        "target": "",
                    },
                ],
                "system_set": [],
                # Declared because `check_delete_scope` refuses a mapping whose
                # far side nobody accepted as destroyable. Not this test's
                # subject — the delete-scope pair below is — but a mapping can no
                # longer be configured without it, which is that guard's point.
                "deletable_roots": ["/var/tmp/example", "/data/yihou"],
            }
        )
    )
    monkeypatch.setenv("ENV_MGR_META", str(meta_path))
    assert _mapping_of(package_root, isolated_meta / "run") == {
        "/var/tmp/example/state": "/var/tmp/example/mirror"
    }


def test_a_run_builds_the_transport_its_mapping_declared(
    package_root: Path, isolated_meta: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the wiring, and the half that had no reader at all.

    `RemoteMapping` has carried `transport` and `target` since it was written and
    `mapping_roots()` dropped both, so a mapping could name a host and nothing
    would ever go there. This asserts the run path turns them into an object on
    `Context.transports`, keyed by the same `local_root` as `mapping` — because a
    transport under a different key is a transport `sync` will never find.

    Nothing connects: constructing an `Ssh` opens no session.
    """
    from env_mgr.remote.connection import Ssh

    meta_path = isolated_meta / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "domains": [],
                "mappings": [
                    {
                        "local_root": "/var/tmp/example/state",
                        "remote_root": "/data/yihou/handoffs",
                        "strength": "weak",
                        "transport": "ssh",
                        "target": "somehost",
                    }
                ],
                "system_set": [],
                # Declared because `check_delete_scope` refuses a mapping whose
                # far side nobody accepted as destroyable. Not this test's
                # subject — the delete-scope pair below is — but a mapping can no
                # longer be configured without it, which is that guard's point.
                "deletable_roots": ["/var/tmp/example", "/data/yihou"],
            }
        )
    )
    monkeypatch.setenv("ENV_MGR_META", str(meta_path))
    layout = layout_for(isolated_meta / "run").create()
    registry = _registry(package_root, layout, Stream(), resume=False, variables={})
    ctx = registry.get("env_mgr")._ctx

    # **True of this fixture, and false as a general rule** — the equality holds
    # because every mapping here is weak. R1b's configuration is the
    # counterexample and it was run: one `strong` mapping leaves `ctx.mapping`
    # empty and `ctx.transports` holding an `Ssh`, and the agent still got its
    # tools, because `_remote_tools` resolves against `far_roots`. So what this
    # asserts is that a **weak** mapping's transport is reachable under the key
    # `sync` will look it up by; it does not say a transport needs a `mapping`
    # entry to be usable.
    assert set(ctx.transports) == set(ctx.mapping), "a weak mapping's transport must share its key"
    transport = ctx.transports["/var/tmp/example/state"]
    assert isinstance(transport, Ssh) and transport.host == "somehost"


def test_a_run_with_no_meta_file_builds_no_transports(
    package_root: Path, isolated_meta: Path
) -> None:
    """The control. Empty is the shipped configuration and stays inert."""
    layout = layout_for(isolated_meta / "run").create()
    registry = _registry(package_root, layout, Stream(), resume=False, variables={})
    assert dict(registry.get("env_mgr")._ctx.transports) == {}


def _meta_with(path: Path, *, far_root: str, deletable: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "domains": [],
                "mappings": [
                    {
                        "local_root": "/var/tmp/example/state",
                        "remote_root": far_root,
                        "strength": "weak",
                        "transport": "ssh",
                        "target": "somehost",
                    }
                ],
                "system_set": [],
                "deletable_roots": deletable,
            }
        )
    )
    return path


def test_the_run_path_refuses_a_delete_outside_the_declared_roots(
    package_root: Path, isolated_meta: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring, not the function. `check_delete_scope` having a unit test says
    nothing about whether anything calls it — which is the gap that let five
    mechanisms sit unwired here.

    It refuses at **composition**: a bad meta file stops the run before a copy is
    attempted, which is the difference between a refusal and a post-mortem.
    """
    from env_mgr.protocols import PrepareRefused

    meta_path = _meta_with(
        isolated_meta / "meta.json",
        far_root="/home/someone-else/work",
        deletable=["/data/yihou"],
    )
    monkeypatch.setenv("ENV_MGR_META", str(meta_path))
    layout = layout_for(isolated_meta / "run").create()
    with pytest.raises(PrepareRefused, match="rsync --delete"):
        _registry(package_root, layout, Stream(), resume=False, variables={})


def test_the_run_path_allows_a_delete_inside_them(
    package_root: Path, isolated_meta: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control. Same file, same route, one field different — without it the
    test above would pass equally well against a run path that refused every
    mapping, or one that failed for some unrelated reason.
    """
    meta_path = _meta_with(
        isolated_meta / "meta.json",
        far_root="/data/yihou/handoffs",
        deletable=["/data/yihou"],
    )
    monkeypatch.setenv("ENV_MGR_META", str(meta_path))
    layout = layout_for(isolated_meta / "run").create()
    registry = _registry(package_root, layout, Stream(), resume=False, variables={})
    assert registry.get("env_mgr")._ctx.mapping == {
        "/var/tmp/example/state": "/data/yihou/handoffs"
    }


# --------------------------------------------------------------------------- #
# Which root a duplicated `local_root` resolves to, and what a bad one costs


def _meta_json(path: Path, mappings: list[dict[str, str]], deletable: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "domains": [],
                "mappings": mappings,
                "system_set": [],
                "deletable_roots": deletable,
            }
        )
    )
    return path


def test_a_strong_root_does_not_become_the_delete_target(
    package_root: Path, isolated_meta: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mapping_roots()` was `far_roots()` narrowed by the weak **key** set.

    `far_roots` keeps the last entry for a `local_root` regardless of strength,
    so declaring one root weak and then strong yielded the **strong** value under
    a weak key. That value is what `sync` hands `rsync -a --delete` and what
    `check_delete_scope` validates: the mount declared strong precisely because
    nothing should be copied to it became the copy's destination.

    Reaching it through the CLI rather than through `Meta` alone, because the
    unit was green — the defect is in which mapping wins, and only a caller that
    declares two shows it.
    """
    meta_path = _meta_json(
        isolated_meta / "meta.json",
        [
            {
                "local_root": "/var/tmp/example/state",
                "remote_root": "/data/yihou/weak",
                "strength": "weak",
                "transport": "ssh",
                "target": "somehost",
            },
            {
                "local_root": "/var/tmp/example/state",
                "remote_root": "/mnt/shared",
                "strength": "strong",
                "transport": "ssh",
                "target": "somehost",
            },
        ],
        deletable=["/data/yihou"],
    )
    monkeypatch.setenv("ENV_MGR_META", str(meta_path))
    layout = layout_for(isolated_meta / "run").create()
    registry = _registry(package_root, layout, Stream(), resume=False, variables={})
    ctx = registry.get("env_mgr")._ctx
    assert ctx.mapping == {"/var/tmp/example/state": "/data/yihou/weak"}
    # CONTROL, and the reason this is two assertions. `far_roots` is *supposed*
    # to keep the strong one — it is the tool surface's map and every mapping is
    # in scope there — so a `mapping_roots` that simply returned `far_roots`
    # unchanged, or one that returned `{}`, would both be caught only by pinning
    # the pair.
    assert ctx.far_roots["/var/tmp/example/state"] == "/mnt/shared"


def test_a_mapping_whose_transport_cannot_sync_is_a_precondition_not_a_traceback(
    package_root: Path, isolated_meta: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sync_transport` raises `ValueError`; `main()` catches five families and
    `ValueError` is none of them.

    So one bad line in an operator's meta file produced a raw traceback and an
    exit code meaning *unexpected failure* — about a typo. `_harness_env` set the
    shape a screen down: a `ValueError` that is really a precondition is re-raised
    as the family `main()` maps to PRECONDITION, naming the file.

    **This does not make a docker mapping work.** `sync_transport` is still the
    only constructor and still refuses one; separating tool reachability from
    sync capability changes what `Context.transports` holds, which is a seam.
    """
    from env_mgr.protocols import PrepareRefused

    meta_path = _meta_json(
        isolated_meta / "meta.json",
        [
            {
                "local_root": "/var/tmp/example/state",
                "remote_root": "/data/yihou/handoffs",
                "strength": "weak",
                "transport": "docker",
                "target": "some-container",
            }
        ],
        deletable=["/data/yihou"],
    )
    monkeypatch.setenv("ENV_MGR_META", str(meta_path))
    layout = layout_for(isolated_meta / "run").create()
    with pytest.raises(PrepareRefused) as exc:
        _registry(package_root, layout, Stream(), resume=False, variables={})
    # The operator has to be able to find the file and the reason in one line.
    assert str(meta_path) in str(exc.value)
    assert "docker" in str(exc.value)
