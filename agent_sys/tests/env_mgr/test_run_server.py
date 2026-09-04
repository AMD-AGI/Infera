# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`run_server` and the registry behind it, against real processes and real ports.

**Nothing here is mocked, and that is the point.** Every property this installer
has is a property of a process it does not own: that it survives its parent,
that its output does not hold a pipe open, that its pid still means what it
meant. A fake `Popen` would agree with whatever the code believed, and the two
defects found while writing this — an empty `/proc/<pid>/cmdline` before exec,
and a 25-second hang from an inherited pipe — are both invisible to one.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from env_mgr.installers.run_server import RunServerInstaller
from env_mgr.recipe import Item, Target
from env_mgr.servers import (
    REGISTRY_ENV_VAR,
    PortHolder,
    ServerRecord,
    owned_servers,
    port_conflict,
    record_spawn,
    records,
    starttime_of,
    stop_all,
)

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="needs procfs"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _alive(pid: int) -> bool:
    """Is `pid` a *running* process?

    **`os.kill(pid, 0)` is not this question** and was the first thing written
    here. It succeeds for a zombie — a process that has terminated and whose
    parent has not reaped it — so every "was it stopped?" assertion passed on
    corpses. These tests are the parent of what they start, so zombies are the
    normal case here.

    Read from `/proc` rather than by calling `env_mgr.servers`: the property
    under test is *the server is no longer running*, and asking the module under
    test whether it thinks so would make the assertion agree with the code by
    construction.
    """
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return data[data.rindex(")") + 2 :].split()[0] != "Z"


@pytest.fixture
def reaper():
    """Kill anything a test started, however the test ended.

    Tests here spawn detached process-group leaders on purpose. Without this a
    failing assertion would leave a listener on a real port of a shared host.
    """
    started: list[int] = []
    yield started
    for pid in started:
        for sig in (15, 9):
            try:
                os.killpg(pid, sig)
            except OSError:
                break
            time.sleep(0.1)


@pytest.fixture
def zone(tmp_path, monkeypatch):
    """A target directory and a registry path, wired as a real run wires them."""
    registry = tmp_path / "servers.json"
    monkeypatch.setenv(REGISTRY_ENV_VAR, str(registry))
    return Target(kind="workspace", name="z", path=str(tmp_path)), registry


def _item(**spec) -> Item:
    spec.setdefault("name", "srv")
    return Item(installer="run_server", importance="required", spec=spec)


def _serve_forever(port: int) -> str:
    return (
        f"exec python3 -c \"import socket,time;"
        f"s=socket.socket();s.setsockopt(1,2,1);s.bind(('127.0.0.1',{port}));"
        f's.listen(5);time.sleep(120)"'
    )


# --------------------------------------------------------------------- the start


def test_a_started_server_is_running_recorded_and_stoppable(zone, reaper):
    target, registry = zone
    port = _free_port()
    item = _item(command=_serve_forever(port), port=port, ready_timeout=15)

    (out,) = RunServerInstaller().install(item, target)
    assert out.level == "ok", out
    pid = out.details["pid"]
    reaper.append(pid)

    # Recorded, with the identity guard filled in.
    (record,) = records(registry)
    assert record.pid == pid
    assert record.port == port
    assert record.starttime == starttime_of(pid)

    # Really listening, not merely reported as such.
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", port)) == 0

    stops = stop_all(registry)
    assert [o.level for o in stops] == ["ok"], stops
    assert not _alive(pid)
    assert records(registry) == []


def test_the_server_outlives_the_process_that_started_it(tmp_path, reaper):
    """The property the whole design turns on, measured across a real exit.

    The installer runs inside `python -m env_mgr`, which exits seconds later. If
    the server did not survive that, the registry would be a list of dead pids.
    """
    port = _free_port()
    registry = tmp_path / "servers.json"
    script = tmp_path / "starter.py"
    script.write_text(
        "import sys;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r});"
        "from env_mgr.installers.run_server import RunServerInstaller;"
        "from env_mgr.recipe import Item, Target;"
        f"item = Item(installer='run_server', importance='required', spec={{'name':'srv',"
        f"'command':{_serve_forever(port)!r},'port':{port},'ready_timeout':15}});"
        f"out = RunServerInstaller().install(item, Target(kind='workspace', name='z',"
        f" path={str(tmp_path)!r}));"
        "print(out[0].level, out[0].details.get('pid'))"
    )
    env = dict(os.environ, **{REGISTRY_ENV_VAR: str(registry)})

    started = time.monotonic()
    # `capture_output=True` exactly as `_run_recipe` does it: this call returning
    # promptly is the assertion that the server's output is not on these pipes.
    proc = subprocess.run(
        ["python3", str(script)], capture_output=True, text=True, env=env, timeout=60
    )
    elapsed = time.monotonic() - started

    assert proc.returncode == 0, proc.stderr
    level, pid_text = proc.stdout.split()
    assert level == "ok", proc.stdout
    pid = int(pid_text)
    reaper.append(pid)

    assert elapsed < 30, (
        f"the starter took {elapsed:.1f}s to return. Measured 2026-09-04: a server "
        f"inheriting the captured pipes keeps them open for its whole life, and "
        f"`subprocess.run` waits for the pipes rather than the child"
    )
    assert _alive(pid), "the server died with the process that started it"
    assert [r.pid for r in records(registry)] == [pid]


def test_a_second_declaration_of_the_same_server_warns_and_starts_nothing(zone, reaper):
    """*Start it once; a duplicate warns* — answered by the registry alone.

    No scope mechanism and no second declaration route: which file declared the
    server does not enter into it, because the registry already knows what is
    up. Asserted on the registry rather than only on the level, because "did
    not start a second one" is the claim and a `warn` with two entries behind it
    would be the failure this is meant to catch.
    """
    target, registry = zone
    port = _free_port()
    item = _item(command=_serve_forever(port), port=port, ready_timeout=15)

    (first,) = RunServerInstaller().install(item, target)
    assert first.level == "ok", first
    reaper.append(first.details["pid"])

    (second,) = RunServerInstaller().install(item, target)
    assert second.level == "warn", second
    assert "already started by this run" in second.message
    assert second.details["pid"] == first.details["pid"]
    assert len(records(registry)) == 1, "it recorded a server it did not start"


def test_a_dead_entry_does_not_block_a_restart(zone, reaper):
    """*Live*, not merely *present*.

    A server that has since died is not a reason to refuse a restart, and the
    entry outlives it — `record_spawn` writes before readiness and nothing
    rewrites the file until shutdown. Without the `starttime` guard here, one
    crashed server would make its name unusable for the rest of the run.
    """
    target, registry = zone
    port = _free_port()
    (dead,) = RunServerInstaller().install(
        _item(command="exit 0", port=port, ready_timeout=5), target
    )
    assert dead.level == "fail"
    assert len(records(registry)) == 1

    (out,) = RunServerInstaller().install(
        _item(command=_serve_forever(port), port=port, ready_timeout=15), target
    )
    assert out.level == "ok", out
    reaper.append(out.details["pid"])


# -------------------------------------------------------------- the two failures


def test_a_server_that_exits_immediately_is_not_started(zone):
    target, registry = zone
    port = _free_port()
    item = _item(command="exit 3", port=port, ready_timeout=10)

    (out,) = RunServerInstaller().install(item, target)
    assert out.level == "fail"
    assert "exited immediately" in out.message
    assert out.details["rc"] == 3
    # Recorded anyway: the entry is written before this is known, on purpose.
    assert len(records(registry)) == 1


def test_a_server_that_never_binds_is_not_started(zone, reaper):
    target, registry = zone
    port = _free_port()
    item = _item(command="exec sleep 60", port=port, ready_timeout=1)

    (out,) = RunServerInstaller().install(item, target)
    reaper.append(out.details["pid"])
    assert out.level == "fail"
    assert "nothing is listening" in out.message
    assert out.details["port"] == port


def test_no_registry_means_no_server_is_started(tmp_path, monkeypatch):
    """Refusing is the point: a server nobody can stop is worse than no server."""
    monkeypatch.delenv(REGISTRY_ENV_VAR, raising=False)
    target = Target(kind="workspace", name="z", path=str(tmp_path))
    port = _free_port()

    (out,) = RunServerInstaller().install(_item(command=_serve_forever(port), port=port), target)
    assert out.level == "fail"
    assert "could never be stopped" in out.message
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", port)) != 0, "it started one anyway"


# ------------------------------------------------------------------ the port rule


def test_an_occupied_port_held_by_the_same_program_warns(tmp_path, reaper):
    """The measured wrinkle: the holder reports as `python3`, and must still warn.

    A process name key would call every Python server the same program; an
    interpreter-path key would too. The key is the **declared** program token,
    so what matters is that `myserver` appears in the holder's command line.
    """
    program = tmp_path / "myserver"
    program.write_text(
        "import socket,time\n"
        "s=socket.socket();s.setsockopt(1,2,1);s.bind(('127.0.0.1',PORT));"
        "s.listen(5);time.sleep(120)\n"
    )
    port = _free_port()
    program.write_text(program.read_text().replace("PORT", str(port)))
    holder = subprocess.Popen(
        ["python3", str(program)], start_new_session=True, stdout=subprocess.DEVNULL
    )
    reaper.append(holder.pid)
    _wait_bound(port)

    verdict = port_conflict(port, f"myserver --port {port}", "srv")
    assert verdict is not None
    assert verdict.level == "warn", verdict
    assert verdict.details["pid"] == holder.pid


def test_an_occupied_port_held_by_a_different_program_fails(tmp_path, reaper):
    port = _free_port()
    holder = subprocess.Popen(
        [
            "python3",
            "-c",
            "import socket,time;s=socket.socket();s.setsockopt(1,2,1);"
            f"s.bind(('127.0.0.1',{port}));s.listen(5);time.sleep(120)",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
    )
    reaper.append(holder.pid)
    _wait_bound(port)

    verdict = port_conflict(port, f"somethingelse --port {port}", "srv")
    assert verdict is not None
    assert verdict.level == "fail", verdict


def test_a_free_port_has_no_verdict():
    assert port_conflict(_free_port(), "anything", "srv") is None


def test_a_holder_whose_command_line_cannot_be_read_fails(tmp_path):
    """The third case: occupied, and the holder is unknowable.

    Measured on this host, a port held by another uid gives no pid from `ss`,
    nothing from `lsof`, and a `/proc/<pid>/fd` that cannot be walked. It cannot
    be *shown* to be the same program, so by the rule it is an error — and this
    is asserted through the seam because arranging a root-owned port inside a
    test is not something to do on a shared machine.
    """
    verdict = port_conflict(
        1234, "serena start-mcp-server", "srv", holder=PortHolder(occupied=True)
    )
    assert verdict is not None
    assert verdict.level == "fail"
    assert verdict.details["holder"] == "unreadable"


@pytest.mark.parametrize(
    "declared",
    [
        "myserver --port 1",
        "exec myserver --port 1",
        "env FOO=1 myserver --port 1",
        "FOO=1 myserver --port 1",
        "nohup myserver",
        "/opt/pkg/bin/myserver",
    ],
)
def test_a_wrapped_command_still_names_its_program(declared):
    """`exec myserver` declares *myserver*, not *exec*.

    Writing `command: exec myserver` is the good way to declare a server — it
    saves a shell process — and a token rule reading `argv[0]` blindly turned it
    into `exec`, which matches nothing, so a duplicate read as a stranger. The
    list of skipped words is short and closed on purpose; a command it does not
    cover degrades to `fail`, never to a false `warn`.
    """
    holder = PortHolder(occupied=True, pid=1, cmdline="/usr/bin/python3 /opt/pkg/bin/myserver")
    verdict = port_conflict(1, declared, "srv", holder=holder)
    assert verdict is not None
    assert verdict.level == "warn", declared


def test_an_unknown_wrapper_fails_rather_than_warning(tmp_path):
    """The direction the closed list is allowed to be wrong in."""
    holder = PortHolder(occupied=True, pid=1, cmdline="/usr/bin/python3 /opt/pkg/bin/myserver")
    verdict = port_conflict(1, "weirdwrapper myserver", "srv", holder=holder)
    assert verdict is not None
    assert verdict.level == "fail"


def test_the_same_program_is_recognised_through_an_interpreter(tmp_path):
    """The wrinkle in one assertion, at the level of the rule rather than a port.

    Both of these holders are `python3` by process name. Only the declaration
    tells them apart, which is why the key comes from the declaration.
    """
    serena_like = PortHolder(
        occupied=True,
        pid=1,
        cmdline="/home/u/.cache/uv/archive-v0/dAP4/bin/python /home/u/.local/bin/serena "
        "start-mcp-server --port 24282",
    )
    stranger = PortHolder(
        occupied=True, pid=2, cmdline="python3 -m http.server 18080 --bind 127.0.0.1"
    )
    same = port_conflict(24282, "serena start-mcp-server", "srv", holder=serena_like)
    other = port_conflict(24282, "serena start-mcp-server", "srv", holder=stranger)
    assert same is not None and same.level == "warn"
    assert other is not None and other.level == "fail"


def test_an_occupied_port_is_not_started_onto(zone, reaper):
    target, registry = zone
    port = _free_port()
    holder = subprocess.Popen(
        [
            "python3",
            "-c",
            "import socket,time;s=socket.socket();s.setsockopt(1,2,1);"
            f"s.bind(('127.0.0.1',{port}));s.listen(5);time.sleep(120)",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
    )
    reaper.append(holder.pid)
    _wait_bound(port)

    (out,) = RunServerInstaller().install(_item(command="elsewhere", port=port), target)
    assert out.level == "fail"
    assert records(registry) == [], "it recorded a server it did not start"


# --------------------------------------------------------------------- the guard


# The guard below is the one place in this feature where being wrong harms
# somebody outside the run: `stop_all` sending SIGTERM then SIGKILL to a process
# `agent_sys` does not own. That is not hypothetical on this host — three serena
# listeners of unknown ownership were measured on 24282-24284 the same day.
#
# **The first version of these two tests used `os.getpid()`** as the stand-in for
# a reused pid, which asserts the right thing but cannot be proven: the mutation
# that makes it fail — removing the guard — signals the test runner. A guard
# proven by a test that cannot be made to go red is a guard nobody has checked.
#
# So: a **sacrificial child**, and a fabricated record carrying its real pid with
# the wrong `starttime`. Pid reuse simulated without waiting for the kernel to
# recycle a number, and nothing aims at pytest.


@pytest.fixture
def sacrifice(reaper):
    """A live process that exists to be aimed at, and that may safely die.

    **`start_new_session=True` is a safety requirement here, not a detail.**
    `stop_all` calls `os.killpg`, which takes a *process-group* id. A child that
    is not a group leader shares the runner's group, so a mutation that removes
    the guard would aim `killpg` at pytest's own group — reintroducing exactly
    the hazard this harness exists to avoid. As its own leader it is the only
    thing in its group.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    reaper.append(proc.pid)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not _alive(proc.pid):
        time.sleep(0.02)
    assert _alive(proc.pid), "the sacrificial child never started"
    return proc


def _record(pid: int, starttime: str, name: str = "stale") -> ServerRecord:
    return ServerRecord(
        name=name, pid=pid, port=None, command="whatever", starttime=starttime,
        cmdline="", started_at=0.0,
    )


def test_the_harness_can_actually_kill_the_sacrifice(tmp_path, sacrifice):
    """The positive control, and the two tests below mean nothing without it.

    They assert that a process is **not** signalled. That is also what they
    would assert if `stop_all` were broken, or if the record never reached it,
    or if `killpg` were aimed at nothing. This one proves the aim is live: with
    the *correct* `starttime`, the very same call kills the very same child.
    """
    registry = tmp_path / "servers.json"
    record_spawn(registry, _record(sacrifice.pid, starttime_of(sacrifice.pid)))

    (out,) = stop_all(registry)
    assert out.level == "ok", out
    assert not _alive(sacrifice.pid)


def test_a_reused_pid_is_never_signalled(tmp_path, sacrifice):
    """A stale entry whose pid has been recycled must not be signalled.

    The record carries the sacrifice's **real pid** and a `starttime` that is
    not its own — which is precisely what a reused pid looks like from the
    registry's side.
    """
    registry = tmp_path / "servers.json"
    real = starttime_of(sacrifice.pid)
    assert real, "cannot run: the sacrifice has no readable starttime"
    record_spawn(registry, _record(sacrifice.pid, str(int(real) + 1)))

    (out,) = stop_all(registry)
    assert out.level == "info"
    assert "no longer ours" in out.message
    assert _alive(sacrifice.pid), "it signalled a process it does not own"


def test_an_unidentifiable_record_is_never_signalled(tmp_path, sacrifice):
    """A record we could not identify even when writing it is never signalled."""
    registry = tmp_path / "servers.json"
    record_spawn(registry, _record(sacrifice.pid, "", name="blank"))

    (out,) = stop_all(registry)
    assert out.level == "info"
    assert _alive(sacrifice.pid), "it signalled a process it does not own"


def test_a_corrupt_line_does_not_strand_the_entries_after_it(tmp_path):
    """A kill mid-write is the case this file exists for, so it must survive one."""
    registry = tmp_path / "servers.json"
    good = ServerRecord("g", os.getpid(), None, "c", "1", "", 0.0)
    registry.write_text('{"name": "half-writ')
    record_spawn(registry, good)
    assert [r.name for r in records(registry)] == ["g"]


# ------------------------------------------------------------------ the manager


def test_owned_servers_stops_on_normal_exit(zone, reaper):
    target, registry = zone
    port = _free_port()
    with owned_servers(registry):
        (out,) = RunServerInstaller().install(
            _item(command=_serve_forever(port), port=port, ready_timeout=15), target
        )
        assert out.level == "ok", out
        pid = out.details["pid"]
        reaper.append(pid)
        assert _alive(pid)
    assert not _alive(pid), "the manager did not stop what the block started"


def test_owned_servers_stops_when_the_block_raises(zone, reaper):
    """A handled error is the other half of the stated guarantee."""
    target, registry = zone
    port = _free_port()
    pid = None
    with pytest.raises(RuntimeError):
        with owned_servers(registry):
            (out,) = RunServerInstaller().install(
                _item(command=_serve_forever(port), port=port, ready_timeout=15), target
            )
            pid = out.details["pid"]
            reaper.append(pid)
            raise RuntimeError("the run failed")
    assert pid is not None
    assert not _alive(pid)


def test_owned_servers_with_no_registry_is_not_an_error():
    with owned_servers(None) as path:
        assert path is None


def _wait_bound(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"nothing came up on port {port}")
