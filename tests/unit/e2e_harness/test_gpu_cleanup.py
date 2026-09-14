###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Safety-boundary tests for exclusive-node ROCm process cleanup."""

from __future__ import annotations

import os
import signal

import pytest

from tests.e2e.harness import gpu_cleanup


@pytest.fixture(autouse=True)
def _avoid_rocm_subprocesses(monkeypatch):
    monkeypatch.setenv("INFERA_E2E_EXPECTED_GPU_COUNT", "1")
    monkeypatch.setattr(gpu_cleanup, "_process_gpu_map", lambda pids: {})


def _process(
    pid: int,
    *,
    uid: int | None = None,
    uids: tuple[int, int, int, int] | None = None,
    name: str = "python3",
):
    uid = os.getuid() if uid is None else uid
    return gpu_cleanup.GpuProcess(
        pid=pid,
        uids=uids or (uid, uid, uid, uid),
        name=name,
        cmdline=f"{name} -m infera.engine.sglang",
        start_time=f"start-{pid}",
    )


def test_cleanup_is_disabled_without_exclusive_ownership(monkeypatch):
    monkeypatch.delenv("INFERA_E2E_EXCLUSIVE", raising=False)
    monkeypatch.setattr(
        gpu_cleanup,
        "_gpu_process_names",
        lambda: pytest.fail("GPU processes must not be inspected on a shared node"),
    )

    assert gpu_cleanup.cleanup_exclusive_gpu_processes() == 0


def test_process_identity_reads_uid_and_start_time_from_proc():
    process = gpu_cleanup._read_process(os.getpid(), "pytest")

    assert process is not None
    assert process.uid == os.getuid()
    assert process.start_time.isdigit()


def test_cleanup_escalates_same_uid_process_from_term_to_kill(monkeypatch):
    process = _process(101, uid=1001)
    waits = iter(([process], []))
    signals = []
    active = True

    def send(processes, sig):
        nonlocal active
        signals.extend((process.pid, sig) for process in processes)
        if sig == signal.SIGKILL:
            active = False

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 1001)
    monkeypatch.setattr(
        gpu_cleanup,
        "_gpu_process_names",
        lambda: {process.pid: process.name} if active else {},
    )
    monkeypatch.setattr(gpu_cleanup, "_read_process", lambda pid, name="": process)
    monkeypatch.setattr(
        gpu_cleanup, "_wait_for_targets", lambda targets, timeout: list(next(waits))
    )
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: {})
    monkeypatch.setattr(gpu_cleanup, "_signal", send)

    assert gpu_cleanup.cleanup_exclusive_gpu_processes() == 1
    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]


def test_cleanup_never_signals_allowlisted_system_process(monkeypatch):
    process = _process(102, uid=1001, name="gpuagent")

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 1001)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {process.pid: process.name})
    monkeypatch.setattr(gpu_cleanup, "_read_process", lambda pid, name="": process)
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: {})
    monkeypatch.setattr(
        gpu_cleanup,
        "_signal",
        lambda processes, sig: pytest.fail("system GPU agents must not be signalled"),
    )

    assert gpu_cleanup.cleanup_exclusive_gpu_processes() == 0


def test_foreign_gpu_owner_is_reported_but_not_signalled(monkeypatch):
    process = _process(103, uid=1002)
    busy = {0: (280 * 1024**3, 288 * 1024**3)}

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 1001)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {process.pid: process.name})
    monkeypatch.setattr(gpu_cleanup, "_read_process", lambda pid, name="": process)
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: busy)
    monkeypatch.setattr(
        gpu_cleanup,
        "_signal",
        lambda processes, sig: pytest.fail("foreign GPU processes must not be signalled"),
    )

    with pytest.raises(
        gpu_cleanup.GpuCleanupError,
        match=rf"{gpu_cleanup.GPU_DIRTY_MARKER}.*pid={process.pid} uid={process.uid}",
    ):
        gpu_cleanup.cleanup_exclusive_gpu_processes()


def test_root_gpu_owner_is_protected_even_when_runner_is_root(monkeypatch):
    process = _process(104, uid=0)
    busy = {0: (280 * 1024**3, 288 * 1024**3)}

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 0)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {process.pid: process.name})
    monkeypatch.setattr(gpu_cleanup, "_read_process", lambda pid, name="": process)
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: busy)
    monkeypatch.setattr(
        gpu_cleanup,
        "_signal",
        lambda processes, sig: pytest.fail("root GPU processes must not be signalled"),
    )

    with pytest.raises(
        gpu_cleanup.GpuCleanupError,
        match=rf"{gpu_cleanup.GPU_DIRTY_MARKER}.*pid={process.pid} uid=0",
    ):
        gpu_cleanup.cleanup_exclusive_gpu_processes()


def test_effective_root_process_is_never_signalled(monkeypatch):
    process = _process(105, uids=(1001, 0, 0, 0))
    busy = {0: (280 * 1024**3, 288 * 1024**3)}

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 1001)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {process.pid: process.name})
    monkeypatch.setattr(gpu_cleanup, "_read_process", lambda pid, name="": process)
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: busy)
    monkeypatch.setattr(
        gpu_cleanup,
        "_signal",
        lambda processes, sig: pytest.fail("effective-root processes must not be signalled"),
    )

    with pytest.raises(
        gpu_cleanup.GpuCleanupError,
        match=rf"{gpu_cleanup.GPU_DIRTY_MARKER}.*uids=1001/0/0/0.*class=root",
    ):
        gpu_cleanup.cleanup_exclusive_gpu_processes()


def test_cleanup_rescans_for_new_same_uid_gpu_processes(monkeypatch):
    first = _process(106, uid=1001)
    second = _process(107, uid=1001)
    active = {first.pid: first}
    signals = []

    def names():
        return {pid: process.name for pid, process in active.items()}

    def read(pid, name=""):
        return active.get(pid) or {first.pid: first, second.pid: second}.get(pid)

    def send(processes, sig):
        signals.extend((process.pid, sig) for process in processes)
        for process in processes:
            active.pop(process.pid, None)
        if first.pid in [process.pid for process in processes]:
            active[second.pid] = second

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 1001)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", names)
    monkeypatch.setattr(gpu_cleanup, "_read_process", read)
    monkeypatch.setattr(gpu_cleanup, "_signal", send)
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: {})

    assert gpu_cleanup.cleanup_exclusive_gpu_processes() == 2
    assert signals == [(first.pid, signal.SIGTERM), (second.pid, signal.SIGTERM)]


def test_signal_revalidates_gpu_pid_and_uses_pidfd(monkeypatch):
    process = _process(108, uid=1001)
    sent = []
    closed = []

    monkeypatch.setattr(gpu_cleanup.os, "pidfd_open", lambda pid: 55)
    monkeypatch.setattr(gpu_cleanup.os, "close", closed.append)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {process.pid: process.name})
    monkeypatch.setattr(gpu_cleanup, "_same_process", lambda candidate: candidate == process)
    monkeypatch.setattr(
        gpu_cleanup.signal,
        "pidfd_send_signal",
        lambda pidfd, sig: sent.append((pidfd, sig)),
    )

    gpu_cleanup._signal([process], signal.SIGTERM)

    assert sent == [(55, signal.SIGTERM)]
    assert closed == [55]


def test_unattributed_busy_vram_fails_with_dirty_node_marker(monkeypatch):
    busy = {2: (270 * 1024**3, 288 * 1024**3)}

    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {})
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: busy)

    with pytest.raises(
        gpu_cleanup.GpuCleanupError,
        match=rf"{gpu_cleanup.GPU_DIRTY_MARKER}.*gpu2=270.0/288.0GiB",
    ):
        gpu_cleanup.cleanup_exclusive_gpu_processes()


def test_unreadable_rocm_process_data_fails_closed(monkeypatch):
    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup, "_wait_for_process_names", lambda timeout: None)

    with pytest.raises(
        gpu_cleanup.GpuCleanupError,
        match=rf"{gpu_cleanup.GPU_DIRTY_MARKER}.*process data is unavailable",
    ):
        gpu_cleanup.cleanup_exclusive_gpu_processes()


def test_unreadable_vram_data_fails_closed(monkeypatch):
    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {})
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: None)

    with pytest.raises(
        gpu_cleanup.GpuCleanupError,
        match=rf"{gpu_cleanup.GPU_DIRTY_MARKER}.*VRAM data is unavailable",
    ):
        gpu_cleanup.cleanup_exclusive_gpu_processes()


def test_vram_wait_observes_busy_then_free(monkeypatch):
    readings = iter(({0: (99, 100)}, {0: (1, 100)}))
    monkeypatch.setattr(gpu_cleanup, "_gpu_vram", lambda: next(readings))
    monkeypatch.setattr(gpu_cleanup.time, "sleep", lambda seconds: None)

    assert gpu_cleanup._wait_for_vram(10) == {}


def test_vram_wait_returns_busy_cards_at_timeout(monkeypatch):
    busy = {0: (99, 100)}
    monkeypatch.setattr(gpu_cleanup, "_gpu_vram", lambda: busy)

    assert gpu_cleanup._wait_for_vram(0) == busy


def test_vram_wait_rejects_partial_card_data(monkeypatch):
    monkeypatch.setenv("INFERA_E2E_EXPECTED_GPU_COUNT", "2")
    monkeypatch.setattr(gpu_cleanup, "_gpu_vram", lambda: {0: (1, 100)})

    assert gpu_cleanup._wait_for_vram(0) is None


def test_rocm_process_json_parser_handles_system_envelope(monkeypatch):
    monkeypatch.setattr(
        gpu_cleanup,
        "_rocm_smi_json",
        lambda *args: {
            "system": {
                "PID17976": "gpuagent, 0, 0, 0, 0",
                "PID22222": "python3, 0, 1234, 0, 0",
            }
        },
    )

    assert gpu_cleanup._gpu_process_names() == {
        17976: "gpuagent",
        22222: "python3",
    }


def _reclaim_node(monkeypatch, *, stopped, containers, vram):
    """Wire a node whose same-uid sweep is a no-op, so only reclaim is exercised."""
    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 1001)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {})
    monkeypatch.setattr(gpu_cleanup, "_gpu_vram", lambda: vram)
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: {})

    def docker(*args, timeout=None):
        if args[0] == "ps":
            return containers
        if args[0] == "stop":
            stopped.append(args[-1])
            return ""
        raise AssertionError(f"unexpected docker call: {args}")

    monkeypatch.setattr(gpu_cleanup, "_docker", docker)


def test_foreign_containers_are_left_alone_unless_reclaim_is_enabled(monkeypatch):
    stopped: list[str] = []
    _reclaim_node(
        monkeypatch,
        stopped=stopped,
        containers="someone_elses_training\trocm/pytorch\t\n",
        vram={0: (280 * 1024**3, 288 * 1024**3)},
    )
    monkeypatch.delenv("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS", raising=False)

    gpu_cleanup.cleanup_exclusive_gpu_processes()

    assert stopped == []


def test_reclaim_stops_foreign_container_holding_vram(monkeypatch):
    stopped: list[str] = []
    _reclaim_node(
        monkeypatch,
        stopped=stopped,
        containers="someone_elses_training\trocm/pytorch\t\n",
        vram={0: (280 * 1024**3, 288 * 1024**3)},
    )
    monkeypatch.setenv("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS", "1")

    gpu_cleanup.cleanup_exclusive_gpu_processes()

    assert stopped == ["someone_elses_training"]


def test_reclaim_does_not_run_on_an_empty_node(monkeypatch):
    stopped: list[str] = []
    _reclaim_node(
        monkeypatch,
        stopped=stopped,
        containers="idle_sidecar\trocm/pytorch\t\n",
        vram={0: (1 * 1024**3, 288 * 1024**3)},
    )
    monkeypatch.setenv("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS", "1")

    gpu_cleanup.cleanup_exclusive_gpu_processes()

    assert stopped == []


def test_reclaim_never_stops_a_kubernetes_pod(monkeypatch):
    stopped: list[str] = []
    _reclaim_node(
        monkeypatch,
        stopped=stopped,
        containers=(
            "k8s_serving_frontend\tnginx\t\n"
            "pod_sidecar\tbusybox\tio.kubernetes.pod.name=serving\n"
            "someone_elses_training\trocm/pytorch\t\n"
        ),
        vram={0: (280 * 1024**3, 288 * 1024**3)},
    )
    monkeypatch.setenv("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS", "1")

    gpu_cleanup.cleanup_exclusive_gpu_processes()

    assert stopped == ["someone_elses_training"]


def test_reclaim_keep_list_protects_named_containers(monkeypatch):
    stopped: list[str] = []
    _reclaim_node(
        monkeypatch,
        stopped=stopped,
        containers="licence_daemon\tvendor/lic\t\nsomeone_elses_training\trocm/pytorch\t\n",
        vram={0: (280 * 1024**3, 288 * 1024**3)},
    )
    monkeypatch.setenv("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS", "1")
    monkeypatch.setenv("INFERA_E2E_RECLAIM_KEEP", "licence_")

    gpu_cleanup.cleanup_exclusive_gpu_processes()

    assert stopped == ["someone_elses_training"]


def test_reclaim_is_skipped_when_docker_cannot_be_queried(monkeypatch):
    monkeypatch.setenv("INFERA_E2E_EXCLUSIVE", "1")
    monkeypatch.setenv("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS", "1")
    monkeypatch.setattr(gpu_cleanup.os, "getuid", lambda: 1001)
    monkeypatch.setattr(gpu_cleanup, "_gpu_process_names", lambda: {})
    monkeypatch.setattr(gpu_cleanup, "_gpu_vram", lambda: {0: (280 * 1024**3, 288 * 1024**3)})
    monkeypatch.setattr(gpu_cleanup, "_wait_for_vram", lambda timeout: {})
    monkeypatch.setattr(gpu_cleanup, "_docker", lambda *args, timeout=None: None)

    assert gpu_cleanup.cleanup_exclusive_gpu_processes() == 0


def test_reclaim_never_stops_inferas_own_containers(monkeypatch):
    """The disagg tier preflights one node while the other may already serve."""
    stopped: list[str] = []
    _reclaim_node(
        monkeypatch,
        stopped=stopped,
        containers=(
            "infera-e2e-disagg-run7-etcd\tetcd\t\n"
            "infera-utest-run7-atom-e2e\trocm/vllm\t\n"
            "tagged_by_label\trocm/vllm\tinfera.e2e.job_tag=run7\n"
            "someone_elses_training\trocm/pytorch\t\n"
        ),
        vram={0: (280 * 1024**3, 288 * 1024**3)},
    )
    monkeypatch.setenv("INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS", "1")

    gpu_cleanup.cleanup_exclusive_gpu_processes()

    assert stopped == ["someone_elses_training"]
