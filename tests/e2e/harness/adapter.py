###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Engine adapter contract + generic worker lifecycle.

    EngineAdapter   maps EngineParams -> a `python -m infera.engine.<engine>`
                    argv + env, and knows engine-specific quirks.
    spawn_worker()  pick a port, build the argv via the adapter, launch the
                    subprocess, and poll /v1/workers until it registers active.
    GpuAllocator    hands out disjoint GPU-index ranges so co-located workers
                    in one test never collide.

Add an engine by subclassing EngineAdapter (see tests/e2e/pd_mixed/sglang/conftest.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from .params import EngineParams

# Terminal reporter + capture manager, set by the e2e conftest so harness lines
# (launch command, setup steps) show live in the run output. Suspending capture
# is essential: a plain terminalreporter.write during a running test lands in
# that test's captured buffer (only replayed on failure), so passing cases would
# print nothing. Falls back to print() when unset (harness used outside e2e).
_reporter = None
_capman = None


def set_reporter(reporter, capman=None) -> None:
    global _reporter, _capman
    _reporter = reporter
    _capman = capman


def emit_reporter_line(msg: str) -> None:
    """Write ``msg`` to the terminal reporter live (capture suspended, flushed),
    or print() with a flush when no reporter is bound."""
    if _reporter is None:
        print(msg, flush=True)
        return
    cm = _capman.global_and_fixture_disabled() if _capman is not None else contextlib.nullcontext()
    with cm:
        _reporter.ensure_newline()
        _reporter.write_line(msg, cyan=True)
        with contextlib.suppress(Exception):
            _reporter._tw.flush()


# Back-compat alias for internal callers.
_emit = emit_reporter_line


async def _run_setup(cmds: tuple[str, ...]) -> None:
    """Run a case's setup shell commands (e.g. an extra ``pip install``) in the
    engine container once before the worker launches. Raises on the first that
    fails so the case fails loudly instead of on a confusing import error."""
    loop = asyncio.get_running_loop()
    for cmd in cmds:
        _emit(f"[e2e setup] {cmd}")
        done = await loop.run_in_executor(
            None,
            lambda c=cmd: subprocess.run(c, shell=True, capture_output=True, text=True),
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"e2e setup command failed (rc={done.returncode}): {cmd}\n"
                f"--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
            )


_GPU_FREE_TIMEOUT = 90


@dataclass
class WorkerHandle:
    worker_id: str
    port: int
    gpu_ids: list[int]
    proc: subprocess.Popen
    log_path: str


class GpuAllocator:
    """Hands out contiguous GPU-index ranges from a fixed pool (per test)."""

    def __init__(self, total: int):
        self._total = total
        self._next = 0

    def take(self, n: int) -> list[int]:
        if self._next + n > self._total:
            raise RuntimeError(
                f"GPU over-allocation: need {n} more, only {self._total - self._next} "
                f"left of {self._total}"
            )
        ids = list(range(self._next, self._next + n))
        self._next += n
        return ids


class EngineAdapter(ABC):
    """Per-engine translation of EngineParams -> launcher argv/env."""

    engine: str = ""  # "sglang" | "vllm" | "atom"
    module: str = ""  # e.g. "infera.engine.sglang"

    @abstractmethod
    def gpus_per_worker(self, params: EngineParams) -> int:
        """How many GPUs a single worker consumes."""

    @abstractmethod
    def build_argv(
        self,
        params: EngineParams,
        *,
        port: int,
        host: str,
        server_ctx: dict,
        gpu_ids: list[int],
    ) -> list[str]:
        """Full `python3 -m <module> ...` argv for one worker."""

    def worker_env(self, params: EngineParams, *, gpu_ids: list[int]) -> dict[str, str]:
        """Worker subprocess env: GPU visibility + any per-case ``extra_env``
        (some models need a specific env var to run; it wins on collisions)."""
        env = {"HIP_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)}
        env.update(dict(params.extra_env))
        return env

    def pick_port(self) -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def worker_id(self, host: str, port: int) -> str:
        return f"{host}:{port}"

    # -- PD-disaggregation (opt-in) -------------------------------------
    # An engine gains a cross-node prefill/decode suite by overriding the two
    # methods below (see tests/e2e/pd_disag/vllm/conftest.py). Engines that
    # haven't implemented it yet inherit the NotImplementedError, which the
    # disagg fixture turns into a clean skip — so PD-mixed is entirely unaffected.
    supports_disagg: bool = False

    def build_disagg_argv(
        self,
        params: EngineParams,
        role,
        *,
        port: int,
        host: str,
        server_ctx: dict,
        advertise_host: str,
        gpu_ids: list[int],
    ) -> list[str]:
        """Full ``python3 -m <module> ...`` argv for one PD worker of ``role``
        (a :class:`~.params.DisaggRole`). Must set the disaggregation transport
        (RDMA KV connector) + a routable ``--advertise-host``."""
        raise NotImplementedError(f"{self.engine} has no PD-disaggregated adapter yet")

    def disagg_worker_env(
        self,
        params: EngineParams,
        role,
        *,
        advertise_host: str,
        gpu_ids: list[int],
        gid_index: str,
    ) -> dict[str, str]:
        """Env for one PD worker: GPU visibility + the RDMA data-plane host/GID +
        any per-case ``extra_env``. Defaults to the mixed env (GPU only)."""
        return self.worker_env(params, gpu_ids=gpu_ids)


async def spawn_worker(
    adapter: EngineAdapter,
    server_ctx: dict,
    params: EngineParams,
    *,
    gpu_ids: list[int],
    procs: list,
    host: str = "127.0.0.1",
) -> WorkerHandle:
    """Run the case's setup commands, then launch one engine worker and wait
    until it registers as active (up to ``params.server_ready_timeout`` seconds).

    Appends the live ``(proc, log_file)`` to the caller-owned ``procs`` list so
    a fixture can tear everything down.
    """
    await _run_setup(params.setup)
    # Transient port collisions: vLLM's EngineCore, Mooncake's transfer engine,
    # etc. pick their INTERNAL ports from the OS ephemeral range via bind(:0),
    # which races against the heavy outbound-connection churn of multi-GPU
    # serving (NCCL / torch.distributed / etcd / nats / Mooncake). The picked
    # port can be grabbed before the server binds it -> "address already in use"
    # and the worker dies before becoming active. Those ports live inside the
    # engine libraries (not ours), so the race can't be prevented at the source;
    # a relaunch re-picks fresh ports and essentially always succeeds. Retry
    # ONLY that specific failure — a real crash (OOM, bad config) is re-raised.
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            return await _launch_once(
                adapter, server_ctx, params, gpu_ids=gpu_ids, procs=procs, host=host
            )
        except RuntimeError as exc:
            if attempt >= attempts or not _is_port_collision(exc):
                raise
            _emit(
                f"[e2e] {adapter.engine} worker hit a transient port collision "
                f"(attempt {attempt}/{attempts}) — relaunching with fresh ports"
            )
            await asyncio.sleep(5)
    raise AssertionError("unreachable")  # loop either returns or raises


def _gpu_vram() -> dict[int, tuple[int, int]]:
    """``{gpu_index: (used_bytes, total_bytes)}`` via rocm-smi; ``{}`` if it
    can't be read. Indices are physical card indices, which match the
    HIP_VISIBLE_DEVICES indices the harness assigns (the container isn't
    launched with a restricted visibility list)."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        data = json.loads(out)
    except Exception:
        return {}
    vram: dict[int, tuple[int, int]] = {}
    for card, info in data.items():
        if not card.startswith("card"):
            continue
        try:
            idx = int(card[len("card") :])
            used = int(info["VRAM Total Used Memory (B)"])
            total = int(info["VRAM Total Memory (B)"])
        except (KeyError, ValueError, TypeError):
            continue
        vram[idx] = (used, total)
    return vram


async def _await_gpus_freed(gpu_ids: list[int], timeout: float = _GPU_FREE_TIMEOUT) -> None:
    """Block until every GPU in ``gpu_ids`` has released its VRAM before the
    next worker launches on them.

    A just-torn-down worker — especially a huge model — can keep holding VRAM
    for tens of seconds after its process exits; starting the next worker on the
    same GPUs then HIP-OOMs. A GPU counts as free once it uses < 5% of its
    capacity (idle baseline is well under that). No-op if rocm-smi can't be read
    (so it never hangs on a non-ROCm host) or on timeout (proceed anyway)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        vram = _gpu_vram()
        if not vram:
            return  # can't measure → don't block
        busy = [g for g in gpu_ids if vram.get(g, (0, 1))[0] >= 0.05 * vram.get(g, (0, 1))[1]]
        if not busy or loop.time() >= deadline:
            return
        await asyncio.sleep(3)


# A worker that died before becoming active because some component failed to
# bind an auto-picked port. Matches the vLLM (ZMQError), Mooncake (tcp_transport),
# and generic socket phrasings.
_PORT_COLLISION_RE = re.compile(r"address already in use|bind:\s*address already in use", re.I)


def _is_port_collision(exc: BaseException) -> bool:
    """True iff the failed worker's log shows an 'address already in use' bind
    failure. The worker's log path is embedded in the RuntimeError message
    (``... see <path>``)."""
    m = re.search(r"see (\S+)", str(exc))
    if not m:
        return False
    try:
        with open(m.group(1), errors="replace") as f:
            return bool(_PORT_COLLISION_RE.search(f.read()))
    except OSError:
        return False


async def _launch_once(
    adapter: EngineAdapter,
    server_ctx: dict,
    params: EngineParams,
    *,
    gpu_ids: list[int],
    procs: list,
    host: str,
) -> WorkerHandle:
    # Reusing GPUs across back-to-back cases: wait for the prior worker's VRAM
    # to be released before launching, so a lingering allocation doesn't OOM us.
    await _await_gpus_freed(gpu_ids)
    port = adapter.pick_port()
    argv = adapter.build_argv(params, port=port, host=host, server_ctx=server_ctx, gpu_ids=gpu_ids)
    env = os.environ.copy()
    env.update(adapter.worker_env(params, gpu_ids=gpu_ids))

    # Persist to the host-mounted /e2e-logs when present (run_tests.sh mounts it),
    # else fall back to /tmp for a bare (unmounted) dev run.
    log_dir = "/e2e-logs" if os.path.isdir("/e2e-logs") else "/tmp"
    os.makedirs(log_dir, exist_ok=True)
    # Tag the log with the task name (engine + case id) and launch time so a
    # persisted host dir keeps a readable, non-clobbering history across runs.
    task = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{adapter.engine}-{params.id()}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"infera-e2e-{task}-{stamp}.log")
    log_file = open(log_path, "w")  # noqa: SIM115 (closed on teardown)

    _emit(f"[e2e launch] {adapter.engine} (gpus={gpu_ids}): {shlex.join(argv)}")
    _emit(f"[e2e log] {log_path}")

    proc = subprocess.Popen(
        argv,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group for clean signal delivery
    )
    entry = (proc, log_file)
    procs.append(entry)

    worker_id = adapter.worker_id(host, port)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + params.server_ready_timeout
    async with httpx.AsyncClient(timeout=5.0) as c:
        while loop.time() < deadline:
            if proc.poll() is not None:
                # Dead before active: reap it so its GPU frees and drop it from
                # the teardown list before failing.
                _terminate_group(proc)
                _close(log_file)
                if entry in procs:
                    procs.remove(entry)
                raise RuntimeError(
                    f"{adapter.engine} worker exited with code {proc.returncode} "
                    f"before becoming active; see {log_path}"
                )
            try:
                r = await c.get(f"{server_ctx['url']}/v1/workers")
                for w in r.json()["workers"]:
                    if w["worker_id"] == worker_id and w["status"] == "active":
                        return WorkerHandle(
                            worker_id=worker_id,
                            port=port,
                            gpu_ids=gpu_ids,
                            proc=proc,
                            log_path=log_path,
                        )
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)

    _terminate_group(proc)
    _close(log_file)
    if entry in procs:
        procs.remove(entry)
    raise TimeoutError(
        f"{adapter.engine} worker {worker_id} did not become active within "
        f"{params.server_ready_timeout}s; see {log_path}"
    )


def _pgroup_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_pgroup_gone(pgid: int, timeout: float) -> bool:
    """Block until every process in the group has exited, or timeout."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pgroup_alive(pgid):
            return True
        time.sleep(0.25)
    return not _pgroup_alive(pgid)


def _reap(proc: subprocess.Popen) -> None:
    """Reap an already-signalled child so it doesn't linger as a <defunct>
    zombie (its parent is us, so init won't collect it)."""
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _terminate_group(proc: subprocess.Popen) -> None:
    """SIGTERM the worker's process group, reap the leader, then wait for the
    rest of the group to exit so GPU memory is freed before the next spawn.

    Gotchas:
    - sglang schedulers sleep ~5s on shutdown while still holding VRAM.
    - The leader's parent is pytest (not init), so it MUST be reaped with
      proc.wait() or it lingers as a zombie. Reparented grandchildren are
      reaped by the container init (`docker run --init`).
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        _reap(proc)
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        _reap(proc)
        return
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    _wait_pgroup_gone(pgid, 10)  # GPU-free barrier before returning


def _close(log_file) -> None:
    try:
        log_file.close()
    except Exception:
        pass


def teardown_workers(procs: list) -> None:
    """Graceful SIGTERM (then SIGKILL) each spawned worker's process group,
    waiting for the whole group (incl. GPU-holding schedulers) to exit."""
    for proc, log_file in procs:
        _terminate_group(proc)
        _close(log_file)
