###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Worker/service placement, decoupled from what to launch.

The PD-mixed suite spawns workers as local subprocesses (see
:func:`tests.e2e.harness.adapter.spawn_worker`). PD-disaggregation instead runs
*every* service in a container on a specific remote node: etcd + the infera
router on node 0, a prefill worker on node 0, and a decode worker on node 1
(KV over Mooncake RDMA). The only thing left on the driver (login) host is the
thin pytest orchestrator + HTTP correctness client — nothing infera runs there.

A :class:`WorkerLauncher` captures only "how/where a process runs"; the engine
adapter still owns "what argv/env to run", so adding an engine (sglang/atom)
never touches placement and adding a placement backend never touches an engine.

:class:`SrunDockerLauncher` is the backend for this repo's SLURM clusters: it
builds the engine image on each node (once), then launches each service as a
detached container pinned to a node via ``srun --nodelist``. The container recipe
mirrors ``infera/tools/preflight/run_preflight_slurm.sh`` (privileged, host
network/IPC, ROCm devices, host ``/boot`` + ``libionic`` for RDMA) for GPU
workers; etcd/router run with just host networking.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

from .adapter import emit_reporter_line

# Repo root (…/tests/e2e/harness/launcher.py -> 4 up). Bind-mounted into each
# container at the SAME path so remote services run THIS working tree (requires
# the repo to live on storage shared across the allocation, e.g. a home/NFS
# mount — the normal case on these clusters).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Host libionic so in-container libibverbs matches the ionic RoCE kernel ABI
# (same mount the preflight uses). Harmless if the image entrypoint ignores it.
_LIBIONIC = "/usr/lib/x86_64-linux-gnu/libionic.so"

# Small public etcd image (same one run_tests.sh uses for the mixed tier).
ETCD_IMAGE = "quay.io/coreos/etcd:v3.5.14"
ETCD_PORT = 2379
ROUTER_PORT = 8000

_SRUN = ["srun", "--overlap", "--nodes=1", "--ntasks=1"]

# The Spur scheduler exposes only a subset of srun (no --overlap/--jobid): it's
# detected by its controller env var. On Spur we place work with plain
# ``srun -N1 -n1 [-p PART] -w NODE`` (each step self-allocates the pinned node);
# on stock SLURM we keep --overlap/--jobid to co-schedule inside an allocation.
_SPUR = bool(os.environ.get("SPUR_CONTROLLER_ADDR"))
_PART = os.environ.get("INFERA_E2E_SLURM_PARTITION")

# ROCm + RoCE flags for GPU worker containers (see module docstring).
_GPU_RDMA_FLAGS = [
    "--privileged",
    "--ipc",
    "host",
    "--shm-size",
    "16gb",
    "--ulimit",
    "memlock=-1",
    "--device",
    "/dev/kfd",
    "--device",
    "/dev/dri",
    "--group-add",
    "video",
    "--group-add",
    "render",
    "-v",
    "/boot:/boot:ro",
]


def _job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


def _srun(node: str, argv: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    if _SPUR:
        cmd = ["srun", "-N1", "-n1"] + (["-p", _PART] if _PART else []) + ["-w", node]
    else:
        cmd = list(_SRUN) + ["--nodelist", node]
        if _job_id():
            cmd += ["--jobid", _job_id()]
    cmd += argv
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@dataclass
class LaunchHandle:
    """A launched container on a node: node/container name + the (host, port) it
    advertises so the readiness poll / clients can reach it."""

    node: str
    container: str
    port: int
    advertise_host: str
    role: str  # "prefill" | "decode" | "router" | "etcd"
    log_path: str
    argv: list[str] = field(default_factory=list)


class WorkerLauncher(ABC):
    """How/where a service runs (placement), independent of the engine."""

    @abstractmethod
    def ensure_image(self, node: str) -> None:
        """Make the engine image available on ``node`` (build/pull as needed)."""

    @abstractmethod
    def start(
        self,
        *,
        node: str,
        argv: list[str],
        env: dict[str, str],
        container: str,
        advertise_host: str,
        port: int,
        role: str,
    ) -> LaunchHandle:
        """Launch one engine worker on ``node`` (non-blocking; readiness is polled
        separately via :func:`wait_workers_active`)."""

    @abstractmethod
    def is_running(self, handle: LaunchHandle) -> bool:
        """Whether the launched container is still up (dead-before-active guard)."""

    @abstractmethod
    def collect_logs(self, handle: LaunchHandle) -> str:
        """Fetch the container's logs (for diagnostics) and persist to its log_path."""

    @abstractmethod
    def stop(self, handle: LaunchHandle) -> None:
        """Tear the container down (frees the node's GPU for workers)."""


class SrunDockerLauncher(WorkerLauncher):
    """Place services as detached docker containers on SLURM nodes via srun.

    Builds ``dockerfile`` -> ``image`` on each node once (skipped if already
    present, unless ``rebuild``). ``model_dir`` (the shared pre-staged model
    tree) is bind-mounted read-only at the same path in every container so the
    in-container model path resolves identically across nodes.
    """

    def __init__(
        self,
        *,
        image: str,
        dockerfile: str,
        model_dir: str | None = None,
        log_dir: str | None = None,
        build_timeout: float = 3600.0,
        start_timeout: float = 300.0,
        shell_entrypoint: bool = False,
    ):
        self.image = image
        self.dockerfile = dockerfile
        # Some engine images set ENTRYPOINT ["/bin/bash"] (e.g. ATOM) rather than
        # the host-ionic injector that execs "$@" (vllm/sglang). For the former we
        # must override --entrypoint bash and pass just "-lc <cmd>", else the CMD's
        # leading "bash" is treated as a script ("cannot execute binary file").
        self.shell_entrypoint = shell_entrypoint
        self.model_dir = model_dir or os.environ.get("INFERA_E2E_MODEL_DIR")
        self.log_dir = log_dir or "/tmp/infera-e2e-logs"
        self.build_timeout = build_timeout
        self.start_timeout = start_timeout
        self._built: set[str] = set()
        os.makedirs(self.log_dir, exist_ok=True)

    # -- cleanup --------------------------------------------------------
    def cleanup_stale(self, nodes: list[str]) -> None:
        """Remove any leftover ``infera-e2e-*`` containers on each node before a
        run. A previous run that was interrupted/crashed can leave containers
        holding the fixed ports (etcd 2379/2380, router 8000, worker 30001/2), so
        the next run's etcd fails with 'address already in use'. Best-effort."""
        for node in dict.fromkeys(nodes):
            _srun(
                node,
                [
                    "bash",
                    "-lc",
                    "docker rm -f $(docker ps -aq --filter name=infera-e2e-) 2>/dev/null || true",
                ],
                timeout=self.start_timeout,
            )

    # -- image ----------------------------------------------------------
    def ensure_image(self, node: str) -> None:
        self.ensure_images([node])

    def ensure_images(self, nodes: list[str]) -> None:
        """Build ``image`` on each node (once per run; ``_built`` memoises).

        Always runs ``docker build`` — docker's layer cache makes an unchanged
        build a fast no-op and auto-rebuilds when the Dockerfile/context changed
        (no manual rebuild flag), and a node reused from an earlier tier already
        has the image cached so this is a quick no-op. ``--network=host`` is
        required so the build's RUN steps (pip) resolve DNS via the host resolver
        (these nodes list ``nameserver 127.0.0.1`` first, unreachable from a
        default bridge build netns). Built per node via ``_srun`` so it works on
        Spur (no ``srun --overlap`` fan-out) and stock SLURM alike."""
        missing = [n for n in dict.fromkeys(nodes) if n not in self._built]
        if not missing:
            return

        for node in missing:
            emit_reporter_line(
                f"[e2e disagg] building {self.image} on {node} from {self.dockerfile} "
                f"(docker cache; first build compiles Mooncake — many minutes)"
            )
            built = _srun(
                node,
                [
                    "docker",
                    "build",
                    "--network=host",
                    "-f",
                    os.path.join(REPO, self.dockerfile),
                    "-t",
                    self.image,
                    REPO,
                ],
                timeout=self.build_timeout,
            )
            if built.returncode != 0:
                raise RuntimeError(
                    f"docker build of {self.image} failed on {node} (rc={built.returncode}).\n"
                    f"--- stdout tail ---\n{built.stdout[-3000:]}\n"
                    f"--- stderr tail ---\n{built.stderr[-3000:]}"
                )
            self._built.add(node)

    # -- low-level container run ---------------------------------------
    def _engine_mounts(self) -> list[str]:
        mounts = ["-v", f"{REPO}:{REPO}"]
        # Mount when set (the dir lives on the compute NODE, not necessarily on
        # this orchestrator host, so don't gate on a local isdir check).
        if self.model_dir:
            mounts += ["-v", f"{self.model_dir}:{self.model_dir}:ro"]
        if os.path.exists(_LIBIONIC):
            mounts += ["-v", f"{_LIBIONIC}:/host-libionic/libionic.so"]
        return mounts

    def _run(
        self,
        node: str,
        container: str,
        image: str,
        docker_args: list[str],
        container_cmd: list[str],
    ) -> None:
        """`srun <node> docker run -d --name <container> <docker_args> <image> <cmd>`.
        Removes any same-named container first so a fixed name never lingers."""
        _srun(node, ["docker", "rm", "-f", container], timeout=self.start_timeout)
        cmd = ["docker", "run", "-d", "--name", container] + docker_args + [image] + container_cmd
        started = _srun(node, cmd, timeout=self.start_timeout)
        if started.returncode != 0:
            raise RuntimeError(
                f"failed to launch container {container} on {node} (rc={started.returncode}).\n"
                f"--- stderr ---\n{started.stderr[-2000:]}"
            )

    def _run_infera(
        self, node: str, container: str, argv: list[str], env: dict[str, str], *, gpu: bool
    ) -> None:
        """Run an infera process (worker or router) in the engine image: keep the
        image ENTRYPOINT (host-ionic inject), cd into the mounted repo, exec argv."""
        docker_args = ["--network", "host"]
        if gpu:
            docker_args += _GPU_RDMA_FLAGS
        docker_args += self._engine_mounts()
        for key, val in {"PYTHONPATH": REPO, **env}.items():
            docker_args += ["-e", f"{key}={val}"]
        inner = f"cd {shlex.quote(REPO)} && exec {shlex.join(argv)}"
        if self.shell_entrypoint:
            # ENTRYPOINT is a bare shell (e.g. ATOM's /bin/bash) -> override it and
            # pass only the shell flags, not a leading "bash" arg.
            docker_args += ["--entrypoint", "bash"]
            self._run(node, container, self.image, docker_args, ["-lc", inner])
        else:
            # ENTRYPOINT execs "$@" (host-ionic injector) -> hand it a full argv.
            self._run(node, container, self.image, docker_args, ["bash", "-lc", inner])

    # -- services -------------------------------------------------------
    def start_etcd(self, *, node: str, container: str, advertise_host: str) -> LaunchHandle:
        argv = [
            "etcd",
            "--advertise-client-urls",
            f"http://{advertise_host}:{ETCD_PORT}",
            "--listen-client-urls",
            f"http://0.0.0.0:{ETCD_PORT}",
        ]
        emit_reporter_line(f"[e2e disagg launch] etcd @ {node} ({advertise_host}:{ETCD_PORT})")
        self._run(node, container, ETCD_IMAGE, ["--network", "host"], argv)
        return LaunchHandle(
            node=node,
            container=container,
            port=ETCD_PORT,
            advertise_host=advertise_host,
            role="etcd",
            log_path=os.path.join(self.log_dir, f"infera-e2e-disagg-{container}.log"),
            argv=argv,
        )

    def start_router(
        self,
        *,
        node: str,
        container: str,
        advertise_host: str,
        etcd_endpoint: str,
        etcd_prefix: str,
        model: str,
    ) -> LaunchHandle:
        # round-robin (workers run --no-enable-kv-events, so kv-aware has no feed);
        # http request transport; etcd discovery. Tokenizer path is required.
        argv = [
            "python3",
            "-m",
            "infera.server",
            "--host",
            "0.0.0.0",
            "--port",
            str(ROUTER_PORT),
            "--discovery-backend",
            "etcd",
            "--etcd-endpoint",
            etcd_endpoint,
            "--etcd-prefix",
            etcd_prefix,
            "--request-transport",
            "http",
            "--router-policy",
            "round-robin",
            "--router-tokenizer-path",
            model,
        ]
        emit_reporter_line(f"[e2e disagg launch] router @ {node} ({advertise_host}:{ROUTER_PORT})")
        emit_reporter_line(f"[e2e disagg cmd] {shlex.join(argv)}")
        self._run_infera(node, container, argv, {}, gpu=False)
        return LaunchHandle(
            node=node,
            container=container,
            port=ROUTER_PORT,
            advertise_host=advertise_host,
            role="router",
            log_path=os.path.join(self.log_dir, f"infera-e2e-disagg-{container}.log"),
            argv=argv,
        )

    def start(
        self,
        *,
        node: str,
        argv: list[str],
        env: dict[str, str],
        container: str,
        advertise_host: str,
        port: int,
        role: str,
    ) -> LaunchHandle:
        emit_reporter_line(f"[e2e disagg launch] {role} @ {node} ({advertise_host}:{port})")
        emit_reporter_line(f"[e2e disagg cmd] {shlex.join(argv)}")
        self._run_infera(node, container, argv, env, gpu=True)
        return LaunchHandle(
            node=node,
            container=container,
            port=port,
            advertise_host=advertise_host,
            role=role,
            log_path=os.path.join(self.log_dir, f"infera-e2e-disagg-{container}.log"),
            argv=argv,
        )

    # -- lifecycle ------------------------------------------------------
    def is_running(self, handle: LaunchHandle) -> bool:
        out = _srun(
            handle.node,
            ["docker", "inspect", "-f", "{{.State.Running}}", handle.container],
            timeout=self.start_timeout,
        )
        return out.returncode == 0 and out.stdout.strip() == "true"

    def collect_logs(self, handle: LaunchHandle) -> str:
        out = _srun(handle.node, ["docker", "logs", handle.container], timeout=self.start_timeout)
        text = (out.stdout or "") + (out.stderr or "")
        try:
            with open(handle.log_path, "w") as f:
                f.write(text)
        except OSError:
            pass
        return text

    def stop(self, handle: LaunchHandle) -> None:
        # Persist logs before removal so a passing OR failing run keeps a record.
        self.collect_logs(handle)
        _srun(handle.node, ["docker", "rm", "-f", handle.container], timeout=self.start_timeout)


async def wait_url_ok(url: str, *, timeout: float, launcher=None, handle=None) -> None:
    """Block until ``url`` answers HTTP 200 (used for the router's /health).

    If a ``launcher``/``handle`` pair is given, a container that dies before the
    URL comes up fails fast (with its logs) instead of waiting to the deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with httpx.AsyncClient(timeout=5.0) as c:
        while loop.time() < deadline:
            if launcher is not None and handle is not None and not launcher.is_running(handle):
                logs = launcher.collect_logs(handle)
                raise RuntimeError(
                    f"{handle.role} on {handle.node} exited before {url} came up; "
                    f"see {handle.log_path}\n--- log tail ---\n{logs[-2000:]}"
                )
            try:
                r = await c.get(url)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    if launcher is not None and handle is not None:
        launcher.collect_logs(handle)
    raise TimeoutError(f"{url} not ready within {timeout}s")


async def wait_workers_active(
    launcher: WorkerLauncher,
    server_url: str,
    handles: list[LaunchHandle],
    *,
    timeout: float,
) -> None:
    """Block until every handle's worker registers ``active`` in ``/v1/workers``.

    Matches a worker by its ``:port`` suffix (tolerant of advertise-host vs the
    exact registered worker_id). If a container dies before becoming active, the
    logs are dumped and the wait fails fast instead of hanging to the deadline.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    pending = {h.container: h for h in handles}
    async with httpx.AsyncClient(timeout=5.0) as c:
        while loop.time() < deadline:
            for h in list(pending.values()):
                if not launcher.is_running(h):
                    logs = launcher.collect_logs(h)
                    raise RuntimeError(
                        f"{h.role} worker on {h.node} exited before becoming active; "
                        f"see {h.log_path}\n--- log tail ---\n{logs[-2000:]}"
                    )
            try:
                r = await c.get(f"{server_url}/v1/workers")
                active = {
                    w["worker_id"]
                    for w in r.json().get("workers", [])
                    if w.get("status") == "active"
                }
                for h in list(pending.values()):
                    if any(wid.endswith(f":{h.port}") for wid in active):
                        emit_reporter_line(f"[e2e disagg] {h.role} @ {h.node} active")
                        pending.pop(h.container, None)
            except (httpx.HTTPError, KeyError, ValueError):
                pass
            if not pending:
                return
            await asyncio.sleep(3)

    late = ", ".join(f"{h.role}@{h.node}" for h in pending.values())
    for h in pending.values():
        launcher.collect_logs(h)
    raise TimeoutError(
        f"PD workers not active within {timeout}s: {late}. See logs under {launcher.log_dir}."
    )
