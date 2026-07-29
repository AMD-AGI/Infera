###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SLURM topology discovery for the cross-node PD-disaggregated e2e suite.

The PD-disaggregated suite places a prefill worker on one node and a decode
worker on another, so the harness needs to know *which* nodes it may use and
each node's routable IP (for ``--advertise-host`` + Mooncake bootstrap). All of
that is auto-discovered from the active SLURM allocation so a run needs no
hand-maintained host list; a small set of optional overrides covers clusters
where auto-discovery guesses wrong.

Optional env overrides (all unset => pure auto-discovery):
  INFERA_E2E_NODE_IPS  ``node=ip,node=ip`` — pin a node's advertise/data-plane
                       IP instead of the ``hostname -I`` auto-pick.
  INFERA_E2E_RESERVATION / INFERA_E2E_SRUN_EXTRA /
  INFERA_E2E_SLURM_QOS_FALLBACK  — scheduler flags, see :func:`srun_argv`.
  INFERA_E2E_GID_INDEX / INFERA_E2E_WORKER_ENV / INFERA_E2E_BUILD_ARGS — the
                       fabric's KV transport (see :func:`kv_transport_env`).

Everything here is best-effort and side-effect free: on a non-SLURM host (or no
allocation) the discovery helpers return empty/None and the suite skips.
"""

from __future__ import annotations

import functools
import os
import re
import shlex
import shutil
import socket
import subprocess

# RoCEv2 GID index default — the ULA (routable) GID on the repo's ionic fabric;
# both the PD bench (MORI_IB_GID_INDEX=1) and regression doc 04 use index 1.
DEFAULT_GID_INDEX = "1"

_SRUN_TIMEOUT = 60

# The Spur scheduler exposes only a subset of srun (no --overlap/--jobid).
_SPUR = bool(os.environ.get("SPUR_CONTROLLER_ADDR"))

# Burst QoS is a FALLBACK, never the default: steps run on the cluster's normal
# QoS until one is refused for a group node limit, then the rest of the run keeps
# the fallback so that stall is paid once instead of per step.
_QOS_FALLBACK = os.environ.get("INFERA_E2E_SLURM_QOS_FALLBACK", "amd-burst-qos")
_QOS_PROBE_TIMEOUT = float(os.environ.get("INFERA_E2E_QOS_PROBE_TIMEOUT", "30"))

# Every disagg step carries this name (traceable to its CI run, and matched by
# ci.yml's `infera-ci-` reclaim filter) and this ceiling. The longest step by far
# is the image build, so the default is generous.
JOB_NAME = "infera-ci-" + (os.environ.get("INFERA_E2E_JOB_TAG") or "disag-local")
STEP_TIME = os.environ.get("INFERA_E2E_SLURM_TIME", "01:00:00")
_QOS_LIMIT_RE = re.compile(
    r"QOSGrp\w*Limit|QOSMax\w*Limit|reached terminal state before allocation|"
    r"Unable to allocate resources",
    re.I,
)
_qos_fallback_on = False


def have_slurm() -> bool:
    """Whether the SLURM client tooling this suite drives is on PATH."""
    return shutil.which("srun") is not None and shutil.which("scontrol") is not None


def in_allocation() -> bool:
    """Whether we're inside a live SLURM allocation (salloc/sbatch)."""
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"))


def _job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


def srun_argv(node: str, *, job: str = "") -> list[str]:
    """The ``srun`` prefix pinning one task to ``node`` — the one place scheduler
    flags are assembled. ``INFERA_E2E_RESERVATION`` keeps a disagg run's many
    short steps on the reserved pair; ``INFERA_E2E_SRUN_EXTRA`` adds site flags."""
    part = os.environ.get("INFERA_E2E_SLURM_PARTITION")
    if _SPUR:
        argv = ["srun", "-N1", "-n1"] + (["-p", part] if part else []) + ["-w", node]
    else:
        argv = ["srun", "--overlap", "--nodes=1", "--ntasks=1", "--nodelist", node]
        if _job_id():
            argv += ["--jobid", _job_id()]
    # Name and timebox every step: unnamed, SLURM labels them after the command
    # ("docker") and leaves them UNLIMITED, so a leaked one is neither traceable
    # to its CI run nor self-expiring. The prefix matches ci.yml's reclaim filter.
    argv += ["-J", job or JOB_NAME, "-t", STEP_TIME]
    reservation = os.environ.get("INFERA_E2E_RESERVATION")
    if reservation:
        argv.append(f"--reservation={reservation}")
    if _qos_fallback_on:
        argv.append(f"--qos={_QOS_FALLBACK}")
    return argv + shlex.split(os.environ.get("INFERA_E2E_SRUN_EXTRA", ""))


def run_on_node(node: str, argv: list[str], *, timeout: float = _SRUN_TIMEOUT):
    """Run ``argv`` on ``node`` and capture its output (never raises on rc!=0)."""
    return subprocess.run(srun_argv(node) + argv, capture_output=True, text=True, timeout=timeout)


def probe_qos(node: str) -> None:
    """Choose this run's QoS once, before the stack comes up. A refused step does
    not fail fast — it queues — so a trivial step is timeboxed instead; if it does
    not land, every later step takes the burst QoS (and fails normally if it too
    is refused). Idempotent, and a no-op with INFERA_E2E_SLURM_QOS_FALLBACK=''."""
    global _qos_fallback_on
    if _qos_fallback_on or not _QOS_FALLBACK:
        return
    # Its own name so the scancel below cannot take the run's other steps with it.
    job = f"{JOB_NAME}-qosprobe-{os.getpid()}"
    try:
        proc = subprocess.run(
            srun_argv(node, job=job) + ["true"],
            capture_output=True,
            text=True,
            timeout=_QOS_PROBE_TIMEOUT,
        )
        if proc.returncode == 0:
            return
        blocked = bool(_QOS_LIMIT_RE.search((proc.stdout or "") + (proc.stderr or "")))
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
    except subprocess.TimeoutExpired:
        blocked, detail = True, [f"still queued after {_QOS_PROBE_TIMEOUT:.0f}s"]
    finally:
        # Killing the srun client does not cancel the queued job; drop it by name
        # on every path (a no-op once it has already run).
        subprocess.run(["scancel", "-n", job], capture_output=True, timeout=_SRUN_TIMEOUT)
    if blocked:
        _qos_fallback_on = True
        print(
            f"[e2e disagg] default QoS unavailable ({' '.join(detail)}) — "
            f"using --qos={_QOS_FALLBACK} for this run",
            flush=True,
        )


def _parse_ip_overrides() -> dict[str, str]:
    """``INFERA_E2E_NODE_IPS='n1=10.0.0.1,n2=10.0.0.2'`` -> ``{n1: 10.0.0.1}``."""
    raw = os.environ.get("INFERA_E2E_NODE_IPS", "")
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if "=" in item:
            node, ip = item.split("=", 1)
            if node.strip() and ip.strip():
                out[node.strip()] = ip.strip()
    return out


@functools.cache
def allocated_nodes() -> list[str]:
    """Nodes this suite may use, prefill first.

    ``INFERA_E2E_NODES`` (a comma list, set by run_tests.sh's disagg dispatcher)
    wins; otherwise expand the allocation's compressed nodelist (e.g.
    ``node[01,03-05]``) via ``scontrol show hostnames``. ``[]`` when neither is
    available (the suite then skips)."""
    override = os.environ.get("INFERA_E2E_NODES")
    if override:
        return [n.strip() for n in override.split(",") if n.strip()]

    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist or not shutil.which("scontrol"):
        return []
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=_SRUN_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _routable(ip: str | None) -> bool:
    # Skip loopback and the per-NIC ionic RDMA subnets (192.168.x) — those aren't
    # routable for control-plane TCP (etcd/router/health) between login + nodes.
    return bool(ip) and not ip.startswith(("127.", "192.168."))


@functools.cache
def node_ip(node: str) -> str | None:
    """A routable IPv4 for ``node`` used as its ``--advertise-host`` (router/etcd
    reach it here) and Mooncake bootstrap address.

    Pinned by ``INFERA_E2E_NODE_IPS`` if given; else the node hostname's
    management IP via DNS/hosts (deterministic + routable). ``hostname -I`` on the
    node is only a fallback (its order is unstable and can surface a non-routable
    RDMA 192.168.x address first)."""
    override = _parse_ip_overrides()
    if node in override:
        return override[node]
    try:
        ip = socket.gethostbyname(node)
        if _routable(ip):
            return ip
    except OSError:
        pass
    if not have_slurm():
        return None
    try:
        out = run_on_node(node, ["hostname", "-I"])
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    ips = [t for t in out.stdout.split() if t.count(".") == 3 and not t.startswith("127.")]
    for ip in ips:
        if _routable(ip):
            return ip
    return ips[0] if ips else None


def gid_index() -> str:
    """RoCEv2 GID index for the RDMA KV transport. Defaults to the ionic ULA GID;
    ``INFERA_E2E_GID_INDEX`` overrides it for a rail that sits elsewhere (mlx5's
    routable GID is index 3 — its 0 and 1 are link-local)."""
    return os.environ.get("INFERA_E2E_GID_INDEX") or DEFAULT_GID_INDEX


def _kv_list(var: str) -> dict[str, str]:
    """``VAR='K=V,K=V'`` -> ``{K: V}`` (``{}`` when unset)."""
    out: dict[str, str] = {}
    for item in os.environ.get(var, "").split(","):
        key, sep, value = item.strip().partition("=")
        if sep and key:
            out[key] = value
    return out


def kv_transport_env() -> dict[str, str]:
    """``INFERA_E2E_WORKER_ENV`` (``K=V,K=V``) merged into every PD worker. Which
    rail carries the KV and how its VRAM registers are fabric properties, so they
    are set per run (see ci.yml's e2e-disag), not baked into an adapter."""
    return _kv_list("INFERA_E2E_WORKER_ENV")


def image_build_args() -> dict[str, str]:
    """``INFERA_E2E_BUILD_ARGS`` (``K=V,K=V``) as image ``--build-arg`` pairs.
    Some fabric constraints only bind at build time: without
    ``MOONCAKE_HIP_DMABUF=1`` the vLLM image has no dma-buf path compiled in."""
    return _kv_list("INFERA_E2E_BUILD_ARGS")


def pd_nodes() -> tuple[str, str] | None:
    """The (prefill_node, decode_node) pair for a 2-node PD run, or ``None`` if
    fewer than two nodes are available. First allocated node hosts prefill."""
    nodes = allocated_nodes()
    if len(nodes) < 2:
        return None
    return nodes[0], nodes[1]
