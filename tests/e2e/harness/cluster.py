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

Everything here is best-effort and side-effect free: on a non-SLURM host (or no
allocation) the discovery helpers return empty/None and the suite skips.
"""

from __future__ import annotations

import functools
import os
import shutil
import socket
import subprocess

# RoCEv2 GID index default — the ULA (routable) GID on the repo's ionic fabric;
# both the PD bench (MORI_IB_GID_INDEX=1) and regression doc 04 use index 1.
DEFAULT_GID_INDEX = "1"

_SRUN_TIMEOUT = 60


def have_slurm() -> bool:
    """Whether the SLURM client tooling this suite drives is on PATH."""
    return shutil.which("srun") is not None and shutil.which("scontrol") is not None


def in_allocation() -> bool:
    """Whether we're inside a live SLURM allocation (salloc/sbatch)."""
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"))


def _job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


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
    argv = ["srun", "--overlap", "--nodes=1", "--ntasks=1", "--nodelist", node]
    if _job_id():
        argv += ["--jobid", _job_id()]
    # Honor a caller-provided reservation (see launcher._RESV) so this probe srun
    # can land on a reserved node instead of being rejected/queued.
    _resv = os.environ.get("INFERA_E2E_RESERVATION")
    if _resv:
        argv += [f"--reservation={_resv}"]
    argv += ["hostname", "-I"]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=_SRUN_TIMEOUT)
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
    """RoCEv2 GID index for the RDMA KV transport (fixed default 1)."""
    return DEFAULT_GID_INDEX


def _gid_routable(gid: str) -> bool:
    """A routable RoCEv2 GID: non-zero and NOT ``fe80`` link-local. Covers both
    IPv4-mapped (``::ffff:a.b.c.d``) and IPv6-ULA (``fc00::/7``, the ionic
    ``fc01:...`` fabric) GIDs — both route cross-node; ``fe80`` times out at
    QP→RTR."""
    g = gid.strip().lower()
    if not g or set(g.replace(":", "")) <= {"0"}:
        return False
    return not g.startswith("fe80")


def _gid_subnet(gid: str) -> str:
    """Coarse rail/subnet key: IPv4-mapped -> /24 (``a.b.c``); IPv6 -> /64 (first
    four hextets). Distinguishes a rail-optimized fabric (each NIC its own subnet)
    from a flat any-to-any fabric."""
    g = gid.strip().lower()
    h = g.replace(":", "")
    if len(h) == 32 and h[:24] == "0" * 20 + "ffff":
        octs = [str(int(h[i : i + 2], 16)) for i in range(24, 32, 2)]
        return ".".join(octs[:3])
    return ":".join(g.split(":")[:4])


def compute_nic_filter(devices: list[tuple[str, str]]) -> str | None:
    """Given ``[(device_name, gid_at_index)]`` for the ACTIVE RDMA ports, return
    the ``MC_TE_FILTERS`` value that pins Mooncake to the right rail(s), or ``None``
    when nothing needs constraining (a plain single-fabric host).

    - Keep only NICs with a routable GID at the index (drops a mgmt NIC whose GID
      there is link-local, e.g. a ConnectX ``mlx5_0``).
    - If those span MULTIPLE subnets (a rail-optimized fabric where ``ionic_i``
      only routes to the peer's ``ionic_i``, and host/aux buffers have no rail
      affinity), pin to a SINGLE deterministic rail so both ends always agree.
    - If they share one subnet (flat fabric), keep all data rails.
    Pure function (no I/O) so it is unit-testable."""
    if not devices:
        return None
    active = [name for name, _ in devices]
    data = [(name, gid) for name, gid in devices if _gid_routable(gid)]
    if not data:
        return None
    subnets = {_gid_subnet(gid) for _, gid in data}
    data_nics = sorted(name for name, _ in data)
    whitelist = [data_nics[0]] if len(subnets) > 1 else data_nics
    if set(whitelist) == set(active):
        return None  # nothing to narrow
    return ",".join(whitelist)


def _rdma_gids_on_node(node: str, gid_idx: int) -> list[tuple[str, str]]:
    """``[(device, gid_at_gid_idx)]`` for ACTIVE RDMA ports on ``node``, read from
    its ``/sys/class/infiniband`` via ``srun`` (the ionic NICs live on the compute
    node, not the login host). ``[]`` on any failure."""
    if not have_slurm():
        return []
    script = (
        "for d in /sys/class/infiniband/*; do "
        '[ -e "$d/ports/1/state" ] || continue; '
        'grep -q ACTIVE "$d/ports/1/state" 2>/dev/null || continue; '
        f'echo "$(basename "$d") $(cat "$d/ports/1/gids/{gid_idx}" 2>/dev/null)"; '
        "done"
    )
    # Spur scheduler exposes only a subset of srun (no --overlap/--jobid); detect
    # it by its controller env var and place work with plain `srun -N1 -n1 -w NODE`
    # (mirrors launcher._srun). Stock SLURM keeps --overlap/--jobid.
    if os.environ.get("SPUR_CONTROLLER_ADDR"):
        part = os.environ.get("INFERA_E2E_SLURM_PARTITION")
        argv = ["srun", "-N1", "-n1"] + (["-p", part] if part else []) + ["-w", node]
    else:
        argv = ["srun", "--overlap", "--nodes=1", "--ntasks=1", "--nodelist", node]
        if _job_id():
            argv += ["--jobid", _job_id()]
    resv = os.environ.get("INFERA_E2E_RESERVATION")
    if resv:
        argv += [f"--reservation={resv}"]
    argv += ["bash", "-lc", script]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=_SRUN_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    devs: list[tuple[str, str]] = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            devs.append((parts[0], parts[1]))
    return devs


@functools.cache
def mooncake_nic_filter(node: str, gid_idx: int) -> str | None:
    """The ``MC_TE_FILTERS`` value to inject into the PD worker env so Mooncake
    binds the right RDMA rail on this cluster (see :func:`compute_nic_filter`).

    Probes ``node`` (identical prefill/decode hardware, so one probe covers both);
    returns ``None`` when no constraint is needed. Cached per (node, index)."""
    return compute_nic_filter(_rdma_gids_on_node(node, gid_idx))


def pd_nodes() -> tuple[str, str] | None:
    """The (prefill_node, decode_node) pair for a 2-node PD run, or ``None`` if
    fewer than two nodes are available. First allocated node hosts prefill."""
    nodes = allocated_nodes()
    if len(nodes) < 2:
        return None
    return nodes[0], nodes[1]
