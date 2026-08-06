###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ROCm / ionic-RoCE RDMA env defaults for the PD KV-transfer backends.

On AMD/ROCm the Mooncake and MoRI transfer engines need the ionic RoCE-v2 GID
index and, for Mooncake, same-name peer-HCA selection. These were previously set
only by launch scripts, so a forgotten knob produced a degraded or broken
transport even though the engine started. Apply them here before the inference
subprocess is spawned. Mooncake itself now keeps HIP for same-host transfers and
automatically chooses RDMA for cross-host targets.

Gated on ROCm (probe ``/dev/kfd``); a no-op on NVIDIA hosts. GID index ``1`` is
the routable RoCE-v2 index on a typical ionic fabric; a fabric that differs
overrides via env. Each transfer engine reads only the vars it recognizes, so
set-defaulting both the Mooncake (``MC_*``) and MoRI (``MORI_*``) vars is
harmless for either backend.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# env var -> default value (applied only if unset). These are the KV-TRANSFER
# engine knobs only. (The RCCL/collectives GID — NCCL_IB_GID_INDEX — is a
# separate concern, irrelevant for intra-node TP, so it's intentionally not here.)
_ROCM_RDMA_DEFAULTS: dict[str, str] = {
    "MC_GID_INDEX": "1",  # Mooncake transfer engine: ionic RoCE v2 GID
    "MORI_IB_GID_INDEX": "1",  # MoRI transfer engine: ionic RoCE v2 GID
}

# Mooncake's two mutually exclusive peer-HCA selection policies.
_MC_DEST_AFFINITY = "MC_ENABLE_DEST_DEVICE_AFFINITY"
_MC_HCA_PEER_AFFINITY = "MC_ENABLE_HCA_PEER_AFFINITY"

# Which of Mooncake's two GPU MR paths to take. Both are compiled into our images;
# this picks one per host. Unset = Mooncake's own auto mode (dma-buf).
_MC_DISABLE_DMABUF = "MOONCAKE_DISABLE_HIP_DMABUF"
# A loaded peer-memory module is what makes bare ibv_reg_mr work on a device
# pointer. ROCm-only here, so the NVIDIA modules the preflight probe also looks
# for are omitted.
_PROC_MODULES = "/proc/modules"
_PEERMEM_MODULES = ("ib_peer_mem", "amdp2p")


def _is_rocm() -> bool:
    """True iff running on ROCm/HIP. Probe ``/dev/kfd`` to avoid importing torch."""
    return os.path.exists("/dev/kfd")


def apply_vllm_aiter_default() -> str | None:
    """Default VLLM_ROCM_USE_AITER=1 on ROCm (set-if-unset). Returns value applied.

    AITER is AMD's optimized kernel library; vLLM defaults its master switch OFF.
    Several ROCm configs REQUIRE it — e.g. MXFP4 MoE models (MiniMax-M2, Kimi)
    have no native vLLM MXFP4 MoE backend and fail hard at load ("No MXFP4 MoE
    backend supports the deployment configuration") unless AITER is on. So on our
    AMD fleet the useful default is ON, not OFF. Operator/env still overrides (set
    VLLM_ROCM_USE_AITER=0 to opt out). ROCm-only; no-op on NVIDIA.
    """
    if not _is_rocm():
        return None
    if os.environ.get("VLLM_ROCM_USE_AITER") not in (None, ""):
        return None  # operator override wins
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    logger.info("VLLM_ROCM_USE_AITER defaulted to 1 (AITER on; override via env)")
    return "1"


def _apply_dest_device_affinity_default() -> str | None:
    """Default Mooncake to the same-named peer HCA. Returns the value applied.

    A no-op on a single-HCA host, and Mooncake falls back to its normal
    selection when the peer has no NIC of that name; on a rail-isolated fabric
    it stops a local rail from picking an unreachable peer rail.
    """
    value = os.environ.get(_MC_DEST_AFFINITY, "")
    if value:
        # Mooncake only tests whether the var is PRESENT, so an explicit "0"
        # would still enable it. Honour the opt-out by removing it entirely.
        if value.lower() in ("0", "false"):
            del os.environ[_MC_DEST_AFFINITY]
        return None
    # Mooncake disables BOTH policies when both are set, which is worse than
    # either alone, so an explicitly configured peer affinity wins.
    if os.environ.get(_MC_HCA_PEER_AFFINITY, "").lower() in ("1", "true"):
        return None
    os.environ[_MC_DEST_AFFINITY] = "1"
    return "1"


def _peermem_loaded() -> bool:
    """True iff a GPU peer-memory kernel module is loaded.

    Unreadable ``/proc/modules`` reports absent: that steers the caller to dma-buf,
    which at worst risks HIP-209 under load, where the other direction cannot
    register VRAM at all.
    """
    try:
        with open(_PROC_MODULES) as f:
            loaded = {line.split()[0] for line in f if line.split()}
    except OSError:
        return False
    return any(m in loaded for m in _PEERMEM_MODULES)


def _apply_gpu_mr_path_default() -> str | None:
    """Pick Mooncake's GPU MR path for this host. Returns the value applied.

    With a peer-memory module loaded, bare ``ibv_reg_mr`` works and is the safer
    path (dma-buf at high util there exhausts a KFD resource -> HIP-209 on a later
    hipModuleLoad). Without one, bare registration fails outright on a device
    pointer, so leave the variable unset and let Mooncake select dma-buf.
    """
    if os.environ.get(_MC_DISABLE_DMABUF, ""):
        return None  # operator override wins
    if not _peermem_loaded():
        return None
    os.environ[_MC_DISABLE_DMABUF] = "1"
    return "1"


def apply_rocm_rdma_env_defaults() -> dict[str, str]:
    """Set ionic-RoCE RDMA env defaults (set-if-unset) on ROCm; no-op elsewhere.

    Call ONCE at engine startup, BEFORE the inference subprocess is spawned, so
    the defaults are inherited by it (and its transfer-engine workers). Returns
    the dict of vars actually applied (empty if none / not ROCm).
    """
    if not _is_rocm():
        return {}
    applied: dict[str, str] = {}
    for key, value in _ROCM_RDMA_DEFAULTS.items():
        if os.environ.get(key) in (None, ""):
            os.environ[key] = value
            applied[key] = value
    affinity = _apply_dest_device_affinity_default()
    if affinity:
        applied[_MC_DEST_AFFINITY] = affinity
    mr_path = _apply_gpu_mr_path_default()
    if mr_path:
        applied[_MC_DISABLE_DMABUF] = mr_path
    if applied:
        logger.info(
            "ROCm RDMA env defaults applied (set-if-unset; override via env): %s",
            applied,
        )
    return applied


_IB_PATH = "/sys/class/infiniband"


def _gid_to_ipv4(gid: str) -> str | None:
    """Return the dotted IPv4 of an IPv4-mapped RoCE GID, else None.

    IPv4-mapped GIDs look like ``0000:...:0000:ffff:ac1e:1b91`` (last 32 bits =
    the IPv4). Link-local (``fe80::``) and pure-IPv6 GIDs return None.
    """
    h = gid.replace(":", "")
    if len(h) != 32 or h[:24] != "0" * 20 + "ffff":
        return None
    try:
        return ".".join(str(int(h[i : i + 2], 16)) for i in range(24, 32, 2))
    except ValueError:
        return None


def _active_rdma_nics(gid_index: int) -> list[tuple[str, str, str]]:
    """List (device_name, ipv4, ipv4_subnet/24) for ACTIVE RoCE NICs with an
    IPv4 GID at ``gid_index``. Sorted by name for determinism (prefill & decode
    on the same host must independently pick the SAME NIC)."""
    out: list[tuple[str, str, str]] = []
    try:
        devs = sorted(os.listdir(_IB_PATH))
    except OSError:
        return out
    for d in devs:
        try:
            state = open(f"{_IB_PATH}/{d}/ports/1/state").read()
            if "ACTIVE" not in state:
                continue
            gid = open(f"{_IB_PATH}/{d}/ports/1/gids/{gid_index}").read().strip()
        except OSError:
            continue
        ip = _gid_to_ipv4(gid)
        if ip is None:
            continue
        subnet = ip.rsplit(".", 1)[0] + ".0/24"
        out.append((d, ip, subnet))
    return out


_NET_PATH = "/sys/class/net"
# vLLM / ATOM host-IP override vars. Both engines' get_ip() defaults to the
# default-route NIC (connect to 8.8.8.8) which, on a multi-homed host (public
# NIC + ionic RoCE), is the PUBLIC NIC — the KV-transfer engine then advertises
# that IP as its segment/bootstrap host and pushes KV over the wrong, non-RDMA
# interface ("block RDMA chunk error -1" / bootstrap engine_id lookup failures).
_KV_HOST_IP_VARS = ("VLLM_HOST_IP", "ATOM_HOST_IP")


def _ifaddr_ipv4(ifname: str) -> str | None:
    """This netdev's IPv4 via SIOCGIFADDR (no ``ip``/netifaces dependency)."""
    import fcntl
    import socket
    import struct

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", ifname[:15].encode())
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, packed)[20:24])
    except OSError:
        return None
    finally:
        s.close()


def _is_private_ipv4(ip: str) -> bool:
    """RFC1918 private, excluding the Docker bridge default (172.17.x)."""
    o = [int(x) for x in ip.split(".")]
    if o[0] == 10:
        return True
    if o[0] == 192 and o[1] == 168:
        return True
    if o[0] == 172 and 16 <= o[1] <= 31 and o[1] != 17:  # 172.17 = docker0
        return True
    return False


# Interface name prefixes that are virtual by construction — container bridges,
# CNI plugins, veth pairs, tunnels. None can carry RoCE, and several are private
# RFC1918, which is exactly what makes them dangerous to a "first private IPv4"
# scan.
_VIRTUAL_IF_PREFIXES = (
    "docker",
    "cni",
    "flannel",
    "cali",
    "cilium",
    "weave",
    "kube-",
    "veth",
    "br-",
    "virbr",
    "tunl",
    "gre",
    "ip6tnl",
    "dummy",
    "bond-dummy",
)


def _private_rail_ipv4() -> str | None:
    """First private (RFC1918) NIC IPv4, sorted by ifname for determinism.

    The KV-transfer P2P-handshake host must be a peer-reachable, NON-public
    address; ``get_ip()`` (route to 8.8.8.8) returns the PUBLIC NIC on a
    multi-homed host.

    Skip virtual interfaces, not just loopback and docker. On a Kubernetes node
    the CNI bridge is private, RFC1918 and — sorted by name — comes FIRST, so a
    naive scan picks `cni0` (10.42.0.1) and pins the KV host IP to an address the
    peer cannot reach. Observed on k3s/flannel: prefill advertised 10.42.0.1 and
    decode 10.42.1.1, each the other node's unreachable bridge gateway. The host
    interface list there begins:

        cni0  docker0  enp105s0  enp121s0 ...

    so ordering alone guarantees the wrong answer. These names are virtual by
    construction and can never be an RDMA rail."""
    try:
        ifs = sorted(os.listdir(_NET_PATH))
    except OSError:
        return None
    for n in ifs:
        if n == "lo" or n.startswith(_VIRTUAL_IF_PREFIXES):
            continue
        ip = _ifaddr_ipv4(n)
        if ip and _is_private_ipv4(ip):
            return ip
    return None


def apply_kv_host_ip_default() -> str | None:
    """Pin VLLM_HOST_IP / ATOM_HOST_IP to this host's RDMA-rail IPv4 (set-if-unset).

    So the PD KV-transfer engine advertises a peer-reachable RDMA-rail address
    for its P2P handshake / bootstrap and segment host — instead of the PUBLIC
    NIC that vLLM's / ATOM's ``get_ip()`` (route to 8.8.8.8) picks on a
    multi-homed host, which sends KV to the wrong interface ("block RDMA chunk
    error -1", bootstrap engine_id lookup failures).

    Resolution order: (1) an IPv4-mapped RoCE GID at ``MC_GID_INDEX`` (exact
    rail); (2) fallback to the first private RFC1918 NIC IPv4 (the ionic rail on
    hosts whose RoCE GIDs are IPv6-only and whose rail IP lives on a NIC distinct
    from the RDMA device's netdev). ROCm-only; no-op if the operator already set
    either var or nothing suitable is found. Run AFTER apply_rocm_rdma_env_defaults().
    """
    if not _is_rocm():
        return None
    if any(os.environ.get(v) for v in _KV_HOST_IP_VARS):
        return None  # operator override wins
    try:
        gid_index = int(os.environ.get("MC_GID_INDEX", "1"))
    except ValueError:
        gid_index = 1
    nics = _active_rdma_nics(gid_index)
    source = f"RoCE GID[{gid_index}]"
    rail_ip = nics[0][1] if nics else None
    if rail_ip is None:  # IPv6-only GIDs / IP not on the RDMA netdev
        rail_ip = _private_rail_ipv4()
        source = "private NIC"
    if rail_ip is None:
        return None
    for v in _KV_HOST_IP_VARS:
        os.environ[v] = rail_ip
    logger.info(
        "KV host IP pinned to RDMA rail %s (via %s) as %s (override via env)",
        rail_ip,
        source,
        "/".join(_KV_HOST_IP_VARS),
    )
    return rail_ip


def apply_mooncake_topology_default(num_gpus: int = 16) -> str | None:
    """Pin Mooncake's per-GPU HCA selection to a SINGLE consistent NIC when the
    host has RoCE NICs spread across MULTIPLE subnets — the default that avoids
    the cross-subnet QP-handshake storm.

    Why: Mooncake's auto-discover assigns each GPU the HCA on its NUMA node. On a
    multi-NIC box whose NICs sit on different subnets (e.g. 172.30.x / 10.245.x /
    172.29.x), prefill GPUs and decode GPUs end up on NICs in DIFFERENT subnets,
    so the RoCE QP→RTR transition can't route and times out ([110]) under
    concurrency — a handshake storm that wedges PD. Pinning every GPU to one NIC
    (same-NIC loopback for same-host PD) makes RTR always resolve → 0 QP errors.

    Set-if-unset (respects an operator ``MC_CUSTOM_TOPO_JSON``), ROCm-only, and a
    no-op unless the NICs actually span >1 subnet (single-subnet hosts don't need
    it). Returns the topology file path written, else None.
    """
    if not _is_rocm():
        return None
    if os.environ.get("MC_CUSTOM_TOPO_JSON"):
        return None  # operator override wins
    try:
        gid_index = int(os.environ.get("MC_GID_INDEX", "3"))
    except ValueError:
        gid_index = 3
    nics = _active_rdma_nics(gid_index)
    subnets = {s for _, _, s in nics}
    if len(nics) <= 1 or len(subnets) <= 1:
        # one NIC, or all on one subnet → NUMA spread is already safe.
        return None
    # Multiple subnets present → pin to the dominant subnet's first NIC
    # (deterministic). Same-host prefill+decode pick the same NIC → loopback.
    from collections import Counter

    dominant = Counter(s for _, _, s in nics).most_common(1)[0][0]
    nic = sorted(d for d, _, s in nics if s == dominant)[0]
    topo = {f"cuda:{i}": [[nic], [nic]] for i in range(num_gpus)}
    topo.update({f"cpu:{i}": [[nic], [nic]] for i in range(num_gpus)})
    fd, path = tempfile.mkstemp(prefix="infera_mc_topo_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(topo, f)
    os.environ["MC_CUSTOM_TOPO_JSON"] = path
    logger.info(
        "Mooncake topology default: NICs span %d subnets %s → pinned all GPUs to "
        "'%s' (subnet %s) to avoid cross-subnet QP-RTR storm; MC_CUSTOM_TOPO_JSON=%s "
        "(override via env)",
        len(subnets),
        sorted(subnets),
        nic,
        dominant,
        path,
    )
    return path
