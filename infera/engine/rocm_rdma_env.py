###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ROCm / ionic-RoCE RDMA env defaults for the PD KV-transfer backends.

On AMD/ROCm the Mooncake and MoRI transfer engines need the ionic RoCE-v2 GID
index (and, for Mooncake, the HIP-transport disable) to actually move KV over
RDMA — without them they silently fall back to TCP or hang. These were
previously set ONLY by the launch scripts (deploy launchers, the SLURM repro
harness), so any launch path that forgot them got a degraded/broken transport
even though the engine "came up". Bake them in as DEFAULTS here, set-if-unset,
so an operator or launcher can still override any of them via the environment.

Gated on ROCm (probe ``/dev/kfd``); a no-op on NVIDIA hosts. GID index ``1`` is
the ionic RoCE-v2 value used across our AMD fleet; a different fabric overrides
via env. Each transfer engine reads only the vars it recognizes, so set-defaulting
both the Mooncake (``MC_*``) and MoRI (``MORI_*``) vars is harmless for either backend.
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
    "MC_DISABLE_HIP_TRANSPORT": "1",  # Mooncake: use RDMA, not the HIP P2P transport
    "MORI_IB_GID_INDEX": "1",  # MoRI transfer engine: ionic RoCE v2 GID
}


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


def _compute_capability() -> tuple[int, int] | None:
    """(major, minor) compute capability of the current GPU, or None.

    CDNA maps arch -> capability: gfx942 (MI300/MI325, CDNA3) = (9, 4);
    gfx950 (MI355X, CDNA4) = (9, 5). Uses torch (the engines already depend on
    it); returns None on CPU-only / no-GPU hosts so callers no-op safely.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return (props.major, props.minor)
    except Exception:
        return None


def is_gfx942() -> bool:
    """True iff the current GPU is gfx942 (MI300/MI325X, CDNA3)."""
    return _compute_capability() == (9, 4)


def _is_dsv4_fp4_model(model_path: str | None) -> bool:
    """True iff ``model_path`` is a local DeepSeek-V4 checkpoint with FP4 experts.

    Double-guards the gfx942 env defaults so they NEVER touch a non-DSv4 model
    (requirement: don't break other models' run path). Reads only ``config.json``
    from a local dir — never downloads. A bare HF repo id (no local dir) or any
    read error returns False (conservative: leave the native path alone).
    """
    if not model_path or not os.path.isdir(model_path):
        return False
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return False
    # DeepSeek-V4 family: model_type deepseek_v4 (or an index_topk sparse-attn
    # config). FP4 experts show up as a fp4/mxfp4 quantization_config.
    model_type = str(cfg.get("model_type", "")).lower()
    is_dsv4 = model_type.startswith("deepseek_v4") or "index_topk" in cfg
    qc = cfg.get("quantization_config") or {}
    quant_blob = json.dumps(qc).lower() if isinstance(qc, dict) else str(qc).lower()
    is_fp4 = "fp4" in quant_blob or "mxfp4" in quant_blob or "e2m1" in quant_blob
    return bool(is_dsv4 and is_fp4)


def apply_dsv4_gfx942_env_defaults(
    model_path: str | None, *, engine: str
) -> dict[str, str]:
    """gfx942 (MI325X) DSv4-FP4 env defaults (set-if-unset). No-op otherwise.

    MI325X/CDNA3 has no working FP4 MoE kernel, so DeepSeek-V4-Pro must dequantize
    its FP4 experts to FP8 at load and run aiter's FP8 blockscale MoE (see the
    ATOM/SGLang ``patch_dsv4_fp4_dequant_gfx942`` source patches baked into the
    images). This flips the master switch for that path — plus the sglang MLA
    backend override (default tilelang crashes gfx942) and the tuned bf16 GEMM
    config — so the operator gets a working MI325X out of the box.

    TRIPLE-GATED so it can't affect MI355X or other models:
      1. arch == gfx942 / capability (9, 4) (MI355X is gfx950 -> skipped);
      2. the model is a local DSv4 checkpoint with FP4 experts (other models ->
         skipped);
      3. every var is set-if-unset (operator/env always overrides).

    Returns the dict of vars actually applied (empty if not applicable). Call
    ONCE at startup BEFORE the engine subprocess is spawned so it's inherited.
    """
    if not is_gfx942():
        return {}
    if not _is_dsv4_fp4_model(model_path):
        return {}

    if engine == "atom":
        defaults = {"ATOM_DSV4_FP4_DEQUANT": "1"}
    elif engine == "sglang":
        defaults = {
            "SGLANG_DSV4_FP4_DEQUANT": "1",
            # Default MLA backend (tilelang) fails TVM compile on gfx942; the
            # pure-triton path works. Nothing auto-overrides it otherwise.
            "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
        }
        # Tuned aiter bf16 GEMM configs for gfx942/cu_num=304 (baked by
        # Dockerfile.sglang.gfx942). Merge with aiter's base file; skip if either
        # is absent so we never point aiter at a missing path.
        csv = os.environ.get(
            "INFERA_DSV4_GEMM_CSV", "/opt/infera/aiter_configs/tuned_dsv4_cu304.csv"
        )
        base = "/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv"
        if os.path.isfile(csv) and os.path.isfile(base):
            defaults["AITER_CONFIG_GEMM_BF16"] = f"{base}:{csv}"
    else:
        return {}

    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if os.environ.get(key) in (None, ""):
            os.environ[key] = value
            applied[key] = value
    if applied:
        logger.info(
            "gfx942 DSv4-FP4 env defaults applied for %s (set-if-unset; "
            "override via env): %s",
            engine,
            applied,
        )
    return applied


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


def _private_rail_ipv4() -> str | None:
    """First private (RFC1918) NIC IPv4, sorted by ifname for determinism.

    The KV-transfer P2P-handshake host must be a peer-reachable, NON-public
    address; ``get_ip()`` (route to 8.8.8.8) returns the PUBLIC NIC on a
    multi-homed host. Skip loopback and the docker bridge."""
    try:
        ifs = sorted(os.listdir(_NET_PATH))
    except OSError:
        return None
    for n in ifs:
        if n == "lo" or n.startswith("docker"):
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
