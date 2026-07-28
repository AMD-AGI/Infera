###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Mooncake KV-registration MODE detector + selector for PD disaggregation.

Mooncake can register the KV cache (device VRAM) for RDMA in three fundamentally
different ways, and the RIGHT one depends entirely on the node's kernel + NIC:

  A. bare ``ibv_reg_mr`` + a peer-memory kernel module (nvidia_peermem / amdgpu
     peer-mem). This is the DEFAULT, no-surprise path when a peer-mem module is
     loaded: registration hands the NIC the GPU pages directly, nothing is pinned
     or duplicated, and every RDMA rail can carry KV. No special image needed.

  B. ``ibv_reg_dmabuf_mr`` (GPUDirect via dma-buf) — the ONLY GPUDirect path when
     there is NO peer-mem module. Whether it is *safe* hinges on the NIC:
       * NIC with ODP (on-demand paging, e.g. Mellanox mlx5): dynamic attach ->
         nothing pinned, KV pool NOT doubled. Viable.
       * NIC without ODP (e.g. AMD Pensando ionic): the driver PINS the whole
         registered region -> the KV pool is duplicated in VRAM and can exhaust a
         KFD resource -> SIGSEGV/HIP-209 on a large pool. NOT viable for full KV.
     dma-buf also needs the engine.so rebuilt with USE_HIP_DMABUF (the stock base
     image compiles it out) -- see deploy/docker/scripts/build_mooncake_dmabuf.sh.

  C. no peer-mem AND no ODP-capable NIC -> neither A nor B is safe for a full KV
     pool. Fall back to capping the KV cache so a pinned dma-buf region fits (or a
     driver-bug workaround). Left as a stub for a follow-up.

This module PROBES the node (peer-mem, per-NIC ODP/vendor/speed/GID, kernel
P2PDMA, GPU topology), then ENUMERATES all three modes -- each marked viable or
blocked (with the reason) -- plus the per-NIC capability matrix and the engine
image's dma-buf capability. Modes are ranked safety-first then bandwidth, so the
top pick is the safest full-KV path, not merely the fastest fabric. It never
launches anything and takes no live transfer measurement (that is
``network.mooncakeperf``); it lays out the OPTIONS and their exact env + launch
flags for a human or agent to choose from. A mode that needs a KV-pool CAP is
flagged for explicit user confirmation.

Output is offered in three formats (all from one probe): a colorized CLI table,
JSON, and Markdown -- print any subset and/or drop all three to files.

Usage:
    python -m infera.tools.preflight.mooncake_mode                 # CLI table (default)
    python -m infera.tools.preflight.mooncake_mode --emit md       # Markdown to stdout
    python -m infera.tools.preflight.mooncake_mode --emit table,json
    python -m infera.tools.preflight.mooncake_mode --out-dir /tmp/mm   # write .json/.md/.txt
    python -m infera.tools.preflight.mooncake_mode --json --quiet > mode.json  # back-compat

Exit code: 0 if the best pick is a viable full-KV mode (A or B, no user cap), 2 if
the only viable path needs a KV cap (C) or nothing is viable -- i.e. 2 means a
human decision or an image rebuild is required.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys

# Reuse the preflight suite's best-effort probe helpers when infera is importable
# (the engine container pip-installs it). Fall back to tiny local copies so a
# single-file copy of this script still runs on a bare node.
try:  # pragma: no cover - import shim
    from .util import have, read_text, run
except ImportError:  # pragma: no cover
    import shutil
    import subprocess

    def have(cmd: str) -> bool:
        return shutil.which(cmd) is not None

    def read_text(path: str) -> str | None:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:  # noqa: BLE001
            return None

    def run(cmd: list[str], timeout: float = 5.0, merge_stderr: bool = True):
        if not have(cmd[0]):
            return None, f"<{cmd[0]} not found>"
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
                text=True,
                timeout=timeout,
                check=False,
            )
            return p.returncode, p.stdout
        except Exception as e:  # noqa: BLE001
            return None, f"<{cmd[0]} failed: {e}>"


_IB = "/sys/class/infiniband"


# --------------------------------------------------------------------------- #
# NIC / RDMA probing
# --------------------------------------------------------------------------- #
def _gid_to_ipv4(gid: str) -> str | None:
    """Dotted IPv4 of an IPv4-mapped RoCE GID (``...:ffff:aabb:ccdd``), else None."""
    h = gid.replace(":", "")
    if len(h) != 32 or h[:24] != "0" * 20 + "ffff":
        return None
    try:
        return ".".join(str(int(h[i : i + 2], 16)) for i in range(24, 32, 2))
    except ValueError:
        return None


def _routable_gid(dev: str) -> tuple[int | None, str | None, str | None]:
    """Pick the routable RoCE v2 GID for ``dev``: (index, addr, kind).

    Prefer an IPv4-mapped RoCE-v2 GID (routable cross-node, what MC_GID_INDEX
    should point at). Fall back to a non-link-local RoCE-v2 GID (a global IPv6
    RoCE-v2, as on ionic). Skip link-local ``fe80::`` (index 0) -- it times out
    cross-node. ``addr`` is the dotted IPv4 for an IPv4-mapped GID, else the raw
    GID string; the caller stores it as gid_ipv4 only when kind is ipv4-mapped.
    Returns (None, None, None) if nothing usable.
    """
    port = f"{_IB}/{dev}/ports/1"
    best_v6: tuple[int, str] | None = None
    try:
        gid_files = sorted(
            os.listdir(f"{port}/gids"), key=lambda x: int(x) if x.isdigit() else 1 << 30
        )
    except OSError:
        return None, None, None
    for gi in gid_files:
        if not gi.isdigit():
            continue
        idx = int(gi)
        gid = (read_text(f"{port}/gids/{gi}") or "").strip()
        if not gid or gid.startswith("fe80") or set(gid.replace(":", "")) == {"0"}:
            continue
        gtype = (read_text(f"{port}/gid_attrs/types/{gi}") or "").strip()
        if "v2" not in gtype.lower() and "RoCE v2" not in gtype:
            # Only RoCE v2 routes on our fabric; v1 GIDs are not usable here.
            continue
        ipv4 = _gid_to_ipv4(gid)
        if ipv4:
            return idx, ipv4, "ipv4-mapped RoCEv2"
        if best_v6 is None:
            best_v6 = (idx, gid)
    if best_v6:
        return best_v6[0], best_v6[1], "global-IPv6 RoCEv2"
    return None, None, None


def _nic_driver(dev: str) -> str:
    try:
        return os.path.basename(os.readlink(f"{_IB}/{dev}/device/driver"))
    except OSError:
        return "?"


def _nic_vendor(dev: str, driver: str) -> str:
    d = driver.lower()
    if "mlx" in d:
        return "mellanox"
    if "ionic" in d:
        return "ionic"
    if "bnxt" in d:
        return "broadcom"
    if "irdma" in d or "i40" in d or "ice" in d:
        return "intel"
    return driver or "?"


def _link_gbps(dev: str) -> float | None:
    rate = (read_text(f"{_IB}/{dev}/ports/1/rate") or "").strip()
    m = re.match(r"([\d.]+)\s*Gb", rate)
    return float(m.group(1)) if m else None


def _numa_node(dev: str) -> int | None:
    v = (read_text(f"{_IB}/{dev}/device/numa_node") or "").strip()
    try:
        n = int(v)
        return n if n >= 0 else None
    except ValueError:
        return None


def _pci_bdf(dev: str) -> str | None:
    """PCI BDF (bus:device.function) backing this RDMA device, e.g. 0000:c1:00.1.

    The BDF is what a dma-buf GPUDirect path routes over (NIC<->GPU P2P is BDF-to-
    BDF), so it is reported for the topology view and to confirm the NIC is a real
    PCI function (not a pure-SW device that could never do GPUDirect)."""
    try:
        return os.path.basename(os.readlink(f"{_IB}/{dev}/device"))
    except OSError:
        return None


def _odp_support(dev: str) -> tuple[bool, bool]:
    """(odp, odp_implicit) for ``dev`` via ``ibv_devinfo -d <dev> -v``.

    ODP (on-demand paging) is the property that lets ibv_reg_dmabuf_mr dynamic-
    attach instead of pinning: with ODP the KV pool is not doubled in VRAM. Parse
    the general_odp_caps block; absence (or no ibv_devinfo) => no ODP."""
    if not have("ibv_devinfo"):
        return False, False
    rc, out = run(["ibv_devinfo", "-d", dev, "-v"], timeout=8.0)
    if rc is None or not out:
        return False, False
    # ODP caps are listed as ODP_SUPPORT / ODP_SUPPORT_IMPLICIT tokens.
    odp = "ODP_SUPPORT" in out
    implicit = "ODP_SUPPORT_IMPLICIT" in out
    return odp, implicit


def _rdma_link_netdev() -> dict[str, str]:
    """ionic_X / mlx5_X -> netdev, from ``rdma link show``."""
    rc, out = run(["rdma", "link", "show"])
    mp: dict[str, str] = {}
    if rc == 0 and out:
        for line in out.splitlines():
            m = re.search(r"(\S+)/\d+\b.*\bnetdev\s+(\S+)", line)
            if m:
                mp[m.group(1)] = m.group(2)
    return mp


def _netdev_ipv4(netdev: str) -> str | None:
    if not netdev or netdev == "?":
        return None
    rc, out = run(["ip", "-o", "-4", "addr", "show", "dev", netdev])
    if rc == 0 and out:
        m = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", out)
        if m:
            return m.group(1)
    return None


def _port_active(dev: str) -> bool:
    return "ACTIVE" in (read_text(f"{_IB}/{dev}/ports/1/state") or "")


def probe_nics() -> list[dict]:
    """One dict per RDMA device: vendor/driver/netdev/ip/speed/gid/odp/bdf/numa."""
    try:
        devs = sorted(os.listdir(_IB))
    except OSError:
        return []
    link = _rdma_link_netdev()
    out: list[dict] = []
    for dev in devs:
        driver = _nic_driver(dev)
        netdev = link.get(dev, "?")
        gid_idx, gid_addr, gid_kind = _routable_gid(dev)
        # gid_ipv4 holds a dotted IPv4 only for an IPv4-mapped GID; a global-IPv6
        # GID (ionic) keeps its raw value in gid_raw and gid_ipv4 stays None.
        is_ipv4 = gid_kind == "ipv4-mapped RoCEv2"
        odp, odp_impl = _odp_support(dev)
        out.append(
            {
                "device": dev,
                "vendor": _nic_vendor(dev, driver),
                "driver": driver,
                "netdev": netdev,
                "ipv4": _netdev_ipv4(netdev),
                "link_gbps": _link_gbps(dev),
                "active": _port_active(dev),
                "pci_bdf": _pci_bdf(dev),
                "numa_node": _numa_node(dev),
                "gid_index": gid_idx,
                "gid_ipv4": gid_addr if is_ipv4 else None,
                "gid_raw": None if is_ipv4 else gid_addr,
                "gid_kind": gid_kind,
                "odp": odp,
                "odp_implicit": odp_impl,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# peer-mem + kernel probing
# --------------------------------------------------------------------------- #
_PEERMEM_MODULES = ("nvidia_peermem", "nv_peer_mem", "ib_peer_mem", "amdp2p")


def probe_peermem() -> dict:
    """Detect a loaded GPU peer-memory kernel module (multi-signal).

    A peer-mem module is what makes bare ibv_reg_mr on a device pointer work
    (mode A). We require POSITIVE evidence to declare it present -- when unsure we
    say absent, which steers the decision toward the dma-buf/cap paths that are
    safe without peer-mem, rather than recommending a mode that would EFAULT.
    """
    evidence: list[str] = []

    # 1. lsmod / /proc/modules for a known peer-mem module.
    mods = read_text("/proc/modules") or ""
    loaded = {ln.split()[0] for ln in mods.splitlines() if ln.split()}
    for m in _PEERMEM_MODULES:
        if m in loaded:
            evidence.append(f"module:{m}")

    # 2. MOFED peer-mem client registry: each registered client (e.g. amdkfd,
    #    nv_mem) shows up as a dir here. This is the strongest signal on MOFED.
    for d in sorted(glob.glob("/sys/kernel/mm/memory_peers/*")):
        name = os.path.basename(d)
        # a client is "active" when its num_alloc_mrs/version is exposed
        evidence.append(f"memory_peers:{name}")

    # 3. per-uverbs peer_mem_clients count (MOFED exposes a nonzero count when a
    #    peer-mem client is attached).
    for pmc in sorted(glob.glob("/sys/class/infiniband_verbs/uverbs*/peer_mem_clients")):
        val = (read_text(pmc) or "").strip()
        if val and val not in ("0", ""):
            evidence.append(f"peer_mem_clients:{os.path.basename(os.path.dirname(pmc))}={val}")

    present = bool(evidence)
    return {"present": present, "evidence": evidence}


def probe_pci_p2pdma() -> str | None:
    """CONFIG_PCI_P2PDMA from the kernel config, if exposed (else None).

    P2PDMA is load-bearing for ionic dma-buf: a no-ODP NIC's dma-buf session is
    stable only when the kernel has P2PDMA compiled in (isKernelDmabufSupported()
    returns true) -- the amd-spur ionic dma-buf died for lack of it, the chi28xx
    ionic dma-buf survived a conc=128 bench because it has it. So detecting this
    correctly gates whether Mode C (capped ionic dma-buf) is viable or a blocker.
    """
    for path in glob.glob("/boot/config-*"):
        txt = read_text(path)
        if txt:
            m = re.search(r"^CONFIG_PCI_P2PDMA=(\S+)", txt, re.MULTILINE)
            if m:
                return m.group(1)
    try:
        with gzip.open("/proc/config.gz", "rt") as f:
            for line in f:
                if line.startswith("CONFIG_PCI_P2PDMA="):
                    return line.strip().split("=", 1)[1]
    except Exception:  # noqa: BLE001
        pass
    # Fallback for the engine container: /boot/config and /proc/config.gz are
    # typically NOT mounted inside the image, but /proc/kallsyms still exposes the
    # compiled-in pci_p2pdma_* symbols. Their presence == CONFIG_PCI_P2PDMA=y for
    # our purpose. Stream it (kallsyms is multi-MB) and stop at the first hit.
    try:
        with open("/proc/kallsyms", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "pci_p2pdma" in line:
                    return "y (from /proc/kallsyms)"
    except OSError:
        pass
    return None


def probe_dmabuf_engine() -> dict:
    """Best-effort: is the installed mooncake engine.so built with USE_HIP_DMABUF?

    dma-buf mode needs the rebuilt engine (the stock base compiles it out). We
    can't import-and-run here, so we locate the mooncake package's engine .so and
    nm-grep for the external symbol ibv_reg_dmabuf_mr (a strings grep is always
    empty -- it is a call symbol, not a literal). Unknown if we can't find it.
    """
    if not have("nm"):
        return {"compiled_in": None, "reason": "nm not available"}
    sos: list[str] = []
    for base in ("/opt/venv", "/usr", sys.prefix):
        sos += glob.glob(f"{base}/**/mooncake/*.so", recursive=True)
        sos += glob.glob(f"{base}/**/Mooncake/**/*engine*.so", recursive=True)
    for so in sorted(set(sos)):
        rc, out = run(["nm", "-D", so], timeout=8.0)
        if rc == 0 and out and re.search(r"ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf", out):
            return {"compiled_in": True, "so": so}
    if sos:
        return {"compiled_in": False, "so": sorted(set(sos))[0]}
    return {"compiled_in": None, "reason": "mooncake engine .so not found"}


def probe_gpus() -> dict:
    """GPU count + gfx + GPU<->NUMA map (best-effort via rocm-smi)."""
    info: dict = {"count": None, "gfx": None, "numa": {}}
    # count: kfd topology nodes with a gpu_id, else rocm-smi
    kfd = glob.glob("/sys/class/kfd/kfd/topology/nodes/*/gpu_id")
    gpu_ids = [p for p in kfd if (read_text(p) or "0").strip() not in ("0", "")]
    if gpu_ids:
        info["count"] = len(gpu_ids)
    rc, out = run(["rocm-smi", "--showtoponuma"], timeout=8.0)
    if rc == 0 and out:
        for m in re.finditer(r"GPU\[(\d+)\].*?Numa Node:\s*(\d+)", out):
            info["numa"][int(m.group(1))] = int(m.group(2))
        if info["count"] is None and info["numa"]:
            info["count"] = len(info["numa"])
    rc, out = run(["rocm-smi", "--showproductname"], timeout=8.0)
    if rc == 0 and out:
        m = re.search(r"gfx\w+", out)
        if m:
            info["gfx"] = m.group(0)
    return info


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
def _fast_rails(nics: list[dict]) -> list[dict]:
    """Active NICs on the node's fastest link tier (the rails PD would prefer)."""
    active = [n for n in nics if n["active"] and n["link_gbps"]]
    if not active:
        return [n for n in nics if n["active"]]
    top = max(n["link_gbps"] for n in active)
    return [n for n in active if n["link_gbps"] == top]


def _p2pdma_present(p2pdma: str | None) -> bool:
    """True when CONFIG_PCI_P2PDMA is compiled in (value 'y'/'m'/kallsyms hit)."""
    if not p2pdma:
        return False
    return p2pdma.strip().lower().startswith(("y", "m"))


def _rail_desc(nics: list[dict]) -> str:
    from collections import Counter

    tags = Counter(
        f"{n['vendor']}@{n['link_gbps']}G" if n["link_gbps"] else n["vendor"] for n in nics
    )
    return ", ".join(f"{c}x {t}" for t, c in tags.items())


def _total_gbps(nics: list[dict]) -> float:
    """Aggregate KV-carrying bandwidth of a NIC set (rails x per-rail speed).

    This is the number the ranking uses to compare fabrics: 8x400G ionic = 3200,
    1x200G mlx5 = 200. It captures WHY mode A/C (all ionic) have far more raw
    bandwidth than mode B (one ODP mlx5) even though B may still rank higher for
    being no-pin/no-cap. Down/no-speed NICs contribute 0."""
    return sum((n["link_gbps"] or 0) for n in nics)


# --------------------------------------------------------------------------- #
# Per-card capability (layer 1: the node facts, one row per NIC)
# --------------------------------------------------------------------------- #
def nic_capability(nics: list[dict], peermem: dict, dmabuf_engine: dict, p2pdma: str | None) -> list[dict]:
    """One capability row per RDMA NIC: can THIS card carry KV, and by which path.

    peer-mem is a node-level property (a loaded kernel module), so it's the same
    for every card; odp is per-card. ``dmabuf_registerable`` = the engine.so has
    ibv_reg_dmabuf_mr compiled in AND (the card has ODP OR the kernel has P2PDMA)
    -- i.e. a dma-buf MR would register and stay alive on this card (no-pin via ODP,
    or pinned-but-stable via P2PDMA). This is the layer-1 view a user/agent reads to
    see, per card, whether peer-mem / dma-buf / ODP is available."""
    pm = peermem["present"]
    engine_ok = dmabuf_engine.get("compiled_in") is not False  # True or unknown
    p2p = _p2pdma_present(p2pdma)
    out: list[dict] = []
    for n in nics:
        if n["gid_index"] is None:
            gid_disp = None
        elif n["gid_ipv4"]:
            gid_disp = f"{n['gid_index']}:{n['gid_ipv4']}"
        else:
            gid_disp = f"{n['gid_index']}:IPv6"
        out.append(
            {
                "device": n["device"],
                "vendor": n["vendor"],
                "link_gbps": n["link_gbps"],
                "active": n["active"],
                "peermem": pm,  # node-level; repeated per row for a self-contained table
                "odp": n["odp"],
                "dmabuf_registerable": bool(engine_ok and (n["odp"] or p2p)),
                "gid_index": n["gid_index"],
                "gid_display": gid_disp,
                "pci_bdf": n["pci_bdf"],
                "numa_node": n["numa_node"],
            }
        )
    return out


def image_capability(dmabuf_engine: dict) -> dict:
    """What the ENGINE IMAGE can do (layer 1). The infera sglang image is EXPECTED
    to support every mode -- bare ibv_reg_mr, ibv_reg_dmabuf_mr (dma-buf compiled
    in), and the HIP-transport gate -- all runtime-decided in one image (the unified
    Dockerfile.sglang). We probe engine.so to CONFIRM the dma-buf half is really
    compiled in; a mismatch means someone is on a stock/older base and should
    rebuild. The probe confirms, it does not define the baseline expectation."""
    compiled = dmabuf_engine.get("compiled_in")
    cap = {
        "expected": "infera sglang image supports ALL modes (bare ibv_reg_mr + "
        "ibv_reg_dmabuf_mr + HIP-transport gate, runtime-decided in one image)",
        "dmabuf_compiled_in": compiled,
        "so": dmabuf_engine.get("so"),
        "note": None,
    }
    if compiled is False:
        cap["note"] = (
            "engine.so has NO ibv_reg_dmabuf_mr -- this is a stock/older base, NOT the "
            "unified infera image. dma-buf modes (B/C) need a rebuild "
            "(deploy/docker/Dockerfile.sglang). Bare ibv_reg_mr (A) still works."
        )
    elif compiled is None:
        cap["note"] = (
            f"could not confirm ({dmabuf_engine.get('reason', 'unknown')}); verify: "
            "nm -D <mooncake engine.so> | grep ibv_reg_dmabuf_mr"
        )
    return cap


# --------------------------------------------------------------------------- #
# Per-mode evaluation (layer 2: enumerate A/B/C, viable or blocked + why)
# --------------------------------------------------------------------------- #
def _engine_warnings(dmabuf_engine: dict) -> list[dict]:
    """Shared dma-buf-engine warnings for modes B/C (need ibv_reg_dmabuf_mr)."""
    compiled = dmabuf_engine.get("compiled_in")
    if compiled is False:
        return [
            {
                "level": "blocker",
                "text": (
                    "engine.so built WITHOUT USE_HIP_DMABUF (ibv_reg_dmabuf_mr not "
                    "compiled in). Needs the unified image rebuild -- see "
                    "deploy/docker/Dockerfile.sglang."
                ),
            }
        ]
    if compiled is None:
        return [
            {
                "level": "verify",
                "text": (
                    "Could not confirm engine.so has USE_HIP_DMABUF "
                    f"({dmabuf_engine.get('reason', 'unknown')}). Verify: "
                    "nm -D <mooncake engine.so> | grep ibv_reg_dmabuf_mr."
                ),
            }
        ]
    return []


def _eval_mode_a(active: list[dict], peermem: dict) -> dict:
    """Mode A: bare ibv_reg_mr + peer-mem. Viable iff a peer-mem module is loaded."""
    gid = next((n["gid_index"] for n in active if n["gid_index"] is not None), 1)
    has_ionic = any(n["vendor"] == "ionic" for n in active)
    devices = [n["device"] for n in active]
    env = {
        "MOONCAKE_DISABLE_HIP_DMABUF": "1",
        "MC_DISABLE_HIP_TRANSPORT": "1",
        "MC_GID_INDEX": str(gid),
        "NCCL_IB_DISABLE": "1",
        **({"RDMAV_FORK_SAFE": "1"} if has_ionic else {}),
    }
    viable = peermem["present"]
    return {
        "mode": "A",
        "key": "A_peermem_ibv_reg_mr",
        "title": "bare ibv_reg_mr + peer-mem (default, no-pin, every rail)",
        "viable": viable,
        "reason": (
            "peer-mem present (" + ", ".join(peermem["evidence"]) + ")"
            if viable
            else "no peer-mem module loaded -- bare ibv_reg_mr would EFAULT on a device pointer"
        ),
        "nic_selection": {
            "devices": devices,
            "rule": "all active rails (cross-node; mooncake pairs by GID subnet). "
            "Single-node loopback: pin ONE device on both legs instead.",
        },
        "env": env,
        "launch_flags": [
            "--disaggregation-ib-device " + ",".join(devices)
            + "   # cross-node: all rails. Single-node: pin ONE (both legs same dev)."
        ]
        if devices
        else [],
        "needs_dmabuf_image": False,
        "needs_user_cap": False,
        "perf": {"rail_gbps": _total_gbps(active), "kv_pool": "full", "pin": "none"},
        "warnings": [],
    }


def _eval_mode_b(active: list[dict], peermem: dict, fast: list[dict], dmabuf_engine: dict) -> dict:
    """Mode B: ibv_reg_dmabuf_mr on an ODP NIC. Viable iff an ODP NIC exists AND the
    engine.so has dma-buf compiled in (unknown counts as viable-with-verify)."""
    odp_nics = [n for n in active if n["odp"]]
    engine_ok = dmabuf_engine.get("compiled_in") is not False
    warnings = list(_engine_warnings(dmabuf_engine))
    pick = sorted(odp_nics, key=lambda n: (-(n["link_gbps"] or 0), n["device"]))[0] if odp_nics else None
    gid = pick["gid_index"] if pick and pick["gid_index"] is not None else 3
    has_non_odp_active = any(not n["odp"] for n in active)

    # perf-regression: the ODP NIC(s) are slower/fewer than the node's fast rails.
    if odp_nics:
        odp_is_fast = all(n["odp"] for n in fast) and len(odp_nics) >= len(fast)
        if not odp_is_fast:
            warnings.append(
                {
                    "level": "perf-regression",
                    "text": (
                        f"KV forced onto ODP NIC(s) [{_rail_desc(odp_nics)}] instead of "
                        f"the fast rails [{_rail_desc(fast)}]. Expect a KV-transfer "
                        "BANDWIDTH REGRESSION -- the price of no-pin dma-buf without "
                        "peer-mem. Confirm the user accepts it."
                    ),
                }
            )
    viable = bool(odp_nics) and engine_ok
    if not odp_nics:
        reason = "no ODP-capable NIC -- ibv_reg_dmabuf_mr would pin+double the pool"
    elif not engine_ok:
        reason = f"ODP NIC {pick['device']} present but engine.so has no dma-buf"
    else:
        reason = f"ODP NIC {pick['device']} ({pick['vendor']}, {pick['link_gbps']}G) -- no-pin dma-buf"
    return {
        "mode": "B",
        "key": "B_dmabuf_odp_nic",
        "title": "ibv_reg_dmabuf_mr (GPUDirect dma-buf) on the ODP NIC (no-pin)",
        "viable": viable,
        "reason": reason,
        "nic_selection": {
            "devices": [pick["device"]] if pick else [],
            "rule": "the fastest ODP NIC (name tiebreak so prefill+decode pick the "
            "SAME one); MC_MS_FILTERS locks KV to it so non-ODP rails never pin.",
        },
        "env": {
            "MOONCAKE_DISABLE_HIP_DMABUF": "0",
            "MC_DISABLE_HIP_TRANSPORT": "1",
            "MC_MS_AUTO_DISC": "0",
            "MC_MS_FILTERS": pick["device"] if pick else "<odp-nic>",
            "MC_GID_INDEX": str(gid),
            "NCCL_IB_DISABLE": "1",
            **({"RDMAV_FORK_SAFE": "1"} if has_non_odp_active else {}),
        },
        "launch_flags": [f"--disaggregation-ib-device {pick['device']}"] if pick else [],
        "needs_dmabuf_image": True,
        "needs_user_cap": False,
        "perf": {
            "rail_gbps": _total_gbps([pick]) if pick else 0,
            "kv_pool": "full",
            "pin": "none",
        },
        "warnings": warnings,
    }


def _eval_mode_c(active: list[dict], p2pdma: str | None, dmabuf_engine: dict) -> dict:
    """Mode C: capped dma-buf on a no-ODP NIC (ionic), gated on kernel P2PDMA.

    ibv_reg_dmabuf_mr on a no-ODP NIC PINS + DOUBLES the KV pool; it is stable only
    when the kernel has CONFIG_PCI_P2PDMA (isKernelDmabufSupported()). Viable iff
    P2PDMA present AND engine has dma-buf; then the pool MUST be capped so weights +
    2x(pin) fit VRAM -- always a user-confirmed tradeoff. Without P2PDMA it's the
    amd-spur dead-end (session dies after the 1st transfer)."""
    p2p = _p2pdma_present(p2pdma)
    engine_ok = dmabuf_engine.get("compiled_in") is not False
    warnings = list(_engine_warnings(dmabuf_engine))
    devices = [n["device"] for n in active]
    gid = next((n["gid_index"] for n in active if n["gid_index"] is not None), 1)
    has_ionic = any(n["vendor"] == "ionic" for n in active)
    viable = p2p and engine_ok

    if not p2p:
        warnings.append(
            {
                "level": "blocker",
                "text": (
                    "kernel CONFIG_PCI_P2PDMA absent "
                    f"(probed: {p2pdma or 'unknown'}) -- no-ODP dma-buf session dies "
                    "after the first transfer (the amd-spur dead-end). No safe KV path."
                ),
            }
        )
        reason = "no kernel P2PDMA -- capped dma-buf can't be made stable here"
    elif not engine_ok:
        reason = "P2PDMA present but engine.so has no dma-buf"
    else:
        reason = f"P2PDMA present ({p2pdma}) -- capped ionic dma-buf is stable"
        warnings.append(
            {
                "level": "user-cap-confirm",
                "text": (
                    "ibv_reg_dmabuf_mr PINS + DOUBLES the KV pool -- it MUST be capped "
                    "so weights + 2x(pin) fit VRAM. Validated (DSv4-Pro TP8 MI355X): "
                    "--mem-fraction-static 0.75 --max-total-tokens 262144 -> 3.6 GB/card "
                    "(2x pin 7.2 GB), avail 140 GB, conc=128 0-fail. This CAP is a "
                    "throughput/context tradeoff -- CONFIRM WITH THE USER and recompute "
                    "per model/TP/VRAM."
                ),
            }
        )
    return {
        "mode": "C",
        "key": "C_cap_kv_dmabuf",
        "title": "capped dma-buf on a no-ODP NIC (ibv_reg_dmabuf_mr, KV pool capped)",
        "viable": viable,
        "reason": reason,
        "nic_selection": {
            "devices": devices,
            "rule": "all active rails; the load-bearing knob is the KV CAP, not NIC choice.",
        },
        "env": {
            "MOONCAKE_DISABLE_HIP_DMABUF": "0",
            "MC_DISABLE_HIP_TRANSPORT": "1",
            "MC_GID_INDEX": str(gid),
            "NCCL_IB_DISABLE": "1",
            **({"RDMAV_FORK_SAFE": "1"} if has_ionic else {}),
        },
        "launch_flags": (
            ["--disaggregation-ib-device " + ",".join(devices)] if devices else []
        )
        + [
            "--mem-fraction-static 0.75   # cap so weights + 2x KV pin fit VRAM",
            "--max-total-tokens 262144   # DSv4-Pro TP8 MI355X; RECOMPUTE per model/TP/VRAM",
        ],
        "needs_dmabuf_image": True,
        "needs_user_cap": True,
        "perf": {"rail_gbps": _total_gbps(active), "kv_pool": "capped", "pin": "2x"},
        "warnings": warnings,
    }


def _rank_key(m: dict) -> tuple:
    """Sort key (descending): safety/correctness first, bandwidth last.

    (viable, no-cap-needed, no-pin, aggregate-bandwidth). This reproduces the
    crsuse decision: mode B on one slow ODP mlx5 (no cap, no pin) OUTRANKS mode C
    on eight fast ionic (needs cap + 2x pin), i.e. we accept a bandwidth hit to
    avoid capping the KV pool. Bandwidth only breaks ties among equally-safe modes."""
    return (
        m["viable"],
        not m["needs_user_cap"],
        m["perf"]["pin"] == "none",
        m["perf"]["rail_gbps"],
    )


def evaluate_modes(
    nics: list[dict], peermem: dict, p2pdma: str | None, dmabuf_engine: dict
) -> list[dict]:
    """Enumerate ALL three modes (A, B, C) with viability + config, in fixed order.

    Unlike the old decide() (which returned a single recommendation), this returns
    every mode so a user/agent sees the full picture: what's viable, what's blocked
    and WHY. Ranking is a separate concern (see _rank_key)."""
    active = [n for n in nics if n["active"]]
    fast = _fast_rails(nics)
    return [
        _eval_mode_a(active, peermem),
        _eval_mode_b(active, peermem, fast, dmabuf_engine),
        _eval_mode_c(active, p2pdma, dmabuf_engine),
    ]


# --------------------------------------------------------------------------- #
# Report / CLI
# --------------------------------------------------------------------------- #
def collect() -> dict:
    nics = probe_nics()
    peermem = probe_peermem()
    p2pdma = probe_pci_p2pdma()
    dmabuf_engine = probe_dmabuf_engine()
    gpus = probe_gpus()
    modes = evaluate_modes(nics, peermem, p2pdma, dmabuf_engine)
    ranked = sorted(modes, key=_rank_key, reverse=True)
    best = ranked[0] if ranked and ranked[0]["viable"] else None
    return {
        "nics": nics,
        "peermem": peermem,
        "kernel_pci_p2pdma": p2pdma,
        "dmabuf_engine": dmabuf_engine,
        "gpus": gpus,
        "nic_capability": nic_capability(nics, peermem, dmabuf_engine, p2pdma),
        "image_capability": image_capability(dmabuf_engine),
        "modes": modes,  # fixed A,B,C order
        "ranked_keys": [m["key"] for m in ranked],
        "recommend": best["key"] if best else None,
    }


_RED = "\033[1;31m"
_YEL = "\033[1;33m"
_GRN = "\033[1;32m"
_BOLD = "\033[1m"
_RST = "\033[0m"


def _c(s: str, color: str, enable: bool) -> str:
    return f"{color}{s}{_RST}" if enable else s


def _warn_tag(level: str, color: bool) -> str:
    if level == "perf-regression":
        return _c("*** PERFORMANCE REGRESSION ***", _RED, color)
    if level == "blocker":
        return _c("*** BLOCKER ***", _RED, color)
    if level == "user-cap-confirm":
        return _c("*** USER CONFIRM: KV CAP ***", _RED, color)
    return _c(f"[{level}]", _YEL, color)


def render_table(report: dict, color: bool) -> str:
    """Human table: layer 1 (node + per-NIC + image facts) then layer 2 (A/B/C)."""
    lines: list[str] = []
    gpus = report["gpus"]

    # ---- layer 1: node facts ----
    lines.append(_c("=== NODE FACTS ===", _BOLD, color))
    lines.append(
        f"GPUs: {gpus.get('count')} x {gpus.get('gfx') or '?'}"
        f"   kernel CONFIG_PCI_P2PDMA={report['kernel_pci_p2pdma'] or 'unknown'}"
    )
    pm = report["peermem"]
    lines.append(
        f"peer-mem module: {'PRESENT' if pm['present'] else 'absent'}"
        + (f"  ({', '.join(pm['evidence'])})" if pm["evidence"] else "")
    )
    img = report["image_capability"]
    di = {True: "compiled-in", False: "NOT compiled-in", None: "unknown"}[img["dmabuf_compiled_in"]]
    lines.append(f"engine image dma-buf (ibv_reg_dmabuf_mr): {di}")
    lines.append(f"  expected: {img['expected']}")
    if img["note"]:
        lines.append("  " + _c(img["note"], _YEL, color))
    lines.append("")

    # ---- layer 1: per-NIC capability ----
    lines.append(_c("=== NIC CAPABILITY ===", _BOLD, color))
    hdr = (
        f"{'NIC':<10}{'vendor':<9}{'Gb/s':<7}{'peermem':<9}{'ODP':<5}"
        f"{'dmabuf':<8}{'GID':<18}{'BDF':<14}{'NUMA':<5}"
    )
    lines.append(hdr)
    for c in report["nic_capability"]:
        lines.append(
            f"{c['device']:<10}{c['vendor']:<9}{str(c['link_gbps'] or '-'):<7}"
            f"{('yes' if c['peermem'] else '-'):<9}{('yes' if c['odp'] else '-'):<5}"
            f"{('yes' if c['dmabuf_registerable'] else '-'):<8}"
            f"{(c['gid_display'] or '-'):<18}{(c['pci_bdf'] or '-'):<14}"
            f"{('-' if c['numa_node'] is None else str(c['numa_node'])):<5}"
            + ("" if c["active"] else "  [DOWN]")
        )
    lines.append("")

    # ---- layer 2: modes, ranked ----
    lines.append(_c("=== MODES (ranked: safety first, then bandwidth) ===", _BOLD, color))
    best = report["recommend"]
    by_key = {m["key"]: m for m in report["modes"]}
    for key in report["ranked_keys"]:
        m = by_key[key]
        star = " ★ best" if key == best else ""
        status = _c("viable", _GRN, color) if m["viable"] else _c("BLOCKED", _RED, color)
        lines.append("")
        lines.append(_c(f"[{m['mode']}] {m['title']}", _BOLD, color) + f"  — {status}{star}")
        lines.append(f"  why: {m['reason']}")
        sel = m["nic_selection"]
        lines.append(f"  NICs: {', '.join(sel['devices']) or '(none)'}")
        lines.append(f"        rule: {sel['rule']}")
        lines.append(
            f"  bandwidth: {m['perf']['rail_gbps']:.0f} Gb/s aggregate"
            f"   KV pool: {m['perf']['kv_pool']}   pin: {m['perf']['pin']}"
            f"   dma-buf image: {'YES' if m['needs_dmabuf_image'] else 'no'}"
        )
        if m["env"]:
            lines.append("  env: " + " ".join(f"{k}={v}" for k, v in m["env"].items()))
        for fl in m["launch_flags"]:
            lines.append(f"  flag: {fl}")
        for w in m["warnings"]:
            lines.append(f"  {_warn_tag(w['level'], color)} {_c(w['text'], _BOLD, color)}")

    lines.append("")
    lines.append(
        _c(
            "NOTE: this enumerates OPTIONS, not an order to launch. Modes needing a KV "
            "CAP require the user to confirm the cap before launching.",
            _YEL,
            color,
        )
    )
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    """Same two layers as the table, as GitHub-flavored Markdown (no ANSI)."""
    p: list[str] = []
    gpus = report["gpus"]
    img = report["image_capability"]
    di = {True: "compiled-in", False: "NOT compiled-in", None: "unknown"}[img["dmabuf_compiled_in"]]
    pm = report["peermem"]

    p.append("# Mooncake KV-registration mode report")
    p.append("")
    p.append("## Node facts")
    p.append("")
    p.append(f"- **GPUs:** {gpus.get('count')} x {gpus.get('gfx') or '?'}")
    p.append(f"- **kernel CONFIG_PCI_P2PDMA:** {report['kernel_pci_p2pdma'] or 'unknown'}")
    p.append(
        f"- **peer-mem module:** {'PRESENT' if pm['present'] else 'absent'}"
        + (f" ({', '.join(pm['evidence'])})" if pm["evidence"] else "")
    )
    p.append(f"- **engine image dma-buf (ibv_reg_dmabuf_mr):** {di}")
    p.append(f"- **image expectation:** {img['expected']}")
    if img["note"]:
        p.append(f"- **note:** {img['note']}")
    p.append("")

    p.append("## NIC capability")
    p.append("")
    p.append("| NIC | vendor | Gb/s | peer-mem | ODP | dma-buf | GID | BDF | NUMA | active |")
    p.append("|-----|--------|------|----------|-----|---------|-----|-----|------|--------|")
    for c in report["nic_capability"]:
        p.append(
            f"| {c['device']} | {c['vendor']} | {c['link_gbps'] or '-'} | "
            f"{'yes' if c['peermem'] else '-'} | {'yes' if c['odp'] else '-'} | "
            f"{'yes' if c['dmabuf_registerable'] else '-'} | {c['gid_display'] or '-'} | "
            f"{c['pci_bdf'] or '-'} | "
            f"{'-' if c['numa_node'] is None else c['numa_node']} | "
            f"{'yes' if c['active'] else 'DOWN'} |"
        )
    p.append("")

    p.append("## Modes (ranked: safety first, then bandwidth)")
    p.append("")
    p.append("| rank | mode | viable | Gb/s | KV pool | pin | dma-buf img | user cap | why |")
    p.append("|------|------|--------|------|---------|-----|-------------|----------|-----|")
    best = report["recommend"]
    by_key = {m["key"]: m for m in report["modes"]}
    for i, key in enumerate(report["ranked_keys"], start=1):
        m = by_key[key]
        star = " ★" if key == best else ""
        p.append(
            f"| {i}{star} | {m['mode']} | {'yes' if m['viable'] else 'BLOCKED'} | "
            f"{m['perf']['rail_gbps']:.0f} | {m['perf']['kv_pool']} | {m['perf']['pin']} | "
            f"{'yes' if m['needs_dmabuf_image'] else 'no'} | "
            f"{'yes' if m['needs_user_cap'] else 'no'} | {m['reason']} |"
        )
    p.append("")

    # per-mode detail blocks (env + flags + warnings)
    for key in report["ranked_keys"]:
        m = by_key[key]
        p.append(f"### [{m['mode']}] {m['title']}")
        p.append("")
        sel = m["nic_selection"]
        p.append(f"- **NICs:** {', '.join(sel['devices']) or '(none)'}")
        p.append(f"- **selection rule:** {sel['rule']}")
        if m["env"]:
            p.append("- **env:**")
            p.append("  ```bash")
            for k, v in m["env"].items():
                p.append(f"  export {k}={v}")
            p.append("  ```")
        if m["launch_flags"]:
            p.append("- **launch flags:**")
            p.append("  ```")
            for fl in m["launch_flags"]:
                p.append(f"  {fl}")
            p.append("  ```")
        for w in m["warnings"]:
            p.append(f"- **{w['level']}:** {w['text']}")
        p.append("")
    return "\n".join(p) + "\n"


def _write_outputs(report: dict, out_dir: str, emit: set[str], color: bool) -> list[str]:
    """Write requested formats into out_dir; return the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    if "json" in emit:
        path = os.path.join(out_dir, "mooncake_mode.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        written.append(path)
    if "md" in emit:
        path = os.path.join(out_dir, "mooncake_mode.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_markdown(report))
        written.append(path)
    if "table" in emit:
        path = os.path.join(out_dir, "mooncake_mode.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_table(report, color=False) + "\n")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="infera.tools.preflight.mooncake_mode", description=__doc__
    )
    p.add_argument(
        "--emit",
        default="table",
        help="comma-separated formats to print to stdout: table,json,md (default table)",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="also write mooncake_mode.{json,md,txt} into this dir (all three formats)",
    )
    p.add_argument("--json", action="store_true", help="back-compat alias for --emit json")
    p.add_argument("--quiet", action="store_true", help="with --json, suppress the human table on stderr")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color")
    args = p.parse_args(argv)

    report = collect()
    color = (not args.no_color) and sys.stdout.isatty()
    emit = {e.strip() for e in args.emit.split(",") if e.strip()}
    if args.json:
        emit = {"json"}

    # stdout
    if emit == {"json"}:
        if not args.quiet:
            sys.stderr.write(render_table(report, color) + "\n")
        print(json.dumps(report, indent=2))
    else:
        chunks: list[str] = []
        if "table" in emit:
            chunks.append(render_table(report, color))
        if "md" in emit:
            chunks.append(render_markdown(report))
        if "json" in emit:
            chunks.append(json.dumps(report, indent=2))
        print("\n\n".join(chunks))

    # optional file drop (always all three formats, machine-consumable)
    if args.out_dir:
        written = _write_outputs(report, args.out_dir, {"json", "md", "table"}, color)
        for path in written:
            sys.stderr.write(f"[mooncake_mode] wrote {path}\n")

    # exit 0 if a viable A/B (full-KV, no user cap) is the best pick, else 2. A
    # C-only node (best needs a KV cap) or an all-blocked node returns 2 so callers
    # know a human decision / rebuild is required.
    by_key = {m["key"]: m for m in report["modes"]}
    best = by_key.get(report["recommend"]) if report["recommend"] else None
    return 0 if (best and not best["needs_user_cap"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
