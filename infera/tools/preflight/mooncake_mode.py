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
P2PDMA, GPU topology), then RECOMMENDS a mode with the exact env + launch flags,
and RED-FLAGS a bandwidth regression when the only viable dma-buf NIC is slower or
fewer than the node's fast rails. It never launches anything and takes no live
transfer measurement (that is ``network.mooncakeperf``); it is the pre-decision
step. The skill that drives it ALWAYS presents the recommendation to the user and
lets them override -- the node can't see intent (a bandwidth hit the user has
already accepted, a spare NIC reserved for something else).

Usage:
    python -m infera.tools.preflight.mooncake_mode            # human report
    python -m infera.tools.preflight.mooncake_mode --json     # machine JSON
    python -m infera.tools.preflight.mooncake_mode --json --quiet > mode.json

Exit code: 0 if a viable mode (A or B) was recommended, 2 if only the stub (C).
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
    """CONFIG_PCI_P2PDMA from the kernel config, if exposed (else None)."""
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


def decide(
    nics: list[dict], peermem: dict, p2pdma: str | None, dmabuf_engine: dict
) -> dict:
    """Choose a mode + env + launch flags + warnings from the probe results."""
    active = [n for n in nics if n["active"]]
    odp_nics = [n for n in active if n["odp"]]
    fast = _fast_rails(nics)
    warnings: list[dict] = []

    # ---- Mode A: peer-mem present -> bare ibv_reg_mr, the default no-surprise path
    if peermem["present"]:
        gid = next((n["gid_index"] for n in active if n["gid_index"] is not None), 1)
        return {
            "mode": "A_peermem_ibv_reg_mr",
            "title": "bare ibv_reg_mr + peer-mem (default)",
            "viable": True,
            "nic_filter": None,  # all rails carry KV; let mooncake auto-discover
            "env": {
                "MOONCAKE_DISABLE_HIP_DMABUF": "1",
                "MC_DISABLE_HIP_TRANSPORT": "1",
                "MC_GID_INDEX": str(gid),
                "NCCL_IB_DISABLE": "1",
            },
            "launch_flags": [],  # no --disaggregation-ib-device: use every rail
            "needs_dmabuf_image": False,
            "warnings": warnings,
            "rationale": (
                "peer-mem module present ("
                + ", ".join(peermem["evidence"])
                + "): ibv_reg_mr registers GPU pages directly, nothing pinned or "
                "doubled, every RDMA rail usable. Stock image."
            ),
        }

    # ---- Mode B: no peer-mem but an ODP NIC exists -> ibv_reg_dmabuf_mr on it
    if odp_nics:
        pick = sorted(odp_nics, key=lambda n: (-(n["link_gbps"] or 0), n["device"]))[0]
        gid = pick["gid_index"] if pick["gid_index"] is not None else 3
        # Perf regression: the only ODP NIC(s) are slower or fewer than the fast
        # rails (e.g. one 200G mlx5 vs eight 400G ionic). All KV is forced onto
        # the ODP NIC -- a real bandwidth downgrade the user must accept.
        odp_is_fast = all(n["odp"] for n in fast) and len(odp_nics) >= len(fast)
        if not odp_is_fast:
            fast_desc = _rail_desc(fast)
            odp_desc = _rail_desc(odp_nics)
            warnings.append(
                {
                    "level": "perf-regression",
                    "text": (
                        f"KV is forced onto the ODP NIC(s) [{odp_desc}] instead of "
                        f"the node's fast rails [{fast_desc}]. Expect a KV-transfer "
                        "BANDWIDTH REGRESSION (fewer and/or slower rails carry the "
                        "whole KV pool). This is the price of no-pin dma-buf without "
                        "peer-mem; confirm the user accepts it."
                    ),
                }
            )
        if dmabuf_engine.get("compiled_in") is False:
            warnings.append(
                {
                    "level": "blocker",
                    "text": (
                        "The installed mooncake engine.so was built WITHOUT "
                        "USE_HIP_DMABUF (ibv_reg_dmabuf_mr not compiled in). dma-buf "
                        "mode needs the rebuilt image -- see "
                        "deploy/docker/scripts/build_mooncake_dmabuf.sh / "
                        "deploy/docker/Dockerfile.sglang.dmabuf."
                    ),
                }
            )
        elif dmabuf_engine.get("compiled_in") is None:
            warnings.append(
                {
                    "level": "verify",
                    "text": (
                        "Could not confirm the engine.so has USE_HIP_DMABUF "
                        f"({dmabuf_engine.get('reason', 'unknown')}). Verify with: "
                        "nm -D <mooncake engine.so> | grep ibv_reg_dmabuf_mr."
                    ),
                }
            )
        has_non_odp_active = any(not n["odp"] for n in active)
        return {
            "mode": "B_dmabuf_ibv_reg_dmabuf_mr",
            "title": "ibv_reg_dmabuf_mr (GPUDirect dma-buf) on the ODP NIC",
            "viable": True,
            "nic_filter": pick["device"],
            "env": {
                "MOONCAKE_DISABLE_HIP_DMABUF": "0",
                "MC_DISABLE_HIP_TRANSPORT": "1",
                "MC_MS_AUTO_DISC": "0",
                "MC_MS_FILTERS": pick["device"],
                "MC_GID_INDEX": str(gid),
                "NCCL_IB_DISABLE": "1",
                # ionic rails present on the box need fork-safety; harmless for mlx5
                **({"RDMAV_FORK_SAFE": "1"} if has_non_odp_active else {}),
            },
            "launch_flags": [f"--disaggregation-ib-device {pick['device']}"],
            "needs_dmabuf_image": True,
            "warnings": warnings,
            "rationale": (
                f"No peer-mem module; NIC {pick['device']} ({pick['vendor']}, "
                f"{pick['link_gbps']}G) supports ODP, so ibv_reg_dmabuf_mr dynamic-"
                "attaches without pinning -> KV pool not doubled. Forced onto this "
                "NIC (MC_MS_FILTERS) so non-ODP rails never pin. Needs the dma-buf "
                "image."
            ),
        }

    # ---- Mode C: no peer-mem and no ODP NIC -> cap-KV workaround (STUB)
    warnings.append(
        {
            "level": "blocker",
            "text": (
                "No peer-mem module AND no ODP-capable NIC. Bare ibv_reg_mr EFAULTs "
                "on a device pointer; ibv_reg_dmabuf_mr would PIN + DOUBLE the KV "
                "pool (SIGSEGV/HIP-209 on a large pool). Neither full-KV path is "
                "safe on this node."
            ),
        }
    )
    return {
        "mode": "C_cap_kv_workaround",
        "title": "cap KV cache to survive a pinned dma-buf region (STUB)",
        "viable": False,
        "nic_filter": None,
        "env": {
            # left for a follow-up: the safe subset is dma-buf with a KV pool small
            # enough that the doubled pin fits, or the driver-bug workaround.
            "MOONCAKE_DISABLE_HIP_DMABUF": "0",
            "MC_DISABLE_HIP_TRANSPORT": "1",
        },
        "launch_flags": [
            # STUB: another agent fills the exact cap (e.g. --max-total-tokens / a
            # reduced --mem-fraction-static so weights + 2x KV pin fit in VRAM).
            "# TODO(cap-kv): --max-total-tokens <N>  # size so weights + 2*KV_pin < VRAM",
        ],
        "needs_dmabuf_image": True,
        "warnings": warnings,
        "rationale": (
            "Fallback path intentionally left as a stub -- fill the KV-cap / driver-"
            "bug workaround here."
        ),
    }


def _rail_desc(nics: list[dict]) -> str:
    from collections import Counter

    tags = Counter(
        f"{n['vendor']}@{n['link_gbps']}G" if n["link_gbps"] else n["vendor"] for n in nics
    )
    return ", ".join(f"{c}x {t}" for t, c in tags.items())


# --------------------------------------------------------------------------- #
# Report / CLI
# --------------------------------------------------------------------------- #
def collect() -> dict:
    nics = probe_nics()
    peermem = probe_peermem()
    p2pdma = probe_pci_p2pdma()
    dmabuf_engine = probe_dmabuf_engine()
    gpus = probe_gpus()
    recommendation = decide(nics, peermem, p2pdma, dmabuf_engine)
    return {
        "nics": nics,
        "peermem": peermem,
        "kernel_pci_p2pdma": p2pdma,
        "dmabuf_engine": dmabuf_engine,
        "gpus": gpus,
        "recommendation": recommendation,
    }


_RED = "\033[1;31m"
_YEL = "\033[1;33m"
_GRN = "\033[1;32m"
_BOLD = "\033[1m"
_RST = "\033[0m"


def _c(s: str, color: str, enable: bool) -> str:
    return f"{color}{s}{_RST}" if enable else s


def render(report: dict, color: bool) -> str:
    lines: list[str] = []
    gpus = report["gpus"]
    lines.append(
        f"GPUs: {gpus.get('count')} x {gpus.get('gfx') or '?'}"
        f"   kernel CONFIG_PCI_P2PDMA={report['kernel_pci_p2pdma'] or 'unknown'}"
    )
    pm = report["peermem"]
    pm_txt = "PRESENT" if pm["present"] else "absent"
    lines.append(f"peer-mem module: {pm_txt}" + (f"  ({', '.join(pm['evidence'])})" if pm["evidence"] else ""))
    de = report["dmabuf_engine"]
    lines.append(
        "dma-buf engine.so (USE_HIP_DMABUF): "
        + {True: "compiled-in", False: "NOT compiled-in", None: "unknown"}[de.get("compiled_in")]
    )
    lines.append("")
    lines.append(f"{'NIC':<10}{'vendor':<10}{'netdev':<9}{'ipv4':<16}{'Gb/s':<7}{'ODP':<5}{'GID':<20}{'BDF':<14}")
    for n in report["nics"]:
        odp = "yes" if n["odp"] else "-"
        if n["gid_index"] is None:
            gid = "-"
        elif n["gid_ipv4"]:
            gid = f"{n['gid_index']}:{n['gid_ipv4']}"  # routable IPv4-mapped
        else:
            gid = f"{n['gid_index']}:IPv6"  # global-IPv6 RoCEv2 (raw GID in --json)
        lines.append(
            f"{n['device']:<10}{n['vendor']:<10}{(n['netdev'] or '-'):<9}"
            f"{(n['ipv4'] or '-'):<16}{str(n['link_gbps'] or '-'):<7}{odp:<5}"
            f"{gid:<20}{(n['pci_bdf'] or '-'):<14}"
            + ("" if n["active"] else "  [DOWN]")
        )
    lines.append("")

    rec = report["recommendation"]
    head = f"RECOMMENDED MODE: {rec['mode']}  —  {rec['title']}"
    lines.append(_c(head, _GRN if rec["viable"] else _RED, color))
    lines.append(f"  why: {rec['rationale']}")
    if rec["nic_filter"]:
        lines.append(f"  NIC: {rec['nic_filter']}")
    lines.append(f"  needs dma-buf image: {'YES' if rec['needs_dmabuf_image'] else 'no'}")
    lines.append("  env:")
    for k, v in rec["env"].items():
        lines.append(f"    {k}={v}")
    if rec["launch_flags"]:
        lines.append("  launch flags:")
        for f in rec["launch_flags"]:
            lines.append(f"    {f}")
    for w in rec["warnings"]:
        if w["level"] == "perf-regression":
            tag = _c("*** PERFORMANCE REGRESSION ***", _RED, color)
        elif w["level"] == "blocker":
            tag = _c("*** BLOCKER ***", _RED, color)
        else:
            tag = _c(f"[{w['level']}]", _YEL, color)
        lines.append("")
        lines.append(f"  {tag} {_c(w['text'], _BOLD, color)}")
    lines.append("")
    lines.append(
        _c(
            "NOTE: this is a recommendation. ALWAYS confirm the mode with the user "
            "before launching — the node cannot see intent (an accepted bandwidth "
            "hit, a NIC reserved for something else).",
            _YEL,
            color,
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="infera.tools.preflight.mooncake_mode", description=__doc__
    )
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--quiet", action="store_true", help="with --json, suppress the human report")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color")
    args = p.parse_args(argv)

    report = collect()
    color = (not args.no_color) and sys.stdout.isatty()
    if args.json:
        if not args.quiet:
            sys.stderr.write(render(report, color) + "\n")
        print(json.dumps(report, indent=2))
    else:
        print(render(report, color))
    return 0 if report["recommendation"]["viable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
