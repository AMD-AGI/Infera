###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Mooncake KV-transfer check: real cross-node buffer transfer over Mooncake's
TransferEngine, measuring what PD would actually get.

Mirrors how sglang actually launches mooncake, cross-checked against the OFFICIAL
Mooncake API (not just one launch script): initialize() takes
    initialize(local_hostname, metadata_server, protocol, device_name)
where device_name is "a comma-separated list of device names to FILTER, or EMPTY
STRING FOR ALL DEVICES" (kvcache-ai Mooncake python-api docs). The engine then
"discovers the topology between CPU/CUDA and RDMA devices automatically and
installs Transport based on the topology" (C++ API docs) -- routing each buffer
to its affine NIC. The working sglang scripts pass no --disaggregation-ib-device,
so its MooncakeTransferEngine calls initialize(host, "P2PHANDSHAKE", "rdma", "")
-- empty = all devices, auto-discovery. We do the same here.

NOTE: passing a device name is NOT an error -- it is the officially supported
device FILTER (equivalently MC_TE_FILTERS / RDMA_DEVICE_NAME), and Infera's PD doc
even recommends matching rails cross-node. So this test deliberately does not
flag a configured ib_device; it just reproduces the default (auto-discovery).

Production env knobs we replicate (Infera rocm_rdma_env defaults / PD doc):
  - MC_GID_INDEX: pin the routable RoCE v2 GID (index 0 is link-local, times out
    cross-node). Mooncake's own default is ~max index and can be wrong.
  - MC_DISABLE_HIP_TRANSPORT=1: force RDMA over the intra-node HIP/XGMI shortcut
    (which advertises an empty segment the peer rejects cross-node).

It catches the silent misconfigurations that make cross-node KV slow or broken:
  - TCP fallback: with RDMA HCAs present Mooncake force-installs RDMA and ignores
    the protocol arg, so exercising TCP needs MC_FORCE_TCP=1.
  - RDMA GID: Mooncake's auto-selection picks a link-local GID that can't route
    cross-node; MC_GID_INDEX must pin the routable RoCE v2 GID.
  - GPU registration + NIC affinity: sglang registers KV in device VRAM
    (batch_register on a cuda pointer). On AMD ionic that bare-VRAM registration
    is the path that can fail, so per-GPU `rdma-gpu{g}` variants register real
    VRAM on GPU g and let Mooncake's auto-discovery route it to that GPU's affine
    NIC -- exactly as production does.

Each measurement reports the env it needed. RDMA is run with the GID pinned as a
correct deployment would (`rdma`, DRAM baseline, and one `rdma-gpu{g}` per GPU,
VRAM), plus `rdma-default` (nothing set, what an operator who forgets the knob
gets) and `tcp`. Because Mooncake caches its global config once per process
(std::call_once for MC_GID_INDEX), each variant runs in its own subprocess with a
clean environment.

INFERA_PREFLIGHT_KV_GPUS caps the per-GPU sweep. Unset (the default) sweeps ALL
local GPUs -- the most production-faithful coverage, but also the slowest: the
node-pair grid is multiplied by (3 + n_gpus) variants, and each mooncake variant
is a fresh subprocess that re-imports and re-initializes the engine. Set it to
e.g. 2 for a fast smoke test -- the ionic bare-VRAM failure is a driver/stack-wide
property that is identical across GPUs, so 1-2 GPUs already reproduce it; reserve
the full sweep for when you need to confirm every GPU's affinity path individually.

Coordination reuses netperf's shared-dir rendezvous: each ordered node pair is
tested both directions; the target registers a buffer stamped with a
per-(gpu,segment) pattern (numpy/torch + register_memory -- allocate_managed_buffer
yields a non-registered address that fails remote reads) and publishes its
address; the initiator batch-reads it (a batch of outstanding requests in
flight, enough to saturate the link on the VRAM path), reports the average
bandwidth, and verifies the bytes it
pulled back match the pattern so an offset/mis-routed-NIC bug is caught rather
than reported green. Runs in-container where Mooncake + the injected host
libionic live.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from ..finding import Finding
from .netperf import (
    _agree_min,
    _barrier,
    _exit_reason,
    _gid_index,
    _mgmt_ip,
    _nics,
    _parse_rdma_errno,
    _touch,
    _wait_file,
)

_RDMA_DEVICE = "ionic_0"  # only used as a fallback for reading the GID index
# GPU (VRAM) geometry mirrors the KV cache: sglang batch_registers device VRAM
# and moves it in ~32 MiB requests, so a 1 GiB region in 32 requests is the real
# per-transfer size the KV path drives.
_GPU_CHUNK = 32 << 20  # 32 MiB per request
_GPU_NCHUNK = 32  # outstanding requests per batch (queue depth); >32 overruns the QP
_GPU_SIZE = _GPU_CHUNK * _GPU_NCHUNK  # 1 GiB VRAM region
# CPU (host DRAM) geometry mirrors PRODUCTION, not a synthetic 1 GiB. The only
# host-DRAM memory sglang registers + RDMAs is the aux/metadata buffers
# (MetadataBuffers, device="cpu"); mooncake send_aux transfers them over RDMA by
# default. Those are a `max_running_requests*2` x per-item region -- each item is
# 64 B..512 B, or up to ~16 KiB (EAGLE hidden_states) -- so the whole region is
# tens of MiB at most and each transfer is <=16 KiB. Registering a 1 GiB host MR
# instead exceeds ulimit -l, so ibv_reg_mr silently ENOMEMs and the tail never
# lands -- a synthetic "data mismatch" production never hits. So we register a
# 64 MiB region (the worst-case aux footprint, still under a normal memlock
# limit) and move it in 16 KiB requests (the largest real aux item).
_CPU_CHUNK = 16 << 10  # 16 KiB per request (largest production aux item)
_CPU_NCHUNK = 32  # outstanding requests per batch
_CPU_SIZE = 64 << 20  # 64 MiB registered host-DRAM region (max aux footprint)
_MIN_SECONDS = 3.0  # keep transferring for at least this long
_PORT_BASE = 17000
_TARGET_TIMEOUT = 120.0  # initiator waits for the target buffer to be published
_DONE_TIMEOUT = 180.0  # target waits for the initiator to finish
_STEP_TIMEOUT = 240.0  # hard cap on one endpoint subprocess


_GID_CACHE: int | None = None


def _ref_gid() -> int:
    """The routable RoCE v2 GID index to pin via MC_GID_INDEX. Production hardcodes
    1; we read it from the first NIC (same value on this fleet) for robustness.
    Cached -- it is read once per spawn and once per finding otherwise."""
    global _GID_CACHE
    if _GID_CACHE is None:
        nics = _nics()
        _GID_CACHE = _gid_index(nics[0] if nics else _RDMA_DEVICE)
    return _GID_CACHE


def _kv_gpus() -> int:
    """Number of local GPUs to sweep (one VRAM buffer per GPU; Mooncake's
    auto-discovery routes each to its affine NIC). Capped by INFERA_PREFLIGHT_KV_GPUS."""
    try:
        import torch
    except ImportError:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        n = torch.cuda.device_count()
    except Exception:
        return 0
    cap = os.environ.get("INFERA_PREFLIGHT_KV_GPUS")
    if cap:
        try:
            n = min(n, int(cap))
        except ValueError:
            pass
    return n


def _variants(ngpu: int) -> list[tuple]:
    """(label, protocol, env_kind, loc, gpu_id, dev). rdma-default deliberately sets
    no GID; rdma-gpu{g} registers VRAM on GPU g (Mooncake auto-routes to its NIC).
    `dev` pins the NIC whitelist (only the CPU baseline pins one; "" = auto). ngpu
    is the cross-rank-agreed GPU count (see run) so every rank matches."""
    nics = _nics()
    # Pin the CPU rdma baseline to one fixed NIC (same index on both ends -> same
    # rail); GPU variants keep "" so Mooncake auto-routes to each GPU's affine NIC.
    cpu_dev = nics[0] if nics else ""
    variants: list[tuple] = [
        ("rdma", "rdma", "gid", "cpu", -1, cpu_dev),
        ("rdma-default", "rdma", "none", "cpu", -1, ""),
        ("tcp", "tcp", "tcp", "cpu", -1, ""),
    ]
    for g in range(ngpu):
        variants.append((f"rdma-gpu{g}", "rdma", "gid", "gpu", g, ""))
    return variants


def _geom(loc: str) -> tuple[int, int, int]:
    """(registered size, per-request chunk, requests per batch) for this buffer
    class: GPU mirrors the KV cache (1 GiB VRAM in 32 MiB requests), CPU mirrors
    the aux/metadata buffers (64 MiB host DRAM in 16 KiB requests)."""
    if loc == "gpu":
        return _GPU_SIZE, _GPU_CHUNK, _GPU_NCHUNK
    return _CPU_SIZE, _CPU_CHUNK, _CPU_NCHUNK


def _chunk_byte(gpu_id: int, i: int) -> int:
    """Distinct byte per (gpu, segment): encodes the GPU tag so a buffer read via
    the wrong NIC, or a segment at the wrong offset, shows up as a mismatch."""
    return ((gpu_id & 0xFF) * 131 + i * 7 + 1) & 0xFF


def _verify(host_arr, gpu_id: int, chunk: int, nchunk: int) -> bool:
    import numpy as np

    for i in range(nchunk):
        seg = host_arr[i * chunk : (i + 1) * chunk]
        if not bool(np.all(seg == _chunk_byte(gpu_id, i))):
            return False
    return True


class _Buf:
    """A registered buffer on host DRAM (loc='cpu', 64 MiB) or a specific GPU's
    VRAM (loc='gpu', 1 GiB, device cuda:gpu_id). `size`/`chunk`/`nchunk` come from
    _geom(loc). Exposes the raw pointer for register_memory plus pattern stamp /
    readback."""

    def __init__(self, loc: str, gpu_id: int) -> None:
        self.loc = loc
        self.gpu_id = gpu_id if loc == "gpu" else -1
        self.size, self.chunk, self.nchunk = _geom(loc)
        if loc == "gpu":
            import torch

            self._torch = torch
            self._t = torch.empty(self.size, dtype=torch.uint8, device=f"cuda:{gpu_id}")
            self.ptr = self._t.data_ptr()
        else:
            import numpy as np

            self._a = np.empty(self.size, dtype=np.uint8)
            self.ptr = self._a.ctypes.data

    def fill(self, value: int) -> None:
        if self.loc == "gpu":
            self._t.fill_(int(value))
            self._torch.cuda.synchronize(self.gpu_id)
        else:
            self._a[:] = value

    def fill_pattern(self) -> None:
        for i in range(self.nchunk):
            v = _chunk_byte(self.gpu_id, i)
            if self.loc == "gpu":
                self._t[i * self.chunk : (i + 1) * self.chunk].fill_(v)
            else:
                self._a[i * self.chunk : (i + 1) * self.chunk] = v
        if self.loc == "gpu":
            self._torch.cuda.synchronize(self.gpu_id)

    def host_bytes(self):
        if self.loc == "gpu":
            return self._t.cpu().numpy()
        return self._a


def _make_buffer(loc: str, gpu_id: int):
    try:
        return _Buf(loc, gpu_id)
    except Exception:
        return None


def _engine(hostname: str, protocol: str, device: str = ""):
    """A Mooncake TransferEngine bound to hostname, or None if unavailable.

    `device` is the 4th initialize() arg (the NIC whitelist). Empty = auto-discover
    and topology-route each buffer to its affine NIC -- exactly like sglang when
    --disaggregation-ib-device is unset, and what the GPU variants use so a per-GPU
    VRAM buffer lands on its index-affine ionic. The CPU baseline instead pins a
    fixed device on BOTH ends: this is a rail-optimized fabric (ionic_i only routes
    to the peer's ionic_i), and a CPU buffer has no GPU affinity, so auto-discovery
    would let the two ends pick different rails and fail. Env (MC_GID_INDEX /
    MC_FORCE_TCP) is set by the parent before this subprocess starts, so it is read
    cleanly at import/initialize here."""
    try:
        from mooncake.engine import TransferEngine
    except ImportError:
        return None
    eng = TransferEngine()
    if eng.initialize(hostname, "P2PHANDSHAKE", protocol, device) != 0:
        return None
    return eng


def _register(eng, ptr: int, size: int) -> bool:
    """register_memory the buffer; auto-detects DRAM vs VRAM from the pointer.
    Returns False when the driver rejects it (the ionic bare-VRAM failure).

    Mooncake's register_memory returns int 0 on success and non-zero on failure
    (confirmed on-cluster: the VRAM path returns non-zero, surfaced as
    register_failed). Some bindings return None on success and raise on error, so
    a non-int (None) is treated as success -- failures still come through as a
    raised exception (caught) or a non-zero int."""
    try:
        ret = eng.register_memory(ptr, size)
    except Exception:
        return False
    return not (isinstance(ret, int) and ret != 0)


def _batch_read(eng, target_hostname: str, local: int, peer: int, chunk: int, nchunk: int) -> bool:
    # One batch of `nchunk` outstanding reads; success is >=0 (matching sglang).
    srcs = [local + i * chunk for i in range(nchunk)]
    dsts = [peer + i * chunk for i in range(nchunk)]
    lens = [chunk] * nchunk
    try:
        return eng.batch_transfer_sync_read(target_hostname, srcs, dsts, lens) >= 0
    except Exception:
        return False


def _target(
    sig: str, hostname: str, host: str, protocol: str, loc: str, gpu_id: int, device: str = ""
) -> None:
    eng = _engine(hostname, protocol, device)
    info = {"ok": False, "host": host, "loc": loc, "gpu": gpu_id}
    buf = None
    if eng is not None:
        buf = _make_buffer(loc, gpu_id)
        if buf is None:
            info["reason"] = "no_gpu"
        else:
            buf.fill_pattern()  # the initiator reads this back and verifies it
            if not _register(eng, buf.ptr, buf.size):
                info["reason"] = "register_failed"
            else:
                # P2PHANDSHAKE ignores the port we pass and binds its own; publish
                # the real one (get_rpc_port) so the initiator connects correctly.
                mgmt = hostname.rsplit(":", 1)[0]
                info = {
                    "ok": True,
                    "host": host,
                    "loc": loc,
                    "gpu": gpu_id,
                    "hostname": f"{mgmt}:{eng.get_rpc_port()}",
                    "addr": buf.ptr,
                }
    tmp = os.path.join(sig, "target.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    os.replace(tmp, os.path.join(sig, "target.json"))
    _wait_file(os.path.join(sig, "done"), _DONE_TIMEOUT)
    del buf  # keep the buffer registered until the initiator is done


def _initiator(sig: str, protocol: str, loc: str, gpu_id: int, device: str = "") -> dict:
    rec: dict = {
        "gb_s": None,
        "gib": 0.0,
        "target": None,
        "loc": loc,
        "gpu": gpu_id,
        "verified": None,
        "reason": None,
        "dev": device,
    }
    if _wait_file(os.path.join(sig, "target.json"), _TARGET_TIMEOUT):
        with open(os.path.join(sig, "target.json"), encoding="utf-8") as fh:
            tgt = json.load(fh)
        rec["target"] = tgt.get("host")
        eng = _engine(f"{_mgmt_ip()}:0", protocol, device)
        if eng is None:
            rec["reason"] = "engine_unavailable"
        elif not tgt.get("ok"):
            rec["reason"] = tgt.get("reason") or "target_unavailable"
        else:
            buf = _make_buffer(loc, gpu_id)
            if buf is None:
                rec["reason"] = "no_gpu"
            elif not _register(eng, buf.ptr, buf.size):
                rec["reason"] = "register_failed"
            else:
                batch = buf.chunk * buf.nchunk  # bytes moved per batch_read
                buf.fill(0)  # sentinel; a successful read overwrites it with the pattern
                if _batch_read(
                    eng, tgt["hostname"], buf.ptr, tgt["addr"], buf.chunk, buf.nchunk
                ):  # warm up
                    moved, t0 = 0, time.monotonic()
                    while time.monotonic() - t0 < _MIN_SECONDS:
                        if not _batch_read(
                            eng, tgt["hostname"], buf.ptr, tgt["addr"], buf.chunk, buf.nchunk
                        ):
                            moved = 0
                            break
                        moved += batch
                    dt = time.monotonic() - t0
                    if moved > 0 and dt > 0:
                        rec["gb_s"] = round(moved / dt / 1e9, 2)
                        rec["gib"] = round(moved / (1 << 30), 1)
                        rec["verified"] = _verify(buf.host_bytes(), gpu_id, buf.chunk, buf.nchunk)
                    else:
                        rec["reason"] = "transfer_failed"
                else:
                    rec["reason"] = "transfer_failed"
    _touch(os.path.join(sig, "done"))
    return rec


def _spawn(
    role: str,
    sig: str,
    hostname: str,
    host: str,
    protocol: str,
    env_kind: str,
    loc: str,
    gpu_id: int,
    dev: str = "",
) -> tuple[int | None, str]:
    """Run one endpoint (target/initiator) in a clean subprocess so Mooncake's
    once-cached global config sees the right env for this variant. Returns
    (exit code, captured output); rc is None on timeout.

    We capture (but don't print) the child's output so we can scrape the
    libibverbs register errno (e.g. EFAULT) out of a VRAM register failure and
    fold it into the finding reason."""
    env = dict(os.environ)
    env.pop("MC_GID_INDEX", None)
    env.pop("MC_FORCE_TCP", None)
    env.pop("MC_DISABLE_HIP_TRANSPORT", None)
    if protocol == "rdma":
        # Documented cross-node requirement (Infera rocm_rdma_env default): force
        # RDMA, not the intra-node HIP/XGMI shortcut, which advertises an empty
        # segment the peer rejects. Set on all rdma variants so rdma-default varies
        # ONLY in the missing MC_GID_INDEX it is meant to demonstrate.
        env["MC_DISABLE_HIP_TRANSPORT"] = "1"
    if env_kind == "gid":
        env["MC_GID_INDEX"] = str(_ref_gid())
    elif env_kind == "tcp":
        env["MC_FORCE_TCP"] = "1"
    spec = {
        "role": role,
        "sig": sig,
        "hostname": hostname,
        "host": host,
        "protocol": protocol,
        "loc": loc,
        "gpu": gpu_id,
        "dev": dev,
    }
    cmd = [sys.executable, "-m", "infera.tools.preflight.network.mooncakeperf", json.dumps(spec)]
    try:
        # Capture (rather than DEVNULL) Mooncake's verbose GID-probe logging: it is
        # not shown, but lets us scrape a register errno on failure. The result
        # itself still travels via result.json.
        cp = subprocess.run(
            cmd,
            env=env,
            timeout=_STEP_TIMEOUT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return cp.returncode, (cp.stdout or "")[-65536:]
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return None, out[-65536:]


def run(dump_path: str, rank: int, world: int, host: str) -> list[Finding]:
    if world < 2:
        return [Finding("info", "mooncake skipped (single node)", {})]
    try:
        import mooncake.engine  # noqa: F401
    except ImportError:
        return [Finding("warn", "mooncake skipped (python bindings not importable)", {})]

    mgmt = _mgmt_ip()
    root = os.path.join(dump_path, "mooncakeperf")
    # Agree the per-GPU sweep count across ranks (min) so every rank iterates the
    # SAME variant set -- otherwise the per-pair barriers desync and hang.
    ngpu = _agree_min(os.path.join(root, "kvgpus"), rank, world, _kv_gpus(), _TARGET_TIMEOUT)
    variants = _variants(ngpu)
    recs: list[dict] = []
    idx = 0
    for s in range(world):
        for c in range(world):
            if s == c:
                continue
            for label, protocol, env_kind, loc, gpu_id, dev in variants:
                sig = os.path.join(root, f"{s}_{c}_{label}")
                os.makedirs(sig, exist_ok=True)
                hostname = f"{mgmt}:{_PORT_BASE + idx}"
                if rank == s:
                    _spawn("target", sig, hostname, host, protocol, env_kind, loc, gpu_id, dev)
                elif rank == c:
                    rc, out = _spawn(
                        "initiator", sig, hostname, host, protocol, env_kind, loc, gpu_id, dev
                    )
                    recs.append(_load_result(sig, label, loc, gpu_id, s, rc, out, dev))
                _barrier(os.path.join(root, "bar", f"{s}_{c}_{label}"), rank, world)
                idx += 1

    if not recs:
        return [Finding("info", "mooncake: no initiator role for this node", {})]
    return [_finding(r, host) for r in recs]


def _load_result(
    sig: str,
    label: str,
    loc: str,
    gpu_id: int,
    target_rank: int,
    rc: int | None = 0,
    out: str = "",
    dev: str = "",
) -> dict:
    try:
        with open(os.path.join(sig, "result.json"), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        rec = {
            "gb_s": None,
            "gib": 0.0,
            "target": None,
            "loc": loc,
            "gpu": gpu_id,
            "verified": None,
            "reason": _exit_reason(rc, out),
            "dev": dev,
        }
    # Enrich a register/transfer failure reason with any errno scraped from the
    # child's output (EFAULT vs ENOMEM etc.), same as the mori path.
    if rec.get("gb_s") is None and rec.get("reason"):
        detail = _parse_rdma_errno(out)
        if detail and detail not in rec["reason"]:
            rec["reason"] = f"{rec['reason']}: {detail}"
    # A transfer that "succeeds" (gb_s set) but fails verification is usually a
    # SILENT registration failure: mooncake's register_memory returns 0 even when
    # libibverbs could not pin the buffer (host-DRAM MR -> ENOMEM [12]), so the
    # buffer's tail never lands and the readback mismatches. Scrape the errno so
    # the finding names the real cause instead of a bare "data mismatch".
    if rec.get("gb_s") is not None and rec.get("verified") is False:
        rec["reg_error"] = _parse_rdma_errno(out)
    rec["label"] = label
    rec["loc"] = rec.get("loc") or loc
    rec["dev"] = rec.get("dev") or dev
    rec["target"] = rec.get("target") or f"rank{target_rank}"
    return rec


def _is_pinned_rdma(label: str) -> bool:
    return label == "rdma" or label.startswith("rdma-gpu")


def _finding(r: dict, host: str) -> Finding:
    label = r["label"]
    msg = f"{r['target']} -> {host} {label}"
    if r["gb_s"] is not None:
        if _is_pinned_rdma(label):
            env = f"MC_GID_INDEX={_ref_gid()}"
        else:
            env = "MC_FORCE_TCP=1" if label == "tcp" else "none"
        verified = r.get("verified")
        detail = {
            "GB/s": r["gb_s"],
            "moved_GiB": r["gib"],
            "loc": r.get("loc"),
            "gpu": r.get("gpu"),
            "env": env,
            "verified": verified,
        }
        if r.get("dev"):
            detail["dev"] = r["dev"]
        if verified is False:
            reason = "data mismatch after transfer"
            if r.get("reg_error"):
                reason += (
                    f"; likely cause: {r['reg_error']} -- register_memory reported "
                    "success but libibverbs could not pin the buffer, so its tail "
                    "never transferred"
                )
            detail["reason"] = reason
            return Finding("fail", msg, detail)
        return Finding("info", msg, detail)
    # rdma-default failing is the expected demonstration (warn, not fail): it shows
    # the operator must set MC_GID_INDEX; the pinned rdma / rdma-gpu{g} / tcp failing is real.
    if label == "rdma-default":
        # No MC_GID_INDEX set (the "forgot the knob" case); Mooncake auto-picks a
        # link-local GID that can't route cross-node. The fix is shown by the
        # `rdma` row above (env=MC_GID_INDEX=...), so just state the cause here.
        return Finding("warn", msg, {"reason": "auto-selected GID is link-local, not routable"})
    fail_detail = {
        "loc": r.get("loc"),
        "gpu": r.get("gpu"),
        "reason": r.get("reason") or "unreachable",
    }
    if r.get("dev"):
        fail_detail["dev"] = r["dev"]
    return Finding("fail", msg, fail_detail)


def _worker() -> None:
    spec = json.loads(sys.argv[1])
    if spec["role"] == "target":
        _target(
            spec["sig"],
            spec["hostname"],
            spec["host"],
            spec["protocol"],
            spec["loc"],
            spec["gpu"],
            spec.get("dev", ""),
        )
    else:
        rec = _initiator(
            spec["sig"], spec["protocol"], spec["loc"], spec["gpu"], spec.get("dev", "")
        )
        tmp = os.path.join(spec["sig"], "result.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        os.replace(tmp, os.path.join(spec["sig"], "result.json"))


if __name__ == "__main__":
    _worker()
