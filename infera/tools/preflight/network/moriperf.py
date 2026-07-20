###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Mori KV-transfer check: real cross-node buffer transfer over Mori's IOEngine
(the production PD transport on this MI355X+ionic fleet, RDMA-only).

Coordination reuses netperf's shared-dir rendezvous: each ordered node pair is
tested both directions. The target registers a buffer and publishes its
EngineDesc + MemoryDesc; the initiator registers the remote engine, then loops
batch_write (one transfer of several in-flight segments, enough to saturate the
link on the VRAM path) for a fixed window and reports the average bandwidth.

Each endpoint runs in its OWN SUBPROCESS (like the Mooncake check). This is not
cosmetic: on this image mori's GPU-VRAM registration can *hard-crash* -- a native
segfault inside RegisterRdmaMemoryRegion (errno 14 EFAULT), which a Python
try/except cannot catch. In-process, that killed the whole rank and deadlocked
every peer at the barrier. Isolated in a child, the crash is contained: the
parent sees the non-zero exit, records it as the finding's reason, releases the
peer, and still reaches the barrier.

Multi-card affinity, matching production exactly: MORI_RDMA_DEVICES is set to ALL
ionic NICs and MORI_IB_GID_INDEX is pinned, just as sglang launches it. Then, in
addition to a host-DRAM baseline, the test sweeps every local GPU and registers
VRAM with (ptr, size, gpu_id, MemoryLocationType.GPU) -- the exact call sglang's
mori conn makes -- so mori's topology auto-match binds GPU g to its affine NIC.
The per-GPU count is AGREED across ranks (min) so every rank runs an identical
variant loop; a lone degraded node cannot desync the barriers.

INFERA_PREFLIGHT_KV_GPUS caps the per-GPU sweep. Unset (the default) sweeps ALL
local GPUs -- the most production-faithful, per-GPU/per-NIC coverage, but also the
slowest: the node-pair grid is multiplied by (1 + n_gpus) variants. Set it to
e.g. 2 for a fast smoke test -- the ionic bare-VRAM failure is a driver/stack-wide
property that is identical across GPUs, so 1-2 GPUs already reproduce it; reserve
the full sweep for when you need to confirm every GPU's affinity path individually.

Each direction also verifies data correctness: the initiator writes a
per-(gpu,segment) pattern into the target's buffer and the target reads it back
and compares, so an offset/rkey/mis-routed-NIC bug that still "succeeds" is
caught as a mismatch rather than reported as a green transfer. Runs in-container
where Mori + the injected host libionic live.
"""

from __future__ import annotations

import base64
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

# GPU (VRAM) geometry mirrors the KV cache sglang registers with
# MemoryLocationType.GPU: a 1 GiB region moved in 32 MiB segments.
_GPU_CHUNK = 32 << 20  # 32 MiB per segment
_GPU_NCHUNK = 32  # segments per transfer (in-flight depth)
_GPU_SIZE = _GPU_CHUNK * _GPU_NCHUNK  # 1 GiB VRAM region
# CPU (host DRAM) geometry mirrors PRODUCTION. The only host-DRAM memory sglang
# registers is the aux/metadata buffers -- mori's conn calls register_memory(...,
# MemoryLocationType.CPU) on them (and RDMAs them when SGLANG_MORI_SEND_AUX_RDMA
# is set). Those are a `max_running_requests*2` x per-item region: each item is
# 64 B..512 B, up to ~16 KiB (EAGLE hidden_states), so the whole footprint is
# tens of MiB and each transfer is <=16 KiB -- NOT 1 GiB. Registering a 1 GiB
# host MR instead exceeds ulimit -l, so ibv_reg_mr silently ENOMEMs and the tail
# never lands -- a synthetic "data mismatch" production never hits. So register a
# 64 MiB region (the worst-case aux footprint, still under a normal memlock
# limit) and move it in 16 KiB segments (the largest real aux item).
_CPU_CHUNK = 16 << 10  # 16 KiB per segment (largest production aux item)
_CPU_NCHUNK = 32  # segments per transfer (in-flight depth)
_CPU_SIZE = 64 << 20  # 64 MiB registered host-DRAM region (max aux footprint)
_MIN_SECONDS = 3.0
_WAIT_MS = 20000  # per-transfer wait_all timeout
_TARGET_TIMEOUT = 120.0
_WROTE_TIMEOUT = 60.0  # target waits for the initiator's final pattern write
_VERIFY_TIMEOUT = 60.0  # initiator waits for the target's readback verdict
_DONE_TIMEOUT = 180.0
_STEP_TIMEOUT = 240.0  # hard cap on one endpoint subprocess (contains a hang/crash)


def _rdma_devices() -> list[str]:
    """All ionic NICs, as production passes to MORI_RDMA_DEVICES; mori then
    topology-matches each gpu_id to its affine NIC."""
    return _nics()


def _kv_gpus() -> int:
    """Number of local GPUs to sweep (one VRAM buffer per GPU, each routed over
    its affine NIC), capped by the NIC count and INFERA_PREFLIGHT_KV_GPUS. The
    cross-rank min of this is what actually drives the loop (see run)."""
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
    return min(n, len(_rdma_devices()))


def _specs(ngpu: int) -> list[tuple[str, str, int]]:
    """(label, loc, gpu_id): a DRAM baseline plus one VRAM buffer per GPU."""
    specs: list[tuple[str, str, int]] = [("cpu", "cpu", -1)]
    specs += [(f"gpu{g}", "gpu", g) for g in range(ngpu)]
    return specs


def _geom(loc: str) -> tuple[int, int, int]:
    """(registered size, per-segment chunk, segments per transfer) for this
    buffer class: GPU mirrors the KV cache (1 GiB VRAM in 32 MiB segments), CPU
    mirrors the aux/metadata buffers (64 MiB host DRAM in 16 KiB segments)."""
    if loc == "gpu":
        return _GPU_SIZE, _GPU_CHUNK, _GPU_NCHUNK
    return _CPU_SIZE, _CPU_CHUNK, _CPU_NCHUNK


def _chunk_byte(gpu_id: int, i: int) -> int:
    """Distinct byte per (gpu, segment): encodes the GPU tag so a buffer that
    lands via the wrong NIC/GPU, or a segment at the wrong offset, mismatches."""
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
    _geom(loc). Exposes the raw pointer for register_memory plus helpers to stamp
    / read back the correctness pattern."""

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
            # Sync first so the device->host copy sees the completed remote RDMA
            # write (the initiator's wait_all + cross-node "wrote" file already
            # ordered it before we get here).
            self._torch.cuda.synchronize(self.gpu_id)
            return self._t.cpu().numpy()
        return self._a


def _make_buffer(loc: str, gpu_id: int):
    try:
        return _Buf(loc, gpu_id)
    except Exception:
        return None


def _engine(host_ip: str):
    """A Mori IOEngine with an RDMA backend, or None if Mori is unavailable.
    Sets MORI_RDMA_DEVICES to every ionic NIC and pins MORI_IB_GID_INDEX, exactly
    as production launches sglang, so per-gpu_id topology matching has all NICs to
    choose from and a routable GID."""
    nics = _rdma_devices()
    if nics:
        os.environ["MORI_RDMA_DEVICES"] = ",".join(nics)
        os.environ.setdefault("MORI_IB_GID_INDEX", str(_gid_index(nics[0])))
    try:
        from mori.io import BackendType, IOEngine, IOEngineConfig, PollCqMode, RdmaBackendConfig
    except ImportError:
        return None
    key = f"preflight-{os.getpid()}-{time.monotonic_ns()}"
    eng = IOEngine(key, IOEngineConfig(host=host_ip, port=0))
    eng.create_backend(BackendType.RDMA, RdmaBackendConfig(1, 1, 1, PollCqMode.POLLING, False))
    return eng


def _register(eng, buf: _Buf):
    """Register buf as sglang's mori conn does: (ptr, size, gpu_id, GPU) for VRAM
    -- mori binds it to GPU gpu_id's affine NIC -- or (ptr, size, -1, CPU) for
    DRAM. Returns the MemoryDesc, or None if registration is rejected. NOTE: on
    this ionic image VRAM registration can segfault rather than raise; the
    subprocess isolation in run() is what turns that into a clean finding."""
    from mori.io import MemoryLocationType

    loc = MemoryLocationType.GPU if buf.loc == "gpu" else MemoryLocationType.CPU
    dev = buf.gpu_id if buf.loc == "gpu" else -1
    try:
        return eng.register_memory(buf.ptr, buf.size, dev, loc)
    except Exception:
        return None


def _target(sig: str, host: str, loc: str, gpu_id: int) -> None:
    eng = _engine(_mgmt_ip())
    info = {"ok": False, "host": host, "loc": loc, "gpu": gpu_id}
    buf = None
    if eng is not None:
        buf = _make_buffer(loc, gpu_id)
        if buf is None:
            info["reason"] = "no_gpu"
        else:
            buf.fill(0)  # sentinel; the initiator overwrites this with the pattern
            mem = _register(eng, buf)
            if mem is None:
                info["reason"] = "register_failed"
            else:
                info = {
                    "ok": True,
                    "host": host,
                    "loc": loc,
                    "gpu": gpu_id,
                    "engine": base64.b64encode(bytes(eng.get_engine_desc().pack())).decode(),
                    "mem": base64.b64encode(bytes(mem.pack())).decode(),
                }
    tmp = os.path.join(sig, "target.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    os.replace(tmp, os.path.join(sig, "target.json"))

    # Correctness handshake: wait for the initiator's final pattern write, read
    # the buffer back and publish the verdict, then keep the buffer alive (wait
    # done) until the initiator confirms it is finished.
    verified = None
    if (
        info.get("ok")
        and buf is not None
        and _wait_file(os.path.join(sig, "wrote"), _WROTE_TIMEOUT)
    ):
        verified = _verify(buf.host_bytes(), gpu_id, buf.chunk, buf.nchunk)
    vtmp = os.path.join(sig, "verify.json.tmp")
    with open(vtmp, "w", encoding="utf-8") as fh:
        json.dump({"verified": verified}, fh)
    os.replace(vtmp, os.path.join(sig, "verify.json"))
    _wait_file(os.path.join(sig, "done"), _DONE_TIMEOUT)


def _write(eng, local_mem, remote_mem, chunk: int, nchunk: int) -> bool:
    offs = [i * chunk for i in range(nchunk)]
    sizes = [chunk] * nchunk
    try:
        uid = eng.allocate_transfer_uid()
        statuses = eng.batch_write([local_mem], [offs], [remote_mem], [offs], [sizes], [uid])
    except Exception:
        return False
    # Two mori builds ship two mutually-exclusive completion APIs, and each image
    # exposes only one of them:
    #   * sglang image: IOEngine.wait_all(statuses, timeout_ms=...) -> StatusCode
    #   * vllm image:   per-transfer TransferStatus.Succeeded()/Failed()/InProgress()
    # Feature-detect so the same check runs unchanged on both. Prefer wait_all when
    # present -- that keeps the sglang-validated path byte-for-byte identical; only
    # builds without it (vllm) fall through to polling the TransferStatus objects.
    if hasattr(eng, "wait_all"):
        try:
            from mori.io import StatusCode

            return eng.wait_all(statuses, timeout_ms=_WAIT_MS) == StatusCode.SUCCESS
        except Exception:
            return False
    # No wait_all: poll each TransferStatus to completion within the same timeout
    # budget; the RDMA backend progresses the CQ itself, so a simple poll is enough.
    deadline = time.monotonic() + _WAIT_MS / 1000.0
    for s in statuses:
        while True:
            try:
                if s.Succeeded():
                    break
                if s.Failed():
                    return False
            except Exception:
                return False
            if time.monotonic() > deadline:
                return False
            time.sleep(0.0005)
    return True


def _initiator(sig: str, loc: str, gpu_id: int) -> dict:
    from mori.io import EngineDesc, MemoryDesc

    rec: dict = {
        "gb_s": None,
        "gib": 0.0,
        "target": None,
        "loc": loc,
        "gpu": gpu_id,
        "verified": None,
        "reason": None,
    }
    if _wait_file(os.path.join(sig, "target.json"), _TARGET_TIMEOUT):
        with open(os.path.join(sig, "target.json"), encoding="utf-8") as fh:
            tgt = json.load(fh)
        rec["target"] = tgt.get("host")
        eng = _engine(_mgmt_ip())
        if eng is None:
            rec["reason"] = "engine_unavailable"
        elif not tgt.get("ok"):
            rec["reason"] = tgt.get("reason") or "target_unavailable"
        else:
            eng.register_remote_engine(EngineDesc.unpack(base64.b64decode(tgt["engine"])))
            remote_mem = MemoryDesc.unpack(base64.b64decode(tgt["mem"]))
            buf = _make_buffer(loc, gpu_id)
            if buf is None:
                rec["reason"] = "no_gpu"
            else:
                buf.fill_pattern()  # every write lands the verifiable pattern remotely
                local_mem = _register(eng, buf)
                batch = buf.chunk * buf.nchunk  # bytes moved per batch_write
                if local_mem is None:
                    rec["reason"] = "register_failed"
                elif _write(eng, local_mem, remote_mem, buf.chunk, buf.nchunk):  # warm up
                    moved, t0 = 0, time.monotonic()
                    while time.monotonic() - t0 < _MIN_SECONDS:
                        if not _write(eng, local_mem, remote_mem, buf.chunk, buf.nchunk):
                            moved = 0
                            break
                        moved += batch
                    dt = time.monotonic() - t0
                    if moved > 0 and dt > 0:
                        # The loop's last successful write already left the full
                        # pattern remotely, so signal the target to verify.
                        rec["gb_s"] = round(moved / dt / 1e9, 2)
                        rec["gib"] = round(moved / (1 << 30), 1)
                        _touch(os.path.join(sig, "wrote"))
                        if _wait_file(os.path.join(sig, "verify.json"), _VERIFY_TIMEOUT):
                            with open(os.path.join(sig, "verify.json"), encoding="utf-8") as fh:
                                rec["verified"] = json.load(fh).get("verified")
                    else:
                        rec["reason"] = "transfer_failed"
                else:
                    rec["reason"] = "transfer_failed"
    _touch(os.path.join(sig, "done"))
    return rec


def _spawn(role: str, sig: str, host: str, loc: str, gpu_id: int) -> tuple[int | None, str]:
    """Run one endpoint in an isolated subprocess so a native mori crash/hang is
    contained. Returns (exit code, captured output); rc is None on timeout.

    stderr is merged into stdout and captured (not shown) purely so we can scrape
    the libibverbs register errno (e.g. errno 14/EFAULT) that mori logs right
    before it segfaults -- otherwise that detail dies with the child."""
    spec = {"role": role, "sig": sig, "host": host, "loc": loc, "gpu": gpu_id}
    cmd = [sys.executable, "-m", "infera.tools.preflight.network.moriperf", json.dumps(spec)]
    try:
        cp = subprocess.run(
            cmd,
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


def _ensure_target_json(
    sig: str, host: str, loc: str, gpu_id: int, rc: int | None, out: str = ""
) -> None:
    """If the target subprocess died before publishing target.json (e.g. a VRAM
    register segfault), publish a not-ok stub so the initiator fails fast instead
    of waiting out _TARGET_TIMEOUT. The reason carries any scraped register errno."""
    d, name = os.path.split(os.path.join(sig, "target.json"))
    try:
        if name in os.listdir(d):
            return
    except OSError:
        pass
    info = {"ok": False, "host": host, "loc": loc, "gpu": gpu_id, "reason": _exit_reason(rc, out)}
    tmp = os.path.join(sig, "target.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    os.replace(tmp, os.path.join(sig, "target.json"))


def _load_result(
    sig: str, label: str, loc: str, gpu_id: int, target_rank: int, rc: int | None, out: str = ""
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
        }
    # A clean (non-crashing) register failure exits 0 with reason "register_failed";
    # enrich it with the errno mori logged so the operator sees EFAULT vs ENOMEM.
    if rec.get("gb_s") is None and rec.get("reason"):
        detail = _parse_rdma_errno(out)
        if detail and detail not in rec["reason"]:
            rec["reason"] = f"{rec['reason']}: {detail}"
    # A transfer that "succeeds" (gb_s set) but fails verification is usually a
    # SILENT registration failure (host-DRAM MR -> ENOMEM); the binding still
    # reports success, so the buffer's tail never lands. Scrape the errno so the
    # finding names the real cause instead of a bare "data mismatch".
    if rec.get("gb_s") is not None and rec.get("verified") is False:
        rec["reg_error"] = _parse_rdma_errno(out)
    rec["label"] = label
    rec["loc"] = rec.get("loc") or loc
    rec["gpu"] = rec.get("gpu", gpu_id)
    rec["target"] = rec.get("target") or f"rank{target_rank}"
    return rec


def run(dump_path: str, rank: int, world: int, host: str) -> list[Finding]:
    if world < 2:
        return [Finding("info", "mori skipped (single node)", {})]
    try:
        import mori.io  # noqa: F401
    except ImportError:
        return [Finding("warn", "mori skipped (python bindings not importable)", {})]

    root = os.path.join(dump_path, "moriperf")
    # Agree the per-GPU sweep count across ranks (min) so every rank iterates the
    # SAME spec set -- otherwise the per-pair barriers desync and hang.
    ngpu = _agree_min(os.path.join(root, "kvgpus"), rank, world, _kv_gpus(), _TARGET_TIMEOUT)
    specs = _specs(ngpu)
    recs: list[dict] = []
    for s in range(world):
        for c in range(world):
            if s == c:
                continue
            for label, loc, gpu_id in specs:
                sig = os.path.join(root, f"{s}_{c}_{label}")
                os.makedirs(sig, exist_ok=True)
                if rank == s:
                    rc, out = _spawn("target", sig, host, loc, gpu_id)
                    _ensure_target_json(sig, host, loc, gpu_id, rc, out)
                elif rank == c:
                    rc, out = _spawn("initiator", sig, host, loc, gpu_id)
                    # The child touches these on success; do it again in the parent
                    # so a crashed/timed-out child still releases the target's waits.
                    _touch(os.path.join(sig, "wrote"))
                    _touch(os.path.join(sig, "done"))
                    recs.append(_load_result(sig, label, loc, gpu_id, s, rc, out))
                _barrier(os.path.join(root, "bar", f"{s}_{c}_{label}"), rank, world)

    if not recs:
        return [Finding("info", "mori: no initiator role for this node", {})]
    return [_finding(r, host) for r in recs]


def _finding(r: dict, host: str) -> Finding:
    label = r.get("label", r.get("loc", "cpu"))
    msg = f"{r['target']} -> {host} rdma/{label}"
    if r["gb_s"] is None:
        return Finding(
            "fail",
            msg,
            {"loc": r.get("loc"), "gpu": r.get("gpu"), "reason": r.get("reason") or "unreachable"},
        )
    verified = r.get("verified")
    detail = {
        "GB/s": r["gb_s"],
        "moved_GiB": r["gib"],
        "loc": r.get("loc"),
        "gpu": r.get("gpu"),
        "verified": verified,
    }
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


def _worker() -> None:
    spec = json.loads(sys.argv[1])
    if spec["role"] == "target":
        _target(spec["sig"], spec["host"], spec["loc"], spec["gpu"])
    else:
        rec = _initiator(spec["sig"], spec["loc"], spec["gpu"])
        tmp = os.path.join(spec["sig"], "result.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        os.replace(tmp, os.path.join(spec["sig"], "result.json"))


if __name__ == "__main__":
    _worker()
