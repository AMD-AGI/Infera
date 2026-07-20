###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Cross-node RoCE bandwidth matrix via ib_write_bw.

Coordinated over the shared dump dir (no ssh / torch.distributed): every node
publishes its RDMA info, then each ordered node pair (server, client) is tested
in turn -- both directions, since links can be asymmetric. Per pair we test the
full NxN NIC grid in N parallel rounds (round r: client nic (a+r)%N connects to
server nic a), so each round uses every NIC at most once. Each client node
returns its measured links; rank 0 assembles them into per-direction matrices.

Must run where ib_write_bw sees the ionic devices (host libionic injected into
the container -- see run_preflight_slurm.sh). OOB (QP exchange) goes over the
routable mgmt IP; the RDMA path uses the device's RoCE v2 GID.
"""

from __future__ import annotations

import errno as _errno
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time

from ..finding import Finding
from ..util import have, read_text, run

_IB = "/sys/class/infiniband"
_BASE_PORT = 18600
_PORT_STRIDE = 64  # per-round port offset, so a killed server's port isn't reused at once
_TEST_TIMEOUT = 15.0  # one ib_write_bw client call
_EXCHANGE_TIMEOUT = 120.0  # wait for all nodes to publish their info
_READY_TIMEOUT = 30.0  # client waits for the server to come up
_LISTEN_TIMEOUT = 20.0  # server waits until its OOB ports are listening
_SETTLE = 3.0  # grace after LISTEN: the OOB port opens ~0.5s in, but ib_write_bw
# needs a few more seconds of RDMA setup before it can actually complete a
# transfer -- signaling too early makes the client connect and fail (empirically
# >=3s is reliable, <=1.5s is not)
_BARRIER_TIMEOUT = 1800.0  # safety net between node pairs
# A link whose measured bandwidth is below this fraction of the node's median is
# flagged as a degraded outlier. Conservative (cross-node RoCE has some natural
# spread + contention), mirroring the relative-outlier check the GPU p2p probe does.
_SLOW_FRAC = 0.5


def _mgmt_ip() -> str | None:
    """Routable IPv4 for out-of-band QP exchange, preferring the 10.x subnet.

    Must not depend on the ``ip`` (iproute2) binary -- some engine images ship
    without it, and returning None here silently breaks the mooncake/mori/netperf
    rendezvous (the OOB handshake address becomes ``None:port``, so every transport
    -- RDMA and TCP alike -- fails). So try ``ip`` first, then ``hostname -I``,
    then resolve our own hostname via ``socket``."""
    ips: list[str] = []
    rc, out = run(["ip", "-o", "-4", "addr", "show"])
    if rc == 0:
        for line in out.splitlines():
            m = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/", line)
            if m and not m.group(1).startswith("127."):
                ips.append(m.group(1))
    if not ips:  # no iproute2: hostname -I prints all configured IPs
        rc, out = run(["hostname", "-I"])
        if rc == 0:
            ips = [
                t
                for t in out.split()
                if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", t) and not t.startswith("127.")
            ]
    if not ips:  # last resort: resolve our own hostname
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if not ip.startswith("127."):
                    ips.append(ip)
        except OSError:
            pass
    return next((ip for ip in ips if ip.startswith("10.")), ips[0] if ips else None)


def _gid_index(dev: str) -> int:
    """Index of the device's RoCE v2 GID, preferring a routable one.

    Routable means a global IPv6 GID or an IPv4-mapped GID (``::ffff:a.b.c.d``,
    i.e. RoCE v2 over an IPv4 NIC -- the index production pins via MC_GID_INDEX).
    NOTE: an IPv4-mapped GID is literally ``0000:0000:0000:0000:0000:ffff:<ipv4>``,
    so it MUST NOT be excluded by a bare ``startswith("0000:")`` -- that filter is
    only meant to skip the *unset* all-zero slot. A link-local ``fe80::`` GID is a
    last-resort fallback (it can't route cross-node)."""
    base = f"{_IB}/{dev}/ports/1"
    fallback = None
    ipv4_mapped = None
    for i in range(16):
        gid = (read_text(f"{base}/gids/{i}") or "").strip()
        typ = (read_text(f"{base}/gid_attrs/types/{i}") or "").strip()
        if not gid or "RoCE v2" not in typ:
            continue
        if set(gid.replace(":", "")) <= {"0"}:  # unset / all-zero slot
            continue
        low = gid.lower()
        if low.startswith("fe80"):
            fallback = i if fallback is None else fallback
        elif low.startswith("0000:0000:0000:0000:0000:ffff:"):  # IPv4-mapped (prod pins)
            # Match the full ::ffff: prefix, not a bare ":ffff:" substring -- a global
            # IPv6 GID may legitimately contain an ffff hextet mid-address and must
            # still be treated as global (returned immediately) below.
            ipv4_mapped = i if ipv4_mapped is None else ipv4_mapped
        else:  # global IPv6 RoCE v2
            return i
    if ipv4_mapped is not None:
        return ipv4_mapped
    return fallback if fallback is not None else 1


def _port(r: int, a: int) -> int:
    return _BASE_PORT + r * _PORT_STRIDE + a


def _nics() -> list[str]:
    try:
        return sorted(x for x in os.listdir(_IB) if x)
    except OSError:
        return []


def detect_local() -> dict | None:
    nics = _nics()
    if not have("ib_write_bw") or not nics:
        return None
    return {"mgmt_ip": _mgmt_ip(), "nics": nics, "gid": _gid_index(nics[0])}


def _parse_bw(out: str) -> float | None:
    """Average GB/s from ib_write_bw output, or None if no results row. The tool
    reports gigabits (--report_gbits); divide by 8 for GB/s to match the KV sections."""
    for line in out.splitlines():
        cols = line.split()
        # results row: #bytes #iters BW_peak BW_avg MsgRate -- take BW_avg (Gb/s).
        if len(cols) >= 5 and all(_is_num(c) for c in cols[:5]):
            return float(cols[3]) / 8.0
    return None


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _client_round(
    server_ip: str, nics: list[str], gid: int, r: int, n: int
) -> dict[int, float | None]:
    """Round r: run all N clients in parallel (client nic (a+r)%N -> server nic a,
    each a distinct NIC/port, so no contention). Return {server_nic a: GB/s|None}."""
    procs = {}
    for a in range(n):
        b = (a + r) % n
        cmd = [
            "ib_write_bw",
            "-d",
            nics[b],
            "-x",
            str(gid),
            "-p",
            str(_port(r, a)),
            "--report_gbits",
            server_ip,
        ]
        procs[a] = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    out: dict[int, float | None] = {}
    for a, p in procs.items():
        try:
            stdout, _ = p.communicate(timeout=_TEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout = ""
        out[a] = _parse_bw(stdout)
    return out


def _server(dev: str, gid: int, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        ["ib_write_bw", "-d", dev, "-x", str(gid), "-p", str(port), "--report_gbits"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _touch(path: str) -> None:
    with open(path, "w"):
        pass


def _wait_file(path: str, timeout: float) -> bool:
    # Poll via os.listdir (readdir), NOT os.path.exists (stat): on NFS a stat of a
    # not-yet-existing path is negatively cached for ~30s, so a peer's freshly
    # created signal file stays invisible; readdir revalidates the directory.
    d, name = os.path.split(path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if name in os.listdir(d):
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def _listen_ports() -> set[int]:
    """Local TCP ports in LISTEN state (from /proc/net/tcp{,6}). Used to confirm
    the servers are up before signaling the client -- probing by connecting would
    steal ib_write_bw's single accept."""
    ports: set[int] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as fh:
                rows = fh.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) > 3 and cols[3] == "0A":  # 0A = TCP_LISTEN
                ports.add(int(cols[1].rsplit(":", 1)[1], 16))
    return ports


def _wait_listen(ports: set[int], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ports <= _listen_ports():
            return True
        time.sleep(0.2)
    return False


def _wait_count(dir_: str, count: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sum(1 for e in os.listdir(dir_) if not e.endswith(".tmp")) >= count:
            return True
        time.sleep(0.5)
    return False


def _serve_pair(sig: str, nics: list[str], gid: int, n: int) -> None:
    for r in range(n):
        ports = [_port(r, a) for a in range(n)]
        procs = [_server(nics[a], gid, ports[a]) for a in range(n)]
        # Signal ready only once the servers are actually listening -- ib_write_bw
        # needs ~1-2s to bring up its RDMA context + OOB socket, and a client that
        # connects before that fails, taking the whole round down with it.
        _wait_listen(set(ports), _LISTEN_TIMEOUT)
        time.sleep(_SETTLE)
        _touch(os.path.join(sig, f"r{r}.ready"))
        _wait_file(os.path.join(sig, f"r{r}.done"), n * _TEST_TIMEOUT + 30)
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def _client_pair(
    sig: str, server_host: str, server_ip: str, client_host: str, nics: list[str], gid: int, n: int
) -> list[dict]:
    recs: list[dict] = []
    for r in range(n):
        up = _wait_file(os.path.join(sig, f"r{r}.ready"), _READY_TIMEOUT)
        bws = _client_round(server_ip, nics, gid, r, n) if up and server_ip else {}
        for a in range(n):
            recs.append(
                {
                    "server": server_host,
                    "s_nic": a,
                    "client": client_host,
                    "c_nic": (a + r) % n,
                    "gb_s": bws.get(a),
                }
            )
        _touch(os.path.join(sig, f"r{r}.done"))
    return recs


def _barrier(bdir: str, rank: int, world: int) -> None:
    os.makedirs(bdir, exist_ok=True)
    _touch(os.path.join(bdir, str(rank)))
    _wait_count(bdir, world, _BARRIER_TIMEOUT)


def _agree_min(exch_dir: str, rank: int, world: int, local: int, timeout: float) -> int:
    """Cross-rank minimum of a local integer, exchanged through the shared dir.

    The KV tests build their per-variant loop from a GPU/NIC count; if ranks
    disagreed (a node with torch/CUDA unavailable, a dead GPU, or a different NIC
    count) they would iterate different label sets and the per-pair file barriers
    would deadlock -- on exactly the degraded node this tool exists to catch. So
    every rank publishes its local count, waits for all `world` (or the timeout),
    and takes the MIN, guaranteeing an identical loop on every rank. Falls back to
    `local` only if nothing is readable (all peers vanished)."""
    os.makedirs(exch_dir, exist_ok=True)
    dst = os.path.join(exch_dir, str(rank))
    with open(dst + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(str(int(local)))
    os.replace(dst + ".tmp", dst)
    _wait_count(exch_dir, world, timeout)
    vals: list[int] = []
    for name in os.listdir(exch_dir):
        if name.endswith(".tmp"):
            continue
        try:
            with open(os.path.join(exch_dir, name), encoding="utf-8") as fh:
                vals.append(int(fh.read().strip()))
        except (OSError, ValueError):
            pass
    return min(vals) if vals else int(local)


def _parse_rdma_errno(text: str) -> str | None:
    """Scrape a memory-registration errno out of a KV worker's captured output.

    The mori/mooncake registration path fails inside libibverbs, which logs the
    errno two different ways:
      - ``RegisterRdmaMemoryRegion failed! errno:14`` / ``ibv_reg_mr ... errno 14``
        (mori) -- the ``errno`` keyword form, and then (for mori) segfaults.
      - ``Failed to register memory 0x...: Cannot allocate memory [12]``
        (mooncake) -- the errno is a bracketed ``[12]`` with NO ``errno`` keyword,
        and the Python ``register_memory`` still returns 0, so nothing else
        signals the failure (the buffer's tail then silently doesn't transfer).
    The bare exit status only tells us *that* it died; the errno tells us *why* --
    14 (EFAULT/"Bad address") is the signature of an unsupported bare-VRAM
    registration, 12 (ENOMEM) a host-DRAM pin that couldn't be satisfied, 22
    (EINVAL) a bad argument. We require a registration keyword on the same line so
    a benign errno/bracket elsewhere in the log isn't misattributed, and decode
    the number via the stdlib errno/strerror tables.

    Returns e.g. ``"register errno 14 (EFAULT: Bad address)"`` or None.
    """
    if not text:
        return None
    hit = None
    for line in text.splitlines():
        if re.search(r"regist|reg_mr|memory region|allocate memory", line, re.I) and re.search(
            r"errno|\[\d+\]", line, re.I
        ):
            hit = line  # keep the last matching line (closest to the failure)
    if hit is None:
        return None
    m = re.search(r"errno[:=]?\s*(\d+)", hit) or re.search(r"\[(\d+)\]", hit)
    if m is None:
        return None
    n = int(m.group(1))
    name = _errno.errorcode.get(n, f"errno{n}")
    try:
        desc = os.strerror(n)
    except (ValueError, OverflowError):
        desc = ""
    return f"register errno {n} ({name}: {desc})" if desc else f"register errno {n} ({name})"


def _exit_reason(rc: int | None, text: str = "") -> str:
    """Human reason for a KV worker subprocess that produced no result JSON.

    ``rc is None`` -> we timed out waiting on it; ``rc < 0`` -> it was killed by a
    signal (``-11`` = SIGSEGV, the ionic bare-VRAM register crash); ``rc > 0`` ->
    it exited non-zero. Any register errno scraped from its output is appended so
    the finding carries the actionable detail (e.g. EFAULT) rather than just
    ``worker_crashed (SIGSEGV)``.
    """
    detail = _parse_rdma_errno(text)
    if rc is None:
        base = "worker_timeout"
    elif rc < 0:
        try:
            base = f"worker_crashed ({signal.Signals(-rc).name})"
        except (ValueError, AttributeError):
            base = f"worker_crashed (signal {-rc})"
    elif rc != 0:
        base = f"worker_exit_{rc}"
    else:
        base = "no_result"
    return f"{base}: {detail}" if detail else base


def run_matrix(dump_path: str, rank: int, world: int, host: str) -> list[Finding]:
    if world < 2:
        return [Finding("info", "netperf skipped (single node)", {})]
    local = detect_local()
    if local is None:
        return [Finding("warn", "netperf skipped (no ib_write_bw or RDMA device)", {})]

    root = os.path.join(dump_path, "netperf")
    nodes_dir = os.path.join(root, "nodes")
    os.makedirs(nodes_dir, exist_ok=True)
    # Atomic publish (tmp + rename) so peers never read a half-written file on NFS.
    dst = os.path.join(nodes_dir, f"{rank}.json")
    with open(dst + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"rank": rank, "host": host, **local}, fh)
    os.replace(dst + ".tmp", dst)
    if not _wait_count(nodes_dir, world, _EXCHANGE_TIMEOUT):
        return [Finding("warn", "netperf: not all nodes exchanged RDMA info", {})]

    table: dict[int, dict] = {}
    for name in os.listdir(nodes_dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(nodes_dir, name), encoding="utf-8") as fh:
            info = json.load(fh)
        table[info["rank"]] = info

    # Both directions: each ordered pair (server s, client c) is its own step,
    # and only one step runs at a time (barrier), so no node is server and client
    # at once.
    recs: list[dict] = []
    for s in range(world):
        for c in range(world):
            if s == c:
                continue
            sig = os.path.join(root, "sig", f"{s}_{c}")
            os.makedirs(sig, exist_ok=True)
            ss, cc = table[s], table[c]
            n = min(len(ss["nics"]), len(cc["nics"]))
            if rank == s:
                _serve_pair(sig, ss["nics"], local["gid"], n)
            elif rank == c:
                print(
                    f"[preflight] netperf {ss['host']} -> {cc['host']} ({n}x{n} nics)...",
                    file=sys.stderr,
                )
                recs += _client_pair(
                    sig, ss["host"], ss["mgmt_ip"], cc["host"], cc["nics"], local["gid"], n
                )
            _barrier(os.path.join(root, "bar", f"{s}_{c}"), rank, world)

    if not recs:
        return [Finding("info", "netperf: no client role for this node", {})]
    failed = sum(1 for r in recs if r["gb_s"] is None)
    level = "fail" if failed else "info"
    findings = [
        Finding(
            level,
            f"netperf: {len(recs) - failed}/{len(recs)} links ok (as client)",
            {"tests": recs},
        )
    ]
    findings += _slow_link_findings(recs)
    return findings


def _slow_link_findings(recs: list[dict]) -> list[Finding]:
    """Warn on links whose bandwidth is a gross outlier below the node's median.

    Without this netperf only FAILs a *fully unreachable* (None) link, so a link
    degraded to a fraction of line rate -- a bad cable/optic, a mis-tuned NIC, a
    wrong queue config -- is reported green. That contradicts the tool's stated
    'compare within a node, outliers warn' philosophy (which the GPU p2p probe
    already implements). A warn (not fail): cross-node RoCE has some natural
    spread, so a lone slow link is a signal to investigate, not a hard stop."""
    vals = [r["gb_s"] for r in recs if r["gb_s"] is not None]
    if len(vals) < 4:  # too few samples for a meaningful median
        return []
    med = statistics.median(vals)
    if med <= 0:
        return []
    out: list[Finding] = []
    for r in recs:
        v = r["gb_s"]
        if v is not None and v < med * _SLOW_FRAC:
            out.append(
                Finding(
                    "warn",
                    f"netperf link slow: {r['server']} nic{r['s_nic']} -> "
                    f"{r['client']} nic{r['c_nic']}",
                    {"gb_s": round(v, 1), "node_median": round(med, 1)},
                )
            )
    return out
