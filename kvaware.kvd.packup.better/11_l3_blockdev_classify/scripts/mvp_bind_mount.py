#!/usr/bin/env python3
"""MVP for BUG #2 — storage_classify mis-parses a bind mount.

No cluster, no GPU, no container, ~1 second. This is the whole root cause, and
it should be the first thing anyone runs.

Three sections:

  1. What findmnt actually prints for a bind mount, and what lsblk does with it.
     Run against the REAL tools if they are present, otherwise printed as the
     recorded observation.
  2. The pre-fix parser vs the post-fix parser, on the same input.
  3. The consequence: pick_io_mode() on an NVMe-backed bind mount, with a fake
     _run that models the REAL lsblk contract (errors on a bracketed argument).
     Without that fidelity the bug slips through — an accept-anything fake makes
     the pre-fix code look correct.

Usage:
    python3 mvp_bind_mount.py
    PYTHONPATH=<infera repo> python3 mvp_bind_mount.py   # also exercises §4
"""

from __future__ import annotations

import shutil
import subprocess
import sys

FAIL = 0


def hdr(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# ----------------------------------------------------------------------
# The parser, before and after. Lifted from infera/kvd/storage_classify.py
# _findmnt() — only the last two statements differ.
# ----------------------------------------------------------------------
def parse_old(line: str):
    parts = line.split()
    if len(parts) < 2:
        return None
    fstype = parts[-1]
    source = " ".join(parts[:-1])
    return source, fstype


def parse_new(line: str):
    parts = line.split()
    if len(parts) < 2:
        return None
    fstype = parts[-1]
    source = " ".join(parts[:-1])
    # THE FIX: findmnt appends a bind mount's subpath in brackets.
    bracket = source.find("[")
    if bracket > 0:
        source = source[:bracket]
    return source, fstype


# ----------------------------------------------------------------------
# 1. what the real tools do
# ----------------------------------------------------------------------
hdr("1. findmnt prints a bind mount's subpath in brackets; lsblk rejects it")

print("Recorded on the node (chi2867, container with -v /mnt/nvme-raid/kvd-long:/kvd-long):")
print()
print("  $ findmnt -no SOURCE,FSTYPE -T /kvd-long")
print("  /dev/md0[/mnt/nvme-raid/kvd-long] ext4")
print()
print("  $ lsblk -no NAME,TRAN,ROTA '/dev/md0[/mnt/nvme-raid/kvd-long]'")
print("  lsblk: /dev/md0[/mnt/nvme-raid/kvd-long]: not a block device      (rc=32)")

if shutil.which("lsblk"):
    print()
    print("Live check on THIS machine — lsblk's reaction to a bracketed argument:")
    r = subprocess.run(
        ["lsblk", "-no", "NAME,TRAN,ROTA", "/dev/sda1[/some/where]"],
        capture_output=True, text=True,
    )
    print(f"  rc={r.returncode}  stderr={r.stderr.strip()!r}")
    if r.returncode == 0:
        print("  NOTE: this lsblk accepted it — unexpected; the recorded rc is 32.")
    else:
        print("  -> non-zero, as recorded. lsblk will not take a bracketed source.")
else:
    print("\n(lsblk not installed here — skipping the live check.)")

# ----------------------------------------------------------------------
# 2. the parser
# ----------------------------------------------------------------------
hdr("2. pre-fix vs post-fix parser, same input")

CASES = [
    ("/dev/md0[/mnt/nvme-raid/kvd-long] ext4", "/dev/md0", "bind mount (the bug)"),
    ("/dev/nvme0n1p1[/kvd-long] ext4", "/dev/nvme0n1p1", "bind mount on NVMe"),
    ("/dev/sda1 ext4", "/dev/sda1", "ordinary mount (must not change)"),
    ("overlay overlay", "overlay", "container overlayfs (must not change)"),
    ("nfsserver:/export nfs4", "nfsserver:/export", "NFS (must not change)"),
    ("[/weird] ext4", "[/weird]", "leading bracket -> find()==0, NOT stripped"),
]

print(f"  {'input':<42} {'OLD':<26} {'NEW':<18} verdict")
print(f"  {'-' * 42} {'-' * 26} {'-' * 18} -------")
for line, want, why in CASES:
    o = parse_old(line)[0]
    n = parse_new(line)[0]
    ok = n == want
    if not ok:
        FAIL += 1
    print(f"  {line:<42} {o:<26} {n:<18} {'ok' if ok else 'MISMATCH'}   ({why})")

print()
print("  The last row is deliberate: the fix uses `bracket > 0`, not `>= 0`, so a")
print("  source that *starts* with '[' is left alone rather than becoming the empty")
print("  string. Off-by-one here would turn a weird case into a crash.")

# ----------------------------------------------------------------------
# 3. the consequence
# ----------------------------------------------------------------------
hdr("3. the consequence: an NVMe-backed bind mount is classified as buffered")


def fake_run_factory(findmnt_line):
    """Models the REAL lsblk contract: it ERRORS on a bracketed argument.
    An accept-anything fake would make the pre-fix code look correct — that
    fidelity is the whole reason this reproduces the bug."""

    def run(cmd, timeout=2.0):
        class R:
            returncode = 0
            stdout = ""
        r = R()
        if cmd[0] == "findmnt":
            r.stdout = findmnt_line
        elif cmd[0] == "lsblk":
            if "[" in cmd[-1]:
                r.returncode = 32
                r.stdout = ""
            else:
                r.stdout = "nvme0n1 nvme 0\n"
        return r

    return run


def simulate(parser, findmnt_line):
    """Mimic classify_storage's chain: findmnt -> parse -> lsblk."""
    src, _fs = parser(findmnt_line)
    r = fake_run_factory(findmnt_line)(["lsblk", "-no", "NAME,TRAN,ROTA", src])
    devices = [l.split()[0] for l in r.stdout.splitlines() if l.split()]
    if not devices:
        return src, [], "unknown device, conservative buffered", False
    return src, devices, "nvme-ssd -> O_DIRECT", True


LINE = "/dev/nvme0n1p1[/kvd-long] ext4\n"
for label, parser in (("OLD (pre-fix)", parse_old), ("NEW (post-fix)", parse_new)):
    src, devs, why, direct = simulate(parser, LINE)
    print(f"  {label}")
    print(f"    mount    = {src}")
    print(f"    devices  = {devs if devs else '[(none)]'}")
    print(f"    rationale: {why}")
    print(f"    io_mode  = {'DIRECT' if direct else 'BUFFERED'}")
    print()

if simulate(parse_old, LINE)[3]:
    print("  !! pre-fix path did NOT reproduce the bug — check the fake's fidelity")
    FAIL += 1
elif not simulate(parse_new, LINE)[3]:
    print("  !! post-fix path did NOT resolve the device")
    FAIL += 1
else:
    print("  -> BUG REPRODUCED and FIX CONFIRMED. Any bind-mounted L3 silently got")
    print("     buffered I/O regardless of the hardware underneath, and the only clue")
    print("     was a WARN that reads like a missing-tool problem.")

# ----------------------------------------------------------------------
# 4. against the real module, if importable
# ----------------------------------------------------------------------
hdr("4. against the real infera.kvd.storage_classify (needs it importable)")
try:
    from pathlib import Path

    from infera.kvd import storage_classify

    orig = storage_classify._run
    try:
        storage_classify._run = fake_run_factory(LINE)
        direct, rationale = storage_classify.pick_io_mode(Path("/kvd-long"))
        info = storage_classify.classify_storage(Path("/kvd-long"))
    finally:
        storage_classify._run = orig
    print(f"  pick_io_mode  -> o_direct={direct}  rationale={rationale!r}")
    print(f"  mount_source  -> {info.mount_source!r}")
    print(f"  devices       -> {[d.dev for d in info.devices]}")
    if direct and info.mount_source == "/dev/nvme0n1p1":
        print("\n  -> the installed module HAS the fix.")
    else:
        print("\n  -> the installed module does NOT have the fix (this is the bug).")
except ImportError as e:
    print(f"  SKIPPED — infera not importable ({e}).")
    print("  Retry with: PYTHONPATH=<infera repo> python3 mvp_bind_mount.py")

hdr("SUMMARY")
if FAIL:
    print(f"  {FAIL} check(s) did not behave as recorded — investigate before trusting this.")
    sys.exit(1)
print("  All checks behaved as recorded.")
print()
print("  What the fix does NOT settle, and both are real:")
print("   (a) the verdict on the reference hardware is still 'buffered', and that is")
print("       CORRECT: md0 is a raid1 of two SATA SSDs and the classifier prefers")
print("       buffered for SATA (readahead win). md0 also has no TRAN of its own.")
print("   (b) a stock container cannot see /dev/md0 at all. Bind-mounting the PATH")
print("       does not expose the DEVICE — you also need --device=/dev/md0.")
sys.exit(0)
