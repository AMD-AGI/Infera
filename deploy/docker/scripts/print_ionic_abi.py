#!/usr/bin/env python3
"""Print the kernel-uverbs ABI range the installed ionic provider declares.

Why this exists: a wrong libionic fails *silently*. libibverbs prints one
warning per device to stderr, ibv_get_device_list then returns 0 HCAs, and
mooncake quietly serves KV over TCP -- every pod stays Ready, every request
succeeds, and the only symptom is that PD is slow. So the build records the
declared range in the build log, and refuses to produce an image whose
provider cannot be inspected at all.

The range lives in the provider's `struct verbs_device_ops`:

    static const struct verbs_device_ops ionic_dev_ops = {
        .name              = "ionic",      <- const char *
        .match_min_abi_ver = N,            <- uint32
        .match_max_abi_ver = N,            <- uint32
        ...

so: find the address of the "ionic" string in .rodata, find the 8-byte
pointer to it in the file, and read the two uint32 that follow.

Measured on 2026-08-28, all four versions currently in repo.radeon.com:

    54.0-149.g3304be71   abi 4..4     (pool line 1.117.1-a-63)
    54.0-187-1           abi 1..1     (pool line 1.117.5-a-77)
    54.0-192-1           abi 1..1     (pool line 1.125.0-a-187)
    54.0-197-1           abi 1..1     (pool line 1.117.5-a-147)

Note the ordering: newer is NOT higher-ABI. Pick by the host, never by
version number. See the Dockerfile block for how.
"""

import glob
import re
import struct
import subprocess
import sys

LIBDIR = "/usr/lib/x86_64-linux-gnu"


def readelf(*args):
    return subprocess.run(
        ["readelf", *args], capture_output=True, text=True, errors="replace"
    ).stdout


def declared_abi(path):
    # Address of the bare "ionic" string in .rodata.
    off = None
    for line in readelf("-p", ".rodata", "-W", path).splitlines():
        m = re.match(r"\s*\[\s*([0-9a-f]+)\]\s+ionic$", line)
        if m:
            off = int(m.group(1), 16)
    if off is None:
        return None
    m = re.search(r"\.rodata\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)", readelf("-S", "-W", path))
    if not m:
        return None
    va = int(m.group(1), 16) + off

    data = open(path, "rb").read()
    needle = struct.pack("<Q", va)
    for i in range(0, len(data) - 16):
        if data[i : i + 8] == needle:
            lo, hi = struct.unpack_from("<II", data, i + 8)
            # The other hit is a string-table entry; the ops struct is the one
            # whose two trailing words look like a sane ABI range.
            if 0 < lo <= hi <= 64:
                return lo, hi
    return None


def main():
    sos = sorted(glob.glob(f"{LIBDIR}/libionic.so.1.*"))
    if not sos:
        print("ERROR: no libionic provider installed in " + LIBDIR, file=sys.stderr)
        return 1
    rc = 0
    for so in sos:
        abi = declared_abi(so)
        if abi is None:
            print(f"ERROR: cannot read ABI range from {so}", file=sys.stderr)
            rc = 1
            continue
        print(f"LIBIONIC_ABI {so.rsplit('/', 1)[-1]} = {abi[0]}..{abi[1]}")
    print(
        "Check against the target node: "
        "cat /sys/class/infiniband_verbs/uverbs<N>/abi_version"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
