# Bench 0 — hipFile sanity

Implementation of the "Bench 0 — hipFile sanity on the testbed" gate.
This is the must-pass-first plumbing check before Bench 1 (microbench),
Bench 2 (kvd-shape synthetic) and Bench 3 (real engine workload).

## What Bench 0 answers

1. Does the hipfile python binding import + open a driver?
2. Does a single `hipFileRead` deliver byte-for-byte the same bytes as
   POSIX `pread` + H2D copy?
3. Is the kernel P2PDMA path live, or is hipFile silently bouncing
   through CPU? (Mooncake-on-ionic failure shape.)
4. Sanity sweep at 4 block sizes × 2 concurrencies — Bench 1 owns the
   full 4 KB..16 MB × 1..128 sweep.

## Build the image first (one-time, 5-8 min)

```bash
bash deploy/docker/scripts/build_hipfile.sh
```

This installs `libhipfile.so` + the `ais-check` binary into `/opt/rocm`
inside the lmsysorg engine image. See `deploy/README.md` for what the
script does and how to opt out of the runtime probe at image-build time.

## Pre-seed a test file

Bench 0 will create the file itself on first run, but if you want a
specific path you can pre-seed it:

```bash
mkdir -p /tmp/hipfile-test-dir
# (or pick an NFS / Vast mount as the path)
```

The script writes a deterministic `i % 256` pattern at the path you give
to `--path` (default 64 MiB). It re-uses an existing file if the size +
first 4 KiB match the expected pattern.

## Run on the testbed

```bash
python -m bench.kvcache.hipfile.bench0_sanity \
    --path /tmp/hipfile-test \
    --path /mnt/<nfs-mount>/hipfile-test \
    --require-p2pdma
```

Repeat `--path` for every target filesystem you want to validate.
`--require-p2pdma` upgrades a "Kernel P2PDMA support: False" result from
`ais-check` to a hard fail (matches `HIPFILE_REQUIRE_P2PDMA=1` in
`deploy/docker/scripts/build_hipfile.sh`).

## Expected output shape

ASCII summary to stdout, JSON to
`bench/kvcache/hipfile/results/bench0_{host}_{date}.json`:

```text
Path: /tmp/hipfile-test  (ext4)
  P2PDMA live: True
  size            equal   c=1 hF/POSIX GB/s   c=8 hF/POSIX GB/s
  -----------------------------------------------------------
  64 KiB          True       0.50 /   0.30      1.20 /   0.80
  1 MiB           True       3.10 /   2.20      6.40 /   4.10
  4 MiB           True       6.80 /   3.50     11.50 /   5.20
  16 MiB          True       9.20 /   3.90     12.80 /   5.40

wrote bench/kvcache/hipfile/results/bench0_<host>_20260601-180000.json
```

The JSON schema:

```json
{
  "host": "<host>",
  "kernel": "6.8.0-111-generic",
  "rocm_version": "7.2.0",
  "libhipfile_path": "/opt/rocm/lib/libhipfile.so",
  "paths": [
    {
      "path": "/tmp/hipfile-test",
      "fstype": "ext2/ext3",
      "p2pdma_live": true,
      "ais_check_output": "ok",
      "sizes": {
        "65536": {
          "equal_bytes": true,
          "concurrencies": {
            "1": {"hipfile_gbs": 0.5, "posix_h2d_gbs": 0.3},
            "8": {"hipfile_gbs": 1.2, "posix_h2d_gbs": 0.8}
          }
        }
      }
    }
  ]
}
```

## Pass / fail criteria

- **Hard fail** (exit non-zero):
  - Any `equal_bytes == false` — hipFile returned corrupt bytes.
  - `--require-p2pdma` set AND `Kernel P2PDMA support: True` not in
    `ais-check` output — silent compat-mode fallback, same failure
    shape as Mooncake-on-ionic.
- **Pass** (exit 0):
  - Every path has `equal_bytes: true` at every size.
  - `ais-check` reports `Kernel P2PDMA support: True` (or
    `--require-p2pdma` was not set and you accept compat mode).

If Bench 0 passes, Bench 1 is the next step (`make bench-hipfile-l3
BENCH=1` once that harness lands). Do **not** start any of the Phase 1+
integration work in `infera/kvd/` before Bench 1 + 2 also pass.

## What this bench does NOT do

- It does not write to the file. Read-only sanity only.
- It does not exercise the kvd code path. The shim is consumed
  directly; kvd integration is Phase 1+.
- It does not run the 4 KB..16 MB × 1..128 throughput sweep. That is
  Bench 1.
- It does not touch any LMCache code path. By scope decision, hipFile
  lands in `infera-kvd` directly, no LMCache intermediate.
