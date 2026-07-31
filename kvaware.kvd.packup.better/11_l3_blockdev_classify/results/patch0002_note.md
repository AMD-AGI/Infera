# Patch 0002 — `storage_classify` mis-parses a bind mount, silently forcing buffered I/O

**File:** `infera/kvd/storage_classify.py` (`_findmnt`)
**Patch:** `0002-storage_classify-bind-mount-subpath.patch`
**Tests:** `test_storage_classify.py` → install to `tests/unit/kvd/` (2 new cases)
**Status:** fixed, tested, A/B-verified on hardware, **not committed** (working
tree of `yihou.dev.glm5.2.mxfp4.experiment`).

## Context — the symptom it cured

Step 5 of this experiment bind-mounted a host directory into the container so
kvd's L3 would sit on a real block device instead of the container overlay:

```
docker run ... -v /mnt/nvme-raid/kvd-long:/kvd-long ...
```

The classifier still reported no device at all:

```
$ python3 -m infera.kvd classify /kvd-long
storage_classify: lsblk returned no devices for source='/dev/md0[/mnt/nvme-raid/kvd-long]'
L3 io_mode: BUFFERED (auto)
  mount    = /dev/md0[/mnt/nvme-raid/kvd-long] (ext4)
  devices  = [(none)]
  rationale: unknown device, conservative buffered
```

## What was wrong

For a bind mount `findmnt` prints the bind subpath in brackets after the device:

```
$ findmnt -no SOURCE,FSTYPE -T /kvd-long
/dev/md0[/mnt/nvme-raid/kvd-long] ext4
```

`_findmnt()` returned that whole string as the device name, and the caller
passed it straight to `lsblk`:

```
$ lsblk -no NAME,TRAN,ROTA '/dev/md0[/mnt/nvme-raid/kvd-long]'
lsblk: /dev/md0[/mnt/nvme-raid/kvd-long]: not a block device      (rc=32)
```

No devices → `rationale: unknown device, conservative buffered`. So **any**
bind-mounted L3 gets buffered I/O regardless of the hardware underneath, and the
only clue is a WARN that reads like a missing-tool problem.

Worth noting the module docstring already *claims* this case is handled:

> "The chain handles md-raid, LVM, dm-crypt, and bind mounts transparently"

It handles the others; bind mounts it did not.

## The fix

Strip the bracketed subpath in `_findmnt` before returning the source. Three
lines; no signature change.

## A/B verification on the real node

Same path, same moment, only this hunk differs (`scripts/abtest.sh`):

```
--- WITHOUT fix (pre-patch behaviour) ---
  mount    = /dev/md0[/mnt/nvme-raid/kvd-long] (ext4)
  devices  = [(none)]
  rationale: unknown device, conservative buffered
  WARN: lsblk returned no devices for source='/dev/md0[/mnt/nvme-raid/kvd-long]'

--- WITH fix ---
  mount    = /dev/md0 (ext4)
  devices  = [md0 (?, ssd)]
  rationale: unknown transport '' (md0), conservative buffered
```

The device now resolves and the WARN is gone.

## What the fix does NOT do

The verdict is still `buffered`, for a **different and legitimate** reason:
`md0`'s `TRAN` column is empty (an md-raid node has no transport of its own —
resolving it means recursing to the member disks, `sda2`/`sdb2`). That is a
separate limitation, not addressed here, and on this box the honest answer is
buffered anyway: md0 is a raid1 of two **SATA** SSDs, and the classifier
deliberately picks buffered for SATA to get the cold-read readahead win.

So patch 0002 fixes *device resolution*, not the io-mode verdict on this
particular hardware.

## Second, independent blocker (environmental, not fixed)

Even with patch 0002, a stock container sees nothing, because `/dev/md0` is not
exposed to it:

```
(container) $ ls -l /dev/md0     -> No such file or directory
(container) $ lsblk /dev/md0     -> not a block device (rc=32)
(host)      $ lsblk -no NAME,TRAN,ROTA /dev/md0  -> md0          0
```

For the A/B above the node was created by hand (`mknod /dev/md0 b 9 0`). A real
deployment that wants accurate L3 classification must pass the device through
(`--device=/dev/md0`, or `--privileged`) in addition to bind-mounting the
directory. Bind-mounting the *path* alone is not enough — that is the trap this
whole step walked into.

## Tests

Two new cases in `tests/unit/kvd/test_storage_classify.py`:

- `test_bind_mount_source_strips_subpath` — end-to-end through `pick_io_mode`,
  with a fake `_run` that models the **real** lsblk contract (errors on a
  bracketed argument). Without that fidelity the bug slips through: an
  accept-anything fake makes the pre-fix code look correct.
- `test_bind_mount_classify_reports_clean_source` — the reported
  `mount_source` must not leak the subpath either; operators read that field.

```
$ python3 -m pytest tests/unit/kvd/test_storage_classify.py -q
47 passed
```

Both fail before the patch, pass after.
