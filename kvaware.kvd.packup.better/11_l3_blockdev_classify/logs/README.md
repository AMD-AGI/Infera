# Raw logs — NONE for this experiment, and none are needed. Read this first.

**There are no `.log` files in this directory.**

This experiment never started an engine. It has no legs, no router, no model
load, no GPU. It bind-mounts a host directory into a container and runs a
classifier against it. There is nothing that produces an engine log, so there is
nothing to copy — this is not a gap like the missing kvd daemon logs elsewhere
in this work, it is simply the wrong shape of artifact for the experiment.

Fabricating stand-in logs would be worse than having none.

## Where this experiment's evidence lives

| Claim | File | Re-runnable? |
|---|---|---|
| `findmnt` prints a bind mount's subpath in brackets; `lsblk` rejects it (rc=32) | `results/mvp_bind_mount.txt` §1 | **yes** — and §1 checks your own `lsblk` live |
| the pre-fix parser returns the bracketed string; the post-fix one strips it | `results/mvp_bind_mount.txt` §2 | **yes**, ~1 s |
| the consequence: an NVMe-backed bind mount is classified `buffered` | `results/mvp_bind_mount.txt` §3 | **yes** |
| the installed module has (or lacks) the fix | `results/mvp_bind_mount.txt` §4 | yes, with `infera` importable |
| A/B on the node: `[(none)]` → `[md0 (?, ssd)]`, WARN gone | `results/step5_nvme_l3.txt`, `results/patch0002_note.md` | on a node, `MODE=node` |
| 47 tests pass with the fix, 2 of them fail without it | `scripts/test_storage_classify.py` | **yes**, ~0.1 s |
| the storage is SATA-behind-md0, and the NVMe is unmounted and someone else's | `results/storage_reality.txt` | survey only, `MODE=node` |

**`results/mvp_bind_mount.txt` is a fresh re-run, not a transcript excerpt.** It
was regenerated on 2026-07-31 by `python3 scripts/mvp_bind_mount.py` and exits
non-zero if any check stops behaving as recorded. That matters: the root cause
here is a pure-Python string-parsing bug with no cluster, no GPU and no
container in the path, so the evidence chain does not depend on any preserved
log at all.

## Recipes that re-derive the claims

**The whole root cause, ~1 second, anywhere:**

```bash
python3 scripts/mvp_bind_mount.py         # or: bash scripts/run.sh   (MODE=desk)
```

**The regression tests, and the proof they are load-bearing:**

```bash
python3 -m pytest scripts/test_storage_classify.py -q      # 47 passed

# ...and that the two new cases actually catch the bug — revert the hunk first:
python3 - <<'PY'
p = "infera/kvd/storage_classify.py"          # your checkout
s = open(p).read()
needle = '    bracket = source.find("[")\n    if bracket > 0:\n        source = source[:bracket]\n'
open(p, "w").write(s.replace(needle, ""))
PY
python3 -m pytest scripts/test_storage_classify.py -q
#   2 failed, 45 passed
#   FAILED ... test_bind_mount_source_strips_subpath
#   FAILED ... test_bind_mount_classify_reports_clean_source
```

A test that passes both before and after a fix proves nothing. These two were
verified to fail pre-fix.

**On a node — the survey and the A/B:**

```bash
MODE=node bash scripts/run.sh
# writes results/storage_survey.observed.txt and results/abtest.observed.txt
```

Or by hand inside a container that has the bind mount:

```bash
findmnt -no SOURCE,FSTYPE -T /kvd-long          # /dev/md0[/mnt/nvme-raid/kvd-long] ext4
lsblk -no NAME,TRAN,ROTA /dev/md0               # md0          0   <- blank TRAN
python3 -m infera.kvd classify /kvd-long        # the full verdict
bash /abtest.sh                                 # both arms, same path, same moment
```

## What is not recoverable

- **The original session's terminal output** for the node-side A/B. What
  survives is quoted verbatim in `results/step5_nvme_l3.txt` and
  `results/patch0002_note.md` — the two `mount = / devices = / rationale:`
  blocks and the WARN line. The surrounding transcript is gone.
- **The `mknod /dev/md0 b 9 0` step** was done by hand at the time and is not
  in any script here; `scripts/run.sh` instead passes `--device` through, which
  is the correct deployment answer rather than the ad-hoc one that was used.
- **Any measurement against real NVMe.** Never taken. See
  `results/storage_reality.txt` — the drives are unmounted and one holds another
  team's data, so this is a resource constraint rather than a missing file.
