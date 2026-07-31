# 11 — kvd's L3 on a real block device, and BUG #2 (`storage_classify` bind mount)

**Ran:** 2026-07-30 14:00–14:30 · **Cost:** ~1 s desk check, or ~2 min on one node (no GPU)
**Verdict:** ◐ **PARTIAL** — a real infera bug found, fixed, tested and A/B-verified; the
original "NVMe L3" question remains **unanswered** for reasons outside the code

## What this experiment answers

**Can kvd's L3 tier be moved off the container overlay onto a real block device,
so that `storage_classify` picks `O_DIRECT` instead of falling back to
buffered?**

The starting state: `--long-path /tmp/kvd-long` lives on the container's
overlayfs, and the classifier reports

```
mount    = overlay (overlay)
devices  = [(none)]
rationale: unknown device, conservative buffered
```

which is expected — overlayfs has no backing block device for `lsblk` to
report. The obvious fix is to bind-mount a host directory. That is where this
experiment starts, and where it immediately hits a bug.

## Result

| | Outcome |
|---|---|
| Bug found in `infera/kvd/storage_classify.py:_findmnt` | ✅ real, fixed, **patch 0002** |
| A/B verified on hardware | ✅ `devices [(none)]` → `[md0 (?, ssd)]`, WARN gone |
| Regression tests | ✅ 2 new, **47 pass** in that file; both new ones fail pre-fix |
| io-mode verdict on this hardware | still `buffered` — and that is **correct** |
| O_DIRECT on genuine NVMe | ❌ **untested** — resource constraint, not a finding |

## The findings that matter

### 1. BUG #2 — any bind-mounted L3 silently got buffered I/O

Rebuilt both containers with `-v /mnt/nvme-raid/kvd-long:/kvd-long`. The
classifier still reported no device at all:

```
storage_classify: lsblk returned no devices for source='/dev/md0[/mnt/nvme-raid/kvd-long]'
L3 io_mode: BUFFERED (auto)
  mount    = /dev/md0[/mnt/nvme-raid/kvd-long] (ext4)
  devices  = [(none)]
  rationale: unknown device, conservative buffered
```

For a bind mount, `findmnt` prints the bind subpath in brackets after the
device:

```
$ findmnt -no SOURCE,FSTYPE -T /kvd-long
/dev/md0[/mnt/nvme-raid/kvd-long] ext4
```

`_findmnt()` returned that **whole string** as the device name and the caller
handed it straight to `lsblk`:

```
$ lsblk -no NAME,TRAN,ROTA '/dev/md0[/mnt/nvme-raid/kvd-long]'
lsblk: /dev/md0[/mnt/nvme-raid/kvd-long]: not a block device      (rc=32)
```

No devices → `rationale: unknown device, conservative buffered`. So **any**
bind-mounted L3 got buffered I/O regardless of the hardware underneath, and the
only clue was a WARN that reads like a missing-tool problem.

Worth noting the module's own docstring already claims this case is handled:

> "The chain handles md-raid, LVM, dm-crypt, and bind mounts transparently"

It handles the others. Bind mounts it did not.

**The fix** strips the bracketed subpath in `_findmnt` — three lines, no
signature change. It uses `bracket > 0` rather than `>= 0` so a source that
*starts* with `[` is left alone instead of becoming the empty string.

**A/B on the node**, same path, same moment, only this hunk differing
(`scripts/abtest.sh`):

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

**Tests.** Two new cases in `tests/unit/kvd/test_storage_classify.py`, 47 in the
file. The first models the **real** `lsblk` contract — it errors on a bracketed
argument. That fidelity is the point: an accept-anything fake makes the pre-fix
code look correct and the bug walks straight through. Verified both fail before
the patch and pass after:

```
$ python3 -m pytest tests/unit/kvd/test_storage_classify.py -q
47 passed                        # with the fix
2 failed, 45 passed              # with the hunk reverted
```

### 2. The verdict is still `buffered`, and that is CORRECT

Patch 0002 fixes **device resolution**, not the io-mode verdict on this
hardware. Two independent and legitimate reasons keep it at buffered:

1. **`md0` has no `TRAN` of its own.** An md-raid node's transport means
   recursing to the members (`sda2`/`sdb2`); `lsblk -no NAME,TRAN,ROTA /dev/md0`
   shows a blank transport column. The classifier reports
   `unknown transport '' (md0)` and stays conservative. Not addressed by 0002,
   and deliberately so.
2. **Even recursed, the members are SATA.** `storage_classify` picks buffered
   for SATA on purpose — slow random-read latency means kernel readahead is the
   only thing keeping throughput up, and `O_DIRECT` deliberately bypasses it.

Before the patch you could not distinguish "conservative because SATA" from
"conservative because blind". Now you can. Same verdict, honest reason.

### 3. The deployment trap: bind-mounting the PATH does not expose the DEVICE

```
(container) $ ls -l /dev/md0     -> No such file or directory
(container) $ lsblk /dev/md0     -> not a block device (rc=32)
(host)      $ lsblk -no NAME,TRAN,ROTA /dev/md0  -> md0          0
```

A stock container sees the files and not the block device behind them, so
`lsblk` fails no matter how good the parser is. For the reference A/B the node
was created by hand (`mknod /dev/md0 b 9 0`).

A deployment that wants accurate L3 classification needs `--device=/dev/md0`
(or `--privileged`) **in addition to** the `-v` mount.

> **This is the real trap of this experiment**, and it is a deployment-recipe
> issue rather than an infera defect. Everyone will bind-mount the path and
> assume that is enough, because it looks complete. `scripts/run.sh` passes the
> device through automatically and reports honestly if it could not.

### 4. The storage is not what its name says

```
/mnt/nvme-raid  ->  /dev/md0 = raid1 of sda2+sdb2  ==  SATA SSDs, NOT NVMe
/dev/nvme0n1..7 ->  8x 7 TB, ext4, ALL UNMOUNTED
                    nvme0n1 already holds another team's kvd-long/kvd-short, 120 GB
```

md0 was chosen deliberately: a real block device that belongs to nobody,
sufficient to test the **classification** question, which was the actual bug.
Mounting an unmounted drive on a shared cluster, or writing into another team's
kvd store, is not ours to do. Details in `results/storage_reality.txt`.

## How to reproduce

**Desk check — no cluster, no GPU, no container, ~1 second. This is the actual
root cause:**

```bash
bash scripts/run.sh                 # MODE=desk is the default
```

It shows what `findmnt`/`lsblk` do (and live-checks your own `lsblk`'s reaction
to a bracketed argument), runs the pre-fix and post-fix parsers side by side,
demonstrates the consequence end-to-end through `pick_io_mode`, probes the
installed module if `infera` is importable, and runs the 47 tests — staging the
packed post-fix copy into a throwaway package if `infera` is not on the path, so
it works anywhere. Output goes to `results/mvp_bind_mount.observed.txt`; the
committed reference `results/mvp_bind_mount.txt` is a *fresh re-run*, not a
transcript excerpt, and the script exits non-zero if any check stops behaving as
recorded.

**On a node, ~2 min, no GPU:**

```bash
MODE=node bash scripts/run.sh
MODE=node HOST_L3=/some/other/dir bash scripts/run.sh
```

Surveys the storage (so you can see *why* the verdict is what it is), starts a
container with both the bind mount and `--device`, warns loudly if the device is
still invisible, runs the A/B, and prints the full classifier output.

**In-repo:**

```bash
git apply scripts/0002-storage_classify-bind-mount-subpath.patch
cp scripts/test_storage_classify.py tests/unit/kvd/
python3 -m pytest tests/unit/kvd/test_storage_classify.py -q     # -> 47 passed
```

`scripts/storage_classify_fixed.py` is the whole post-fix module, for dropping
into a container that predates the fix without a rebuild.

## Gotchas specific to this experiment

- **The `-v` mount is only half of it.** Pass `--device=<dev>` too, or the
  classifier is blind whatever the parser does. This is the single most likely
  thing to get wrong when reproducing.
- **`buffered` is not a failure.** On SATA it is the intended answer. The bug
  was never the verdict; it was the blindness. Do not "fix" the SATA rule.
- **Test fakes must model the real tool's contract.** An `lsblk` fake that
  accepts a bracketed argument makes the pre-fix code pass. The included test
  returns rc=32 for a bracketed target, exactly as the real binary does.
- **`bracket > 0`, not `>= 0`.** A source starting with `[` must be left alone
  rather than truncated to nothing. The MVP has a case for it.
- **`/mnt/nvme-raid` is not NVMe.** Believe `lsblk`, not the mount point name.
- **Shared-cluster hygiene.** Do not mount the unmounted NVMe drives and do not
  write into `/dev/nvme0n1`'s existing `kvd-long`. `scripts/run.sh` mounts
  nothing and formats nothing — it only classifies a path that is already
  mounted.
- **The `io_mode` can be forced** (`--io-mode direct`) but that changes the flag,
  not the substrate. Forcing `direct` onto SATA is very likely a pessimisation.

## What this does NOT prove

1. **Nothing about NVMe O_DIRECT performance.** The headline item — "put L3 on
   real NVMe and measure" — was **not done**. The drives are unmounted and one
   holds another team's data. This is a resource constraint, and no claim in
   either direction is supported.
2. **It does not show the fix changes any verdict on this hardware.** Before and
   after, the answer is `buffered`. What changed is that the device resolves and
   the misleading WARN is gone. On genuinely NVMe-backed bind-mounted storage
   the fix *would* flip the verdict to `O_DIRECT` — that is demonstrated only in
   the MVP's simulation (`results/mvp_bind_mount.txt` §3, and the unit test),
   never on real NVMe hardware.
3. **It does not fix the md-raid transport gap.** `md0` still reports
   `unknown transport ''`. Recursing an md node to its members to resolve
   transport is a separate limitation, untouched here.
4. **No performance measurement of any kind.** No throughput, no latency, no
   comparison of buffered vs direct on anything. Only classification.
5. **The L3 read path under load is untested.** A separate experiment showed the
   write path works (573 MB resident) and that the daemon recovers its long
   region across its own restart. Neither is exercised here.
6. **One host, one filesystem, one mount type.** ext4-on-md0-via-bind. LVM,
   dm-crypt, ZFS, and NFS-backed L3 were not tried, though the module claims to
   handle several of them.
7. **The patch is uncommitted** as of 2026-07-30, sitting in the working tree of
   branch `yihou.dev.glm5.2.mxfp4.experiment`. Whether it has landed since is
   **unknown** from this packup's evidence — check the branch.

## Environment (verbatim in every packup so this folder stands alone)

**Cluster access.** Jump host `root@149.28.124.225`, then `ssh <node>`. Key-based,
no password appears in any script here.

```bash
J(){ ssh -o StrictHostKeyChecking=no root@149.28.124.225 \
       "ssh -o StrictHostKeyChecking=no $1 '$2'"; }
```

**Nodes** (8× AMD Instinct MI355X / gfx950 each, 128 threads, 3023 GB RAM):

| Host | Data-plane IP | amdgpu | Kernel |
|---|---|---|---|
| chi2879 | 10.2.122.10 | 6.16.13 | 6.8.0-124-generic |
| chi2867 | 10.2.122.44 | 6.16.13 | 6.8.0-107-generic |

**Fabric:** ionic RoCE v2, 8 rails/node (`ionic_0`…`ionic_7`), all PORT_ACTIVE.
Module `26.03.3.001`, NIC firmware `1.117.5-a-77`, routable GID at **index 1**
(hence `MC_GID_INDEX=1`). chi2879→chi2867 RTT 0.069 ms.

**Image:** `infera/engine-sglang:pd-unified`
sha256 `f8ec2d627392435b7cf4c97e47b93a3b36588bec43864a1758b7c0dc9405bd18`
(sglang 0.5.15.post1, torch 2.9.1+rocm7.2.0, ROCm 7.2.0). A **local build**, not
on a registry — the Infera PR #19 rebuild that makes mooncake cross-node RDMA
work. Distribute with `docker save ... | ssh <dst> docker load`.

**infera repo:** branch `yihou.dev.glm5.2.mxfp4.experiment`, commit `362192e7`.

**Models (absolute paths on the shared VAST NFS mount `/mnt/vast`):**
- GLM-5.2-MXFP4 — `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (408 GB, 282 shards,
  `GlmMoeDsaForCausalLM`, 78 layers, 256 experts)
- Qwen3-1.7B — `/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

**Kit staging dir:** `/mnt/vast/c_huggingface/glm52_kvexp` — must be on the
shared FS so both nodes' containers see the same copy.

**Host `libionic` injection is mandatory.** Without it RDMA silently degrades to
TCP. Verify **inside** the container: `ibv_devinfo | grep -c PORT_ACTIVE` → `8`.

**Secrets:** cluster SSH only (key-based). No registry login needed (local
image). etcd runs unauthenticated on the data-plane IP.

## Traps that bite every experiment here

**1. `docker exec -d $CTR bash -lc '...'` does not persist.** The detached
login-shell form exits and takes the child with it. Symptom: no process, no log
file, no error. Bit us twice (router, kvd daemon). **Always** stage a script
file and run `docker exec -d $CTR bash /the_script.sh`, or use
`docker exec -d $CTR env VAR=... bash /script`.

**2. Nested ssh quoting silently mangles variables.** In
`ssh jump "ssh node '...$f...'"` the OUTER shell expands `$f`. Stage a script
file instead of fighting the quoting.

**3. Cold start is 6-12 min and looks like a hang.** GLM-5.2 loads 408 GB.
Watch the log growing (`wc -l`); don't kill it.

**4. Three kvaware ports collide when two workers share a host** — unrelated
code paths, fixing one does nothing for the others:

| Port | Default | Failure |
|---|---|---|
| sglang `--kv-events-config` block | from `free_tcp_port_block` | deterministic same base → `ZMQError: Address already in use` (this is **patch 0001**) |
| `--kv-events-bind` | `tcp://0.0.0.0:5557` | identical on every leg; 2nd fails to bind |
| `--kv-snapshot-port` | `8801` | **the nastiest** — leg prints `ready to roll`, *then* dies during etcd registration. Looks healthy; worker never appears in `/v1/workers`. |

**5. `--mem-fraction-static` is TP-dependent.** `0.85` suits TP8 (51 GB/GPU of
weights). At **TP4** weights double to 102 GB/GPU and 0.85 OOMs — use `0.70`.

**6. `--hicache-ratio` sizes the host pool off the KV pool.** Default 2.0 on a
small model tried to allocate **355 GB per DP rank**. Use `--hicache-size <GB>`
(absolute) instead.

**7. Never probe a PD leg directly.** `curl` to a leg's own port just hangs — a
PD leg only serves through the pair. Use the router, and use a differential run
(flip one thing, hold the rest) to isolate.

**8. Shared cluster hygiene.** Don't prune images, don't mount other people's
drives, don't `docker rm` a container you can't prove is yours
(`docker inspect` → Binds/Env/Created).
