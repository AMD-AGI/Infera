# Patches

Three fixes. **The first two are load-bearing** — without them the experiment does not
run to completion, so they are part of the reproduction, not incidental cleanup. The
third is operational hygiene that will bite anyone who runs a longer workload.

Each is a `diff -u` against the predecessor kit
(`glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/scripts/`), so applying them
to that kit yields this kit's `scripts/`. The fixed scripts are also shipped whole in
`scripts/` — you do not need to apply anything if you use those directly.

**To apply** (`-p1`, from a directory holding the three predecessor scripts):

    K=<repo>/glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/scripts
    mkdir /tmp/replay && cp $K/{glm52_leg.sh,start_leg.sh,reset_merged.sh} /tmp/replay/
    cd /tmp/replay && for p in <kit>/patches/000*.patch; do patch -p1 < $p; done

**Verified:** doing exactly that reproduces this kit's `scripts/glm52_leg.sh`,
`scripts/start_leg.sh` and `scripts/reset_node.sh` **byte-for-byte** (`diff` clean).
Note 0003 renames `reset_merged.sh` → `reset_node.sh`.

---

## 0001 — `glm52_leg.sh`: discover `MC_GID_INDEX` instead of hardcoding 1

**What.** Replaces `export MC_GID_INDEX=1` with a discovery function that picks the
first GID index on the first active ionic device that is neither empty nor link-local
(`fe80::`). Falls back to the hardcoded `1` if discovery finds nothing.

**Why.** The index is **node-dependent**, and the kit hardcoded one node's value. Both
nodes expose two RoCE v2 GIDs per ionic port, at different indices:

| node | idx0 | idx1 | idx2 |
|---|---|---|---|
| chi2879 | `fe80::` link-local | **`fd93::` routable** | empty |
| chi2867 | `fe80::` link-local | **empty** | **`fd93::` routable** |

Verified identical across all 8 NICs on each node, and independently via `show_gids`
(`n_gids_found=2`, indices 0 and 2 on chi2867).

**Context — the symptom it cured.** With the hardcoded `1`, the **decode** leg on
chi2867 died during init on all 8 DP ranks, while the prefill leg on chi2879 had zero
errors. That asymmetry is the whole diagnosis:

    rdma_context.cpp:1132  GID is NULL, please check your GID index by specifying MC_GID_INDEX
    rdma_context.cpp:200   Failed to open device ionic_4 on port with GID 1
    rdma_transport.cpp:932 Disable device ionic_4
    RuntimeError: Mooncake Transfer Engine initialization failed.
    RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)

**How applied.** Edit in `scripts/glm52_leg.sh`, marked `BENCH DELTA`. Verify before
booting a leg:

    bash -c 'source <(sed -n "/^IB_DEVICES=/,/^MC_GID_INDEX=/p" scripts/glm52_leg.sh); echo $MC_GID_INDEX'
    # chi2879 -> 1 ; chi2867 -> 2

**Do not "fix" this by using idx0.** The link-local GID is not routable across the
fabric and crashes MoRI at `ionic.cpp:414`.

**Upstream status.** Not filed. This is a deployment-script fix in our own kit, not an
sglang or mooncake defect — mooncake correctly reports the empty GID and refuses.

---

## 0002 — `start_leg.sh`: frozen sweep config + prefill `mem-fraction-static` 0.88 → 0.80

**What.** Two changes in one file:
1. Freezes the bench config: `CTX=262144`, `ISL=8192`/`TP=8` (→ chunk 65536, 8192 per
   rank at dp8), `CUDA_GRAPH_BS=128`, `MAX_RUNNING=2048`, and re-copies the leg script
   into the container on every launch.
2. **`GMU_P` = 0.80 for prefill** (decode stays 0.85).

**Why (the GMU part).** At ISL 155,000 × conc 32 the prefill leg aborted:

    rocdevice.cpp:3582  HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 1203306 MB
    Fatal Python error: Aborted

**It is not KV exhaustion.** The scheduler lines immediately before the abort read
`token usage: 0.01–0.05` — the KV pool was essentially empty. What was large was
`#pending-token: 2,486,426` at `#queue-req: 16`. With DP-attention at dp8 each rank
holds its own 8192-token chunk activations, and a 155K prompt is 19 chunks; under the
prefill-delayer's batching the transient activation peak exceeds what
`1 − mem_fraction_static` leaves outside the static reservation.

**The direction is the counter-intuitive part.** Prefill activation OOM is fixed by
*lowering* `mem-fraction-static`, which is the **opposite** of the decode-side retract
fix (raise it, for more KV room). Diagnose by phase:

| phase | symptom | fix |
|---|---|---|
| decode | retract / `get_cpu_copy NotImplementedError` | **raise** mem-fraction-static |
| prefill | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` / `Aborted` | **lower** it |

Consistent with the prior first-hand result on this stack (DSv4 dp8 prefill OOMed at
0.90, clean at 0.85); this prompt is far longer, hence one step further to 0.80.

**Context.** Cost: KV pool 3,260,672 → 2,829,952 tokens/rank (−13 %) for activation
headroom 32.5 → 54.6 GB (+68 %). After the change, p90/c32 ran **32/32 clean** and c64
and c128 followed with **zero** `HSA_STATUS_ERROR`.

Decode was deliberately left at 0.85 — it did not crash (its last healthy log line is
*after* the prefill abort, so it died in the cascade), and changing both would have made
the fix a two-variable change.

**Caveat this creates.** The p50 rounds ran at 0.88 and the p90 rounds at 0.80, so the
two pairs are **not** a controlled comparison of each other. Recorded in `notes.md`.

**Upstream status.** Not filed — a tuning value, not a defect.

---

## 0003 — `reset_node.sh`: kvd `--long-bytes` 512G → 64G

**What.** Lowers kvd's L3 spill budget to match `--max-bytes`, and clears
`/tmp/kvd-long` on reset.

**Why.** `--long-path /tmp/kvd-long` is **inside the container**, so kvd's L3 tier
spills into the container's writable layer on the node's **root disk** — not onto
`/mnt/vast`. The kit's 512 GB budget is larger than the free space on `/`.

**Context.** After round 8, `bench_run`'s writable layer had reached **263 GB**, `/` hit
**100 %** of 838 GB, and every `docker exec` began failing with:

    OCI runtime exec failed: write /tmp/runc-processNNN: no space left on device

which reads as a docker fault rather than as our own cache. Reclaiming needed care on a
shared node: 243 GB in our `bench_run` tier plus 20 GB in our exited `merged_run` — both
verified ours via `docker inspect` before removing anything. Nothing belonging to
another user was touched.

**How applied.** `KVD_LONG_GB` in `scripts/reset_node.sh`, default 64. A longer workload
(Case A: 89 % cache-hit target over 67 minutes) writes far more than this sweep did, so
512G would refill the disk mid-run.

**Better long-term fix, not taken here.** Point `--long-path` at `/mnt/vast` instead of
container-local `/tmp`. Not done in this kit because it changes the storage medium under
the L3 tier (NFS vs local NVMe) and would make these numbers non-comparable to every
prior validated run. Flagged in `notes.md` as a recommendation.

**Upstream status.** Worth raising: kvd's default `--long-bytes` should be sanity-checked
against actual free space on the target filesystem at startup, and the self-check already
detects `source='overlay'` (it logs `storage_classify: lsblk returned no devices for
source='overlay'`) — so it has the information needed to warn.

---

## Addendum (folded in after Phase 2, 2026-08-01)

This kit was sealed before a `pkill` incident that happened during Phase-2 setup.
Two files have since been refreshed from the workspace; **the Phase-1 results are
unaffected** (the incident occurred after the last round).

**`pkill -f infera.kvd` also kills the ENGINE.** `-f` matches a *regex* and `.` is a
wildcard, so `infera.kvd` matches the engine's own `--infera-kvd-socket` argument.
The engine dies silently — no HSA error, no traceback, just a clean exit and the
router quietly at `active_workers: 1`.

| file | change |
|---|---|
| `scripts/reset_node.sh` | kill-list patterns now escaped: `'-m infera\.kvd '` etc. |
| `scripts/restart_kvd.sh` | **new** — restarts only the daemon, anchored on `-m infera\.kvd ` (escaped dot, trailing space), and asserts the engine count is unchanged afterwards |

**Use `restart_kvd.sh`, never a bare pkill.**
