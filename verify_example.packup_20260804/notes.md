# Notes — traps, wrong turns, and what this run could not answer

Ordered by how much they would cost someone repeating this. The first three each
**produce no error**.

---

## 1. `preflight_rdma.sh mode` does not run on this image (gap, not fixed)

**What.** The kit's first documented command fails immediately:

```
[preflight] registration-mode probe on chi2835
/opt/venv/bin/python3: No module named infera.tools.preflight.mooncake_mode
EXIT=1
```

**Why.** The module exists in the repo (`infera/tools/preflight/mooncake_mode.py`, 1132
lines, added by `fe01159`) but not in `infera/engine-sglang:merged-e` — the image was
built before it landed:

```bash
docker run --rm --entrypoint bash $IMAGE -c 'ls /opt/infera/infera/tools/preflight/'
# __init__.py __main__.py cli.py finding.py firmware gpu host network report.py
# run_preflight_slurm.sh storage util.py          <- no mooncake_mode.py
```

**How it was worked around.** Bind-mount the repo's module over the installed package
(`REPRODUCE.md` §2). It then runs correctly and reports mode A on both nodes.

**Not fixed, deliberately.** This is the image lagging the repo, not a script bug. The
script is correct; the fix is to rebuild or publish an image that carries the module. Left
as a note so the next person recognises the error in ten seconds instead of debugging a
"missing dependency".

---

## 2. The image lacks GLM52_P1V3 (gap, patched at runtime)

Covered fully in `patches/README.md` §6. The one-line version: `merged-e` has
`_p1v2_trim` (P1V2) but not `_p1v2_clip` (P1V3), and without P1V3 the decode leg crashes
minutes into an agentic workload. Patched inside the container and verified in the
**bytecode**.

The trap adjacent to this one, worth stating separately: **a freshly created container has
no `.pyc` yet**, so `rm .../dsa_indexer*.pyc` after patching reported "No such file". That
is the *safe* case. The dangerous case is patching a container that has been running — the
`.pyc` is then the image's, dated at image-build time, and the engine keeps using it. Always
check the `.pyc` mtime against the patch time.

---

## 3. The kit cannot express a per-node rail list (limitation, not fixed)

**What.** `RDMA_IB_DEVICES` is a single global value in the wrapper, consumed by both
legs. Our two nodes differ:

| node | ACTIVE rails |
|---|---|
| chi2835 | `ionic_0..7` (8) |
| chi2879 | `ionic_0..4,6,7` (7 — `ionic_5` PORT_DOWN) |

**Why it matters.** The kit's own `cluster/README.md` says *"Run it on **each** node — the
two can legitimately differ"*, and *"a rail that is physically down must not be listed, or
every transfer targeting it fails"*. Both are true, and together they are unsatisfiable
with one value.

**What was done.** The wrapper sets the **intersection** (7 rails). This is safe — every
named device is `PORT_ACTIVE` on both nodes — but the prefill leg then runs 7 of its 8
usable rails. **The cost is unmeasured.**

**Why not fixed.** Fixing it means a schema change (per-node override, or deriving the
list on each node inside `leg.sh` the way the reference launcher does). That is a design
decision about the kit's config surface, beyond the scope of "make this run and fix what's
broken". Flagged for the kit's owner.

---

## 4. Nested ssh + `docker exec` + quoting silently no-ops

**What.** This looked like it worked and did nothing:

```bash
ssh $JUMP "ssh $N \"docker exec $CTR bash -c \\\"pkill -9 -f 'infera.engine.sglang'\\\"\""
```

Reported no error. `rocm-smi --showpids` afterwards still listed eight
`sglang::scheduler` processes holding 274 GB/card.

**Why it matters.** The subsequent relaunch would have hit a live engine — port already
bound, VRAM already taken — and the failure would have surfaced as a port-allocation bug.

**How it was resolved.** Stage a **script file** and run that
(`scripts/relaunch_decode.sh`). Every non-trivial remote command in this kit is a file for
this reason.

**Related, same family:** `pkill -f` takes a **regex**, so `-f 'infera.kvd'` also matches
the engine's own `--infera-kvd-socket` argument — the "kill the kvd daemon" pattern kills
the engine too. `scripts/teardown_prev.sh` escapes the dots.

---

## 5. The 51.9 % cache figure is a metric-definition artifact

**What.** `analyze.py` reports `server cache hit : 51.90 %` for a run whose reference is
88.1 %. Chasing that gap took the longest of anything in this session.

**It is not the deployment.** Three independent measurements agree the cache is working:

| measurement | value |
|---|---|
| per-request ratio, **p50** (n=175) | **88.2 %** |
| prefill engine's own token totals, `#cached-token / #new-token` (268 batches) | **49.7 %** |
| `cache_probe.py`: same prompt twice | round 1 `cached=None`, round 2 `cached=1280 / 1301` |

**Why the numbers differ.** They are three different questions:

- **88.2 %** — median over requests *that have the field*. This is what the reference kit
  reports, and the only apples-to-apples comparison. 58 of 233 records carry no
  `usage_prompt_cache_read_tokens`; **53 of those 58 are `turn_index=0`** — first turns,
  which have no prefix to hit by construction.
- **51.9 %** — `analyze.py` counts the missing field as 0, dragging the median down. It is
  arithmetically a *session-weighted* figure dominated by first turns.
- **49.7 %** — token-weighted over the whole run, first turns included. It agrees with
  51.9 % because they ask the same question, and it is the honest answer to "what fraction
  of all prompt tokens were served from cache".

**Which to quote.** 88.2 %, when comparing against the reference or against a prefix-reuse
target — and say "per-request median, turns with a prefix". Quote 49.7 % for "fraction of
prompt tokens not recomputed".

**What this does NOT establish.** Whether `analyze.py`'s definition is wrong or merely
different is a judgement about that script, which is **ours**, not the customer's. The
customer's own kit has a separate, documented defect here (its README claims the reported
figure is the server's realized hit rate; the code computes it from the trace file's
`hash_ids` and never asks the server) — see the reference packup. Not re-litigated.

---

## 6. `cached: None` on the smoke completion is expected

`smoke` prints `usage : 25 prompt / 134 completion | cached None`. That is a cold,
never-seen 25-token prompt — there is nothing to hit. It is **not** evidence that
`--enable-cache-report` is missing.

Established by sending the same prompt twice (`scripts/cache_probe.py`): round 1
`cached=None`, round 2 `cached=1280`. The second request is the test; a single request can
never be.

---

## 7. kvd wrote 47,556 entries and read back zero

After the bench:

```json
{"entries": 47556, "sets_total": 80248, "evictions_total": 32692,
 "gets_total": 0, "hits_total": 0, "misses_total": 0}
```

The reference run recorded the same shape (`+57,870 sets / 0 gets`). **Why prefill never
reads back is NOT ESTABLISHED** — that was true in the reference kit and this run adds no
evidence either way. The measurement that would answer it: correlate kvd `gets` against
the engine's HiCache prefetch decisions over the window, which needs per-request HiCache
tier accounting neither run captured.

`gets = 0` on the **decode** leg is different — it is by design; infera skips kvd wiring
on a PD decode leg, and the kit sets `DECODE_KVD=0` to match.

---

## 8. What this run does NOT establish

Stated plainly, because the temptation is to read a green run as broader than it is.

- **Nothing about `cluster.dmabuf.sh`.** Both nodes have `ib_peer_mem`, so only the mode-A
  path was exercised. Wrapper B is untested by this run.
- **Nothing about `preflight_rdma.sh fabric`.** Only the `mode` subcommand ran. The
  cross-node bandwidth probe was not exercised.
- **Nothing about `engine/bench.sh`.** The throughput sweep was skipped in favour of the
  customer bench, which was the spec's actual ask.
- **Nothing about the placeholder image.** `inferaimage/infera-sglang:0.2.0` does not
  exist on this cluster; every finding is against `merged-e`. A published image would
  carry `mooncake_mode` and P1V3, which would make gaps #1 and #2 disappear — and might
  change the entrypoint contract that fix #1 hangs off.
- **Nothing about long-horizon stability.** The longest continuous window was 913 s
  (the bench). The reference kit's crash-without-P1V3 happened at 125 s and 766 s, so this
  window does clear that bar — but a multi-hour run was not attempted.
- **No A/B on any fix.** Each kit fix is justified by the reading it corrects, not by a
  controlled before/after deployment. Fix #1 is the exception: both configurations were
  run (`scripts/verify_mount_revert.sh`, 8 devices vs 0 + warning).
- **The rail asymmetry is uncontrolled** (§3), and its effect on KV transfer time is
  unmeasured.

---

## 9. Shared-cluster hygiene observed

- Both nodes' slurm holds belong to `yeandy-debug`. **Never `scancel`** — only our own
  containers and processes were touched.
- Ownership of the pre-existing `bench_run` deployment was **proven before removal**:
  `docker inspect` showed the exact bind set our own launcher creates
  (`/mnt/vast` + our libionic mount) and its logs were under our own
  `/mnt/vast/c_huggingface/bench_20260801/`.
- Nothing was pruned. No image removed, no other user's container touched, no `/tmp`
  cleaned.
- kvd L3 capped at 64 GB, not the kit's higher defaults — it writes to the container's
  writable layer on the **node root disk**, and an oversized budget has filled a node's
  root filesystem on this cluster before, after which every `docker exec` fails with
  "no space left on device" and the node reads as broken rather than full.
