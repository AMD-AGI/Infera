# Patches

Six changes were load-bearing for this run: **five kit fixes** (all in
[`kit_fixes.diff`](kit_fixes.diff), against `examples/sglang_1p1d_glm5.2/` at `e2d462a`)
and **one runtime patch** to the engine ([`apply_p1v3.py`](apply_p1v3.py)).

All five kit fixes share a failure mode, which is the single most expensive bug class on
this stack:

> **They fail silently.** The leg boots, serves, and returns a clean run — of the wrong
> deployment, or with a check that reports the wrong thing. No error, no warning.

| # | file(s) | what it prevents |
|---|---|---|
| 1 | `common.sh`, `preflight_rdma.sh` | host RDMA library mounted where nothing reads it → **zero RDMA devices**, KV over a 5-20× slower transport |
| 2 | `common.sh` | a device count of 0 printed without comment → #1 goes unnoticed |
| 3 | `engine/smoke.sh` | router-policy check that prints **empty whatever the policy is** |
| 4 | `common.sh`, `engine/smoke.sh` | `00` readings, and `binary file matches` instead of the matched text |
| 5 | `engine/smoke.sh` | MTP read from a 5-sample tail → **healthy leg reported as degenerate** |
| 6 | engine source (runtime) | decode leg **crashes** under MTP + DP-attention on an idle rank |

---

## 1. HOST_RDMA_MOUNT — the mount point must be the one the entrypoint reads

**What.** `common.sh:start_container` mounted the host provider library at a path it
derived itself:

```bash
mounts+=(-v "$HOST_RDMA_LIB:/host-rdma/$(basename "$HOST_RDMA_LIB"):ro")
```

`preflight_rdma.sh` did the same. The fix introduces `HOST_RDMA_MOUNT`, keeps that
expression as the default, and lets the wrapper name the path its image actually reads.

**Why it matters.** The `infera-sglang` image's ENTRYPOINT reads exactly one path:

```bash
SRC=/host-libionic/libionic.so
if [ -e "$SRC" ]; then ... cp -f "$SRC" "$TGT"; ldconfig; fi
```

`/host-rdma/libionic.so.1` is not that path, so the injection **no-ops**. The container
keeps its own `libionic.so.1.0.54.0-149`, which does not speak the host `ionic_rdma`
kmod's ABI. Measured first-hand on the first bring-up:

```
libibverbs: Warning: Driver ionic does not support the kernel ABI of 1 (supports 4 to 4)
            for device /sys/class/infiniband/ionic_0        [× all 8 cards]
No IB devices found
```

**And the leg still booted and served.** `MC_FORCE_TCP` and `GID is NULL` were both 0 —
because mooncake never got far enough to print either. Every one of the kit's own RDMA
checks passed on a deployment with no RDMA at all.

**How it was caught.** By reading the `RDMA PORT_ACTIVE visible in container: 0` line
that `start_container` prints — which the unfixed kit prints without comment (see #2).

**Verified, both directions**, with `scripts/verify_mount_revert.sh` on chi2835:

| configuration | device count | warning |
|---|---|---|
| wrapper sets `HOST_RDMA_MOUNT=/host-libionic/libionic.so` | **8** | none |
| `HOST_RDMA_MOUNT` omitted → kit default | **0** | printed |

**Why the default was left alone rather than hardcoded to the image's path.** An earlier
revision of this fix pinned the mount point to `/host-libionic/libionic.so` in
`common.sh`. That is an `infera-sglang` property, not a kit-wide truth, and hardcoding it
binds the kit to one image. The variable belongs in the wrapper, where every other
site-specific value lives.

---

## 2. Warn when the container sees zero RDMA devices

**What.** `start_container` ended with:

```bash
echo -n "  RDMA PORT_ACTIVE visible in container: "
docker exec "$CTR" bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE || echo 0'
```

Now the value is captured, printed, and — when 0 — followed by a warning naming the
likely cause and the command that confirms it.

**Why it matters.** Zero devices is not an error condition anywhere in the stack:
mooncake falls back to a transport that works and is merely 5-20× slower, and the leg
boots normally. The number is the *only* signal, and it scrolls past inside a bring-up
that is otherwise all-green. This is what makes #1 expensive rather than annoying.

**Context.** #1 and #2 are one defect and its detector. With #1's default deliberately
left image-agnostic, a site that forgets `HOST_RDMA_MOUNT` gets exactly the silent
failure — so the warning is what makes the flexible default safe.

---

## 3. The router policy check never matched

**What.**

```bash
grep -o 'router-policy=[a-z-]*' /tmp/router.log      # before
```

The Rust router does not emit that string. It dumps a `Config{...}` struct containing
`router_policy: "kv-aware"`, ANSI-colourised. Three mismatches at once: hyphen vs
underscore, `=` vs `: `, and unquoted vs quoted. Plus the log carries binary bytes, so a
bare `grep` prints `binary file matches` rather than the text.

**Why it matters.** The line printed **empty regardless of the actual policy** — it could
not distinguish `kv-aware` from `round-robin` from a router that failed to parse the flag.
A check that cannot fail is not a check.

**Fix.** `strings` → strip ANSI → match the struct field. Also added the tokenizer-loaded
count: kv-aware **silently degrades to load-only routing** if the tokenizer did not load,
so the policy line alone is insufficient.

**Verified live:** `router router_policy: "kv-aware"   tokenizer-loaded=1`.

---

## 4. `grep -c … || echo 0`, and binary logs

**What.** Two idioms, several sites.

`grep -c` **already prints `0`** on no match, and exits 1. The `|| echo 0` then appends a
*second* zero:

```
chi2835    MC_FORCE_TCP=00   GID-is-NULL=00
```

`00` is not a number to any downstream test, and reads as a typo rather than a bug.
Replaced with `grep -c …; true`.

Separately, the engine logs contain binary bytes. `grep` switches to "binary file
matches" mode and prints **no matched text** — so any `grep -o` pattern silently returns
nothing. Every log read now goes through `strings` first.

**Also fixed here:** the labels. `echo -n "  $h MC_FORCE_TCP="` followed by output
arriving over ssh interleaves, and the block rendered as a bare column of digits with no
way to tell which check produced which. Values are captured into variables and printed
with a single `printf`.

Before / after, same deployment:

```
0                                                    chi2835  MC_FORCE_TCP=0  GID-is-NULL=0  RDMA-devices-in-container=8
0                                       ->           chi2879  MC_FORCE_TCP=0  GID-is-NULL=0  RDMA-devices-in-container=7
0
0
```

---

## 5. MTP read from a 5-sample tail

**What.**

```bash
grep -o 'accept len: [0-9.]*' $DLOG | tail -5      # before
```

Now: the full distribution — `n`, p10, **median**, p90, and the count/percentage at 4.00.

**Why it matters.** The kit's own README states the rule: *"An MTP acceptance length of a
steady 4.00 is bad news, not a good result."* A 5-sample tail cannot distinguish "steady"
from "the last five happened to be 4.00".

**How it was caught — this fired on us.** A `smoke` run after the bench printed:

```
accept len: 4.00
accept len: 4.00
accept len: 4.00
accept len: 4.00
accept len: 3.17
```

By the kit's documented reading, that is a degenerate repetition loop. The full
distribution says otherwise:

```
n=1265  p10=2.01  MEDIAN=2.88  p90=3.84  at-4.00=59 (4.7%)
```

**Median 2.88 — squarely in the healthy 2–3 band.** 4.7 % of batches sit at 4.00, which
matches the reference run's 3.8 % (`par8`, engine mean 2.76). The tail had simply landed
on a run of them.

**Context.** Had this not been investigated, the honest conclusion from the kit's own
instructions would have been "MTP has degenerated" — on a leg that was working correctly.
A check whose false-positive rate is ~5 % per invocation is worse than no check, because
it gets believed.

---

## 6. GLM52_P1V3 — runtime patch to the engine (NOT a kit fix)

**What.** `apply_p1v3.py`, applied inside the decode container. Inherited verbatim from
`../par8.glm52.dpaoff.packup_20260803/patches/`; the same fix lives in the repo at
`deploy/docker/patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py`.

**Why it was needed here.** The image predates the patch. Measured:

```bash
docker run --rm --entrypoint bash infera/engine-sglang:merged-e -c \
  'F=.../dsa_indexer.py; grep -c _p1v2_trim $F; grep -c _p1v2_clip $F; grep -c _p1v2_rows $F'
# 4    <- GLM52_P1V2 present
# 0    <- GLM52_P1V3 ABSENT
# 0
```

Without it the decode leg crashes under MTP + DP-attention on an **idle** rank with
`Expected lengths.size(0) == B to be true` — reproduced twice in the reference kit at
125 s and 766 s into agentic runs. The bug needs MTP **and** DPA **and** an idle rank
simultaneously, which a fixed-shape sweep never produces but an agentic workload does
constantly.

**How applied.**

```bash
docker cp apply_p1v3.py glm52_pd:/tmp/ && docker exec glm52_pd python3 /tmp/apply_p1v3.py
# -> patched OK - GLM52_P1V3 occurrences: 3
bash relaunch_decode.sh      # reap, delete the .pyc, relaunch via the kit's engine/leg.sh
```

**Verification is on the BYTECODE, not the source:**

```
-rw-r--r-- 44181 2026-08-04 08:01:49 .../__pycache__/dsa_indexer.cpython-310.pyc
_p1v2_rows in BYTECODE: 1
_p1v2_clip in BYTECODE: 1
```

The `.pyc` mtime (08:01:49) is after the patch (07:58), and both markers are in the
compiled object. A running engine has already imported the old module, so a patch script
printing `patched OK` proves only that the *file on disk* changed. This exact trap — source
patched, bytecode stale — has invalidated a full experiment on this stack before.

**Deliberate deviation, stated plainly.** The kit's design is "the image is the artifact";
patching a live container is not how it is meant to be deployed. The correct fix is to
rebuild the image from `deploy/docker/Dockerfile.sglang`, which applies the whole DSA
patch set at build time **and verifies the bytecode markers**. That was skipped here
because it costs 1-2 h per node and is orthogonal to what this run set out to test. Any
production deployment should carry the patch in the image, not at runtime.
