# Patches — what / why / how / context

Five `git format-patch` files: the whole PR. Apply on `origin/main` @ `8692fb4`:

```bash
git checkout -b <branch> 8692fb4
git am patches/0001-*.patch patches/0002-*.patch patches/0003-*.patch \
       patches/0004-*.patch patches/0005-*.patch
```

Result is a tree hashing `a16d0dce342be853e0369681f8fae7fde84d6b2a` — identical
to the branch this experiment ran from. Verified against a fresh clone.

---

## 0001 — `fix(net)`: randomise the `free_tcp_port_block` scan start

**What.** `infera/common/net.py`. `free_tcp_port_block()` scanned downward from a
fixed base (`ip_local_port_range.low - count`) and released its probe sockets
before returning. Now it tries 64 **random** bases in the same sub-ephemeral
window first, falling back to the original exhaustive downward scan.
Adds `tests/unit/common/test_net_port_block.py` (4 tests).

**Why.** The reservation cannot be exclusive — the probe binds `127.0.0.1:P`
while the caller's real listener binds `0.0.0.0:P` (zmq `tcp://*`), so holding it
would lock out our own child. With a fixed start, collisions were not merely
possible but **deterministic**: two engines launched on one host both scanned
from the same place and both saw the same block free.

**How applied.** Plain `git am`; touches one function plus a new test file.

**Context — the symptom it cured.** A PD pair on one host: the second leg died
with

```
zmq.error.ZMQError: Address already in use (addr='tcp://*:32765')
```

Reached only from the **kv-aware** path — `sglang/worker.py:77`, when
`enable_kv_events` is set and SGLang binds one KV-event publisher per DP rank at
`base + rank`.

**Two alternatives were tried and rejected against live MVPs**, which is why the
fix looks the way it does:

- *Hold the reservation until the child binds.* MVP showed the child then fails
  with `errno 98` — we would be locking out our own subprocess.
- *`0.0.0.0` + `SO_REUSEADDR` reservation.* MVP showed a second probe can still
  take the same port; the reservation isn't exclusive, so it buys nothing.

**Blast radius.** `free_tcp_port_block` has exactly **one** caller in the repo
(`infera/engine/sglang/worker.py:77`). vLLM and ATOM use the *other* function,
`free_tcp_port()`, which this patch does not touch.

---

## 0002 — `fix(kvd)`: strip the bind-mount subpath from the `findmnt` source

**What.** `infera/kvd/storage_classify.py`. `_findmnt()` returned the raw
`findmnt` source, which for a bind mount carries the bind subpath in brackets:

```
/dev/md0[/mnt/nvme-raid/kvd-long]
```

Now only the part before `[` is kept. Adds 2 cases to
`tests/unit/kvd/test_storage_classify.py`.

**Why.** The caller hands that string to `lsblk`, which answers *"not a block
device"*. The failure is **silent**: classification falls back to buffered I/O
even on hardware that qualifies for O_DIRECT.

**How applied.** Plain `git am`.

**Context — the symptom it cured.** Exactly the normal kvd deployment: an L3
long-tier directory bind-mounted into the engine container. A container-hosted
kvd would never use O_DIRECT on its NVMe tier, with no error to say why.

**A trap inside the test.** The first version of the new test **passed on the
broken code** — the fake `_run` helper returned success for `lsblk` regardless of
its argument. It had to be taught the real contract (real `lsblk` exits non-zero
on a bracketed target) before it could fail. If you extend these tests, revert
the fix and confirm they go red.

**Blast radius.** `storage_classify` is private to the kvd daemon
(`infera/kvd/` + the preflight tools); no engine imports it. But the daemon is
**shared by all three engines**, so vLLM's kvd L3 path benefits identically.

One second-order effect checked deliberately: `mount_source` is also fed to
`_nconnect_for_nfs` / `_rsize_wsize_for_nfs`, which compare it against
`/proc/mounts` with `src == source`. `/proc/mounts` **never** carries the bracket
form (verified), so stripping moves that comparison from *always fails* to *can
succeed* — a fix for NFS too, not a regression. There is also a `target == mp`
fallback.

---

## 0003 — `docs(kvd)`: record SGLang KV-cache offload support

**What.** `manual/features/feature_matrix.md` + `manual/features/kv_cache_offload.md`.
Flips SGLang kvd from 🚧 to ✅ and adds a SGLang section to the offload page.

**Why.** The manual said kvd is *"vLLM only (for now)"*. `InferaKvdBackend`
(a SGLang `HiCacheStorage` backend) has been in the repo since **`Infera
v0.1.0`** — i.e. it predates the doc that says it doesn't exist.

**How applied.** Plain `git am`; docs only.

**Context.** ⚠️ **This reverses `29a69ca`** ("docs: kvd offload is vLLM-only",
jiejing, 2026-07-24). That commit gave no rationale; the most likely reason is
that CI's kvd e2e covers **only vLLM**
(`tests/e2e/pd_mixed/vllm/test_mixed_kvd.py` — there is no SGLang equivalent), so
the SGLang path was untested rather than absent. This experiment tested it on
hardware: the backend connects on all 8 DP ranks, serves, and survives an engine
restart (+102 hits / 0 new sets). **Worth confirming with the original author in
review** so the doc doesn't ping-pong. Adding a SGLang kvd e2e case would settle
it permanently — not done here, out of scope.

AIC GPU-Direct remains vLLM-only; SGLang reads via the daemon's POSIX path.

---

## 0004 — `build(sglang)`: the kv-aware + kvd deployment image

**What.** New `deploy/docker/Dockerfile.sglang.kvaware-kvd` (also copied to
`../dockerfiles/`).

**Why.** Not different image *contents* — `Dockerfile.sglang` already builds
engine + wrapper + kvd daemon + router + statctl, and one image runs all four
roles. This adds the **contract**: a digest-pinnable base, a build-time
self-check, and the operational defaults recorded in the image instead of in
someone's shell history.

**How applied.** Plain `git am`. Build with:

```bash
docker build -f deploy/docker/Dockerfile.sglang -t <base> .
docker build -f deploy/docker/Dockerfile.sglang.kvaware-kvd \
  --build-arg INFERA_SGLANG_IMAGE=<base> -t infera/engine-sglang:kvaware-kvd .
```

**Context.** The self-check imports `InferaKvdBackend`,
`wire_infera_kvd_backend`, `attach_to_radix_cache`, `statctl`, and asserts
`free_tcp_port_block` no longer returns a fixed base. It **discriminates** —
10 distinct bases post-fix, 1 pre-fix (measured). So a build that loses the kvd
adapter (bad merge, or a base bump that moves `HiCacheStorage`) fails at build
time rather than starting fine and serving with no L3.

`PYTHONHASHSEED=0` is baked in because for kvd it is a **correctness**
requirement, not tuning: unset, it is random per process, the two PD legs hash
the same prompt differently, and every restart orphans the whole L3 cache.

---

## 0005 — `docs(serving)`: the operator guide

**What.** New `manual/serving/kvaware_kvd_operations.md` + a toctree entry.

**Why.** The feature pages give the flags. Nothing said which of them exist only
to make an experiment cheap, what production should set instead, how to check a
switch actually took effect, or which numbers are safe to tune.

**How applied.** Plain `git am`; docs only.

**Context.** The section that matters most is the verification one, because all
three features fail *silently* by degrading to a slower-but-correct path — so
"it works" is not evidence. Each check therefore has a negative case. The
sharpest: in the earlier investigation a reuse phase ran **2.7× faster** with
kvd's counters flat at **zero** — the win was the in-GPU radix cache, not kvd.

Also documents the trap that `--kv-overlap-weight` defaults to `1.0` and the
per-role weights fall back to it, so a PD deployment that doesn't set them
explicitly behaves nearly round-robin and looks like kv-aware "does nothing".

The recommended weights (20.0 prefill / 2.0 decode) are labelled as a documented
starting point, **not** benchmarked optima — see `../notes.md` §10.
