# 03 — `free_tcp_port_block` gave every caller the same port block (BUG #1)

**Ran:** 2026-07-30 (MVP re-run 2026-07-31) · **Cost:** ~1 s desk check, or ~8 min on one node
**Verdict:** ❌ decode leg dead — and the cause is a **real infera bug**, now fixed and tested

## What this experiment answers

Round r1's two configuration mistakes are fixed: the hicache host pool is
bounded with `--hicache-size 8` (8.00 GB/rank, was 354.94) and the legs' ports
are 1000 apart. So: **does a 1P1D pair with kvaware come up now?**

No. And this time it is not a misconfiguration — it is a bug in
`infera/common/net.py`.

Configuration: Qwen3-1.7B, one node (chi2879), 1P1D — prefill TP4 GPU0-3
:30000, decode TP4 GPU4-7 :31000, infera router :8100, `infera.kvd` daemon on
`/tmp/kvd/kvd.sock`, `KVD=1 KVAWARE=1 POLICY=kv-aware`.

## Result

**Progress:** the host pool is bounded, and kvd genuinely connects for the first
time —

```
[DP*] Allocating 8.00 GB host memory for hierarchical KV cache.      <- was 354.94
[DP1 TP1 EP1] Creating dynamic storage backend 'infera-kvd'
[DP1 TP1 EP1] infera-kvd adapter connected to /tmp/kvd/kvd.sock (model=qwen3, compat_key=tp0of1_pp0of1)
[DP1 TP1 EP1] Tree cache initialized: impl=HiRadixCache hierarchical=True
```

prefill = ready. **decode Rank-0 scheduler died:**

```
zmq.error.ZMQError: Address already in use (addr='tcp://*:32765')
RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)
```

## The finding that matters

Both legs' `--kv-events-config` carried the **same** endpoint port:

```
--kv-events-config {"publisher": "zmq", "endpoint": "tcp://*:32764", "topic": "kv-events"}
```

With `--dp-size 4`, sglang binds one KV-event publisher per DP rank at
`base + attn_dp_rank`:

```
leg A (prefill) -> 32764, 32765, 32766, 32767
leg B (decode)  -> 32764, 32765, 32766, 32767      <- every one collides
```

The crash lands on `32765` = base+1, DP rank 1's publisher.

That base comes from `infera/common/net.py:free_tcp_port_block(count)`, which
before the fix **scanned downward from a fixed start** (`ip_local_port_range.low
− count` = 32768 − 4 = **32764**) and **released its probe sockets in `finally`
before returning**. Two callers therefore saw the identical free block and both
returned it.

> **Not a rare race — deterministic.** Ten consecutive calls return the same
> number, every time:
>
> ```
> OLD bases: [32764, 32764, 32764, 32764, 32764, 32764, 32764, 32764, 32764, 32764]
> distinct: 1 -> FAIL (bug reproduced)
> ```

It is reached only from `worker.py:77`, i.e. **only when `enable_kv_events` is
on**. This is a kvaware-path bug, which is why nobody had hit it before.

Note the symptom string is the *same* as round r1's, but the address is not:
r1 crashed on `tcp://127.0.0.1:30235` (loopback, from sglang's own `--port`
arithmetic), this one on `tcp://*:32765` (wildcard, from infera's
`free_tcp_port_block`). Different code, different fix. Do not conflate them.

### The fix, and why it is "randomise" and not the two obvious options

Try `_PORT_BLOCK_TRIES = 64` randomly-chosen bases in `[1024, low − count]`
first, then fall back to the original exhaustive downward scan so the function
still never fails while a block is genuinely available. Signature and socket
semantics unchanged.

```
>>> [free_tcp_port_block(4) for _ in range(6)]
[12457, 4265, 24310, 16567, 28874, 29547]
```

Both more-obvious fixes were **tested on the node and rejected on evidence**.
Recorded so nobody re-proposes them:

**1. Hold the reservation until the child binds.** Rejected — the probe binds
`127.0.0.1:P` but the sglang child binds `0.0.0.0:P` (zmq `tcp://*`):

```
reserved 127.0.0.1:29600
child bind 0.0.0.0:29600 -> BLOCKED errno=98  <-- holding would break our own child
```

Holding the block locks out the very process it was reserved for.

**2. Reserve on `0.0.0.0` with `SO_REUSEADDR`** so the child can take it over.
Rejected — then the reservation is not exclusive either:

```
reserved 0.0.0.0:29700 (SO_REUSEADDR, no listen)
  other leg probe  -> OK   <-- BAD, collision still possible
  our child        -> OK
```

Exclusivity and let-the-child-take-over are mutually incompatible here, so the
reservation **must** be released before returning. What remains fixable is the
**determinism of the scan start**.

Both were caught by running a six-line MVP *before* shipping a change. That is
the whole reason neither shipped.

**A third thing that was tried and is the wrong knob:** `--kv-events-bind`.
It controls *infera's own* publisher socket, not the port passed to sglang —
`worker.py:77` ignores it for the `--kv-events-config` computation. It does fix
a *different*, adjacent collision (its default `tcp://0.0.0.0:5557` is identical
on every leg), which is why later rounds give each leg its own 5557 / 5657.
Two separate port problems one flag apart; fixing one does nothing for the
other.

## How to reproduce

**Desk check — no cluster, no GPU, ~1 second. This is the actual root cause:**

```bash
bash scripts/run.sh                 # MODE=desk is the default
```

It reproduces the bug against an inlined copy of the pre-fix loop body,
confirms the fix against the patched `net.py` (needs `infera` importable — set
`PYTHONPATH` to a checkout), demonstrates both rejected alternatives live, and
runs the 4 regression tests. Output goes to
`results/mvp_port_block.observed.txt`; the committed reference is
`results/mvp_port_block.txt`.

**Live 1P1D on one node, ~8 min:**

```bash
MODE=live NETPY=old bash scripts/run.sh   # reproduce r2's dead decode leg (FAILS by design)
MODE=live NETPY=new bash scripts/run.sh   # both legs up, distinct endpoints
```

`NETPY=old` leaves the container's stock (pre-fix) `net.py` alone; `NETPY=new`
`docker cp`s `scripts/net_fixed.py` over `/opt/infera/infera/common/net.py`.
The image predates the fix, so that copy is how it was applied without a
rebuild.

**In-repo:**

```bash
git apply scripts/0001-free_tcp_port_block-randomise-scan-start.patch
cp scripts/test_net_port_block.py tests/unit/common/
python3 -m pytest tests/unit/common/test_net_port_block.py -q     # -> 4 passed
```

## Gotchas specific to this experiment

- **32764 is not a magic number** — it is `ip_local_port_range.low − count`.
  On a box with a different range you get a different base, and the collision
  is exactly as total. That is the point: whatever the value, *both* callers
  get it. If your MVP prints something other than 32764, check
  `/proc/sys/net/ipv4/ip_local_port_range` before assuming the script is wrong.
- **`dp_size > 1` is what makes it fatal.** At `dp_size=1` both legs still get
  the same base, but only rank 0 binds, and whichever leg loses simply fails on
  the base itself rather than on base+1. The symptom shifts; the bug does not.
- **Turning kvaware off makes the bug vanish** — `free_tcp_port_block` is
  unreachable without `enable_kv_events`. That is a diagnostic signal, not a
  fix, and it costs you kvd on the decode leg as a side effect.
- **`MODE=live NETPY=new` passing does not mean requests work.** Both legs up is
  a *startup* result. Serving over same-host mooncake RDMA fails for a
  completely unrelated reason (cross-rail loopback) — separate problem,
  separate fix.
- **The patch was uncommitted** as of 2026-07-30, sitting in the working tree of
  branch `yihou.dev.glm5.2.mxfp4.experiment`. Whether it has landed since is
  **unknown** from this packup's evidence — check the branch.
- **Raw engine logs for this round no longer exist** (container removed before
  the GLM-5.2 runs). This matters less here than elsewhere: the root cause is a
  pure-Python function, and `results/mvp_port_block.txt` is a *fresh* re-run,
  not a transcript excerpt. See `logs/README.md`.

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
