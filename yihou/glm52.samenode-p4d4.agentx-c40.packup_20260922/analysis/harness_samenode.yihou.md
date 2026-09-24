# Same-node harness — vendoring, patches, and verification

Goal: make the bench harness launch prefill **and** decode on the single node
`crsuse2-m2m-276`, without touching one byte of the tracked repo. This documents
every change, the first-hand evidence for it, and the verification output.

All work lives under
`bench/glm5p2_pd/results/yihou-samenode-p4d4/` (`<ws>` below). The tracked
harness under `bench/glm5p2_pd/` is unmodified.

## Layout produced

```
<ws>/scripts/bench-harness/     vendored copy of bench/glm5p2_pd (scripts + tools/ + eval/), patched
<ws>/scripts/topology.yihou.tsv two rows, both 276 / 10.245.152.249, prefill first
<ws>/scripts/config.yihou.sn.p4d4.sh  the run config (sources the vendored config.sh)
<ws>/bin/ssh                    ssh shim routing 276 through `spur exec`
<ws>/patches/topology.py.diff   unified diff of the only modified vendored file
<ws>/patches/SHA256SUMS.txt      sha256 of every vendored/created file
<ws>/analysis/harness_samenode.yihou.md  this file
```

Vendored from `bench/glm5p2_pd/` excluding `results/`, `.cache/` (83 MB, regenerated
by agentx_bench), and `__pycache__/`. `eval/` is 17 KB and was kept. `diff -rq`
against the repo source confirms **`tools/topology.py` is the only file that
differs** — everything else is byte-identical.

## 1. topology.py — relaxed the uniqueness check

`load()` raised `SystemExit("node and data_ip must be unique")` on
`node in nodes or data_ip in ips` (repo `tools/topology.py:46-47`). Node 276 has
exactly one physical IPv4 (`10.245.152.249` on `ens3`, verified below), so a
prefill row and a decode row collide on both.

Patch (`patches/topology.py.diff`): replaced the `nodes`/`ips` sets with a
`seen_rows: set[tuple[role,node,data_ip]]` and reject only an **exact duplicate
row** (same role+node+data_ip twice). Everything else is untouched: role
whitelist, node-name regex, IPv4 check, at-least-one-of-each, and the
`index`/`instance` numbering (`index = line-2`, per-role counter) — so every port
`launch.sh` derives from `index` is unchanged.

Verification:
```
repo original  → rc=1  "node and data_ip must be unique"   (rejects, as before)
vendored       → rc=0                                       (accepts)
vendored rows  → 0  prefill-0  prefill  crsuse2-m2m-276  10.245.152.249
                 1  decode-0   decode   crsuse2-m2m-276  10.245.152.249
duplicate row  → rc=1  "duplicate topology row role=prefill ..."   (still rejected)
missing decode → rc=1  "at least one prefill and one decode are required"
```

## 2. ssh shim — routing 276 through `spur exec`

276 refuses ssh (`AllowUsers ubuntu root`; we are `yihou`). The only transport is
`spur exec 165913 bash -c '<cmd>'` (runs as yihou, pwd=/, docker available).

`<ws>/bin/ssh` is a drop-in `ssh`. Prepend `<ws>/bin` to `PATH` and the harness
calls it unchanged. Behaviour:

- Parses and discards ssh option flags (full `takes-arg` letter set; the harness
  only uses `-o` + `-n`), takes the first non-option as host, collects the rest as
  the command.
- **host == crsuse2-m2m-276** → `exec spur exec 165913 bash -c "<joined cmd>"`.
  Real ssh joins the post-host argv with single spaces and runs it through one
  remote shell (one parse). We reproduce that exactly: `spur exec` **preserves
  argv boundaries** (verified: `printf '[%s]' a 'b c' d` → `[a][b c][d]`), so
  `bash -c` receives the joined command as a single argument and parses it once.
- **any other host** → `exec <real ssh> "$@"` with the original argv preserved, so
  137/138/etc. still work.
- `-n` → append `</dev/null` to the spur exec (stdin from /dev/null, ssh's `-n`).
- Exit status and stderr propagate from the remote command.

Override knobs: `SPUR_SHIM_JOB` / `SPUR_SHIM_NODE`.

Verification (`PATH=<ws>/bin:$PATH`):
```
ssh -o BatchMode=yes crsuse2-m2m-276 'hostname; docker ps --format {{.Names}}'
   → crsuse2-m2m-276 / yzhou_model
ssh -n ... crsuse2-m2m-276 'exit 7'                → rc=7           (exit code)
ssh ... crsuse2-m2m-276 "$(printf '%q ' echo 'a b' 'c;d')"  → "a b c;d"
   (the launch.sh remote_command() %q string survives exactly ONE parse == ssh)
ssh ... crsuse2-m2m-276 'ls /no/such/yihou/path'   → stderr ls error, rc=2
ssh ... crsuse2-m2m-nonexistent-yihou 'hostname'   → REAL ssh "Could not resolve", rc=255
ssh -n ... crsuse2-m2m-276 'cat; echo X'           → X, rc=0, no hang (empty stdin)
# wait_healthy.py's exact pattern (argv list + shlex.join one string) via subprocess:
["ssh","-o","BatchMode=yes",...,"crsuse2-m2m-276","docker inspect --format {{.State.Status}} yzhou_model"]
   → stdout "running", rc=0
```

`wait_healthy.py` (`remote()` at `tools/wait_healthy.py:25-29`) passes ssh options
as a list and the remote command as one `shlex.join` string — the shim satisfies
it. It splits `--ssh-options` with `shlex.split` (`:127`), so `-o BatchMode=yes …`
arrives as the flags the shim already skips.

### Known limitation — `check_nodes.sh` does not work through the shim

`spur exec` has **no stdin option** (`spur exec --help`: only `--controller`), and
stdin is not forwarded (verified: `printf x | spur exec … 'read v; echo $v'` →
empty). Every launch-path caller is safe: `launch.sh` runs engine.sh `</dev/null`;
`stop.sh` and `preflight.sh` use `ssh -n`; `wait_healthy.py` and `agentx_bench.sh`
run commands that never read stdin. **Only `check_nodes.sh`** pipes a heredoc into
`ssh … bash -s` (`check_nodes.sh:45-46`); that script cannot run through the shim.
It is not on the deliverable path — GPU-idle checks are done with `rocm-smi` over
`spur exec` directly. Not worked around; recorded honestly.

## 3. REMOTE_BENCH_DIR — vendored path is visible on 276

`launch.sh:101` runs `bash "$REMOTE_BENCH_DIR/engine.sh"` on the node. `/home` is
an NFS mount on 276 (verified: `mount | grep /home` → `…:/volumes/… on /home type
nfs (rw…)`, `HOME=/home/yihou`), and the vendored `engine.sh` is reachable there
(verified: `spur exec … ls <ws>/scripts/bench-harness` succeeds). Set
`REMOTE_BENCH_DIR=<ws>/scripts/bench-harness` at launch time.

## 4. Config and topology files

`scripts/topology.yihou.tsv` — two rows, both `crsuse2-m2m-276` / `10.245.152.249`,
prefill first → index 0, decode → index 1.

`scripts/config.yihou.sn.p4d4.sh` — based on
`glm52.p8d8.agentx-sweep.packup_20260920/scripts/config.yihou.p4d4.sh`. Deltas:

- `CONTROL_NODE`/`BUILDER_NODE=crsuse2-m2m-276`, `CONTAINER_PREFIX=glm52-pd-yihou-sn-p4d4`.
- `PREFILL_GPU_DEVICES=0,1,2,3`, `DECODE_GPU_DEVICES=4,5,6,7` (separate vars, both
  resolve at launch.sh source time where `role` is unset).
- **Shared RDMA map — BOTH legs on `ionic_0-3` (identical).** *Supersedes the
  earlier per-role split (prefill ionic_0-3 / decode ionic_4-7), which was
  withdrawn.* A controlled perftest (leader, 2026-09-22, `rounds/002-rdma-loopback/`,
  reproduced on 276 **and** 137) showed RDMA between two *different* ionic devices
  on the same host passes no traffic, while same-device loopback runs at ~40 GB/s
  and the cross-host control at 42 GB/s with the identical command. The
  discriminator is *different device*, not *different socket* (`ionic_0<-ionic_1`,
  both NUMA0, also fails). So both legs take `ionic_0-3`; every matched rank pair
  then transfers same-device→same-device, and `MC_ENABLE_DEST_DEVICE_AFFINITY=1`
  + `PD_DP_RANK_AFFINITY=1` (both left at config.sh defaults) produce exactly that
  `rank i ↔ ionic_i ↔ rank i` pairing. **Accepted cost:** decode GPUs 4-7 are
  NUMA1 while `ionic_0-3` are NUMA0, so decode KV DMA crosses the socket — accepted
  to get a working run. Bring-up fallback documented in the config: point all four
  keys at `ionic_0` on both legs. The `$role`-branch *mechanism* is retained (both
  branches resolve to the same value today) and stays documented — evidence it is
  expressible: `engine.sh:15` sets `role="$1"`, `engine.sh:32` sources `CONFIG`
  after that, so the config is sourced with `$role` set (confirmed by dry-run).
  Explicit `[[ -z "${RDMA_DEVICE:-}" ]]` guard, never `${VAR:=…}` (a `}` inside the
  JSON truncates it); the guard also lets a command-line `RDMA_DEVICE=` win.
- **Per-role AITER JIT cache root** (added for the same-node hazard) — `engine.sh:91`
  resolves `${AITER_JIT_CACHE_ROOT:-/tmp/aiter-jit-$(id -u)}/$image_key` and bind-
  mounts it. That was per-*machine* in every cross-node deployment; with both legs
  on one host the two containers would share one cache dir and can race on first
  compile. The config branches on `$role` to give each leg its own root
  (`/tmp/aiter-jit-yihou-sn-{prefill,decode}`; both carry `yihou`).
- `DECODE_NO_CUSTOM_AR=1` → `DECODE_EXTRA_ARGS=--disable-custom-all-reduce`
  (mandatory for this shape; here defaulted ON, not left as an open probe).
- Sources the vendored **`config.sh`** (fast mode: `AGENTX_DURATION=1200`, warmup
  1/lane, HiCache off) via an absolute `BENCH_DIR` into `<ws>/scripts/bench-harness`.
- `DSA_PREFILL_BACKEND=DSA_DECODE_BACKEND=tilelang`; `DSA_TOPK_BACKEND=""` cleared
  **after** the source. Note: `config.sh` defines **no** `DSA_TOPK_BACKEND` default
  (unlike `config.full.sh:89`), so it is already unset post-source; the explicit
  clear only guards a command-line value. `engine.sh:181` drops the flag when empty.
- `DECODE_SIMULATE_ACC_LEN` left unset → `config.sh:80` `${VAR-3.61}` keeps 3.61,
  and a command-line `DECODE_SIMULATE_ACC_LEN=` (empty, set) disables it.
- `JSON_MODEL_OVERRIDE_ARGS` inherited from `config.sh` default
  `{"index_share_for_mtp_iteration":false}` (not touched).

### Config-resolution dry-run (sourced the way engine.sh does)

```
role=prefill: RDMA={"0":"ionic_0",…,"3":"ionic_3"}  MC_TE_FILTERS=ionic_0..3  GPUs 0,1,2,3
role=decode : RDMA={"0":"ionic_0",…,"3":"ionic_3"}  MC_TE_FILTERS=ionic_0..3  GPUs 4,5,6,7
   (both roles now resolve to the SAME ionic_0-3 map; AITER root per role:
    prefill=/tmp/aiter-jit-yihou-sn-prefill  decode=/tmp/aiter-jit-yihou-sn-decode)
both: DECODE_EXTRA_ARGS=--disable-custom-all-reduce  DSA_*=tilelang  DSA_TOPK_BACKEND=empty(drop)
      PREFILL_HICACHE=0 DECODE_HICACHE=0  DECODE_MTP=1 steps5 topk1 draft6
      DECODE_SIMULATE_ACC_LEN=3.61  JSON_MODEL_OVERRIDE_ARGS={"index_share_for_mtp_iteration":false}
      IMAGE=infera-sglang:v0519-yihou-0917-nextnfix-hicache  CONTROL_NODE=276
      TP4/DP4  mem_fraction 0.85  AGENTX_DURATION=1200 warmup/lane=1  SGLANG_OPT_USE_TOPK_V2=false
override: RDMA_DEVICE=custom_override exported first  → wins (custom_override)
empty   : DECODE_SIMULATE_ACC_LEN= exported first     → engine.sh:149 sim OFF
```

### Ports (launch.sh: base + row index)

| role | index | engine | bootstrap | kv-event | snapshot |
|---|---|---|---|---|---|
| prefill | 0 | 29001 | 28998 | 25557 | 28801 |
| decode  | 1 | 29002 | 28999 | 25558 | 28802 |

Distinct per role, as `launch.sh:39-43` derives them.

## Node facts verified first-hand on 276 (via spur exec)

- data IP `10.245.152.249` is on `ens3`; `engine.sh:82-83` auto-resolves the NIC to
  `ens3` (so `DATA_NIC` is left unset, matching the cross-node reference).
- `/lib/x86_64-linux-gnu/libionic.so` present → `HOST_RDMA_LIB` mount (config.sh
  default) is satisfiable.
- model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` present.

## Extra ssh call-site coverage (leader follow-up, all verified)

1. **`stop.sh:33` / `preflight.sh:45` use `ssh -n`.** The shim treats `-n` as a
   bare flag (opt `n` not in the takes-arg set) and additionally sets stdin from
   `/dev/null`; it is not mistaken for the host. Verified: `ssh -n … 276 'exit 7'`
   → rc=7; `ssh -n … 276 'cat; echo X'` → `X`, rc=0, no hang.
2. **`agentx_bench.sh:70`** — `bash -lc "pgrep -cf '(^|/)aiperf profile ' || true"`
   (login shell + nested single quotes). Sending it the way `ssh_run` actually
   does — `printf '%q'` per argv item, one joined string — through the shim
   returns `0` on stdout, rc=0 (the expected "no orphans"). The nested quotes and
   the `|| true` survive the single remote parse intact.
3. **`agentx_bench.sh:9` `REPO="$DIR/../.."` → `<ws>`**, mounted `-v REPO:REPO` in
   the client container on 276. Verified `spur exec … ls -d <ws>` succeeds (it is
   under the NFS `/home`), and `cd <ws>/scripts/bench-harness && cd ../.. && pwd`
   resolves to `<ws>`. InferenceX/OUT_DIR/cache defaults sit under `<ws>` so the
   single REPO mount covers them; the `/shared_nfs` model gets its own mount.
4. **`tools/agentx_env.py:38-41` and `tools/wait_healthy.py:25-29`** both call bare
   `ssh` via `subprocess.run(["ssh", *shlex.split(opts), node, shlex.join(cmd)])`
   — argv list, command as one string. They find the shim on `PATH`; verified end
   to end (`docker inspect … yzhou_model` → `running`, rc=0). `--ssh-options` is
   `shlex.split` so the `-o …` flags arrive as flags the shim skips.

Also noted: `stop.sh` runs `rocm-smi --showpids --showmemuse` once per topology
row, so with two rows on one node it runs twice on 276 — harmless, just duplicated
output.

## Same-node defects found and fixed

Three defects surface only when prefill and decode share one host. Each is fixed in
the vendored harness and is inert (default no-op) cross-node.

1. **Topology uniqueness check** (`tools/topology.py`, §1) — rejected two rows with
   the same node/IP. Relaxed to reject only a duplicated `(role,node,ip)` row.
2. **Engine port-block overlap** (`launch.sh`, `ENGINE_PORT_STRIDE`). SGLang derives
   an internal port block from `--port` (a fixed `ZMQ_TCP_PORT_DELTA` offset plus
   `port_base+0..N-1`). With two legs one port apart on one host those blocks overlap
   and the second leg dies at `zmq.error.ZMQError: Address already in use
   (addr='tcp://127.0.0.1:29236')`. Fixed by striding the *engine* port only
   (`ENGINE_PORT_BASE + index*ENGINE_PORT_STRIDE`; default 1 → cross-node byte-for-
   byte unchanged; `256` in this config). Confirmed necessary in production.
3. **RCCL start race (this task).** `launch.sh` started both rows back-to-back; on
   one host the two legs' RCCL communicators initialise concurrently and one dies:
   `rccl .../p2p.cc:256 NCCL WARN hipIpcGetMemHandle failed : invalid argument` /
   `NCCL error: unhandled cuda error`. It is a RACE, not a config error — the failing
   leg follows start ORDER, not the GPU set or the role:

   | round | node | started 1st | started 2nd | leg that failed |
   |---|---|---|---|---|
   | 7 / 8 | 276 | prefill GPUs 0-3 | decode GPUs 4-7 | **prefill** |
   | 9 | 276 | prefill GPUs 4-7 | decode GPUs 0-3 | neither |
   | 15 | 137 | prefill GPUs 0-3 | decode GPUs 4-7 | **decode** |

   Rounds 7/8 and 15 have the identical GPU assignment yet a *different* leg fails, so
   it is neither the GPU set nor the role; cross-node the two inits are on different
   machines and cannot race. By-hand fix (round 16): start decode only after prefill
   answers `/health` 200 → zero segfaults, both legs `fired up`.

   **Codified:** `launch.sh` now serialises same-node startup. Before starting a row
   whose node already has a started leg, it polls that leg's `/health` via
   `wait_healthy.py` (the same poller, not a second one). Design choice — health-gate
   over a fixed sleep: model-load time varies with page cache, so a sleep is a guess;
   the health poll adapts and is exactly the verified round-16 behaviour. Knobs:
   `SAME_NODE_START_GATE=health|off` (default `health`); `ENGINE_START_STAGGER_S`
   (default 0) adds a fixed same-node sleep for operators who prefer a delay. The gate
   fires **only** when a node repeats in the topology, so cross-node is behaviourally
   unchanged. Verified by a stubbed logic dry-run (no launch): same-node inserts a
   `/health` poll of the prefill leg before the decode start; cross-node starts both
   legs back-to-back with no gate; `off` disables it; `stagger=2` adds the sleep then
   gates. The end-of-loop `wait_healthy` over all legs is unchanged.

## Same-node shared-resource survey (leader follow-up)

With both containers on one host, what do they share and where can they collide?
Surveyed `engine.sh`'s docker args first-hand:

- **Writable shared HOST path — only the AITER JIT cache.** `engine.sh:102-103`
  bind-mounts exactly three host paths: `$MODEL:ro` and `$HOST_RDMA_LIB:ro`
  (read-only, no collision) and `$aiter_jit_cache → /aiter-jit` (writable). The
  AITER cache is the *only* writable host path and the *only* thing under host
  `/tmp` either container touches. It was per-machine before; on one host the two
  legs would share `/tmp/aiter-jit-$(id -u)/<image_key>` and race on first compile.
  **Fixed:** `AITER_JIT_CACHE_ROOT` is now per role
  (`/tmp/aiter-jit-yihou-sn-{prefill,decode}`). No other host `/tmp` is mounted, so
  every other `/tmp` write (Triton, torchinductor, HIP/rocm caches) lives inside
  each container's private filesystem and **cannot** collide.
- **`--ipc host` (`engine.sh:96`) → shared SysV IPC + `/dev/shm`.** The two
  containers join the host IPC namespace, so POSIX/SysV shared-memory segment names
  (torch/nccl/sglang) are a theoretical collision surface. Unchanged: this was
  identical in cross-node runs (one container per node), sglang keys its shm by
  port, and altering it is a separate variable. Flagged as a watch-item, not a
  demonstrated collision.
- **`--network host` + identical `INFERA_NODEPORT_RANGE=30000-32767` on both legs
  (`engine.sh:131`).** The *fixed* ports (engine/bootstrap/kv/snapshot) are already
  distinct per row (base+index), but the *dynamic* nodeport range is shared, so two
  engines on one host could draw the same ephemeral port. Whether infera coordinates
  allocation (via etcd) to avoid that is **not verified first-hand** — suspended. If
  bring-up hits a bind error in 30000-32767, the mitigation is to split the range
  per role (e.g. prefill 30000-31383 / decode 31384-32767). Not pre-emptively
  changed — one variable at a time.

## Unresolved risks (suspended, not concluded)

- **Cross-socket decode DMA (accepted, not a risk to correctness).** Shared
  `ionic_0-3` means decode GPUs 4-7 (NUMA1) DMA over NUMA0 NICs. Accepted for a
  working run; a performance deviation, recorded, not optimised now. The earlier
  "broken rail symmetry" risk is **resolved** — the perftest showed different-device
  transfers move no traffic, so identical device lists are the *only* working
  choice, not a compromise.
- **Image on 276.** `infera-sglang:v0519-yihou-0917-nextnfix-hicache` transfer is a
  separate teammate's task; not verified present here.
- **check_nodes.sh** unusable through the shim (see §2). Not on the launch path.

No launch was performed. Bring-up is the leader's call.
