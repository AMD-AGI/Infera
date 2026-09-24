# Same-node 1P1D P4D4 — GLM-5.2 MXFP4, AgentX CONC=40 fast

**Both the prefill leg and the decode leg on ONE machine.** Every previous
GLM-5.2 P/D run in this repo was cross-node; this kit is the first same-node
one, and it exists because machines got scarce.

One AgentX fast point at concurrency 40, 1,226.8 s, on `crsuse2-m2m-137`
(MI355X ×8), GLM-5.2-MXFP4, prefill GPUs 0-3 / decode GPUs 4-7, TP4+DP4 with
DP-attention on both legs, decode MTP EAGLE (5 steps / 6 draft / topk 1), KV over
mooncake, HiCache **off**, **simulated acceptance `SGLANG_SIMULATE_ACC_LEN=3.61`**.
Run finished 2026-09-22 10:19 UTC.

| metric | value |
|---|---|
| requests profiled | **843** |
| aiperf request errors | **0 / 843 = 0.000 %** |
| `records_error_dropped` | 2 (`InvalidInferenceResultError`) |
| **total throughput** | **82,291 tok/s** |
| **per-GPU throughput** | **10,286 tok/s/chip** (8 GPUs) |
| TTFT p50 / p90 | 15.03 s / 64.73 s |
| ITL p50 | 10.97 ms |
| interactivity p50 | 91.2 |
| `spec_accept_length` (4 decode ranks) | 3.725 / 3.575 / 3.450 / 3.625 → mean **3.59** |
| server GPU cache hit rate | 87.7 % |

## What this kit actually establishes

**Same-node 1P1D works, and it takes three fixes that are invisible cross-node.**
All three were found the hard way, and all three live in the **vendored** harness
under `scripts/bench-harness/` — the tracked repo is untouched.

1. **`tools/topology.py:load()` rejects the shape.** It raised on
   `node in nodes or data_ip in ips`, and one host offers exactly one of each.
   Relaxed to reject only an exactly-duplicated `(role, node, ip)` row.
2. **SGLang's internal port block overlaps.** `server_args.py:836,846-861`
   derives `port_base` from `--port` and reserves
   `port_base+0..NUM_DERIVED_PORTS-1`. `launch.sh` handed out
   `ENGINE_PORT_BASE+index`, so the legs got 29001/29002 — **9 of 10 ports
   overlapped** — and prefill died at
   `zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:29236')`.
   Fixed with `ENGINE_PORT_STRIDE` (default 1, so cross-node is unchanged; 256 here).
3. **The two legs' RCCL inits race.** Concurrent 4-GPU communicator init on one
   host kills one leg with
   `rccl .../p2p.cc:256 NCCL WARN hipIpcGetMemHandle failed : invalid argument`.
   Fixed by gating each leg's start on the previously-started leg's `/health`.

**And it establishes what same-node KV transfer really does:** it does **not**
use RDMA. Mooncake advertises the GPU segment as `"rdma,hip"` regardless of the
`MOONCAKE_PROTOCOL` string, matches the **port-stripped host**, and prefers
`hip` (priority 4) over `rdma` (2) — so KV moves over **GPU-IPC on XGMI**. Proven
by transfer, not inferred: a client with *zero* RDMA transport read a segment a
server had registered under `protocol="rdma"`.

## The result node is 137, not 276

The task named `crsuse2-m2m-276`. A controlled differential proved **276
segfaults this workload and 137 does not**, with kernel, amdgpu and ROCm
versions **identical** on both — so it is machine state, not software. The user
approved the move. Rounds 12-14 in `working_process.md` carry the evidence:
a solo leg on 276 still segfaults (co-location exonerated), a wiped JIT cache
still segfaults (cache exonerated), and the identical run on 137 serves.

`analysis/hip_ipc_limits.yihou.md` localises 276's crash to
`scheduler.py:1894` — `self.schedule_stream.cuda_stream ==
self.forward_stream.cuda_stream`, a bare native HIP stream-handle read. A
segfault *there* is the signature of an already-corrupted HIP context surfacing
at the next native touch. **Root cause on 276 is NOT established.**

## SCOPE LIMITS — read before quoting any number

1. **Correctness is waived by construction.** Simulated acceptance forces the
   accept *count*, not which tokens are right; the deployment emits garbled text
   (the smoke test returned `1!Sans!Sans!…`). The acceptance gauge ≈3.59 reports
   the value it was told to report and carries **no** correctness information.
   Comparable to the 20260920 packups, **not** to a correctness run.
2. **`is_multinode: true` in `agentx_conc40.json` is wrong** — AgentX infers it.
   Both legs were on one host; see `results/workers.json`.
3. **No paired cross-node control at this exact config** — 136 and 138 were
   occupied by other tenants.
4. **`index_share_for_mtp_iteration=false`** is carried (the `config.sh`
   default), needed to avoid the issue.md §3.3 memory fault.
5. **Round 9's "GPU swap fixed it" reading was wrong and is retracted** — the
   swap did not fix the race, it lost it differently. See `notes.md`.

## Navigation

| path | what is there |
|---|---|
| `results/RESULT.md` | THE numbers, the deployment table, and the scope limits |
| `results/agentx_conc40.json` | the raw AgentX artifact |
| `results/spec_accept.decode.txt`, `workers.json`, `server-info.*.json` | live runtime snapshots taken while the deployment was up |
| `REPRODUCE.md` | ordered, copy-pasteable reproduction |
| `environment.md` | hardware + software env, and the **137 vs 276 comparison** |
| `notes.md` | gotchas, wrong turns, and the retracted conclusions |
| `working_process.md` | every round, in order, including the dead ends |
| `spec/mission.md` | the task of record |
| `analysis/` | six investigation notes (mooncake same-host, upstream research, HIP IPC limits, harness, hardware, image transfer) |
| `scripts/bench-harness/` | the **patched** harness that ran this |
| `patches/` | the diffs vs the tracked repo |
| `logs/key-excerpts.md` | the exact log lines each conclusion rests on |

**Large artifacts are deliberately not packed** (user decision): the 576 MB
`server_metrics_export.json`, the 28 MB timeslices CSV, and the full per-round
server logs (up to 17 MB each) stay in the experiment workspace at
`bench/glm5p2_pd/results/yihou-samenode-p4d4/rounds/`, which is gitignored.
`logs/key-excerpts.md` carries the load-bearing lines.
