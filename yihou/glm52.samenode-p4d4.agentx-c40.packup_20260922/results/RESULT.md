# RESULT — GLM-5.2 same-node 1P1D P4D4, AgentX CONC=40 fast

**Phase A deliverable. Achieved 2026-09-22 on `crsuse2-m2m-137`.**

Both the prefill leg and the decode leg ran on **one machine**. Raw artifact:
`rounds/017-agentx-c40/out/agentx_conc40.json`.

## The deployment

| | |
|---|---|
| node | `crsuse2-m2m-137`, data IP `10.245.153.247` — **both legs, one host** |
| prefill | GPUs 0-3, TP4 / DP4, DP-attention, `ionic_0-3`, engine port 29001 |
| decode | GPUs 4-7, TP4 / DP4, DP-attention, MTP EAGLE 5/6/topk1, `ionic_0-3`, engine port 29257 |
| router | `10.245.153.247:28000`, kv-aware, `PD_DP_RANK_AFFINITY=1` |
| image | `infera-sglang:v0519-yihou-0917-nextnfix-hicache` |
| model | GLM-5.2-MXFP4, `mem_fraction_static` 0.85, `kv_cache_dtype` fp8_e4m3 |
| HiCache | **OFF** on both legs (Phase A) |
| acceptance | **simulated, `SGLANG_SIMULATE_ACC_LEN=3.61`** |
| decode extra | `--disable-custom-all-reduce` (mandatory for this P4D4 shape) |

`/v1/workers` confirmed both workers on the same IP:
`10.245.153.247:29001` (prefill) and `10.245.153.247:29257` (decode).

## The numbers

| metric | value |
|---|---|
| concurrency | 40 |
| duration | 1,226.8 s |
| requests profiled | **843** |
| aiperf request errors | **0 / 843 = 0.000 %** |
| `records_error_dropped` | 2 (`InvalidInferenceResultError`) |
| **total throughput** | **82,291 tok/s** |
| **per-GPU throughput** | **10,286 tok/s/chip** (8 GPUs) |
| input throughput | 81,685 tok/s |
| output throughput | 607 tok/s |
| TTFT p50 / p90 / p95 | 15.03 s / 64.73 s / 128.04 s |
| E2E latency p50 / p90 | 22.14 s / 88.45 s |
| ITL p50 / p90 | 10.97 ms / 13.41 ms |
| interactivity p50 | 91.2 |
| input tokens p50 | 95,409 |
| output tokens p50 (actual) | 346 |
| server GPU cache hit rate | 87.7 % |
| KV pool | 18,034,688 tokens, 98 % peak usage |

## `spec_accept_length` — all four decode ranks

| dp_rank | `spec_accept_length` | `spec_accept_rate` |
|---|---|---|
| 0 | 3.725 | 0.545 |
| 1 | 3.575 | 0.515 |
| 2 | 3.450 | 0.490 |
| 3 | 3.625 | 0.525 |

Mean ≈ **3.59**, consistent with the forced 3.61. **All four ranks moved**, so
none of these is the idle-gauge artifact the mission warns about.

## SCOPE LIMITS — read before quoting any number

1. **Correctness is waived by construction.** `SGLANG_SIMULATE_ACC_LEN=3.61`
   forces the accept *count*, not which tokens are right, so the deployment
   emits garbled text — the smoke test returned `1!Sans!Sans!…`. The
   acceptance gauge reading ≈3.59 carries **no** correctness information; it
   reports the value it was told to report. These are timing numbers under
   forced acceptance, comparable to the 20260920 packups, **not** to a
   correctness run.
2. **The node is 137, not 276.** The mission named `crsuse2-m2m-276`; a
   controlled differential proved 276 segfaults this workload while 137 runs it
   clean, on identical kernel/driver/ROCm. The user approved the move. See
   `working_process.md` rounds 12-14.
3. **`is_multinode: true` in the JSON is wrong.** AgentX infers it; both legs
   were on one host. Trust the deployment table above and `/v1/workers`.
4. **One non-reference setting is carried:**
   `index_share_for_mtp_iteration=false` (the `config.sh` default), needed to
   avoid the issue.md §3.3 memory fault.
5. **The start stagger is load-bearing.** The two legs must not initialise their
   RCCL communicators concurrently — see the defect list below. This run
   staggered them by hand; `launch.sh` now gates on `/health`.
6. **No cross-node control at this exact config.** 136 and 138 were occupied by
   other tenants, so the same-node numbers are not paired with a cross-node run
   of the identical harness.

## The three same-node defects this run had to solve

All three are invisible cross-node and all three are fixed in the **vendored**
harness only; the tracked repo is untouched.

1. **`tools/topology.py:load()` rejected the shape** — it raised on
   `node in nodes or data_ip in ips`, and a single host offers one of each.
   Relaxed to reject only an exactly-duplicated `(role, node, ip)` row.
2. **SGLang's internal port block overlapped.** `server_args.py:836,846-861`
   derives `port_base` from `--port` and reserves
   `port_base+0..NUM_DERIVED_PORTS-1`. `launch.sh` handed out
   `ENGINE_PORT_BASE+index`, so the legs got 29001/29002 and overlapped 9 of 10
   ports; prefill died at `zmq.error.ZMQError: Address already in use
   (addr='tcp://127.0.0.1:29236')`. Fixed with `ENGINE_PORT_STRIDE` (default 1;
   256 here).
3. **The two legs' RCCL inits race.** Concurrent 4-GPU communicator init on one
   host kills one leg with
   `rccl .../p2p.cc:256 NCCL WARN hipIpcGetMemHandle failed : invalid argument`.
   Proven a race, not a misconfiguration: rounds 7/8 and 15 have the **identical**
   GPU assignment yet a **different leg** fails. Fixed by gating each leg's start
   on the previously-started leg's `/health`.

## What KV transfer actually uses — and why the rail question is moot

Same-host KV does **not** ride RDMA. Mooncake advertises the GPU segment as
`"rdma,hip"` regardless of the `MOONCAKE_PROTOCOL` string (proven by a
discriminator transfer: a client with **zero** RDMA transport read a segment a
server had registered under `protocol="rdma"`), `isHipReachableTarget` compares
the **port-stripped host** of the segment name so both legs match, and
`hip` outranks `rdma` (4 > 2). So KV moves over **GPU-IPC on XGMI**.

Consequences, both recorded as hard rules:

- **Never set `MC_DISABLE_HIP=1`** — it would force the legs onto RDMA.
- **Never set `MC_USE_HIP_IPC=0`** — fabric/VMM mode segfaults `register_memory`
  at *every* buffer size on this stack.
- `MC_DISABLE_HIP_TRANSPORT=1` in `config.sh:89` is a **dead name** — absent
  from the binary, so it is a no-op. Do not "fix" it into the real spelling.

This also makes the measured ionic limitation irrelevant **to this shape**:
same-host RDMA between two *different* ionic devices moves zero traffic
(reproduced on 276 and 137), but same-node KV never touches the NIC.
