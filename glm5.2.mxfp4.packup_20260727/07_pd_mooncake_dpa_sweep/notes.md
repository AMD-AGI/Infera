# Notes — 07 PD mooncake RDMA + DP-attention sweep

## What this adds over 03

03 proved GLM-5.2 mooncake **RDMA** PD works at conc=64 with **pure TP8** (no DP-attention). 07 turns
DP-attention ON and pushes to conc=2048. The reference for the high-conc recipe is the DSv4 R4 sweep
(`legacy.infera/infera.fuck/sglang_single_r4_20260707_080726`), whose key finding transfers directly:

> Pure TP8 satisfies low/mid concurrency but **collapses at conc≥256** (DSv4: c256 dropped to 50% of
> baseline). The fix is EP + **DP-attention** (`--dp8 --enable-dp-attention --ep-size 8`), after which
> c256→c1024 scale linearly. Also needs `SGLANG_DP_USE_GATHERV=1`, `--enable-prefill-delayer`
> (`--prefill-delayer-max-delay-ms 5000`), and `--chunked-prefill-size = ISL × TP`.

We keep the **GLM-5.2 DSA-ROCm recipe** (tilelang indexer, `--kv-cache-dtype fp8_e4m3`, NO
`--attention-backend dsv4`, NO page-256 — those are DSv4-specific and fight GLM's auto-config) and
graft only the DP-parallelism structure on top. So 07 = (03's GLM DSA + mooncake RDMA env) + (R4's DP
parallelism args).

## Symmetric DP on both legs is mandatory (not a choice)

mooncake transfers the KV cache tensors shard-for-shard from prefill to decode. The shard layout is a
function of `dp_size × tp_size`. If prefill ran DP8 and decode ran TP8-only (or vice-versa), the KV
tensor each rank produces would not line up with the rank that consumes it → wrong KV / crash. So both
legs run the identical `--dp-size 8 --enable-dp-attention --ep-size 8`. The prefill-delayer is the only
asymmetry (prefill-only), and that's fine — it batches *incoming prefills*, it doesn't touch KV layout.

## Capacity knobs — the parameters the task explicitly called out

The user said "注意配置 max_req_seq 和 context Len 等影响产能的参数". Concretely:

- **`--context-length 32768`** (down from the 400000 used in phases 1-3). The 1k/1k workload needs
  ≤2k tokens/request; a 400k context just reserves KV headroom we'll never use and *shrinks the usable
  request slots*. Lowering it is what lets the KV pool hold ~1550 concurrent 2k-token requests.
- **`--max-running-requests 2048`** — the scheduler's concurrent-request cap. Sized to the top of the
  ladder so no point is throttled by the server cap (the client `--max-concurrency` is what actually
  drives load per point).
- **`--chunked-prefill-size 65536`** = ISL(8192) × TP(8): the DP recipe wants one prefill chunk to
  span the whole DP group.
- **`--cuda-graph-max-bs 128`**: capture graphs up to bs 128; larger decode batches replay eager
  (correct, slightly slower — acceptable at the very top of the ladder).
- **`--mem-fraction-static` 0.88 prefill / 0.85 decode**: DP-attn prefill at high conc is prone to an
  HSA allocator OOM, so DP prefill wants a *lower* memfrac than a colocated server would (memory note
  from the DSv4 DP work). Decode slightly lower still.

Result: prefill KV pool **3.26M tokens**, decode **3.10M tokens**. Peak observed KV usage across the
whole sweep was **0.15** (15%) — KV was never the bottleneck; compute/queueing was.

## The conc=2048 "dip" is saturation, not failure

Throughput climbs monotonically 64→1024 (2188 → 12855 out tok/s) then dips at 2048 (11970). This is the
expected saturation knee, not an error:

- At conc=1024 the decode leg runs near its efficient batch size; adding more in-flight requests past
  ~1024 stops adding parallel work and just deepens the queue.
- conc=2048 submits 4096 prompts against a decode pool that steady-states around ~1500 × 2k-token
  slots, so median TTFT jumps to 17.5 s and p99 to 40 s (requests wait for a slot). But **all 4096
  complete** — no drops, no retracts.
- If you needed conc=2048 to also *improve* throughput, you'd scale out decode (more decode nodes /
  1P2D) rather than push a single decode leg past its knee.

## The "5257 retract / abort / watchdog" false alarm

A naive `grep -icE 'retract|abort|Watchdog'` on the decode log returns 5258 — alarming at first. It is
**entirely spurious**:

- Every `Decode batch` log line contains the field `#retracted-req: 0`, so `grep retract` matches
  5257 benign status lines. Filtering for a *non-zero* count (`#retracted-req: [1-9]`) returns **zero**.
- The 1 `abort` hit is the `abort_on_priority_when_disabled=False` server-arg string; the 1 `Watchdog`
  hit is the `--watchdog-timeout` arg echo. Neither is a runtime event.

Real error scan (`KVTransferError`, `session not alive`, `CUDA error`, `out of memory`, non-zero
retractions) = **0 across the entire sweep**. Lesson: on sglang decode logs, always filter retractions
by non-zero value, not by substring.

## Gotcha — `docker exec -d … bash -lc '…'` does not persist

Launching a leg (or the router) with `docker exec -d $CTR bash -lc 'ENV=… bash /script'` silently
fails to keep the process alive — no log file is ever created. The reliable form is
`docker exec -d $CTR env VAR=… VAR2=… bash /script` (pass env via `env`, invoke the script directly).
For the router (which needs shell redirection) write a tiny `/run_router.sh` inside the container and
`docker exec -d $CTR bash /run_router.sh`. Both `up_dpa.sh` and REPRODUCE.md use the working forms.

## Transport

Same as 03 — real mooncake RDMA, dmabuf OFF (`MOONCAKE_DISABLE_HIP_DMABUF=1`, bare ibv_reg_mr +
amdgpu peermem), `MC_DISABLE_HIP_TRANSPORT=1`, all 8 ionic NICs, `RDMAV_FORK_SAFE=1`. Confirmed via
8× `rdma_context.cpp HIP dmabuf disabled` in the prefill log and 0 `MC_FORCE_TCP`. DP-attention runs
cleanly across the RDMA PD boundary at every concurrency point.
