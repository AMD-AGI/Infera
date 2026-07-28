# HANDOFF — PD + MTP decode hang (Bug 2, open)

State as of 2026-07-28 ~10:40 UTC. Written to survive a context compact: everything
needed to resume is here, no prior conversation required.

---

## One-line status

Bug 1 (the `lengths.size(0) == B` top-k crash) is **FIXED and shipped**. A **second,
independent defect** blocks PD when MTP runs on the decode leg: the server boots clean and
the crash is gone, but the **first routed request deadlocks**. Not fixed. Not started.

## Live cluster state (verify before trusting — nodes may have been released)

| What | Value |
|---|---|
| Job / node A | `4540` → `crsuse2-m2m-207`, IP **10.245.156.172** — container `dbg` |
| Job / node B | `4614` → `crsuse2-m2m-197`, IP **10.245.158.91** — container `pd_spur` |
| Currently running | **PD DPA-only** (the PASSING config): prefill on 197, decode on 207, both healthy (`health=200`), 10 schedulers each |
| Router | in `pd_spur`, last good on port **8006** (`/tmp/router5.log`) |
| Patch applied? | **YES** in both containers (`/tmp/apply_fix.py` was run; backup at `dsa_indexer.py.orig`) |

```bash
export DOCKER_CONFIG=/tmp/dockercfg    # before EVERY docker call
spur exec 4540 bash -c "... docker exec dbg ..."       # decode / mix node
spur exec 4614 bash -c "... docker exec pd_spur ..."   # prefill node
```

## What is PROVEN about the hang

**It is not Bug 1.** The `RuntimeError: Expected lengths.size(0) == B` never reappears
under PD with the patch. The decode leg completes the full disaggregation warmup —
all 8 DP ranks return 200 with `spec_accept_length: 2.0`, `spec_verify_ct: 4` — and prints
`The server is fired up and ready to roll!` (`logs/pd_decode_30001_dpa_mtp.log`,
`logs/pd_decode_v3.log`). It then hangs on the first *routed* request. No crash, no error,
no timeout — a hard deadlock.

**It is not the transport.** Hung run vs passing run are byte-identical on transport
(`evidence/transport_evidence.txt`): both `disaggregation_ib_device='mlx5_0'`, 8× rdma,
0 hip, 0 tcp, **0 ionic** (all 9 RDMA devices visible; mooncake picked only mlx5_0),
0 `KVTransferError`. The DSv4 kit's classic traps (hip IPC across PD instances, ionic
instability, dmabuf compiled out) all do not apply.

**It is not the patch.** Same patched image + same script, flipping only `MTP=1`→`MTP=0`
on the decode leg → PD passes 4/4 and 256/256. The same patched code also runs 256/256
single-node with DPA+MTP fused.

**The stall signature** (`py-spy dump` on all 8 ranks, sampled twice 6 s apart, zero
movement — this is the key artifact):

```
DP0,3,5,7:  broadcast              (torch/distributed/distributed_c10d.py:2841)
DP2,4,6:    all_gather_into_tensor (torch/distributed/distributed_c10d.py:4056)
DP1:        init_forward_metadata  (dsa_backend.py:746)
              prepare_for_draft_extend  (base_spec_worker.py:163)
              _draft_extend_for_decode  (eagle_worker_v2.py:921)
              forward_batch_generation  (eagle_worker_v2.py:1259)
              event_loop_overlap_disagg_decode (decode.py:1848)   <- PD-specific event loop
```

7 ranks are inside a collective; DP1 never arrives. Classic ragged-collective deadlock.

## The most promising lead (not yet tested)

`base_spec_worker.py:161-163` — `init_forward_metadata` is called **conditionally**:

```python
can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(forward_batch)
if not batch.forward_mode.is_idle() and not can_cuda_graph:
    draft_model_runner.attn_backend.init_forward_metadata(forward_batch)   # <- DP1 stuck in here
```

**Hypothesis:** `can_run_graph()` and/or `is_idle()` evaluate **differently per DP rank**
(different batch size / seq lens / idle-ness on that rank). Ranks where the condition is
False skip straight to the collective; DP1 takes the eager path and blocks inside
`init_forward_metadata` at `dsa_backend.py:746`:

```python
max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
```

`.item()` is a GPU→CPU sync. If the peers are already in a collective holding the stream,
that sync cannot retire → deadlock. Under DP-attention this branch **must** be taken
uniformly across ranks or not at all.

Why PD and not single-node: PD decode uses a different event loop
(`event_loop_overlap_disagg_decode`, `decode.py:1848`) and batches are shaped by KV
arrival from the prefill leg, so per-rank divergence in batch shape/idleness is far more
likely than in the single-node overlap loop.

**Corroborating comment right below** (`base_spec_worker.py:164-167`) — upstream already
knows DSA + DP padding is fragile here:

> *"Planned pre-pad; do NOT opt into post-pad re-plan. DSA's indexer cannot rebuild its
> deep_gemm schedule_meta on a DP-padded batch (the `_batch_size == batch_size`
> assertion, see #27091)"*

**sglang issue #27091 is the thing to read first.**

## Concrete next steps (in order)

1. **Confirm the divergence.** Add a rank-tagged log right before the `if` in
   `base_spec_worker.py:161`: print `dp_rank, forward_mode, is_idle(), can_cuda_graph,
   bs, seq_lens.shape`. Re-run PD with MTP; the expectation is that the ranks split
   exactly along the `can_cuda_graph` value, matching the py-spy split (DP1 vs the rest).
   *This is a 1-line instrumentation and settles the hypothesis outright.*
2. **If confirmed** — make the branch collective-uniform. Options, cheapest first:
   a. Force `can_cuda_graph=False` on all ranks under PD+DPA (all take the eager path).
   b. All-reduce the predicate across the DP group before branching, so every rank agrees.
   c. Avoid the `.item()` sync in `dsa_backend.py:746` on this path — `seq_lens_cpu` is
      already a host mirror; if it is reliably populated, the `.max().item()` on the GPU
      tensor in the `else` branch is what to eliminate.
3. **If not confirmed** — bisect the batch: reproduce with `--dp-size 2` (much smaller
   surface, faster boot) and dump all ranks' `forward_batch` shapes at the stall.
4. Check whether `--speculative-attention-mode prefill` (vs the default `decode`) changes
   the dispatch — it selects a different backend for draft-extend
   (`dsa_indexer.py::_uses_dsa_attention_backend`).

## How to reproduce the hang

Full commands in `REPRODUCE.md` §5. The short version — same as the passing PD run but
with `MTP=1` on the decode leg:

```bash
# decode (node 207), the ONLY change vs the passing config is MTP=1:
docker exec -d dbg env ROLE=decode MY_IP=10.245.156.172 P_IP=10.245.158.91 \
  MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 PORT=30001 \
  DPA=1 MTP=1 DMABUF=1 CTX=32768 GMU=0.80 TORCHINDUCTOR_COMPILE_THREADS=4 \
  LOG=/home/yihou/glm52_fix/logs/pd_decode_mtp_debug.log bash /pd_leg_spur.sh
# wait ~10 min for "ready to roll", start a router on a FRESH port, then probe -> hangs
```

Diagnose with:
```bash
docker exec dbg bash -c 'for p in $(ps -eo pid,args --no-headers | \
  grep -oE "^ *[0-9]+ .*sglang::scheduler_DP[0-9]" | awk "{print \$1}"); do
    N=$(ps -p $p -o args= | grep -oE "DP[0-9]"); \
    echo "$N: $(py-spy dump --pid $p 2>&1 | sed -n 5p | xargs)"; done'
```
(py-spy is already installed in `dbg`.) Sample twice ~6 s apart — identical output = hang.

## Traps that will cost you a boot cycle if forgotten

1. **Never `pkill -f launch_server`** — it orphans the scheduler tree, leaks ~82% VRAM,
   and the next server wedges mid-boot with all ranks alive but the detokenizer silent.
   Use `docker rm -f <ctr>` + recreate; confirm `rocm-smi --showmemuse` reads 0% first.
   Re-`docker cp` the scripts and re-run `apply_fix.py` after recreating.
2. **`TORCHINDUCTOR_COMPILE_THREADS=4` is mandatory on a cold Inductor cache.** Without
   it, 8 DP ranks spawn 264 compile workers on 236 cores and deadlock during warmup in a
   *different* place (DP0-2 in `synchronize` inside Inductor, DP3-7 at a collective,
   stalled compiling `@torch.compile`'d `select_top_k_tokens`, `spec_utils.py:274`).
   Do not confuse that boot-time deadlock with this request-time one.
3. **Router:** fresh `--port` AND `--prometheus-port` on every restart; `pkill -9 -f
   launch_router` first. Once the router marks decode unhealthy it will not re-register —
   restart the router after decode reports `health=200`.
4. Cold start is ~9-10 min. 8 live `sglang::scheduler_DP*` = working, not hung.

## Unverified oddity (may be nothing, may bite later)

Scheduler subprocesses carry **no `MC_*` env vars at all** — the launcher has all four
(`MC_GID_INDEX`, `MC_MS_FILTERS`, `MC_MS_AUTO_DISC`, `MC_DISABLE_HIP_TRANSPORT`) but
`/proc/<scheduler pid>/environ` has zero. Mooncake initialises *inside* the scheduler and
still picked mlx5_0 + rdma correctly (presumably via `--disaggregation-ib-device`, not
env). Identical in both the hung and passing runs, so it does not explain this bug — but
on a node where mlx5 is not the default pick it could matter. Not investigated.

## Files

- Hung-run logs: `logs/pd_decode_30001_dpa_mtp.log`, `logs/pd_decode_v3.log`,
  `logs/pd_decode_v2.log` (v2 = the run killed by the Inductor deadlock)
- Passing PD run: `logs/pd_decode_dpaonly.log`, `logs/pd_prefill_30000.log`,
  `results/pd_dpaonly_c64.{log,jsonl}`
- Transport comparison: `evidence/transport_evidence.txt`
- Bug 1 (fixed) analysis: `NOTES_rootcause_and_fix.md`
- Shipped patch: `../../deploy/docker/patches/sglang_dsa/` in the `infera` repo,
  branch `worktree-dsa-hip-dp-rows-fix`, commit `9bcec49`
