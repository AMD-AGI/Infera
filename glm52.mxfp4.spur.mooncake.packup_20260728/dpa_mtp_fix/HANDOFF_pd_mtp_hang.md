# HANDOFF — PD + MTP decode hang (Bug 2)

State as of 2026-07-28 ~11:35 UTC. Written to survive a context compact: everything
needed to resume is here, no prior conversation required.

---

## One-line status

Bug 1 (the `lengths.size(0) == B` top-k crash) is **FIXED and shipped**. Bug 2 (PD + MTP
decode deadlock) is **ROOT-CAUSED AND REPRODUCED**; a candidate fix is written and under
test. See "ROOT CAUSE — PROVEN" below; the older hypothesis sections are kept, marked
refuted, for the audit trail.

## ROOT CAUSE — PROVEN (2026-07-28 11:24 UTC)

**A DP-ragged branch in `base_spec_worker.prepare_for_draft_extend`.**

```python
can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(forward_batch)
if not batch.forward_mode.is_idle() and not can_cuda_graph:      # <-- PER-RANK predicate
    draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
```

`can_cuda_graph` is always False on this HIP build (draft-extend graph never captured),
so the branch reduces to `if not is_idle()`.

Rank-tagged instrumentation (`scripts/instrument_divergence.py`) gave a measured
invariant, **56/56 samples, zero mismatches**:

```
is_idle(rank) == (global_num_tokens_cpu[rank] == 0)
```

So the branch is taken by exactly the ranks that hold work this step. When every rank has
work — `gnt=[4,4,4,4,4,4,4,4]` — all 8 take it and the step completes (this is why boot,
warmup, and the 4-prompt probe all pass). When a PD decode step has work for a **subset**
— `gnt=[0,0,4,0,0,0,0,0]` — one rank enters `init_forward_metadata` and seven skip it.
That call is not collective-free (DSA indexer metadata + a `.max().item()` device sync),
so the busy rank blocks there while its peers run ahead into the next collective:

```
last probe line before the hang:   gnt=[0,0,4,0,0,0,0,0]   dp=2 eager=True, all others eager=False
py-spy, sampled twice, identical:  DP2:        init_forward_metadata (dsa_backend.py:746)
                                   DP1,3,4,5,7: all_gather_into_tensor
                                   DP0,6:       broadcast
```

The stuck rank **follows whichever rank owns the work** — DP1 in the original capture,
DP2 in the re-reproduction — confirming it is a ragged-collective race, not a fixed rank.

**Why single-node mix passes and PD fails:** in the mix loop every rank gets work each
step, so `gnt` is uniformly non-zero. On the PD decode leg batches are shaped by KV
arrival from the prefill leg, so partial-occupancy steps are routine.

**Candidate fix** (`scripts/fix_bug2.py`): derive the predicate from the
global state every rank already has, so no extra communication is needed —

```python
_gnt = getattr(forward_batch, "global_num_tokens_cpu", None)
_needs_metadata = max(_gnt) > 0 if _gnt else (not batch.forward_mode.is_idle())
if _needs_metadata and not can_cuda_graph:
    draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
```

Plus a required companion guard: idle ranks now arrive with a zero-row batch, and stock
`dsa_backend.init_forward_metadata` calls `seq_lens_cpu.max()` unguarded →
`RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0`
(observed: DP6 crashed exactly there on the first attempt). `fix_bug2.py` guards both
`max()` sites with a `numel() > 0` check.

### Fix status: PARTIAL — necessary but NOT sufficient

With both parts applied the server boots clean and all ranks do enter the call
(verified: the stuck rank's frame is `base_spec_worker.py:182`, the patched line), **but
the first routed request still deadlocks**:

```
DP0,2,3,4,6,7: all_gather_into_tensor
DP1:           broadcast
DP5:           init_forward_metadata (dsa_backend.py:752)
```

`dsa_backend.py:752` is the **`else` arm** of the `max_seqlen_k` branch — i.e. the
`seq_lens_cpu is None` path, which does `seq_lens.max().item()` on a **GPU** tensor
(a real device sync), versus the `if` arm's cheap host read.

### The surviving divergence — CONFIRMED (2026-07-28 12:26 UTC)

The arm probe (`scripts/instrument_v2.py`) caught it. Last step before the hang,
`gnt=[0,0,0,0,0,0,4,0]` — only DP6 holds work:

```
7 idle ranks: mode=IDLE            seq_lens_cpu_is_None=False numel=0 -> arm=HOST      (cheap host read)
DP6:          mode=DRAFT_EXTEND_V2 seq_lens_cpu_is_None=True  numel=1 -> arm=GPU_SYNC  (blocking .item())
```

and py-spy, sampled twice with zero movement:

```
DP6:              init_forward_metadata (dsa_backend.py:767)   <- seq_lens.max().item()
DP0,1,3,4,5,7:    all_gather_into_tensor
DP2:              broadcast
```

`dsa_backend.py:767` is literally `int(forward_batch.seq_lens.max().item()) + draft_token_num`.

**So the causal chain is now complete and each link is measured:**

1. A PD decode step gives work to a subset of DP ranks (`gnt` mostly zeros) — routine on
   the decode leg because batches are shaped by KV arrival from prefill.
2. `is_idle(rank) == (gnt[rank] == 0)` (56/56 samples) → the busy rank gets
   `DRAFT_EXTEND_V2`, the rest get `IDLE`.
3. Forward mode decides whether the host mirror exists: `IDLE` → `seq_lens_cpu` present;
   `DRAFT_EXTEND_V2` → `seq_lens_cpu is None` (`gpu_only`, `base_spec_worker.py:112`;
   see the "needs_cpu_seq_lens=False nulls the host mirror for spec-v2 relay batches"
   comment at `dsa_backend.py:763`).
4. Missing mirror forces the `else` arm → `seq_lens.max().item()`, a **GPU→CPU sync**,
   while the idle peers take the free host path and run ahead into the next collective.
5. The syncing rank can never retire its `.item()` because the peers hold the stream in a
   collective it has not joined → **hard deadlock**.

**Why `fix_bug2.py` was not enough:** it equalised *whether* ranks call
`init_forward_metadata`, but not *what they do inside it*. The expensive/blocking arm is
still selected per-rank, by forward mode.

**Implication for the real fix.** Making the branch uniform is not sufficient; the
`.item()` on this path has to go, or be made uniform. Cheapest ordering to try:

a. **Avoid the sync entirely on the spec-v2 relay path.** `max_seqlen_k` only sizes the
   page-table slice. A safe upper bound already known on the host (e.g. the batch's
   `seq_lens_cpu` before it was nulled, or `context_length`/page-table width) would remove
   the D2H read. This is the most promising: it deletes the hazard rather than
   synchronising around it.
b. **Restore the host mirror for `DRAFT_EXTEND_V2`** so both modes take the HOST arm —
   i.e. make `gpu_only` False on this path. Costs one H2D per step, which is what the
   `needs_cpu_seq_lens=False` optimisation was avoiding, so measure before adopting.
c. Force every rank onto the same arm (all sync, or none), which keeps the cost but
   removes the raggedness. Least attractive — it makes idle ranks pay a device sync.

Note (c) is what a naive reading of "make it uniform" suggests, and it would work, but (a)
is strictly better if a host-side bound is available.

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

## ~~The most promising lead~~ — REFUTED 2026-07-28 11:15 UTC

> **STATUS: the `can_cuda_graph` divergence hypothesis below is DISPROVEN.**
> Do not spend time on it. Evidence and what it means are in the next section;
> the original text is kept only so the reasoning trail is auditable.

`base_spec_worker.py:161-163` — `init_forward_metadata` is called **conditionally**:

```python
can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(forward_batch)
if not batch.forward_mode.is_idle() and not can_cuda_graph:
    draft_model_runner.attn_backend.init_forward_metadata(forward_batch)   # <- DP1 stuck in here
```

**Hypothesis (WRONG):** `can_run_graph()` and/or `is_idle()` evaluate **differently per DP rank**
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

## WHY THE ABOVE IS REFUTED (evidence, 2026-07-28 11:15 UTC)

`can_cuda_graph` **cannot** differ per rank here, because on this build it is
**always `False` on every rank**: the draft-extend CUDA graph is never captured at all.

```bash
# in BOTH the hung PD run and the PASSING single-node mix run:
grep -c "Capture draft extend CUDA graph begin" logs/pd_decode_30001_dpa_mtp.log  # -> 0
grep -c "Capture draft extend CUDA graph begin" logs/mix_dpa_mtp_fix1.log         # -> 0
grep -c "Capture draft decode CUDA graph begin" logs/pd_decode_30001_dpa_mtp.log  # -> 1
```

Only the draft **decode** graph is captured; the draft **extend** graph is not. Reason
(`eagle_worker_v2.py:441-482`): the capture is gated on

```python
supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and graph_supported_backend
supports_hip_aiter_draft_extend_graph = _is_hip and isinstance(
    self.draft_attn_backend, AiterMultiStepDraftBackend)
if self.draft_extend_attn_backend and (_is_npu or _is_xpu
        or supports_cuda_draft_extend_graph or supports_hip_aiter_draft_extend_graph):
```

On this stack `is_hip()==True` / `is_cuda()==False` (verified in-container), so
`supports_cuda_draft_extend_graph` is False; and with the DSA/tilelang draft backend
`draft_attn_backend` is **not** an `AiterMultiStepDraftBackend`, so
`supports_hip_aiter_draft_extend_graph` is False too. Net:
`cuda_graph_runner_for_draft_extend is None` → `can_cuda_graph` is falsy on all 8 ranks →
**every rank takes the eager `init_forward_metadata` path.** The branch is already
collective-uniform. It is not the divergence.

Corollary: fix option 2(a) from the old plan ("force can_cuda_graph=False on all ranks")
is a no-op — that is already the state.

**What the py-spy split then means.** Ranks were in *three* different places
(broadcast / all_gather_into_tensor / init_forward_metadata) — that is not "7 waiting for
1 straggler", it is ranks executing **different collective sequences**. Since they all
enter the same eager branch, the divergence happens either *inside*
`init_forward_metadata` or *before* draft-extend is reached (i.e. the ranks are not even
running the same batch/phase). That is what to instrument next.

Also note the batch itself IS DP-synced upstream: `get_next_disagg_decode_batch_to_run`
ends with `dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)`
(`disaggregation/decode.py`), which all-gathers `global_num_tokens` and emits idle
batches so every rank agrees on shape/mode. So plain "one rank got an empty batch" is
also unlikely — unless that sync is itself being skipped or is what one group is stuck in.

## Control experiment (rules out DPA and PD themselves)

Run 2026-07-28 11:05-11:15 UTC on the same patched image, same two-node PD, MTP **off**,
sustained load (not just a 4-prompt probe):

| conc | completed | out tok/s | req/s | median TPOT | median TTFT |
|---:|---:|---:|---:|---:|---:|
| 64  | 256/256   | 2051 | 2.00 | 27.3 ms | 2052 ms |
| 128 | 512/512   | 3743 | 3.66 | 31.2 ms | 2347 ms |
| 256 | 1024/1024 | 6243 | 6.10 | 36.8 ms | 3487 ms |

1792/1792 requests, zero failures, clean scaling. **PD + DP-attention + the Bug 1 patch
are healthy under real load.** MTP on the decode leg is the sole trigger.
(Raw: `results/ctrl_dpaonly_c{64,128,256}.jsonl`.)

## Concrete next steps (in order)

1. ~~Confirm `can_cuda_graph` divergence~~ — **done, refuted.** See above.
2. **Find where the ranks actually diverge.** Two probes are written and ready:
   - `/home/yihou/glm52_fix/instrument_divergence.py` — rank-tagged log of
     `(dp_rank, forward_mode, is_idle, can_cuda_graph, bs, global_num_tokens_cpu,
     can_run_dp_cuda_graph)` immediately before the `if` in `base_spec_worker.py:161`.
     Still useful: it now answers *"do all ranks even reach draft-extend, with the same
     mode and bs?"* rather than the refuted question.
   - `/home/yihou/glm52_fix/instrument_dsa_stall.py` — ENTER / GOT_MAXSEQLEN markers
     inside `dsa_backend.init_forward_metadata`, so the hung rank's **last** printed line
     names the exact blocking statement.
   Both are idempotent, verify their anchor matches exactly once, `py_compile` the result,
   and support `--revert`. Install with `docker cp` + `docker exec python3 /tmp/<probe>.py`.
3. If the ranks turn out to reach draft-extend with identical shapes, suspect the
   collective *inside* the DSA indexer / MoE a2a under `draft_tp_context` — compare the
   `draft_tp_context(self.draft_worker.draft_runner.tp_group)` group against the one the
   non-spec path uses; a spec-only subgroup that not every DP rank joins would produce
   exactly this three-way split.
4. Bisect with `--dp-size 2` (much smaller surface, ~faster boot) once a probe confirms
   the shape of the divergence.
5. Check whether `--speculative-attention-mode prefill` (vs the default `decode`) changes
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
2. **`TORCHINDUCTOR_COMPILE_THREADS=4` is NOT sufficient on a cold Inductor cache.**
   Without any limit, 8 DP ranks spawn 264 compile workers on 236 cores and deadlock
   during warmup in a *different* place from Bug 2 (DP0-2 in `synchronize` inside
   Inductor's `_make_launchers`, DP3-7 at a `broadcast`, stalled compiling
   `@torch.compile`'d `select_top_k_tokens`, `spec_utils.py:274`).
   Do not confuse that boot-time deadlock with the request-time one.

   **Observed 2026-07-28 12:11 UTC: it recurred WITH `=4` set**, in a freshly recreated
   container (cold cache), signature identical:
   ```
   DP0,1,2: synchronize (torch/cuda/__init__.py:1083)
              _make_launchers (torch/_inductor/runtime/triton_heuristics.py:646)
              precompile -> _wait_futures -> _compile_to_module
   DP3..7:  broadcast (torch/distributed/distributed_c10d.py:2841)
   ```
   Boot hung >17 min (a healthy boot is ~9-10 min) with `health=503` throughout and no
   exception in the log. Recovery that works:
   ```bash
   TORCHINDUCTOR_COMPILE_THREADS=1
   TORCHINDUCTOR_CACHE_DIR=/home/yihou/glm52_fix/inductor_cache   # persist across containers
   TRITON_CACHE_DIR=/home/yihou/glm52_fix/triton_cache
   ```
   Putting the caches on a host bind-mount is the real fix — after one successful boot the
   compile step is warm and the race cannot recur. **Distinguishing the two deadlocks:**
   boot-time = ranks in `synchronize`/Inductor frames and the server never reaches
   `ready to roll`; Bug 2 = server IS ready, `health=200`, and it hangs only when a
   request is routed, with one rank in `init_forward_metadata`.
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
