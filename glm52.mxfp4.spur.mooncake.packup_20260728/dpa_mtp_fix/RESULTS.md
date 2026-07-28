# Results — GLM-5.2 DSA "DPA + MTP" fix (Bug 1)

**Run date:** 2026-07-28 UTC · **Engine:** sglang 0.5.15.post1 · **HW:** 8× MI355X (gfx950), ROCm 7.2.0
**Nodes:** crsuse2-m2m-207 (mix + PD decode), crsuse2-m2m-197 (PD prefill)

---

## Headline

**Bug 1 is FIXED and verified on single-node mix.** The
`RuntimeError: Expected lengths.size(0) == B` crash that made `--enable-dp-attention` and
EAGLE MTP mutually exclusive on gfx950 is gone; the two features now run fused, and the
combination is *faster* than either alone.

A **second, unrelated defect** was uncovered while validating PD: with MTP on the decode
leg, PD comes up cleanly but hangs on the first routed request. That is **not** the Bug 1
crash (different failure mode, different code path) and is **not fixed** here.

## PASS — single-node mix (the fix target)

| Check | Before fix | After fix | Evidence |
|---|---|---|---|
| DPA8 + MTP boot + first batch | **crash** `lengths.size(0) == B` | **ready to roll**, no crash | `logs/mix_dpa_mtp_baseline.log` vs `logs/mix_dpa_mtp_fix1.log` |
| Correctness probe | n/a (crashed) | **4/4** | Paris / Beijing / 4 / Jupiter, coherent CoT |
| DP flags took | — | `enable_dp_attention=True, dp_size=8, ep_size=8` | `logs/mix_dpa_mtp_fix1.log` |
| Spec-dec active | — | **accept len median 3.86** of 4 (n=251, min 2.35, p25 3.69, max 4.00) | `evidence/accept_len.txt` |
| conc=64, ISL/OSL 1k/1k, 256 prompts | n/a | **256/256, 0 failed** | `results/mixfix_c64.{log,jsonl}` |

conc=64 numbers (`results/mixfix_c64.log`):

```
Successful requests:              256
Benchmark duration (s):           82.37
Total token throughput (tok/s):   6365.03
Output token throughput (tok/s):  3182.52
Median TTFT (ms):                 863.67      (mean 1021.00, P99 4201.17)
Median TPOT (ms):                 17.22       (mean 17.54, P90 21.44, P99 28.78)
```

**Fusing the two features beats either one alone on decode latency.** Against the parent
kit's PD single-feature baselines (different topology, so treat as directional):

| Config | Median TPOT |
|---|---|
| PD DPA-only (parent kit, conc=128) | 31.3 ms |
| PD MTP-only (parent kit, conc=128) | 19.2 ms |
| **mix DPA+MTP fused (this run, conc=64)** | **17.2 ms** |

## PASS — regressions (the fix breaks nothing)

| Config | Result | Evidence |
|---|---|---|
| Single-node **DPA-only** (`DPA=1 MTP=0`) | **4/4**, `dp_size=8 ep_size=8` | `logs/regr_dpa_only.log` |
| Single-node **MTP-only** (`DPA=0 MTP=1`) | **4/4**, accept len 2.50 – 3.52 (n=4, probe only) | `logs/regr_mtp_only.log`, `evidence/accept_len.txt` |

Both guards in the patch (`0 < q_offset < q_fp8.shape[0]` and
`topk_result.shape[0] == q_offset`) are false when no DP padding exists, so the non-DP
path is unchanged by construction — and measured to be so.

## PASS — 2-node PD, decode leg DPA8-only

| Check | Result |
|---|---|
| Correctness probe via router | **4/4** |
| Transport | 8× `installTransport, type=rdma` on `mlx5_0`; **0** hip, **0** tcp, **0** ionic, **0** `KVTransferError` |
| conc=64 (1k/1k, 256 prompts) | **256/256**; 4268 tok/s total, median TPOT 27.31 ms, median TTFT 1699 ms |

Evidence: `evidence/transport_evidence.txt`, `logs/pd_decode_dpaonly.log`,
`logs/pd_prefill_30000.log`, `results/pd_dpaonly_c64.{log,jsonl}`.

## NOT PASSING — 2-node PD with MTP on the decode leg (separate defect)

**What works:** the decode leg with `DPA=1 MTP=1` **starts fine** — full disaggregation
warmup returns 200 on all 8 DP ranks, each reporting `spec_accept_length: 2.0`,
`spec_verify_ct: 4`, and prints `The server is fired up and ready to roll!`
(`logs/pd_decode_30001_dpa_mtp.log`, `logs/pd_decode_v3.log`). **The Bug 1 crash does not
reappear under PD** — that is the acceptance criterion for this fix and it is met.

**What fails:** the first *routed* request hangs (no crash, no error — a hard deadlock).
`py-spy dump` across all 8 ranks, sampled twice 6 s apart with zero movement:

```
DP0,3,5,7: broadcast              (torch/distributed/distributed_c10d.py:2841)
DP2,4,6:   all_gather_into_tensor (torch/distributed/distributed_c10d.py:4056)
DP1:       init_forward_metadata  (dsa_backend.py:746)   <- .max().item(), a GPU->CPU sync
             prepare_for_draft_extend (base_spec_worker.py:163)
             _draft_extend_for_decode (eagle_worker_v2.py:921)
             event_loop_overlap_disagg_decode (decode.py:1848)
```

**Why we believe this is NOT the DSA fix's doing:**
1. The stalled frame is DSA *metadata construction* + EAGLE draft-extend scheduling, not
   the indexer top-k path. The patch touches only three spots in `dsa_indexer.py`.
2. The same patched image, same script, **only `MTP=1`→`MTP=0`** on the decode leg →
   PD passes 4/4 and 256/256. Transport config is byte-identical between the two runs
   (see the side-by-side counts in `evidence/transport_evidence.txt`).
3. The same patched code runs 256/256 requests on a single node with DPA+MTP fused.

**Ruled out — transport is not the cause.** Both the hung and the passing run show
identical, correct transport: `disaggregation_ib_device='mlx5_0'`, 8× rdma, 0 hip/tcp,
0 ionic (all 9 devices are visible in the container, mooncake picked only mlx5_0),
0 `KVTransferError`. The DSv4 kit's classic traps (hip IPC across PD instances, ionic
instability, dmabuf compiled out) all do not apply here.

**Open question (unverified, worth a look):** the scheduler subprocesses carry **no
`MC_*` env vars** at all — the launcher process has all four
(`MC_GID_INDEX`, `MC_MS_FILTERS`, `MC_MS_AUTO_DISC`, `MC_DISABLE_HIP_TRANSPORT`) but
`/proc/<scheduler pid>/environ` has zero. Mooncake initialises inside the scheduler, yet
still selected mlx5_0 and rdma correctly — presumably via `--disaggregation-ib-device`
rather than env. Consistent across both runs, so it does not explain the MTP difference,
but it could bite on a node where mlx5 is not the default pick. Not investigated.

## Also observed — two operational traps (not defects in the fix)

Both are documented with what/why/how in `NOTES_rootcause_and_fix.md` §6:

1. **`pkill -f launch_server` orphans the scheduler tree and leaks ~82% VRAM**, wedging
   the next server mid-boot. Fix: `docker rm -f` and recreate the container between
   config changes; confirm `rocm-smi --showmemuse` reads 0% first.
2. **Cold Inductor cache + 8 DP ranks deadlocks the PD warmup** (264 compile workers on
   236 cores; DP0-2 stuck in `synchronize` inside Inductor while DP3-7 wait at a
   collective, stalled in `@torch.compile`'d `select_top_k_tokens`, the EAGLE helper).
   Fix: `TORCHINDUCTOR_COMPILE_THREADS=4`. This one cost a full boot cycle to diagnose.

## Reproduction

See `REPRODUCE.md` (exact commands), `ENVIRONMENT.md` (hardware/software/paths/secrets),
`NOTES_rootcause_and_fix.md` (what/why/how/context of the bug and the fix),
`PLAN.md` (the plan this work followed), `TASK_SPEC_original_todo.md` (the original task
spec — note its root-cause hypothesis was wrong on four counts, see NOTES §2).
