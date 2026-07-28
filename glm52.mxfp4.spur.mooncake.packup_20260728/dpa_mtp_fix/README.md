# GLM-5.2 DSA: fixing the "DP-attention + MTP" crash on gfx950 (Bug 1)

**Ran:** 2026-07-28 UTC · **Engine:** sglang 0.5.15.post1 · **HW:** 8× AMD MI355X (gfx950), ROCm 7.2.0
**Model:** GLM-5.2-MXFP4 · **Cluster:** crsuse spur (nodes 207 + 197)

Fixes the crash that made `--enable-dp-attention` and EAGLE MTP mutually exclusive for
GLM-5.2 DSA on AMD gfx950:

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```

## Task

Fix "Bug 1" as specified in `TASK_SPEC_original_todo.md` (the task spec handed to this
run). User constraints: **single-node mix first, PD after**; **patch the source inside the
running container first**, emit a `.patch` only once there is verified progress; do not
rebuild the image to test an edit.

## Result at a glance

| Config | Correctness | Stress | Status |
|---|---|---|---|
| **Single-node mix, DPA8 + MTP** (the fix target) | **4/4**, accept len median **3.86**/4 | **256/256** @ conc=64, median TPOT **17.2 ms** | ✅ **FIXED** |
| Regression: single-node DPA-only | 4/4 | — | ✅ unregressed |
| Regression: single-node MTP-only | 4/4, accept len 2.50–3.52 (n=4) | — | ✅ unregressed |
| 2-node PD, decode DPA8-only | 4/4 | 256/256 @ conc=64 | ✅ pass |
| 2-node PD, decode DPA8 **+ MTP** | boots clean, warmup 200 ×8 ranks, **Bug 1 crash gone** | — | ⚠️ **hangs on first routed request — separate defect, NOT fixed** |

Fusing DPA + MTP is also *faster* than either alone: median TPOT 17.2 ms vs the parent
kit's 31.3 ms (DPA-only) and 19.2 ms (MTP-only).

**Honest scope:** Bug 1 (the top-k row-count crash) is fixed and verified. PD with MTP
surfaces a *different* failure — a collective deadlock in the EAGLE draft-extend metadata
path — which this delivery does **not** fix and does not claim to. Evidence that it is
independent of this patch is in `RESULTS.md`.

## Root cause in one paragraph

`dsa_indexer.py::_get_topk_paged` — the CUDA path slices its MQA-logits inputs to the
**real** row count (`q_fp8[:q_offset]`), so `logits.shape[0] == lengths.shape[0]`; the
**aiter/HIP** path does not slice and sizes `logits` from the **DP-padded**
`q_fp8.shape[0]`. The padding-restore right after was gated `if not _is_hip`, consistent
with the CUDA slice but leaving HIP with no path that produces matching shapes. Under
DP-attention the hidden states are padded, so top-k saw `score.shape[0] != lengths.shape[0]`
and asserted. **Measured proof** (`evidence/dsa_rows_measured.txt`, 27 distinct padded
shapes over a full conc=64 run):

```
mode=ForwardMode.DRAFT_EXTEND_V2  q_fp8=(36, 32, 128)  q_offset=32  lengths=(32,)  -> mqa_q=(32, ...)
mode=ForwardMode.DRAFT_EXTEND_V2  q_fp8=(40, 32, 128)  q_offset=28  lengths=(28,)  -> mqa_q=(28, ...)
mode=ForwardMode.IDLE             q_fp8=(4,  32, 128)  q_offset=1   lengths=(1,)   -> mqa_q=(1,  ...)
```

`DRAFT_EXTEND_V2` is the dominant padded mode (26 of the 27 padded shapes); `IDLE` is the
extreme case (4 padded rows vs 1 real). `TARGET_VERIFY` is never padded.

The fix makes HIP obey the CUDA contract: slice to the real rows, then restore padding
(~5 functional lines, no new kernel).

⚠️ The original task spec's root-cause hypothesis was **wrong on three counts** (PD is not
required; it crashes in the *fused* top-k not the unfused one; the CUDA `seqlens_expand`
clamp it proposed is already present). It was *right* that draft-extend is where the
mismatch lives. See `NOTES_rootcause_and_fix.md` §2 — that correction is the most reusable
part of this kit.

## Reading order

1. **`NOTES_rootcause_and_fix.md`** — the what/why/how/context: the crash, the four
   corrections to the original analysis, the real root cause with measured evidence, the
   fix, and two operational traps (VRAM leak from `pkill`, Inductor cold-cache deadlock).
   Most valuable file here.
2. **`RESULTS.md`** — every number, with the evidence file backing each claim, plus an
   explicit account of what did *not* pass.
3. **`ENVIRONMENT.md`** — hardware, RDMA fabric, image sha256, driver/ROCm/torch versions,
   absolute paths of non-git dependencies, secrets (none).
4. **`REPRODUCE.md`** — exact commands, from holding a node to the conc stress.
5. `PLAN.md` — the plan this work followed. `TASK_SPEC_original_todo.md` — the input spec.

## Folder map

```
patches/
  dsa_indexer_hip_dp_rows.patch   unified diff vs the image's stock file
  dsa_indexer.patched.py          whole patched file (bind-mount drop-in)
  apply_fix.py                    idempotent applier, verifies each anchor matches once
scripts/
  mix_leg.sh          single-node mix server, DPA/MTP switches   (new, this run)
  pd_leg_spur.sh      2-node PD leg, mlx5+dmabuf transport recipe (reused, unchanged)
  probe.py            4-prompt correctness probe                  (reused)
  sweep_dpa.sh        concurrency stress                          (reused)
evidence/
  crash_traceback_baseline.txt  the original crash, full traceback
  dsa_rows_measured.txt         SGLANG_DEBUG_DSA_ROWS output — the root-cause proof
  transport_evidence.txt        rdma/hip/tcp/mlx5/ionic counts, hung vs passing run
  accept_len_*.txt              spec-dec really firing
results/
  mixfix_c64.{log,jsonl}        single-node mix DPA+MTP stress (raw bench output)
  pd_dpaonly_c64.{log,jsonl}    PD DPA-only stress
logs/                           full server logs, all 9 runs (untrimmed, 7.8 MB)
```

## Dependencies not in git

- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (~408 GiB, shared NFS)
- Image tar `/home/yihou/infera.yihou.sglang.1.0.tar` (27 GB, NFS)
- Original experiment workspace `/home/yihou/glm52_fix/` (preserved as-is, not deleted)
- **No secrets/tokens required** — public base image, model on shared NFS, spur identity
  automatic, node→node image move via NFS tar (ssh between compute nodes is banned).

## Upstream

The defect is **ROCm-specific** — the CUDA path was always correct — so it is *not*
covered by sglang PR #30378/#30427, which fixed the CUDA-side padded-row *values*
(`seqlens_expand_kernel` clamp, already present in this image) rather than the HIP-side
row *count*. Worth filing upstream against the aiter paged-MQA integration.
