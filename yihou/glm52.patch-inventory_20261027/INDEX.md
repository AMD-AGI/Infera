# GLM-5.2 PD work — inventory of yihou's fixes and optimization patches (2026-09-18 .. 2026-10-27)

Built 2026-10-27 ~07:00Z for an agent that will open upstream PRs (sglang, Infera, aiter, Mooncake). Every row was
checked against a file (pack-up README/notes, the patch itself, git history, or upstream `main` fetched with `gh api`
on 2026-10-27, sglang main = `e4d4a9a7de`). Items marked **[unverified]** were not checked first-hand.

Paths: `PK = yihou/` in this repo (branch `dev/pd_opt/glm_5.2_agentx`; pack-up commits listed per row);
`NFS = /shared_nfs/yihou_glm52` (reachable from crsuse2-m2m-135 only); `SG = /home/yihou/dev/git.16-19/sglang.pd.pp.dcp`
(local sglang worktree, branch `dev/yihou/glm52_dcp_pd`, commits authored `yihou <yihou@amd.com>`).
All diffs are in `patches/`. Config/tuning results (no code) are in `config_findings.md`.

## 0. Read this first — correctness

- **Every throughput number in this file used simulated MTP acceptance** (`SGLANG_SIMULATE_ACC_LEN` 3.61 at K5, 2.99 at K3).
  Simulated acceptance does not look at output content. No number here is a correctness result.
- **llying 10-27 (Infera commits 27740072, 19a9c82d, file `llying/prefill-opt-20261027/README.zh-CN.md`)**: the llr35 stack
  garbles in PD with real acceptance (200-question probe 10 garbled; long-context needle 13/64 at 16k, 2/16 at 86k).
  Bisect round 1 (each variant = llr35 minus one item, HiCache off): **main cause = prefill `SGLANG_PP_COMM_OVERLAP=1`**
  (a config flag; without it smoke 6/6, probe 0.975 / 0 garbled). Long contexts still degenerate without it (second
  problem, open). Decode custom all-reduce off: short-prompt garbles 7 -> 1. Removing KF / FP8 draft / index_share showed
  no change, but those were tested with PP comm overlap still on (masked) — llying says this does not clear them.
  Round 2 (base = llr35 minus PP comm overlap; variants clean-k5, clean-k3, no-car, no-KF, no-GEMM-csv, no-idxshare,
  no-async) started 06:35Z 10-27, **no result at the time of writing**.
- **Code in the llr35 stack** (site file `PK/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/sites/llr35.site.env`, read):
  K01 (KF overlay: `YIHOU_KF_SPARSE/INDEXER/MOE=1` + `SGLANG_OPT_USE_TOPK_V2=false`), A01 (both aiter BF16 GEMM csvs),
  I07 (kit hooks), and llying's kit `patches/deepseek_v2.py` on the prefill (same change as S13). Config in that stack:
  custom AR on, `SGLANG_PP_COMM_OVERLAP=1`, index_share true, K3, draft FP8 (`SGLANG_GLM_NEXTN_MOE_PTPC=1`), decode mem 0.95,
  `--stream-interval 30`, `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`, `--pp-async-batch-depth 1`, C96.
  **Not** in the llr35 stack: all v0.5.21 mounts (S01-S09), the DCP branch (D*), S10-S12, S14, Mooncake patches.
- Earlier correctness facts: v0.5.19 decode custom all-reduce corrupted the speculative path with real acceptance
  (`PK/glm52-mtp-garbled-decode-rootcause.packup_20260918`); on the v0522-fdbafb60 image `--enable-fused-qk-norm-rope
  --enable-aiter-allreduce-fusion` with custom AR gave garbage on node 190, culprit flag not isolated (OPT_LEDGER A2);
  cx90 correctness 22/23 on the v0.5.21 pf90 stack (S01, S02, T01 mounted) and 23/23 with OffloaderV2 G3/G2 + S06
  (`PK/glm52.v0521-c48-followups.packup_20261015`, `PK/glm52.v0521-offloaderv2.packup_20261015`); DCP branch E2
  (DP2 x DCP2 decode, graph) returned garbled outputs on 3/200, 13/240, 27/320 requests, not bisected
  (`PK/glm52.dcp4-pd.packup_20261022/patches/README.md`).

## 1. Summary table (ours = authored by yihou, in any of his sessions)

Kind: BF = bug fix, P = performance, F = feature, T = tooling/diagnostic, X = experiment-only. "Up?" = recommended for
upstream as-is (Y), after rework (R), no (N).

| id | title | target | base | kind | owner | upstream PR / status | Up? | patch file(s) |
|---|---|---|---|---|---|---|---|---|
| S01 | KV-metrics ZMQ PUSH send non-blocking (drop on EAGAIN) | sglang | v0.5.21 image 9d755dd6 | BF | yihou | none; main still blocking (`kv_events_publisher.py:90`) | R | S01-*.diff |
| S02 | ROCm `_absorb_bmm_config` -> None (triton LLVM abort at M=256) | sglang | v0.5.21 image | BF (workaround) | yihou | none | N (report as issue) | S02-*.diff |
| S03 | Mooncake KV sender: per-shard transfer log + `YIHOU_XFER_SHARD=dest/room/mix` (hashed room) | sglang | v0.5.21 image | P + T | yihou | #42163 (YAMY1234, open) does room sharding with splitmix64 | R (merge into #42163 discussion) | S03-*.diff |
| S04 | `YIHOU_NONFINAL_EVENT` switch for the non-final-chunk wait event | sglang | v0.5.21 image (incl. Infera wait-event patch) | X | yihou | related: dorado269 #33970 (open, Aug), #43299 (open) | N | S04-*.diff |
| S05 | EagerRunner idle: also init the active forward-context attention backend (index_share MAF fix) | sglang | v0.5.21 image; upstream main 93088901 | BF | yihou | **#45483 open** (dorado269) | Y (filed) | S05-*.v0521.diff, S05-*.pr45483.patch |
| S06 | OffloaderV2 keeps `is_shuffled` on device copies | sglang | v0.5.21 image; main | BF | yihou | **#45484 open** (dorado269) | Y (filed) | S06-*.v0521.diff, S06-*.pr45484.patch |
| S07 | `prefill_complete` validation reads `attn_dp_size` | sglang | v0.5.21 image | BF | yihou | none (candidate written, not filed); main line 208 unchanged | Y | S07-*.diff |
| S08 | Decode admission: running requests keep >= share of KV pool (runshare v1/v2) | sglang | v0.5.21 image | F | yihou | none | N (measured worse) | S08-*.v1/.v2.diff |
| S09 | DSA graph variant from a host lower bound (skip `seq_lens.max().item()`) | sglang | v0.5.21 image | P | yihou | none | N (no gain) | S09-*.diff |
| S10 | DCP pack: device-cached pack metadata + high-priority gather stream, no default-stream wait | sglang | overlay 93034eba91 | P | yihou | none (memory: PR TODO); related #44724 open | R | S10-*.diff |
| S11 | Mooncake transfer-queue shard by hashed bootstrap_room when the destination needs DCP relayout | sglang | overlay 93034eba91 | P | yihou | #42163 open (same idea, not ours) | R (comment on #42163) | S11-*.diff |
| S12 | Decode loop: `torch.cuda.empty_cache()` on every rank when free < 6 GiB (every 8 iters) | sglang | ab7a898b77; port to v0522-fdbafb60 | BF (workaround) | yihou | none | R (needs root cause) | S12-*.diff, S12b superseded |
| S13 | deepseek_v2: `config.hidden_size` instead of `embed_tokens.embedding_dim` on non-first PP ranks | sglang | 93034eba91 (= commit ab7a898b77) | BF | yihou commit; **same change in llying's kit v6-pp4 dated earlier** | none; main still has the bug (`deepseek_v2.py:3013,3048`) | Y (agree attribution with llying) | S13-*.patch |
| S14 | Online MXFP4 for the GLM-5.x MTP draft (`quark_mxfp4`) | sglang | v0522-fdbafb60 image (sglang e9b22a0818) | F | yihou | none | R (accuracy lower) | S14-*.diff |
| D10-D31 | DCP-on-PD series (relayout draft KV, ROCm DCP decode behind DCP1 prefill, split decode indexer, DP2xDCP2 fixes, deferred-KV release, KEEP_KV_B_BF16, tests) | sglang | fda84f80b7 + PR #44322 merge 4792dc0e52 | BF/P/F/T | yihou (session sglang-pd-pp-dcp-fa) | none filed; depends on #44322 (open) | per commit, see §3 | D-dcp-branch/00NN-*.patch |
| M01 | Mooncake PR #4340 (HIP IPC sub-allocation relocation) ported to faae8dd4 | Mooncake | faae8dd4 | BF (port) | port by yihou; fix by salexspb | #4340 open | N (comment/backport only) | M01-*.diff |
| M02 | `MC_HIP_STREAM_PRIORITY=high`: high-priority HIP copy streams | Mooncake | faae8dd4 + M01 | P | yihou | none | R (needs e2e) | M02-*.diff |
| M03 | One batched HIP peer-copy kernel per batch | Mooncake | faae8dd4 + M01 | P | yihou | none | N (never built) | M03-*.diff |
| K01 | KF decode kernel overlay (sparse MLA, indexer logits + top-k, MoE, DCP sparse/indexer, gfx950 fused indexer) | aiter / sglang (kernels) | v0.5.21/v0522 sglang via import hook | P | yihou KF session (KernelForge-generated) | none | R (re-integrate as real kernels) | K01-*.NEWFILES.diff |
| A01 | aiter BF16 GEMM tuned csvs for GLM-5.2 decode on gfx950 (75 + 49 shapes) | aiter | image v0522-fdbafb60 aiter | P (config data) | yihou | none | R (after correctness round 2) | A01-*.NEWFILES.diff |
| I01 | bench `engine.sh`: role-scoped `*_EXTRA_ARGS` / `*_EXTRA_ENV`, AITER AR fusion conditional | Infera | 5a342acf | T | yihou | none | Y (harness) | I01-*.patch |
| I02 | bench wiring for router `INFERA_PD_DP_RANK_AFFINITY` (1/0 -> true/false) | Infera | 5a342acf | BF | yihou (hand port of llying 99fa0406 bench half) | none | Y (if 99fa0406 lands) | I02-*.patch |
| I03 | bench same-node 1P1D: topology dedupe, `ENGINE_PORT_STRIDE`, same-node start gate | Infera | bench @ 0923 | BF | yihou | none | Y | I03-*.diff |
| I04 | router: resolve stash conflicts; PD DP-rank affinity ported to `pick_targets`/sessions API | Infera | 28121cb4 | BF | yihou | none | R (check vs current router) | I04-*.patch |
| I05 | docker: v0.5.21 recipe + GLM-5.2 patch re-cut; v0521-mc4340 (#4340) recipe; pipefail fix; mc4340prio | Infera | 28121cb4 | F/BF | yihou | none | R (superseded by v0522 recipe in #210?) [unverified] | I05a-d |
| I06 | bench harness v0521 lineage: `DECODE_PID_OFFSET`, `NODE_RW_ROOT`, `IPC_MODE_LEGACY`, cpusets, role mounts, agentx_env checks | Infera | 28121cb4 | BF/T | yihou | none | R | I06a/b |
| I07 | sglang_2p1d kit hooks: `DECODE_CUSTOM_AR`, `DECODE_ENV_EXTRA`, `PREFILL_ENV_EXTRA`, `DECODE_MOUNT_EXTRA` (list) | Infera | 06796e13 examples/sglang_2p1d_glm5.2 + llying kit-v6-pp4 | T | yihou | none | Y | I07-*.diff |
| T01-T07 | diagnostics: Mooncake regbase log, DSA seed guard, top-k guards, DCP xfer stats, memlog, decode bench port, hipxfer LD_PRELOAD shim | sglang / Mooncake / tooling | various | T | yihou | none | N | T0N-*.diff |
| X01-X02 | synthetic reduced-expert models (8 / N experts) to free HBM | sglang | v0.5.21 image | X | yihou | none | N (non-faithful model) | X0N-*.diff |

Not ours (listed so nobody files them as ours): §5.

## 2. Details — sglang, v0.5.21 era (image `infera-sglang:v0521-yihou-1010-mc4340`, ID 9d755dd6)

Base for S01-S09, T01-T03, X01-X02: the image's own sources, extracted at
`NFS/v0521cap_20261012/src/image-9d755dd6/sglang/srt` (sglang tree 9f5d8712 + image-local diff, per
`PK/glm52.v0521-idxshare-maf-fix.packup_20261021/patches/README.md`; server reports 0.5.21.dev20261008+g11b1fb21de).
Diffs in `patches/` were regenerated for this inventory as `diff -u <image file> <mounted file>` and compared with the
pack-up diffs (S01, S02, S06, T01 identical in +/- lines). Applied as read-only bind mounts by the harness
(`NFS/v0521dbg_20261010/harness/mounts-v0521/src/`, tables `mounts.yihou.*.tsv`). "Applies to main" = `patch --dry-run`
against sglang main e4d4a9a7de files fetched 10-27.

### S01 — KV-metrics PUSH send non-blocking
- Files: `srt/managers/scheduler_components/kv_events_publisher.py`.
- Problem / root cause: v0.5.21 PUSH socket for auxiliary KV metrics connects to an endpoint nothing binds in this
  deployment; once the ZMQ queue fills, the blocking `sock_send` wedges the prefill scheduler event loop (round c44a hung;
  py-spy 2026-10-09, ledger R001).
- Fix: `flags=zmq.NOBLOCK`, catch `zmq.Again`, warn once, drop telemetry.
- Evidence: c44a hang -> c44b came up past it; all later v0.5.21 runs carried it (e.g. e2g 22,114, g3x1 25,524).
  Ref: `PK/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/patches/README.md` §2 (repo commit 56421666), re-cut as
  DCP-branch commit 36bc86517d (D-dcp-branch/0008).
- Upstream: no PR. main e4d4a9a7 still calls `sock_send(...)` with flags=0; the diff applies to main without offset.
- Upstreamability: small; dropping metrics silently should be a reviewed choice (maybe a counter + debug log). Needs a
  unit test (unbound PUSH + full HWM must not block). Remove the YIHOU tag.

### S02 — `_absorb_bmm_config` returns None on ROCm
- Files: `srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py`.
- Problem: v0.5.21's hand-built absorbed-bmm tile config at M=256 aborted triton's LLVM backend (`Sequence.h:275
  iota_range` assertion) in a prefill rank (round c44b crashed, ledger R003). Workaround returns None so aiter picks its
  own config (v0.5.19 behaviour).
- Evidence: c44b crash -> no crash after; carried in all v0.5.21 runs. Re-cut as D-dcp-branch/0009 (4d1d6a647e).
- Upstream: no PR. main still has the function (`forward_mla_rocm.py:156`); diff applies. Whether the abort still
  happens with main's triton is **not tested**.
- Upstreamability: this disables a perf path; upstream wants either a triton bug report or a guarded config
  (e.g. cap BLOCK sizes at the failing M). File as an issue with the repro, not as this diff.

### S03 — Mooncake sender transfer log + hashed-room shard (v0.5.21)
- Files: `srt/disaggregation/mooncake/conn.py` (diff includes T01 regbase logging, since xferfix2 was built on it).
- What: `YIHOU_XFER_LOG=1` per-shard per-minute stats (queue wait, `batch_transfer_sync` calls/blocks/GiB/wall);
  `YIHOU_XFER_SHARD=dest` (default = upstream: sum of dest session ports % 4) | `room` | `mix`
  (`((room*0x9E3779B97F4A7C15) mod 2^64 >> 32) % 4`). Root cause found: with `dest` each prefill DP used one of the 4
  transfer workers; `room % 4` equals the DP rank, so plain room sharding does not spread either.
- Evidence: g4tune R2 (mix) vs R1 (dest), same node 250: prefill transfer_kv avg 2.16 vs 9.98 s, A2 22,854 vs 21,262
  (n=1 each). Not adopted by the user then (`PK/glm52.v0521-g4tune-xfer.packup_20261020/patches/README.md`, notes §2).
- Upstream: #42163 "[PD] Shard Mooncake transfers by bootstrap room" (YAMY1234, open) — same idea with splitmix64.
  Our data point (room % dp == DP rank, so the hash is needed) is useful as a review comment there.
- Upstreamability: logging part is local tooling; do not upstream the env names.

### S04 — non-final chunk wait-event switch
- Files: `srt/disaggregation/prefill.py`. Env `YIHOU_NONFINAL_EVENT=orig|launch|off`, minute log `YIHOU-NONFINAL`.
- Base note: the image already carries Infera's wait-event patch (`deploy/docker/patches/sglang_disagg/
  patch_mooncake_early_send_wait_event.py`, liyingli/xiaobochen); our diff is relative to that, so it differs from the
  pack-up's `0003` diff (which is vs plain upstream 11b1fb21de).
- Evidence: e2f (`room` shard + `launch`) 19,864 fresh, best TTFT, score not improved (decode-bound) (R036-R042,
  oai pack-up README §4). g4tune ran `orig` only.
- Upstream: N. Related open PRs: dorado269 #33970 (Aug, "[PD] Make the mooncake KV transfer wait on the prefill forward
  that wrote the pages"), #43299 (not ours).

### S05 — EagerRunner idle batch inits the active attention backend (index_share MAF) — PR #45483
- Files: `srt/model_executor/runner/eager_runner.py` (2 imports + 4 lines).
- Root cause (code read + debug agent + call-site guards; `PK/glm52.v0521-idxshare-maf-fix.packup_20261021/README.md`):
  with `index_share_for_mtp_iteration=true`, a seedless PD batch forces all DP ranks to eager drafting; the idle DP rank's
  draft steps run under `ForwardContext(attn_backend=draft_attn_backend.attn_backends[i])` while `_execute_idle` only
  re-inits `model_runner.attn_backend` (the draft-extend backend); the DSA indexer reads the stale step backend,
  ranks 2048-wide dummy logits with stale lengths (up to 104,740) -> `coop_topk_kernel` over-read -> decode
  "Memory access fault".
- Evidence: 8 faulting rounds without the fix; idxf1a A1+A2 0 MAF (A2 21,814), dbge6a 0 BAD/0 MAF; replicates
  idxf1b/c 21,538 / 20,516 (tune.p4p4 §3). index_share itself: llr3 +0.9 % / TPOT -3.4 % (config_findings).
- Upstream: **#45483 OPEN** (dorado269, opened 10-21). Patch vs main 93088901 in `S05-*.pr45483.patch`; PR body in
  `PK/glm52.v0521-baseline-xnode-g3g4.packup_20261022/patches/pr45483.body.md`. Also cherry-picked as D-dcp-branch/0020.
- Note: D27 (da6e383513, "skip the indexer on an eager idle DP rank on ROCm too") addresses another idle-rank fault in
  DP2 x DCP2 — related, separate.

### S06 — OffloaderV2 keeps `is_shuffled` — PR #45484
- Files: `srt/utils/offloader.py` (`_CpuParamOffloader.create_device_tensor`).
- Root cause (code read): `.to("cuda")` drops the `is_shuffled` attribute; aiter fused_moe/MXFP4 selects the kernel via
  `getattr(w, "is_shuffled")`, so onloaded MoE layers take the non-preshuffled path.
- Evidence: correctness 23/23 with OffloaderV2 G8/G3/G2 + patch; perf rounds (G3 best g3x1 25,524). **No unpatched round
  was run**, so the effect of the patch alone is not measured (`PK/glm52.v0521-offloaderv2.packup_20261015/patches/README.md`).
- Upstream: **#45484 OPEN** (dorado269). v0522 version with CPU fail-first test = D-dcp-branch/0021 (c9394c114f).
  main e4d4a9a7 offloader has no `is_shuffled` handling; v0521 diff applies with offset.

### S07 — `prefill_complete` validation vs attention DP
- Files: `srt/arg_groups/pd_disaggregation_hook.py` (1 line).
- Root cause: `handle_deprecated_dp_attention` rewrites `--dp-size 4 --enable-dp-attention` to `attn_dp_size=4,
  dp_size=1, enable_dp_attention=False` before `_validate_prefill_complete`, which computes attention DP from the old
  fields -> 1 -> startup ValueError "does not yet support attention tensor parallelism greater than one".
- Evidence: pc48 prefill exit 1 at startup -> pc48b up, 2 x 3600 s C48 runs completed.
- Upstream: no PR; candidate text `PK/glm52.v0521-prefill-complete.packup_20261014/patches/UPSTREAM-PR-CANDIDATE.md`
  (proposes `attn_dp_size = cfg.attn_dp_size`). main e4d4a9a7 line 208 still has the old expression; our diff applies
  (offset -5). Needs an arg-resolution unit test. The policy itself was not a win (config_findings).

### S08 — runshare: decode admission keeps a KV share for running requests
- Files: `srt/disaggregation/decode.py` (`pop_preallocated`). Env `YIHOU_DECODE_RUNNING_KV_SHARE`, v2 adds
  `YIHOU_RUNSHARE_MAX_WAIT_S` (skip + anti-starvation).
- Evidence: v1 C40 warm 21,271 vs 22,114 (-3.8 %); v2 C56 A2 17,611 vs 18,016 (-2.2 %), C64 15,070 vs 15,108 (-0.3 %).
  `PK/glm52.v0521-runshare-highconc.packup_20261014`. Do not upstream.

### S09 — DSA graph variant from a host lower bound
- Files: `srt/speculative/eagle_utils.py`, `srt/layers/attention/graph_variants.py`. Env `YIHOU_DSA_HOST_VARIANT=1`.
- Evidence: sync removed (R054); warm R2 21,681, no measurable gain (R057). Do not upstream.

## 3. Details — sglang, DCP / v0.5.22 era

### S10 — DCP pack: cached device metadata + priority stream (the decisive DCP transfer fix)
- Base: overlay 93034eba91 (`SG`), file `srt/disaggregation/common/dcp_pack.py`; `patch --dry-run` passes on 93034eba91.
- Root cause (instrumented, `[yihou-xfer]` sub-timers): per call, `try_pack_dcp_src` did a pageable H2D of row indices,
  `copy_mla_rows_into_pack` built `torch.tensor(metadata, device=...)` (another pageable H2D), and the gather stream
  waited on the producer/default stream, so each pack waited 40-78 ms behind prefill compute (91 % of send time).
- Fix (`YIHOU_PACK_NOWAIT=2`): pinned + non_blocking row indices; per-(ptrs, widths, strides) metadata cached on device
  with pack offset = prefix * n computed in a new triton kernel; gather on `torch.cuda.Stream(priority=-1)`; no
  default-stream wait.
- Evidence: ms/pack 77.6 (dcpx8c) -> 40.9 (no-wait only, dcpx9) -> 0.6 (dcpx10); KV transfer p50 12.3 s -> 0.93 s;
  score 19,997 -> 24,753 (+23.8 %), cross-node 135->136 DCP4, C56, n=1. Kernel output equality vs stock on one GPU
  (`patches/tests/t_pack2.*` in `PK/glm52.dcpx-pack-fix-best.packup_20261023`, repo commit 6d45fc2d; n=1/777/8192 equal).
- Caveats: which half (cached metadata vs priority stream) matters is **not separated**. Ordering now relies on
  `TransferKVChunk.wait_event` or on the chunk being queued after the forward; under a PP prefill (overlap off) the early
  cached-prefix send has no wait event — **correctness under PP + no-wait is unchecked**. The diff contains
  instrumentation (`time.perf_counter`, env modes 1/2) that must be removed for upstream.
- Upstream: no PR (memory file `glm52-dcp-pack-fix-upstream-pr-todo.md`: user asked to file it, title prefix [AMD], from
  fork dorado269, no Co-Authored-By). main e4d4a9a7 still does `torch.as_tensor(src_token_indices, device=...)`,
  `wait_stream(producer_stream())`, `synchronize()` (dcp_pack.py) and `torch.tensor(metadata, device=...)`
  (`kernels/ops/kvcache/pd_dcp_gather.py:63`); hunk 2 of our diff does **not** apply to main — needs a rebase.
  Related open: #44724 "[Fix] disaggregation: Select the staging device in DCP packing threads".

### S11 — hashed-room transfer-queue shard for DCP destinations
- Base: overlay 93034eba91 `srt/disaggregation/mooncake/conn.py`; applies to main e4d4a9a7 with offset.
- Root cause: with a DP1 (DCP) decode every room has the same destination set, so `sum(dst ports) % 4` is constant and
  all chunks of a prefill rank serialize on one worker; `room % 4` equals the prefill DP rank
  (`data_parallel_controller.py:883`), so it is constant too (dcpx7 showed no change).
- Fix (`YIHOU_DCP_SHARD=room`, only when a destination `requires_dcp_relayout`): Fibonacci-hash the room.
- Evidence: py-spy 1 -> 2-4 active workers per rank; transfer p50 21.3 -> 11.3 s; 19,691 -> 21,302 (dcpx6 -> dcpx7b).
- Upstream: #42163 (not ours, open) shards by room with splitmix64 for all destinations; comment there with this data
  rather than a competing PR. `T04` = S11 + per-minute xfer stats (tooling).

### S12 — decode `empty_cache` on every rank when device free < 6 GiB
- Base: sglang ab7a898b77 (overlay `NFS/dcp_pd_20261021/src/sglang-ab7a898b77-memlog-ec2.yihou`), file
  `srt/disaggregation/decode.py`, two 4-line hunks in both decode event loops; `git apply --check` passes on ab7a898b77
  (checked again 10-27 with `patch --dry-run`); also applies to main with offset. Port to the v0522-fdbafb60 image
  (sglang e9b22a0818) in `S12-*.port-v0522-fdbafb60.diff` (`NFS/llrepro_20261023/ec_port.yihou/README.md`).
- Root cause (measured with T05 memlog): at decode mem 0.88/0.92 the torch allocator's reserved-but-free blocks grow
  under burst prealloc until the device is full -> `HSA_STATUS_ERROR_OUT_OF_RESOURCES`. A TP0-only empty_cache (S12b)
  did not help (other ranks still grew).
- Evidence (same-node DCP4, llying v0522 kit S2 config, `PK/glm52.dcp4-decode-gmu.packup_20261023`, commit 2ecf34d3):
  without fix 0.88/0.92 abort; with fix 0.88 = 27,685 (S2n; 0.85 runs 25,941 / 25,433), 0.92 = 22,540 (S2m). n=1 each.
  The v0522-fdbafb60 port was **never run** (OPT_LEDGER P25 OPEN).
- Upstreamability: a periodic `empty_cache` is a workaround with a latency cost; upstream would want the cause
  (fragmentation from burst prealloc). Gate behind a server arg if filed.

### S13 — deepseek_v2 hidden size on non-first PP ranks
- Commit ab7a898b77 (author yihou, 2026-10-22 10:34Z) = `S13-*.patch`; identical hunks to llying's kit file
  `2p1d-kit/v6-pp4/patches/deepseek_v2.py` whose diff header is dated 2026-10-21 17:12Z
  (`PK/glm52.llrepro-pp4-dpa4.packup_20261024/scripts/ds.patch`). **Attribution: llying's kit had it first; agree with
  llying before filing.**
- Root cause: on a PP stage that does not own the embedding, `embed_tokens` is a `PPMissingLayer` without
  `embedding_dim` -> AttributeError at model construction (gfx950 + 256 experts branch).
- Evidence: PP4 prefill starts (ppdc1, `PK/glm52.ppdc-pp4-dcp4.packup_20261023/notes.md` §1); every PP4 run since.
- Upstream: main e4d4a9a7 still reads `self.embed_tokens.embedding_dim` (lines 3013, 3048); patch applies with offset.

### S14 — online MXFP4 for the GLM MTP draft
- Base: image v0522-fdbafb60 (sglang e9b22a0818), files `srt/configs/model_config.py`, `srt/model_loader/weight_utils.py`
  (`.orig` copies in `PK/glm52.llrepro-k3-mtp.packup_20261026/mxfp4_draft/patched/`). Widens Qwen3.5-only requantization
  gates to `GlmMoeDsaForCausalLMNextN`; used with `--speculative-draft-model-quantization quark_mxfp4`.
- Evidence: llr34 47,282 vs llr32 47,447 (-0.3 %, neutral); draft memory 6.92 -> 5.07 GB; real acceptance on node 190
  K5: MXFP4 3.16-3.42 vs bf16 3.547/3.550 (12 prompts). FP8 draft (existing env, config_findings) kept instead.
- Upstreamability: InferenceX disallows MXFP4 drafts (ledger); accuracy cost measured; low priority.

### D10-D31 — DCP-on-PD sglang branch (`SG`, branch `dev/yihou/glm52_dcp_pd`)
Base: sglang `origin/main` fda84f80b7 + merge 4792dc0e52 of PR #44322 ("[Feat] [AMD] dsa: Enable DCP on the triton
sparse-MLA backend", EricKing626, **OPEN**, not ours; `D-dcp-branch/0000-*.NOT-OURS.diff`). Patches = `git format-patch
4792dc0e52..93034eba91` (31 commits, author yihou), copied from `PK/glm52.dcp4-pd.packup_20261022/patches/` (commit
2326727c); all 31 SHAs verified present in `SG`. Verification column from that README ("CPU" = unit test in the commit;
"fail-first" = fails on old code). All new behaviour is env-gated, default off.
0001-0007 are Infera recipe patches re-cut as commits (not ours, §5); 0008/0009 = S01/S02; 0020 = S05 cherry-pick;
0025 = local debug log (T).

| id | sha | subject (files) | kind | verification / evidence | upstream note |
|---|---|---|---|---|---|
| D10 | a6bdc7894d | [PD] dcp: Relayout a DCP-sharded DSA draft KV like the target (disagg common/mooncake/mori/nixl conn, test) | BF | CPU fail-first; GPU pdBe-r1 PASS (math 29 x2, passkey 8K/34K/96K) | needs #44322 |
| D11 | 228263d116 | [PD] dsa: Allow a ROCm DCP decode behind a DCP1 prefill (model_overrides/deepseek_v2.py, test) | F | pdBe-r1 | needs #44322 |
| D12 | ae529c96bf | [PD] dsa: SGLANG_DSA_DCP_PD_DROP_MTP_SEED (environ, eagle_disaggregation) | T/F | CPU | A/B switch; probably N |
| D13, D16, D29 | 906b0915a9, 646dc699f3, efa41449fe | [Test] DP-attn prefill -> DCP4; PP prefill -> DCP4; DP-attn prefill -> DP2xDCP2 | test | CPU | ship with D10 |
| D14, D15, D18, D24, D30, D31 | a78ff55a8b, a04c3f8e0c, 6ffb90ae6b, 0347a6d47e, a6ea78e322, 93034eba91 | [AMD] dsa: page-split DSA decode top-k for DCP (new `dsa_dcp_split_indexer.py`, `_triton.py`), env gate `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER` / `_SPLIT_INDEXER_TRITON`, owned pages once per forward, fused pack+merge, radix-select merge | P | benchrun T15/T16: merge 46 us/layer; vs replicated ref 34K +0.5..+1.5 ms (slower), 150K -2.6..-4.2 ms, 200K -4.4/-6.6/-9.2 ms (N16/24/32). Gate PASS 40/40 | R: only wins at long context; squash 14/15/18/24/30/31 into one PR |
| D17 | 6ee35f3617 | [PD] pp: Gate non-final prefill chunk sends on the forward stream (prefill.py) | BF | CPU fail-first; PP never run on GPU with it | Y candidate; relate to #43299 / #33970 |
| D19 | c4eda0343e | [PD] decode: Release deferred KV and skip transfer polling in unison across attn-TP ranks (decode.py, 2 tests) | BF | CPU, 4/5 new tests fail on old code; GPU cancel repro not run | Y candidate |
| D21 | c9394c114f | [AMD] offloader: Keep is_shuffled (+ CPU test) | BF | CPU fail-first | = S06 / #45484 |
| D22 | 636cc68dd0 | [AMD] dsa: SGLANG_DSA_DCP_KEEP_KV_B_BF16 (deepseek_weight_loader, test) | P | benchrun T10 -0.4/-1.6/-1.7/-1.7 ms at N 8/16/24/32, 34K; weights +0.34 GB/GPU | R |
| D23 | b17da53725 | [Fix] attention: Pad a batch-major padded-extend output on its token dim (graph_utils, test) | BF | DP2xDCP2 crash gone (mixrun R13) | Y candidate |
| D26 | e9e1432ce6 | [Fix] dsa: no warning on an idle DP rank's DCP prepare (forward_mla_rocm) | BF (log) | — | Y (tiny) |
| D27 | da6e383513 | [Fix] dsa: Skip the indexer on an eager idle DP rank on ROCm too (dsa_indexer, test) | BF | DP2xDCP2 idle MAF gone (mixrun R18) | Y candidate |
| D28 | 5aacce1f1c | [Fix] forward-batch: Pad extend_seq_lens_cpu into a new list (forward_batch_info, test) | BF | eager x2 + graph 5/5, 40/40, no fault (R24/R25) | Y candidate |

Results with the branch: DCP4 PD best scored 18,321/chip (r5b A1 19,400; commit 2326727c message); cross-node with S10+S11 dcpx11
25,197; PP4 prefill + DCP4 ppdc2 C72 31,128 (all sim acceptance). Open: E2 (DP2xDCP2 graph) garbled outputs, not bisected.
Not upstream-checked against main for each commit [unverified]; rebase before filing.

## 4. Details — Mooncake, aiter, KF, Infera, tooling

### M01 — Mooncake PR #4340 port to faae8dd4
- Fix authored by salexspb (#4340 "[TE] Fix HIP and NVLink IPC relocation for suballocated buffers", OPEN); yihou ported it
  to faae8dd4, hand-rebasing the `transfer_engine_impl` hunks. Files: `hip_transport.{h,cpp}`, `transfer_engine_impl.cpp`.
- Root cause (R018/R019): HIP opens an IPC handle at the allocation base; faae8dd4 relocated with the registered pointer,
  so KV for interior registrations would land at offset (ptr - base), silently. Proven with a GPU te_pair test (interior
  region correct only with #4340). In production runs REGBASE (T01) showed `interior=0`, so the bug was not triggered.
- Used in every image since v0521-mc4340 and in the v0522 `-destpin` build (identical diff, `cmp` 10-27).
- Upstream: comment on #4340 with the faae8dd4 port; no new PR.

### M02 — high-priority Mooncake HIP copy streams (`MC_HIP_STREAM_PRIORITY=high`)
- File `src/transport/hip_transport/stream_pool.cpp`. Evidence: microbench peer-copy stall behind long kernels 200 -> 11 ms
  (R039/R043). **Not production-tested** (image mc4340prio built, `config.yihou.mc4340prio.sh` prepared, not run).
- Same mechanism as S10's priority gather stream (which was measured, but not separated from the metadata cache).

### M03 — batched HIP peer-copy kernel (`MC_HIP_BATCH_KERNEL=1`): never built; moot after R035 (blocks are large).

### K01 — KF decode kernel overlay (`yihou_kf`)
- Owner: yihou's KernelForge session (`/home/yihou/kf-glm52-yihou`, pack-ups `glm52.kf-decode-kernels-e2e.packup_20261023`
  and `glm52.kf-dcp-and-fused-indexer.packup_20261025`); kernels are KernelForge-generated candidates. Copy used here =
  `.../packup_20261025/overlay`, md5-identical to the overlay mounted in llr35 (`NFS/kf_e2e_20261023/kf/e2e/overlay`).
- Mechanism: `sitecustomize.py` + post-import hooks monkey-patch `triton_sparse_mla_fwd`, `aiter_paged_mqa_logits`,
  `DSATopKBackend.topk_transform`, `AiterRunnerCore.run` (+ DCP variants, `Gfx950FusedIndexer`), shape-gated, with a
  one-time CHECK against the original. **Not a diff against sglang**; `K01-*.NEWFILES.diff` adds the overlay files.
- Evidence: llr5 40,777 vs llr3 37,743 (+8.0 %, step p50 70.1 -> 55.3 ms); KF top-k engaged with
  `SGLANG_OPT_USE_TOPK_V2=false` llr7 +1.8 %; DPA C56 TPOT 14.7-15.3 -> 11.2-11.3 ms (KF pack-up 10-23). DCP KF: -5 %
  (indexer selector cap 8192 overflows on real inputs; 1-3 rows differ in 11/24 calls — **not exact**). Fused indexer
  1.36x kernel, e2e not measurable; its hook was a no-op on the llying image (llr22).
- Correctness: CHECK once per shape (sparse MLA identical, MoE SNR ~51 dB); overlay validated only on K5 verify rows,
  llr35 is K3 (llying note). In llr35 stack; bisect round 1 inconclusive for KF (masked).
- Upstream path: re-implement winning kernels in aiter (sparse MLA decode, paged MQA logits, MoE) with tests; the overlay
  itself is not upstreamable.

### A01 — aiter BF16 GEMM tuned csvs (gfx950, cu_num 256)
- `yihou_glm52_dpa_bf16_tuned_gemm.csv` (75 rows: merged c96 + K3 shapes, e.g. o_proj N=6144 K=16384) and
  `yihou_glm52_ptpc_bf16_tuned_gemm.csv` (49 rows: N=1024 K=6144, N=6144 K=512, shapes only with the FP8 draft), tuned on
  node 190 with aiter `gemm_a16w16_tune.py --with-hipblaslt`; mounted into `aiter/configs/model_configs/` (aiter merges
  model_configs csvs, log confirmed). Source `PK/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/aiter_retune/` (commit ce786d27).
- Evidence: llr32 47,447 vs llr29 47,014 (+0.9 %, = run spread ±0.9 %), step p50 43.3 -> 42.1 ms; 2nd csv llr35 -0.2 %
  (neutral, 0 untuned BF16 GEMM left). In the llr35 stack; "no-GEMM-csv" is a round-2 bisect variant (pending).
- Upstreamability: aiter accepts per-model tuned csvs; tuned M are exact decode M values only; re-tune on aiter main.
  Wait for llying's round 2.

### I01-I07 — Infera harness / kit / docker
- I01 `engine.sh` role-scoped extra args/env (`PK/glm52-mtp-garbled-decode-rootcause.packup_20260918/patches/`): enabled
  the R05/R06 bisect that found `--disable-custom-all-reduce` as the v0.5.19 garble cure; no-op by default. Adapted from a
  parallel session's patch; their `NCCL_SOCKET_IFNAME` hunk deliberately excluded.
- I02 bench wiring for llying's router affinity (99fa0406, not ours): without it the router container gets no `-e` flags and
  the feature is dead code; translates 1/0 to true/false (clap rejects 1/0). C32/C40 had 0 affinity-503s.
- I03 same-node 1P1D harness: topology accepts one host; `ENGINE_PORT_STRIDE` (stride 1 overlapped 9 of 10 derived ports ->
  ZMQ "Address already in use"); same-node `/health` start gate (necessary, not sufficient); mirrored in `agentx_env`.
- I04 router conflict resolution + PD DP-rank affinity on upstream `pick_targets`/sessions (router did not build).
- I05 docker recipes (bundle `infera-src.yihou.28121cb4..702af10f.bundle` in the oai pack-up): v0.5.21 base + GLM-5.2
  patch re-cut (dsa_page_table_rows, NextN fusion arch, wait-event; drops #32196/#41395/#41756/#37152 now upstream);
  v0521-mc4340 (M01); pipefail artifact-check fix; mc4340prio (M02). Whether current Infera main (v0522 recipe, #210)
  supersedes these: **[unverified]**.
- I06 harness lineage diffs vs `bench/glm5p2_pd` @ 28121cb4 (`DECODE_PID_OFFSET` against CLR same-PID IPC guard and ROCr
  `@hsa<pid>` socket collision, R012/R020/R022; `NODE_RW_ROOT`; `IPC_MODE_LEGACY`/`ENGINE_PID_HOST`/`DEBUG_AGENT`; per-leg
  cpusets R045; role-only mounts R050; agentx_env knob validation). Defaults keep the command line byte-identical.
- I07 kit hooks on `examples/sglang_2p1d_glm5.2` (Infera 06796e13 + llying kit-v6-pp4): `DECODE_CUSTOM_AR=1` strips
  `--disable-custom-all-reduce`; `DECODE_ENV_EXTRA` / `PREFILL_ENV_EXTRA`; `DECODE_MOUNT_EXTRA` (space-separated list).
  Diff = `NFS/llrepro_20261023/{worker,common}.sh.before-kfhook` vs current kit. Note `DECODE_CUSTOM_AR` turns on a
  feature implicated in garbling.
- Harness-only, no diff extracted: `config.yihou.dcpx.sh` (DCPX_* knobs incl. prefill EAGLE flags `DCPX_PMTP`, graph bs
  list), `install_pp.yihou.sh` (agentx_env TP*PP fix) in `PK/glm52.ppdc-pp4-dcp4.packup_20261023/`; analysis tools
  (`llsum`, `bneck`, `qseries`, `isl_series`, `guard_llr`, `jsonl_score`, `queue_runs`) in the llrepro pack-ups.

### T01-T07, X01-X02
T01 Mooncake conn regbase interior-registration log (diagnostic for M01; all runs interior=0). T02 DSA seed guard (0 bad
entries in ~1.24 M rows). T03 DSA indexer top-k guard v5 (found the BAD k-only calls for S05; the `sgl_kernel/top_k.py`
SKGUARD files are in `PK/glm52.v0521-idxshare-maf-fix.packup_20261021/scripts/mounts-src/`, no image original extracted, no
diff). T04 S11 + xfer stats. T05 decode memlog (allocator state per Decode-batch line). T06 decode internal bench port +
shape capture (for `infera.dev.yihou.sglang.bench.fast.script`, Infera #164 draft). T07 `hipxfer_trace.yihou.c`
LD_PRELOAD timing shim. X01/X02 synthetic 8-/N-expert models (non-faithful; showed prefill KV is the lever: 25,905 /
25,297 synthetic vs 22,306).

## 5. Not ours (do not file as ours)

| item | owner | where it appears |
|---|---|---|
| Router PD DP-rank affinity (rust) 99fa0406 | llying (`origin/llying/bench/sglang_glm5p2_agentx`) | `patches/NOT-OURS-llying-99fa0406-*.patch` |
| Router decode-input-demand R2 (#188), decode-request-weight (#213), v0522 image + 2p1d kit (#210) | llying, merged | llr kit |
| llying kit v6-pp4 (incl. `patches/deepseek_v2.py` PP fix, see S13), Mooncake `MC_*` env defaults | llying | llrepro pack-ups |
| Mooncake dest-local-rail, dest-pinned-hca, ib-timeout patches | llying | `patches/NOT-OURS-llying-mooncake-*.diff` |
| Mooncake #4340 | salexspb (open) | M01 is our port |
| sglang #42553 GTT HiCache | jiejingzhangamd, merged 10-12 | used as config |
| sglang #37152 HiCache JIT copy rounds | jiejingzhangamd, merged 09-19 | v0519 image |
| sglang #42163 room shard | YAMY1234, open | compare S03/S11 |
| sglang #44322 triton sparse-MLA DCP | EricKing626, open | base of D-series |
| GLM NextN fused-shared-experts arch fix 4350d37c5 | xiaobochen-amd fork | v0519/v0521 images |
| Infera sglang recipe patches (dsa_indexer HIP DP padded rows, dsa_dp_sync, dsa_page_table_rows, wait event, implicit ROCm rejection sampling, responses tool args, HiCache staged write-back) | liyingli / xiaobochen-amd / leiwei12 (git log); yihou co-authored parts in Aug (Infera #58, #79) | D-dcp-branch 0001-0007 |
| ATOM PR #209 items (MLA bmm per-M configs, MXFP4/FP8 draft idea, TOPK_FORCE_PATH) | ChengYao-amd, merged | llr28 neutral |
| `SGLANG_GLM_NEXTN_MOE_PTPC` (FP8 draft experts), `SGLANG_PP_COMM_OVERLAP`, `--pp-async-batch-depth` | existing code in the v0522-fdbafb60 image; origin not traced [unverified] | config_findings |

## 6. Earlier upstream PRs by yihou (fork dorado269), before this window — still open

`gh search prs --author dorado269` 10-27: sglang #33059 "Fix DSA indexer aiter (HIP) padding mismatch under DP-attention"
(open), #33968 "[ROCm] Fix HiCache host-pool allocator: hipHostRegister's device pointer is not the host VA" (open), #33970
"[PD] Make the mooncake KV transfer wait on the prefill forward that wrote the pages" (open); #33973 / #33643 closed.
Infera #58 / #79 (GLM-5.2 PD + DPA + EAGLE enablement, agentic fixes) merged. These belong to the July-August phase and
are listed only so the PR agent does not duplicate them.

## 7. Suggested PR order (by evidence x effort)

1. Shepherd #45483 (S05) and #45484 (S06) — filed, need review/CI.
2. S07 prefill_complete validation — 1 line + unit test.
3. S13 PP hidden size — 2 lines; agree attribution with llying first.
4. S10 DCP pack fix — biggest measured gain (+23.8 %); rebase on main, strip instrumentation, add the equality test, check
   the PP no-wait ordering question first.
5. S01 non-blocking KV-metrics send (with a test); S11/S03 as review data on #42163.
6. D-series bug fixes (D17, D19, D23, D27, D28) after #44322 lands; split indexer (D14..D31) as one perf PR.
7. A01 aiter csvs and K01 kernels only after llying's correctness round 2.
