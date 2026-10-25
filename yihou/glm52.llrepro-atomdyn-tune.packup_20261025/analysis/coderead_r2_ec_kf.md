# R2 (verified)

## R2 verification: claims re-checked

**Answer to the user's question:** R2 was never switched off in the llying kit. `engine/up.sh:51` hardcodes `--decode-input-demand on`. The router startup line reads `decode_input_demand: On` in every llr run, llr1 through llr8. Our ppda1 did not have R2 because the env never set it; ppda3 added it.

I re-ran every read myself, with bounded commands. Two small helper files were written in yihou paths: `135:/tmp/yihou/{llr7,ppda3}-router.clean.log` (router logs with colour codes stripped) and `/shared_nfs/yihou_glm52/llrepro_20261023/yihou-r2v-dp.pl`. Nothing was deleted and no running container was touched.

### Claim-by-claim

**1. How R2 was enabled in ppda3**
- `config.yihou.dcpx.sh:144-148`: `DCPX_R2` becomes `ROUTER_EXTRA_ENV += INFERA_R2_DECODE_DEMAND=$DCPX_R2`. **CONFIRMED.**
- `run.v3x.yihou.sh`: the `-e` list is built at line 75 and spliced into the router `docker run` at line 78. **CONFIRMED.** The report said 75-77, which is a minor line-number slip.
- `env.ppda3-135x136.yihou.sh:12` contains `DCPX_R2=on`. **CONFIRMED.**
- Dryrun line 29 (`dcpx_20261022/packup/stage4.yihou/results/dryruns/dryrun.ppda3-135x136.yihou.txt`) has `INFERA_PD_DP_RANK_AFFINITY=false`, `INFERA_R2_DECODE_DEMAND=on`, image `infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin`, `infera.server`, `--router-policy kv-aware`, and no `--decode-input-demand`. **CONFIRMED.**
- The router binary is `/usr/local/bin/infera-router` (first log line). **CONFIRMED.**
- Container `glm52-v0521-yihou-135-ppda3-router` is exited, and its env includes `INFERA_R2_DECODE_DEMAND=on`. **CONFIRMED.**
- There is no router log under `runs/ppda3`; `server-logs` holds only `decode-0.a1.log` and `prefill-0.log`. The copy is at `135:/tmp/yihou/ppda3-router.yihou.log` (29.5 MB). **CONFIRMED.**
- The startup line `experiments=Experiments { prefill_guard_completion: false, decode: On, ... }` is in the log. **CONFIRMED.**
- The `routing_experiments` lines carry `demand_mode=On`. **CONFIRMED**, but these are per-candidate lines, not one per pick: 40,224 candidate lines, of which 10,056 have `selected=true`.
- Selected picks per rank: dp0 2406, dp1 2708, dp2 2451, dp3 2491. **CONFIRMED.**
- 8,437 of the 10,056 selected candidates have `legacy_selected=false`. **CONFIRMED.**

**2. Router in the llying kit (llr7)**
- Image `infera-sglang:v0522-fdbafb60`, Id `sha256:f69f49e3...`, created 2026-10-23T15:24:01Z. **CONFIRMED.**
- The router is `infera-router`, launched via `python3 -m infera.server --router-backend rust`. **CONFIRMED.**
- Infera lives at `/opt/infera/infera` (also `/opt/venv/.../site-packages/infera`). `_version.py` says `__version__ = '0.0.0'` and `__commit_id__ = None`. **CONFIRMED.** The actual commit is unknown.
- `up.sh:44-52` launch arguments, including `--decode-input-demand on "${router_extra[@]}"`. **CONFIRMED.**
- `llr7.site.env:9` has `ROUTER_EXTRA_ARGS=` (empty). **CONFIRMED.**
- The `yihou-llr-router` container has 0 `INFERA_*` env entries, and its Args contain `decode-input-demand on`. **CONFIRMED.**
- The startup config has `router_policy: "kv-aware"`, `pd_prefill_guard_release: Completion`, `session_affinity: Prefill`, `decode_input_demand: On`, and no `pd_dp_rank_affinity` field. **CONFIRMED.**

**3. Does that router support R2?**
- `--help` shows `--decode-input-demand <off|shadow|on>`, env `INFERA_DECODE_INPUT_DEMAND`, default `off`. **CONFIRMED.**
- Literal grep counts in the binary: `INFERA_R2_DECODE_DEMAND` 0, `INFERA_PD_DP_RANK_AFFINITY` 0, `INFERA_DECODE_INPUT_DEMAND` 1. **CONFIRMED.** That our env name would be "silently ignored" on this image is an inference; I did not test it.
- No kit change is needed. **CONFIRMED**, and it holds more widely than the report said: all of llr1 through llr8, not just llr5-7.
- llr7 decode picks (`policy pick role=Decode`): 10,838 total; dp0 2612, dp1 2991, dp2 2717, dp3 2518. **CONFIRMED.**
- The word "demand" appears only once in the llr7 router log, in the config line. This router does not log the demand cost per pick.
- Whether `picked=` is the decision after demand weighting is **UNVERIFIED**: no source was read.
- The "ppda1 equal 2379×4" figure comes only from the comment in `prep_ppda3.yihou.sh:3`. **CONFIRMED** that this is the source. That same comment wrongly says llying runs `INFERA_R2_DECODE_DEMAND=on`; llying actually uses `--decode-input-demand on`.

**Other router differences**
- ppda3 logs `session affinity configuration mode=Off ttl=3600`, against llr7's `session_affinity: Prefill`. **CONFIRMED.**
- ppda3 has `prefill_guard_completion: false`, against llr7's `pd_prefill_guard_release: Completion`. **CONFIRMED.**

**4. Per-rank decode table**
I recomputed it from both logs and every number matches the report. **CONFIRMED.** The claim that per-rank averages are "within about ±5 %" is approximate: llr7 average running requests span 9.0 to 9.9, about -6 % / +3 % around the mean.

### Corrected final report

| | ppda3 (ours) | llr7 (llying kit) |
|---|---|---|
| How R2 is enabled | env `INFERA_R2_DECODE_DEMAND=on` (`DCPX_R2=on`) | CLI `--decode-input-demand on`, hardcoded at `up.sh:51` |
| Image | `infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin` | `infera-sglang:v0522-fdbafb60` |
| Router binary | `/usr/local/bin/infera-router` | `/usr/local/bin/infera-router` (version `0.0.0`, commit None) |
| R2 shown in the router log | `decode: On`; per candidate `demand_mode=On` | `decode_input_demand: On` |
| Final decode picks dp0/dp1/dp2/dp3 | 2406 / 2708 / 2451 / 2491 (selected=true, total 10,056) | 2612 / 2991 / 2717 / 2518 (total 10,838) |
| Session affinity | `mode=Off` | `Prefill` |
| Prefill guard | `prefill_guard_completion: false` | `pd_prefill_guard_release: Completion` |

- In ppda3, 8,437 of 10,056 final decode choices differ from the legacy choice (`legacy_selected=false`).
- In fdbafb60 the equivalent switch is `--decode-input-demand` / env `INFERA_DECODE_INPUT_DEMAND` (default `off`). Our env name does not appear in that binary.
- Decode-log per-rank averages (sampled `Decode batch` lines, warm-up included):

| Run | DP | samples | avg running reqs | avg token usage | share of samples with usage ≥ 0.95 | prealloc | transfer | queue |
|---|---|---|---|---|---|---|---|---|
| llr7 | DP0 | 1812 | 9.0 | 0.707 | 0.168 | 0.43 | 3.97 | 0 |
| llr7 | DP1 | 1809 | 9.9 | 0.715 | 0.190 | 0.38 | 4.04 | 0 |
| llr7 | DP2 | 1812 | 9.9 | 0.707 | 0.166 | 0.38 | 4.16 | 0 |
| llr7 | DP3 | 1814 | 9.6 | 0.713 | 0.174 | 0.42 | 3.80 | 0 |
| ppda3 | DP0 | 1304 | 12.6 | 0.709 | 0.145 | 0.29 | 2.74 | 0.10 |
| ppda3 | DP1 | 1296 | 12.1 | 0.702 | 0.149 | 0.26 | 2.60 | 0.08 |
| ppda3 | DP2 | 1301 | 11.8 | 0.702 | 0.142 | 0.42 | 2.55 | 0.04 |
| ppda3 | DP3 | 1297 | 11.6 | 0.703 | 0.154 | 0.34 | 2.83 | 0.11 |

  Every rank in both runs is at usage ≥ 0.95 in 14-19 % of samples. These whole-run averages show no rank that is persistently out of line.

**Open questions**
- Does llr7's `picked=` reflect the demand-weighted decision? Answering it needs a read of the router source or a `shadow`-mode log.
- Is there short-term rank imbalance at the moments a rank is full? Answering it needs a per-timestamp comparison across ranks.
- Which Infera commit is inside each image?

Files used for the checks:
- `/shared_nfs/yihou_glm52/llrepro_20261023/kit/sglang_2p1d_glm5.2/engine/up.sh`
- `/shared_nfs/yihou_glm52/llrepro_20261023/llr7.site.env`
- `/shared_nfs/yihou_glm52/llrepro_20261023/work/runs/llr7-c80/logs/yihou-llr-router.log`
- `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/config.yihou.dcpx.sh`
- `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/run.v3x.yihou.sh`
- `/shared_nfs/yihou_glm52/g4tune_20261019/env.ppda3-135x136.yihou.sh`
- `135:/tmp/yihou/ppda3-router.yihou.log` and the cleaned copies in `135:/tmp/yihou/`
- `/shared_nfs/yihou_glm52/llrepro_20261023/yihou-r2v-dp.pl`

# EC (verified)

Verification of the EC report. All reads were bounded and read-only, and I touched no containers.

## Claim-by-claim verdict

**§1 The patch**
- Pack-up `yihou/glm52.dcp4-decode-gmu.packup_20261023/` is in commit 2ecf34d3, with patches 01/02 (`metrics_reporter.py`) and 03 (`decode.py`): CONFIRMED.
- Base sglang `ab7a898b77` on image `infera-sglang:v0522-d395c3b3-1020-mc4340`: CONFIRMED (`environment.md`).
- Overlay `.../src/sglang-ab7a898b77-memlog-ec2.yihou`: CONFIRMED. Its decode.py contains "yihou gmu" twice. The diff against `sglang-ab7a898b77` is two 4-line insertions after lines 3074 and 3121 (8 changed lines).
- Code and condition (every 8th loop iteration, every rank, free memory < 6 GiB): CONFIRMED (the patch file, and OPTIMIZATION.md §1).
- "03 alone was not run": CONFIRMED (OPTIMIZATION.md §1).

**§2 Porting to the kit image**
- The image's sglang is at `e9b22a0818`: CONFIRMED (`git rev-parse` in the image). The working tree has local edits, but decode.py is not among them.
- `ec_port.yihou/decode.kit-e9b22a08.orig.py` is identical to the image file (md5 `f20af4ff...`): CONFIRMED.
- Kit decode.py vs ab7a898b77: CONFIRMED. The 121 is the full diff output length; the actual changed lines (`<`/`>`) number 101.
- Anchor at lines 3049 and 3096, `import torch` at line 33, and the port is exactly the two hunks: CONFIRMED (diff of orig vs ec2).
- "Compiles with py_compile": UNVERIFIED (not re-run). A `__pycache__/` directory exists in `ec_port.yihou/`, which is consistent with it.
- "No other `process_decode_queue()` callers": UNVERIFIED. I only checked decode.py, which has exactly two.
- `worker.sh:146` takes a single `-v "$DECODE_MOUNT_EXTRA"`: CONFIRMED. llr5, llr6, llr7 and llr8 `site.env:12` all use that slot for the KF overlay.
- Added by me: the KF overlay contains no decode.py and no hooks on disaggregation, empty_cache or process_decode_queue (grep). It is loaded via `DECODE_ENV_EXTRA="PYTHONPATH=/yihou_overlay ..."` (`llr8.site.env:13`), so the two mounts do not collide at the file level.
- The kit decode is DPA4 (`--tp-size n --dp-size n --enable-dp-attention`, `worker.sh:111`), and `--mem-fraction-static 0.92` is hard-coded at `worker.sh:112`: CONFIRMED.

**§3 What the other session measured**
- Every value in the abort table and the patched-run table: CONFIRMED (OPTIMIZATION.md §2). One correction: the report says "both runs completed", but **three** patched runs completed. It left out S2o (burst node v3-019, SDMA=0, 0.92, 24,114 tok/s/chip, which they marked "not comparable").
- Their mechanism: CONFIRMED. In S2e allocated grew +0.8 GiB, reserved +15.7 GiB, free fell 16.36 -> 0.92 GiB, and the abort came 2 s later. They say the link to the HIP error is "inferred from timing".
- `expandable_segments` and `garbage_collection_threshold` did not help: CONFIRMED.
- S2k (TP0 only): CONFIRMED. TP0 recovered; ranks 1-2 aborted.
- Their recommendation "0.88 + patch; 0.92 needs a cost measurement first": CONFIRMED.

**§4 The llr8 log**
- Start-up memory table: CONFIRMED for TP0 (pool lines 398-412; 3.80 GB after draft decode at line 1414; 2.22 GB at lines 1959-1960; llr7 10.46 GB at lines 1879-1880).
  - The capture costs are the same in both runs: 10.51, 1.17 and 1.58 GB.
  - Side note: target-verify capture begins at 23.73 GB in llr7 and 15.49 GB in llr8, which is higher than the "Memory pool end" value. The table is still correct as quoted.
- The warning comes from `triton_load_watch.py` and reads free memory with `empty_cache=False`; threshold 1.0 at `environ.py:1214`: CONFIRMED. I did not re-check the exact 98-115 line range.
- Warning counts, 144 in llr8 vs 0 in llr7: CONFIRMED.
- First warning at line 2110, 04:22:53, 0.89 GiB on all four ranks, before "fired up" at 04:22:54: CONFIRMED.
- 0.89 GiB through 04:51, then 0.77-0.97 at 04:54: CONFIRMED.
- **Partly WRONG on the low readings:** the 0.15 GiB on TP0 is at **04:55:21 (line 14211)**, not 04:55:23. The 0.07 GiB readings on TP0/TP1 and 0.78-0.93 on TP2/TP3 at 04:55:23 (lines 14225-14229) are CONFIRMED.
- Abort at line 14249: `rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES`, then "Fatal Python error: Aborted": CONFIRMED.
- **WRONG:** "The string 'Available Free mem : 0 MB' is not in this log". Line 14249 ends with `Code: 0x1008 Available Free mem : 0 MB`; the report's grep output was truncated.
- First Decode batch at 04:48:38 (line 9657); no retractions (0 lines with retracted-req >= 1): CONFIRMED.
- Per-rank running requests and token usage just before the abort: the exact ranges depend on the window chosen. In the last 40 Decode-batch lines I read DP0 7-21 / 0.63-0.80, DP1 7-13 / 0.66-0.80, DP2 9-13 / 0.68-0.77, DP3 8-10 / 0.70-0.81. That is the same picture as the report's: no retractions and usage below 0.82.
- No allocator allocated/reserved lines in the log: CONFIRMED (none found).

## Corrected final report

1. **The patch:** `patches/03-ec2-allrank.decode.diff` in commit 2ecf34d3, `yihou/glm52.dcp4-decode-gmu.packup_20261023/`. It targets sglang `ab7a898b77` on image `v0522-d395c3b3-1020-mc4340`, and the live overlay is `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-ab7a898b77-memlog-ec2.yihou`. It adds the following after `self.process_decode_queue()` in both decode event loops:
   ```python
   self._yihou_it = getattr(self, "_yihou_it", 0) + 1
   if self._yihou_it % 8 == 0 and torch.cuda.mem_get_info()[0] < 6 * 2**30:
       torch.cuda.empty_cache()
   ```
   This runs on every rank. The measured builds also carried patches 01 and 02; "03 alone was not run".

2. **Port to the kit image** `infera-sglang:v0522-fdbafb60` (sglang `e9b22a0818`):
   - The kit's decode.py differs from ab7a898b77 by 101 changed lines, so the old file must not be mounted.
   - The two hunks apply without conflict at lines 3049 and 3096. The ported file `ec_port.yihou/decode.kit-e9b22a08.ec2.py` is exactly orig plus the two hunks.
   - The kit's only decode mount slot (`worker.sh:146`, a single `-v`) is taken by the KF overlay (`llr*.site.env:12`). The KF overlay works through PYTHONPATH and has no decode.py, so there is no file-level conflict.
   - To apply it, `worker.sh:146` has to accept several mounts; the report's loop proposal works for that and needs the user's OK to edit the kit. Then mount `decode.kit-e9b22a08.ec2.py` onto `/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py:ro`.
   - Also: the kit hard-codes `--mem-fraction-static 0.92` at `worker.sh:112`.
   - The patch has only been measured on DCP4 (TP4 EP4 DCP4), never on DPA4 (`worker.sh:111`) or at 0.95.

3. **Their measurements:** same-node P4D4, DCP4 decode, custom AR on, C80, one 1800 s run each.
   - Without the patch, 0.87, 0.88 and 0.92 all aborted with HSA OUT_OF_RESOURCES (S2j; S2e/h/i/k; S2g092/a/b).
   - With the patch:
     - S2n at 0.88: 27,685 tok/s/chip.
     - S2m at 0.92: 22,540.
     - S2o at 0.92 on a burst node: 24,114, not comparable.
   - The 0.85 references without the patch were 25,941 and 25,433.
   - In S2e, reserved grew +15.7 GiB against +0.8 GiB allocated before the abort. They state the link to the HIP error is "inferred from timing".
   - Their recommendation: 0.88 + patch; 0.92 needs a cost measurement first.

4. **llr8 log** (`yihou-llr-decode.log`):
   - On TP0, available memory after capture was 2.22 GB vs 10.46 GB in llr7. The capture costs were identical, so the 8.2 GB difference is the larger KV pool (135.21 + 2.01 GB vs 126.99 + 1.88 GB).
   - Low-free warnings (with `empty_cache=False`): 144 in llr8, 0 in llr7. They read 0.89 GiB on all ranks from 04:22:53, just before serving started.
   - Lowest readings: 0.15 GiB on TP0 at 04:55:21 (line 14211), then 0.07 GiB on TP0/TP1 at 04:55:23. TP2/TP3 read 0.78-0.93 GiB.
   - Abort at line 14249 (04:55:26): `HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 0 MB`. That was about 7 minutes after the first Decode batch (04:48:38). There were no retractions and token usage was at or below 0.81.
   - The log has no allocator allocated/reserved data, so what consumed the memory is not visible in it.

**Open (UNVERIFIED):**
- Whether `empty_cache` frees enough at 0.95 on DPA4. The < 6 GiB condition would be true from the start of serving at that setting, so the patch would fire every 8 iterations.
- The cost of those calls.
- Whether the kit file compiles with the port. `py_compile` was not re-run; the `__pycache__/` directory is consistent with it.

Files: `/shared_nfs/yihou_glm52/llrepro_20261023/ec_port.yihou/` (decode.kit-e9b22a08.orig.py, decode.kit-e9b22a08.ec2.py, 03-ec2-allrank.decode.kit-e9b22a08.diff, README.md); `/home/yihou/dev/git.16-19/infera.glm52.pd/yihou/glm52.dcp4-decode-gmu.packup_20261023/OPTIMIZATION.md`.

# KF (verified)

Verdict: the report's cause holds up, with three small corrections. For TARGET_VERIFY steps with bs ≤ 8 (rows ≤ 48), the image's built-in gfx950 fused DSA indexer handles the call and returns early. `_get_topk_paged`, `aiter_paged_mqa_logits` and `topk_transform` are never reached, so the KF indexer patches cannot fire there.

I checked this against the image itself. The copied file `/shared_nfs/yihou_glm52/kf_e2e_20261023/coderead_fdbafb60/dsa/dsa_indexer.py` has the same md5 as `/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py` in `infera-sglang:v0522-fdbafb60` (7d2667ed…).

**Claim by claim**

1. Dispatch at `dsa_indexer.py:2033-2045` — CONFIRMED. Inside `forward_cuda` (starts at :1954), `if self.use_gfx950_fused_indexer:` calls `_gfx950_fused_decode`. If that returns non-None, it returns right there via `maybe_capture_indexer_topk`. The standard `_get_topk_paged` call for decode, verify and draft_extend_v2 is at :2262-2270. `aiter_paged_mqa_logits` is at :1408 (`_get_topk_paged` is defined at :1209).
2. The gate accepts only decode/idle or target_verify (:722-726) — CONFIRMED.
3. "It returns None only if in_graph" — WRONG as stated. `in_graph` is one of many None returns. Others are `metadata is None` (:719), `force_unfused_topk`, CP metadata, no bf16 copy, LoRA, pool page size, layer-split pool, rows outside `[1, MAX_ROWS]` (:778-779), seqlens or page-table mismatch (:783-789), and more. What is correct:
   - `in_graph` is `_is_in_piecewise_or_breakable_cuda_graph()`, called at :1986-1988.
   - That function is defined in `dsa_prefill_cuda_graph.py:34-35` as `is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph()`.
   - The verify capture is `backend=full` (decode log :419). So "the fused path also runs under full capture" is consistent with the code, but I did not read what those two helpers return under full capture.
4. `MAX_ROWS = 48  # bs 8 x num_draft_tokens 6` in `fused_decode.py:31` (in the image), imported at :712 and checked at :778 — CONFIRMED. I found no env knob for it in `dsa/`.
5. Enable condition at :464-469 — CONFIRMED: `not use_dsa_indexer_fusion and gfx950_shape_ok and gfx950_weights_ok and gfx950_fused_indexer_runtime_ok()`. :472-473 tests `enable_dsa_fused_indexer is True` only to log why the path is not used; the comment there reads "unset, this path is a default". Whether setting that arg to False turns the fused path off is UNVERIFIED.
6. "KF installed in all 4 TP procs, lines 296-311" — CONFIRMED with a minor correction. The patch lines for pids 299-302 run from :296 to :315 (the `forward_{extend,decode}` lines are :312-315). The "installed" lines are spread over :2-32 and :344-350.
7. "gfx950 fused DSA indexer enabled" on 4 ranks — CONFIRMED (:347, 348, 349, 351, not a contiguous :347-351).
8. Decline "row count is outside [1, 48]" at :442-448 — CONFIRMED. These lines fall in the bs=64 capture (rows 384). `_decline_fused` (:103-107) logs only the first occurrence of each reason, so these lines say nothing about smaller bs.
9. Target-verify capture (begin :419, end :1105-1112, bs list [1..8, 10, 12, …, 64]) — CONFIRMED:
   - bs 10-20 (rows 60-120): `indexer_logits` and `indexer_topk` log "-> KF" (pid 299 at :838-973).
   - bs 8 down to 1 (:988-1080): only `sparse_mla … -> KF`, with no indexer lines at all.
   - Addition: rows 132-384 in verify log "-> orig" because `INDEXER_MAX_ROWS = 128` (`__init__.py:150`, checked at :171 and :220).
10. Draft-extend capture (begin :1417, end :1879) holds all the rows ≤ 48 indexer "-> orig" lines (:1606-1709) and rows 60-120 "-> orig" (:1502-1605) — CONFIRMED. Draft extend is not target_verify, so `_verify` is false: the wrapper at `__init__.py:89-98` sets the mode from `fb.forward_mode.is_target_verify()`. Draft extend is also excluded by the fused gate (:722-726).
11. The `_verify` wrapper and import binding are not the cause — CONFIRMED as far as the logs go. When `_get_topk_paged` is reached in verify (rows 60-120), both patches pick KF. That would not happen if the wrapper or the patched name binding were broken.

**Corrected report**
- **Cause:** with bs ≤ 8 in target verify, `forward_cuda` (`dsa_indexer.py:2033-2045`) serves the indexer through `_gfx950_fused_decode` and returns before `_get_topk_paged` (:2268). The fused path is on by default when shape, weights and runtime qualify (:464-469) and the log shows it enabled on all 4 ranks (:347-351). It takes rows in [1, 48] (`MAX_ROWS = 48`, `fused_decode.py:31`; check at `dsa_indexer.py:778`).
- **KF indexer window:** KF applies only where verify reaches the standard path *and* rows ≤ `INDEXER_MAX_ROWS = 128`. With EAGLE's 6 tokens per request that is rows 60-120 (bs 10-20). Rows 132+ (bs ≥ 22) log "orig" because of KF's own 128-row cap, not the fused indexer.
- **The rows ≤ 48 "orig" lines** all come from the draft-extend capture (:1606-1709), which is correctly non-verify.

**Open questions**
- I did not read what the two `is_in_*_cuda_graph` helpers return under `backend=full`. The inference that the fused path runs there rests on the log: no indexer lines at bs ≤ 8 in verify capture while `sparse_mla` lines do appear.
- Whether `--enable-dsa-fused-indexer false` (or equivalent) disables the fused path. To decide, read how `gfx950_fused_indexer_runtime_ok()` and the exec config consume that arg.
- How KF compares with `Gfx950FusedIndexer` at rows ≤ 48 is not measured.
- The task's "33 % of steady-state steps have rows ≤ 48" figure is not re-checked here.
- After capture there are `indexer_topk rows=1` and `rows=4 -> orig` lines (:2122, :2135). Those row counts are not multiples of 6, and I have not established which mode produced them.

I created nothing and deleted nothing. The only containers I ran were two `--rm` code-read containers, `yihou-coderead-vf1` and `yihou-coderead-vf2`, on 136 (CPU only).