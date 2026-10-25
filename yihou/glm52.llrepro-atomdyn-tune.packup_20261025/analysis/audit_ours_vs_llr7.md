# Evidence audit: R2, measured gains missing from llr7, tokenizer/detokenizer workers (2026-10-24)

Our current best run is llr7 at 41,679 tok/s/chip (poll.md:1300). It is llying's kit plus three of our changes: KF decode kernels, decode custom all-reduce, and decode `TOPK_V2=false`.

- **R2:** it is llying's router feature, not ours, and llr7 already has it on.
- **Missing gains:** none of our measured gains is cleanly both missing from llr7 and applicable to its configuration. Two have partial evidence; the rest are DCP-only, v0.5.21-only, or were never measured on their own.
- **Workers:** the llr7 decode runs 4 tokenizer workers and 1 detokenizer worker. Detokenizer 4 was measured neutral in llr4.
- **Bottleneck (question 4):** not answered here. It depends on traces from llr7r, which is still running.

## 1. R2

**What it is.** R2 is the Infera router's decode input-demand routing (`--decode-input-demand`), written by llying.
- It landed on `origin/main` as e54e9c62 (llying-001, 2026-10-19, PR #188).
- The CLI flag is defined at `origin/main:infera/server/args.py:130`; the router reads it as the env `INFERA_DECODE_INPUT_DEMAND` (`origin/main:rust/router/src/config.rs:78`).
- Earlier env-only versions were also liyingli's: 8cb2914e (09-24), f3560cb1, 338926e5, d913b48c, 565461b3, c8b0069f.
- On our branch the older env name `INFERA_R2_DECODE_DEMAND` lives at `rust/router/src/routing_experiments.rs:97`.
- My earlier statement that R2 was our own optimization and was missing from the llr runs was wrong. The ledger already records the correction at poll.md:1305.

**Which runs had it**

| run | score | R2 | evidence |
|---|---|---|---|
| llr1 / 2 / 3 / 5 / 6 | 37,392 / 35,997 / 37,743 / 40,777 / 40,937 | ON | router log line 2: `decode_input_demand: On` |
| llr4, llr8 | – | ON | same router log line |
| **llr7 (best)** | **41,679** | **ON** | router log line 2; turned on by the kit itself (`engine/up.sh:51` passes `--decode-input-demand on`) |
| llr7r | running | not yet known | no log directory yet |
| g3x1, dpax1, ppda1, ppda2 (our own stack) | 25,524 / 26,321 / 34,288 / – | OFF | the dryruns contain no R2 or decode-input-demand string |
| ppda3 (our own stack) | 36,543 (+6.6 % over ppda1, poll.md:1278) | requested; whether it took effect is unverified | `DCPX_R2=on` became `-e INFERA_R2_DECODE_DEMAND=on` (`config.yihou.dcpx.sh:144-148`, dryrun:29); no CLI flag passed and no router log found |

The +6.6 % for ppda3 is a measured difference. That R2 caused it is not proven.

## 2. Our measured improvements that are not in llr7

The llr7 configuration is PP4 TP1 prefill, DPA4 decode, image v0522-fdbafb60. Values for llr7 come from the `server_args` lines in the decode and prefill0 logs, from `llr7.site.env`, and from the kit's `worker.sh`.

**(a) Measured gain, absent from llr7, and applicable to its configuration.** No item cleanly qualifies. Two have partial evidence:

| item | measurement | llr7 | caveat |
|---|---|---|---|
| Prefill mem fraction 0.90 | pf90 vs q1: 23,127 vs 22,306, +3.7 % (A2 only) | 0.88 (`worker.sh:90`) | The two runs were on different nodes, with the old TP4 DPA prefill. Whether it carries over to PP4 is not known. |
| OffloaderV2 G3 + `is_shuffled` patch | G3 beat G4 cross-node at C56: A1/A2 25,524 / 25,376 vs 24,976 / 24,398 (`G/README.md:28-30`) | not used (`offload_group_size -1`, no offloader mount) | We never showed a gain over no offload; the pack-up calls that spread "within what this kit cannot attribute" (offloaderv2 README:39). The patch is for v0.5.21. |

**(b) Measured gain, but not applicable to llr7.** Each is DCP-only, v0.5.21-only, or was not adopted:

| item | measurement | why it does not apply |
|---|---|---|
| dcp_pack fix (cached metadata + priority stream) | 19,997 → 24,753 (+23.8 %), pack 77.6 → 0.6 ms (`opt.readme.md:12-17`) | DCP decode only |
| Hashed transfer-queue shard | 19,691 → 21,302; transfer p50 21.3 → 11.3 s (`opt.readme.md:26-28`) | Only engages for a DCP relayout |
| Decode CUDA graph max bs 96 with 13 capture sizes | +1.8 %, inside the run-to-run spread; free memory 24.1 → 28.5 GB | Measured on DCP decode only |
| `YIHOU_XFER_SHARD=mix` | +7.5 %; transfer_kv average 2.16 vs 9.98 s (`g4tune-xfer README:20-29`) | v0.5.21; user chose not to adopt it on 10-20 |

**(c) Absent from llr7, but never measured on its own.** These are bundled settings or defaults:

| item | ours | llr7 |
|---|---|---|
| GTT host pool | in the 10-20 rule bundle | no GTT or USERPTR log line or env; the actual allocation mode is unverified |
| HiCache size per rank | 260 | 250, with `write_back` |
| `disaggregation_decode_extra_slots` / `host_receive_threshold` | 48 / 0.0 | 0 / 1.0 |
| Decode graph max bs / max running requests | 64 / 96 | 256 / 256 |
| Decode mem fraction 0.95 | llr2 result ambiguous; llr8 VOID (out of resources, poll.md:1303) | 0.92 |
| `HSA_USERPTR_FOR_PAGED_MEM=0` on the prefill | ppda3 | absent |
| `HSA_NO_SCRATCH_RECLAIM` | prefill 0 / decode 1 | 1 on both legs |
| Prefill mounts (conn, dcp_pack, deepseek_v2) and overlay 93034eba91 | ppda3 | overlay 8dee3f6bdb; the two overlays were never compared at code level |

**(d) Already present in llr7**

| item | measured effect |
|---|---|
| Cross-node RDMA 135 → 136 | +14–20 % over same-node |
| index_share | +0.9 %, TPOT 20.5 → 19.8 ms |
| KF decode kernels | +8.0 % (llr3 → llr5) |
| Decode custom all-reduce | +0.4 %, TPOT −4.8 % |
| `TOPK_V2=false` on decode | +1.8 % |
| R2 | (section 1) |
| Decode tokenizer workers 4 | no consistent effect |

PP4 prefill, decode EAGLE 5/1/6, and the Mooncake envs `MC_IB_TIMEOUT` / `MC_ENABLE_DEST_LOCAL_RAIL` / `MC_MAX_MR_SIZE` are also present. Those Mooncake envs are llying's own kit defaults (his original kit, `worker.sh:47-51`), so poll.md:1305 is wrong to list them as ours only.

## 3. Tokenizer and detokenizer workers

| run | decode tokenizer / detokenizer | prefill | source |
|---|---|---|---|
| g3x1 (our v0.5.21 best) | 4 / 4 | 1 / 1 | `decode-0.a1.log:23` |
| llr3 | 4 / 1 | 1 / 1 | decode log line 23 |
| llr4 | 4 / 4 | 1 / 1 | decode log line 23 |
| **llr7** | **4 / 1** | 1 / 1 | decode log line 25 |

- **Tokenizer 4 is in llr7.** It comes from the kit (`worker.sh:112`, decode only). On its own it measured A1 +1.7 % and A2 −2.3 % (poll.md:369,371), recorded as no consistent effect.
- **Detokenizer 4 is not in llr7**; the kit never passes the flag. Its only isolated measurement is llr4 vs llr3: 37,601 vs 37,743 (−0.4 %), same TPOT, TTFT p90 +9 %. The pack-up recorded it as "neutral, not kept".
- **The 4+4 setting was your 10-15 rule, not a measured gain** (poll.md:511). No completed 4+4 vs 1/1 comparison was found from that era.
- **To add detokenizer 4 to the kit:** append `--detokenizer-worker-num 4` to `DECODE_EXTRA_ARGS` in the site env, as `llr4.site.env:9` does.
- **One line in the handoff is wrong:** `HO:54` describes tw90 as 4/4, but tw90 used detokenizer 1.

## Unverified items (labelled, not concluded)

- Whether R2 actually took effect in ppda3, and which router build the destpin image carries.
- Whether the fdbafb60 image contains idlefix and the noblock / noabsorbcfg / nonfinal patches; the Dockerfile was not read.
- How the llr7 image allocates the host pool.
- Whether the fdbafb60 Mooncake honours `MC_IB_TIMEOUT` and `DEST_LOCAL_RAIL`.
- Where the extra_slots and host_receive_threshold difference comes from.
- Prefix-hit values for llr5–7, and why the llr runs hit 0.41–0.50 while ppda hit 0.70–0.74.
- Whether ofg2h4 or any later ofg*h round produced a clean 4+4 vs 1/1 comparison.

Scratch scripts from the audit are in `/tmp/yihou/audit/` (c.sh, c2.sh, c3.sh).

# Verified sub-reports

## A_R2

**R2 provenance and per-run status: verified audit (2026-10-24, read-only)**

Every claim in the input report checked out; I found nothing wrong. Two points stay open, both about ppda3 (see the end).

**1. R2 is llying's router feature, not ours.**
- `e54e9c62` is by llying-001 <liyingli@amd.com>, dated 2026-10-19: "feat(router): add opt-in Decode input-demand routing; key session affinity on X-Session-ID (#188)". `git branch -r --contains` lists it on `origin/main`.
- The related commits are all by liyingli:
  - `f3560cb1` (10-09), "add opt-in decode input-demand routing".
  - `338926e5` (10-15).
  - `d913b48c` (10-16), "expose decode demand ... as launcher flags".
  - `565461b3` (10-21), "--decode-request-weight".
  - `8cb2914e` (09-24) and `c8b0069f` (10-10, "R2 decode hit credit").
- The env name `INFERA_R2_DECODE_DEMAND` lives in `rust/router/src/routing_experiments.rs:97` on branch `dev/pd_opt/glm_5.2_agentx`. That file does not exist on `origin/main`.
- The `origin/main` router reads `INFERA_DECODE_INPUT_DEMAND` (`origin/main:rust/router/src/config.rs:78`).
- The CLI flag `--decode-input-demand` is defined at `origin/main:infera/server/args.py:130` and passed through at `infera/server/launch_rust.py:134`.

**2. R2 status per run**

| run | score (ledger line) | R2 | evidence |
|---|---|---|---|
| llr1 | 37,392 (poll.md:1281) | ON | `llr1-c80/logs/yihou-llr-router.log` line 2: `decode_input_demand: On` |
| llr2 | 35,997 (poll.md:1284) | ON | same line 2 |
| llr3 | 37,743 (poll.md:1289) | ON | same line 2 |
| llr4 | not checked | ON | same line 2 |
| llr5 | 40,777 (poll.md:1296) | ON | same line 2 |
| llr6 | 40,937 (poll.md:1298) | ON | same line 2 |
| llr7 | 41,679 (poll.md:1300) | ON | router log line 2 `decode_input_demand: On`; it comes from kit `engine/up.sh:51` `--decode-input-demand on` |
| llr8 | not checked | ON | same line 2 |
| llr7r | running | UNVERIFIED | `work/runs/llr7r-c80/logs` does not exist yet |
| g3x1 | 25,524 at C56 (poll.md:1178) | OFF according to dryrun | `dryrun.g3x1-135x136.yihou.txt` has 0 matches for R2_DECODE, DECODE_INPUT_DEMAND or decode-input-demand; it has only `INFERA_PD_DP_RANK_AFFINITY=true` |
| dpax1 | 26,321 (poll.md:1232) | OFF according to dryrun | 0 matches; `INFERA_PD_DP_RANK_AFFINITY=true` |
| ppda1 | 34,288 (poll.md:1278, as the ppda3 baseline) | OFF according to dryrun | 0 matches; `INFERA_PD_DP_RANK_AFFINITY=false` |
| ppda2 | not checked | OFF according to dryrun | 0 matches; `INFERA_PD_DP_RANK_AFFINITY=false` |
| ppda3 | 36,543 (poll.md:1278) | set ON, but whether it took effect is UNVERIFIED | see below |

ppda3 evidence:
- `env.ppda3-135x136.yihou.sh:1,12` sets `DCPX_R2=on`.
- `harness/config.yihou.dcpx.sh:144-148` turns that into `ROUTER_EXTRA_ENV=INFERA_R2_DECODE_DEMAND=on`.
- `dryrun.ppda3-...txt:29` starts the router with `-e INFERA_R2_DECODE_DEMAND=on` on image `infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin`, and passes no `--decode-input-demand` flag.

**3. Current best is llr7 (41,679), and R2 is ON there.**
llr7 is the llying kit plus KF decode kernels, decode custom AR, and `TOPK_V2=false` (poll.md:1296/1298/1300). R2 is on because the kit itself enables it (`up.sh:51`), not because we added it. Every llr run has R2 ON according to its router startup log. The dryruns show R2 off in our own-stack runs g3x1, dpax1, ppda1 and ppda2. ppda3 is the only own-stack run that tried to turn it on. The earlier claim that R2 was our own optimization and was missing from the llr runs was wrong, and the ledger already corrects it at poll.md:1305.

**Open items about ppda3**
- **Was R2 actually active?** We set the branch env name (`INFERA_R2_DECODE_DEMAND`), not the main-router name or the CLI flag. It depends on which router build is inside the destpin image, and I could not check that (docker on the nodes was off-limits, and no ppda3 router log was found). A ppda3 router startup log, or the router source inside that image, would settle it.
- **What caused the +6.6 % over ppda1?** poll.md:1278 measures ppda3 at +6.6 % over ppda1. That is a measured difference. That R2 caused it is not proven.

## B1

# B1 verification: the measured items from 10-12 to 10-21 (v0.5.21), and whether each is in llr7

I re-read every claim below myself. All reads were read-only. I did not use docker and did not write any files.

Verdict: the report is largely correct. Four corrections are needed (section 0). Every other row was CONFIRMED with the sources given, apart from the rows labelled UNVERIFIED.

## 0. Corrections to the report under review

1. **Wrong path:** `bench-harness/config.sh:53,67,69,93`. The real file is `G/scripts/harness/config-chain/config.sh`. Its line numbers and values match:
   - `:53` is `PREFILL_HSA_NO_SCRATCH_RECLAIM` default 0
   - `:67` is `DECODE_GRAPH_MAX_BS` default 64
   - `:69` is `DECODE_HSA_NO_SCRATCH_RECLAIM` default 1
   - `:93` is `SGLANG_OPT_USE_TOPK_V2` default false
2. **Item 13 cites the wrong lines for the transfer numbers.** "transfer_kv p50 ~1.1 s (same-node ~7.5 s)" is at `G/notes.md:34`. The same-node range is 6.2–7.5 s at `G/README.md:23`.
3. **The handoff file contradicts the tw90 run on detokenizer workers.** `HO:54` describes tw90 as "decode tokenizer/detokenizer 4 workers". The tw90 pack-up README (`c48-followups README:10`) says `detokenizer_worker_num` was left at its default of 1. The report's item 9 is correct: tw90 changed only the tokenizer. `HO:54` is the inaccurate line.
4. **Item 15, GTT host pool: one point upgraded from UNVERIFIED to confirmed.**
   - The llr7 prefill log has zero lines matching `hipHostMalloc|USERPTR|GTT` (I ran `grep -c` over the whole of `yihou-llr-prefill0.log`).
   - `HSA_USERPTR_FOR_PAGED_MEM` does not appear anywhere in `K/engine/worker.sh` (grep).
   - What the llr7 image actually does when allocating the host pool remains UNVERIFIED. I did not read the Dockerfile.

## A. Changes we measured, with their llr7 status

The llr7 values come from `server_args` at line 25 of `yihou-llr-decode.log` and `yihou-llr-prefill0.log`, from `llr7.site.env`, and from `K/engine/worker.sh` (mtime 10-24 00:52, before llr7 started).

| # | item (knob) | measured (source) | verdict then | in llr7 |
|---|---|---|---|---|
| 1 | Prefill mem 0.88 → 0.90 | pf90 A2 23,127 vs q1 22,306 (+3.7 %), on different nodes (`CON:15-17`) | 0.90 kept (`DR:19` shows 0.90) | **Absent.** The PP4 prefill uses 0.88 (`worker.sh:90`; `PL7`). Different topology. |
| 2 | Decode max_running 64 → 96 | Bundled with mem 0.95, "not isolated" (`HO:74`) | kept (`DR:27` = 96) | **Different:** 256 (`DL7`; `worker.sh:69`) |
| 3 | HiCache size per rank | 50 vs 450: 20,112 vs 22,114 (−9.1 %). 360: 21,342. Nodes and builds differ (`CON:52-53`). Set to 260 for the GTT pool (`LED:1053`). | 450, then 260 | **Different:** 250 with `write_back` (`worker.sh:100-101`; `PL7`) |
| 4 | `write_through_selective` | wts48 21,226 vs q1 22,306 (−4.8 %) (`CON:54`) | reverted | **Absent.** llr7 uses `write_back`, which we never measured on its own. |
| 5 | runshare v1/v2 | v1 −3.8 %; v2 −2.2 % / −0.3 % (`CON:48-49`) | reverted | Not applicable |
| 6 | `prefill_complete` decode policy | A2 +4.4 %, A1 −5.6 %, sign inconsistent (`CON:50`) | abandoned | Not applicable |
| 7 | OffloaderV2 G8/G4/G3 + `is_shuffled` patch | G8: A1 +2.4 %, A2 −1.3 %. Prefill KV +6.9 / +15.6 / +21.7 / +32.9 % (`CON:114`). The G4 / G3 / no-offload spread is "within what this kit cannot attribute" (`offloaderv2-highconc README:39`). Cross-node C56: G3 A1/A2 25,524 / 25,376 vs G4 24,976 / 24,398 and 24,362 / 24,896 (`G/README.md:28-30`). | G3 kept (`DR:19` group size 3) | **Absent.** `offload_group_size: -1` (`PL7`). `worker.sh:139-149` has no offloader mount. |
| 8 | Offload prefetch step 2 | 21,342, inside the same-node band 20,516–21,814 (`G/README.md:23,25`; `G/notes.md:5-6`) | not kept | Not applicable |
| 9 | Decode tokenizer workers 4 | tw90 vs pf90 on the same node: A1 +1.7 %, A2 −2.3 %, "no consistent effect" (`c48-followups README:27`) | user rule | **Present:** `tokenizer_worker_num: 4` (`DL7`; `worker.sh:112`) |
| 9b | Decode detokenizer workers 4 | Not isolated in this window. Became a user rule on 10-15 (`offloaderv2-highconc README:10`). Later measured as llr4 37,601 vs llr3 37,743, "neutral, not kept" (spur `glm52.llrepro-pp4-dpa4.packup_20261024/README.md:13`). | rule | **Absent:** `detokenizer_worker_num: 1` (`DL7`). g3x1 had 4 (`runs/g3x1/server-logs/decode-0.a1.log:23`). |
| 10 | `YIHOU_XFER_SHARD=mix` vs `dest` | A2 22,854 vs 21,262 (+7.5 %); transfer_kv avg 2.16 vs 9.98 s (`g4tune-xfer README:20-29`) | **not adopted** (user decision 10-20) | Not applicable. g3x1's xferfix2 mount was "log-only" (`LED:1185`). |
| 11 | Chunked prefill 65536 | A1 21,092. A2 crashed with HSA_STATUS_ERROR_OUT_OF_RESOURCES (`G/README.md:26`). g4xc1 was stopped in warm-up (`g4tune-xfer README`). | reverted | Not applicable. llr7 prefill uses chunk 8192 and max_prefill 8192 (`PL7`). |
| 12 | index_share true + idlefix | idxf1a 21,814 vs g4n3 22,045 (−1.0 %), "not a performance claim" (`idxshare-maf-fix README:16,77-78`). Without the fix, the decode hit a MAF 23–100 min into load (`:21`). | rule | **index_share present** in both legs (`DL7`/`PL7`, set in `SE7:9`; overrides the `false` at `worker.sh:70`). In the llr kit it measured +0.9 %, TPOT 20.5 → 19.8 ms (`LED:1289`). **idlefix in the llr7 image: UNVERIFIED.** 0 MAF observed in llr3 and llr7 (`LED:1289,1300`). |
| 13 | Cross-node RDMA, 135 → 136 | +14–20 % vs same-node mean 21,289. ISL 4–7 % lower, not normalised (`G/README.md`). transfer_kv p50 ~1.1 vs ~7.5 s (`G/notes.md:34`). | kept | **Present.** 135 → 136, GPU4-7, 8 rails (`SE7`). |
| 14 | Concurrency | Same-node C52 A2 23,367 (A1 23,048). G3 C60 23,905, C64 22,860 (`G/README.md`). | C56 | Not applicable. llr7 runs C80. |
| 15 | 10-20 rule bundle: GTT pool + HiCache 260 + index_share + decode custom AR | Bundle only (`LED:1053`). `HO:74`: custom AR and GTT 260 "not isolated / not shown to help". | rule | **GTT pool:** no GTT/USERPTR log line or env in llr7 (see correction 4); the mode itself is UNVERIFIED. **Custom AR present:** `disable_custom_all_reduce: False` (`DL7`). Later measured as llr6 +0.4 %, TPOT 16.5 → 15.7 ms (`LED:1298`). |

## B. Settings in g3x1 that we never measured on their own

| item | g3x1 (source) | llr7 (source) | status |
|---|---|---|---|
| Decode cuda graph max bs | 64 (`DR:27`; `config-chain/config.sh:67`) | 256 (`DL7`) | differs |
| Decode mem | 0.95 (`DR:27`) | 0.92 (`DL7`). llr2 at 0.95 was ambiguous (`LED:1284`); llr8 at 0.95 aborted with OUT_OF_RESOURCES (`LED:1303`). | differs |
| `decode_extra_slots` / `host_receive_threshold` | 48 / 0.0 (`decode-0.a1.log:23`) | 0 / 1.0 (`DL7`) | differs; origin UNVERIFIED |
| `SGLANG_OPT_USE_TOPK_V2` | false on both legs (`DR:19,27`) | decode false (`SE7`; llr7 +1.8 % vs llr6, `LED:1300`); prefill true (`worker.sh:42`) | partial |
| `HSA_NO_SCRATCH_RECLAIM` | prefill 0, decode 1 (`DR:19,27`) | 1 on both legs (`worker.sh:40`) | differs, never measured |
| Mooncake env | `MC_TE_FILTERS` ionic_0-3, `MC_ENABLE_DEST_DEVICE_AFFINITY=1` (`DR:19,27`) | 8 rails, plus `DEST_LOCAL_RAIL=1`, `IB_TIMEOUT=18`, `MAX_MR_SIZE=1G` (`worker.sh:45-51`) | the extra vars are llr7-only; none measured by us |
| Router | `INFERA_PD_DP_RANK_AFFINITY=true`, kv-aware 20/2 (`DR:30`) | kv-aware 20/2, `session_affinity: Prefill`, `guard_release: Completion`, `decode_input_demand: On` (router log line 2; `up.sh:50-51`). No affinity env on the docker line (`up.sh:46`). | DP-rank affinity is not applicable to PP4 (one prefill DP rank; `config.yihou.dcpx.sh:124`). R2 is llying's feature and is ON (`LED:1305`). |
| EAGLE 5/1/6 | decode (`DR:27`) | decode 5/1/6 (`DL7`); the prefill also runs EAGLE (`PL7`; `worker.sh:77-78`) | decode same |
| Image-carried patches (noblock / noabsorbcfg / nonfinal) | our mounts | **UNVERIFIED in this pass.** I did not read the fdbafb60 Dockerfile. | open |

Two more facts from llr7, read in this pass:
- The llr7 prefill mounts a patched `deepseek_v2.py` from the kit (`worker.sh:144`).
- KV capacity:
  - llr7 prefill PP4 pool: 10,738,496 tokens (`prefill0.log:336-339`).
  - llr7 decode: 2,858,496 tokens per rank (`decode.log:398-401`).
  - g3x1: 3,092,288 per prefill DP rank and 2,634,048 per decode rank (`HO:34`).

## C. Answers

- **Q3, CPU tokenizer/detokenizer workers:** llr7 runs tokenizer 4 but detokenizer 1 (`DL7`). Detokenizer 4 was measured once, in llr4: neutral (37,601 vs 37,743). In the v0.5.21 era, 4/4 was a user rule, not a measured gain.
- **Measured items of ours that are not in llr7:**
  - OffloaderV2 G3: the G3 > G4 margin was measured, but the gain over no offload is not attributable.
  - Prefill mem 0.90: measured across nodes, and the prefill topology is different.
  - The GTT host pool (no log line or env in llr7): only ever measured inside the bundle.
  - Everything else that showed a gain is either already in llr7 (cross-node, index_share, custom AR, tokenizer 4) or was reverted / not adopted.
- **"R2" means two different things in our files:** `SHARD=mix` in `g4tune-xfer README:21`, and the router's `--decode-input-demand` in `LED:1278` and `LED:1305`.

**Open items:**
- whether idlefix is present in the llr7 image
- the host-pool allocation mode in the llr7 image
- whether the image's patches are equivalent to our mounts
- the origin of the `extra_slots` / `host_receive_threshold` difference

## B2

# B2 verification of the 2026-10-22..24 audit (read-only; no docker, nothing modified)

**Verdict:** most of the report holds. One claim is wrong (item 7 says ppda3 had custom AR off; it was on). Three details needed tightening (dcpx3 bundle, prefix-hit range, llr7 start time), and one claim is stronger than the report said (the MC_* envs are in llying's original kit). One cited commit could not be checked; it is not needed for the conclusion.

Path abbreviations:
- `WT` = spur006:~/dev/git/yihou_wt_tune_20261022/yihou/
- `LED` = spur006:.../debug/poll.md (the ledger)
- `LLR` = /shared_nfs/yihou_glm52/llrepro_20261023
- `RUNS` = /shared_nfs/yihou_glm52/v0521dbg_20261010/runs

## Corrections to the report

1. **Item 7 is WRONG.** It says "ppda3 runs 0.95 without custom AR". The `RUNS/ppda3/server-logs/decode-0.a1.log` server_args show `disable_custom_all_reduce': False` and `mem_fraction_static': 0.95`. So ppda3 ran mem 0.95 with custom AR **on** and without KF (ppda3 = ppda1 + R2, `WT/glm52.llrepro-pp4-dpa4.packup_20261024/README.md:17`). The crash happened only when KF was added on top of custom AR (llr8).
2. **Item 15: the dcpx3 bundle did not include MC_MAX_MR_SIZE.**
   - `LED:1190` lists dcpx3's rail changes as `MC_ENABLE_DEST_LOCAL_RAIL=1, MC_IB_TIMEOUT=18`.
   - `MC_MAX_MR_SIZE=1 GiB` came in earlier, with dcpx1 (`LED:1185`).
3. **Item 15 is stronger than the report said: the MC_* envs are llying's own kit defaults.**
   - They appear in his original kit as packed 10-22: `origin/dev/pd_opt/glm_5.2_agentx:yihou/glm52.llying-dcp-s1s3.packup_20261022/scripts/kit/orig/examples/sglang_2p1d_glm5.2/engine/worker.sh:47-51`.
   - The llr kit sets them for every role, outside the role branches: `LLR/kit/sglang_2p1d_glm5.2/engine/worker.sh:36-56`.
   - So `LED:1305` is wrong to list them as ours-only.
   - The report's commit 01f25323 is a yihou commit from 10-24 00:04, "llying 4+4 C80 reproduced …". My attempt to grep its worker.sh timed out, so that sub-claim is UNVERIFIED. It is not needed: the original-kit evidence above settles the point.
4. **Prefix-hit range.** `WT/glm52.llrepro-pp4-dpa4.packup_20261024/results/llr/summary.txt` gives GPU hit 0.492 / 0.504 / 0.411 / 0.488 for llr1-4. The correct range is 0.41-0.50, not 0.41-0.49. I did not read llr5-7 hit values, so they are UNVERIFIED. ppda1/2/3 are 0.698 / 0.728 / 0.735 (`results/ppda3/summary.tsv`).
5. **llr7 start time.** The llr7 router started at 2026-10-24T02:54:10Z (`yihou-llr-router.log:2`), not about 03:00. worker.sh mtime is 00:52:23, so the kit file still predates the run.

## Confirmed facts

**R2 (Infera router decode input-demand routing):**
- R2 on main is commit e54e9c62 by llying-001, 2026-10-19 (#188).
  - The CLI flag is at `origin/main:infera/server/args.py:130`.
  - The env name is at `origin/main:rust/router/src/config.rs:78`.
- The branch has an older env-only version, `HEAD:rust/router/src/routing_experiments.rs:97`. Blame puts it at 8cb2914e by liyingli, 2026-09-24, so it is not ours either.
- R2 is on in llr7:
  - kit `up.sh:51` passes `--decode-input-demand on`.
  - The router logs for llr1, llr3 and llr7 all print `decode_input_demand: On`.
- `LED:1305` already records this correction.

**llr7 is our best run: 41,679** (`WT/glm52.llrepro-kf-ar-topk.packup_20261024/README.md:15`). llr8 (llr7 + decode mem 0.95) is VOID (`LED:1303`).

**llr7 server_args (from the llr7 run logs):**

| | decode | prefill |
|---|---|---|
| tokenizer workers | 4 | 1 |
| detokenizer workers | 1 | 1 |
| mem_fraction_static | 0.92 | 0.88 |
| max_running_requests | 256 | 256 |
| cuda_graph_max_bs_decode / cuda_graph_bs_decode | 256 / None | – |
| disaggregation_decode_extra_slots | 0 | – |
| custom AR | on (`disable_custom_all_reduce False`) | off |
| speculative algorithm | EAGLE | EAGLE |
| index_share override | on | on |
| pp_size / tp_size | – | 4 / 1 |
| schedule_policy | – | shortest-prefill-first |
| HiCache | – | write_back, hicache_size 250 |
| cpu_offload_gb / offload_group_size | – | 0 / -1 |
| chunked_prefill_size / max_prefill_tokens | – | 8192 / 8192 |

- The decode's tokenizer 4 comes from kit `worker.sh:112`, which is in the decode branch.
- `llr7.site.env` mounts only the decode KF overlay, has `DECODE_ENV_EXTRA` with `SGLANG_OPT_USE_TOPK_V2=false`, and sets `DECODE_CUSTOM_AR=1` and 8 rails.

**Per-item effects:**

| # | Item | Measured effect (source) | In llr7 |
|---|---|---|---|
| 1 | Hashed transfer-queue shard | dcpx6→dcpx7b 19,691→21,302; KV transfer p50 21.3→11.3 s; room%4 changed nothing (`dcpx-pack-fix-best/opt.readme.md:26-28`). Engages only for a DCP relayout. | Absent (no prefill mount) |
| 2 | dcp_pack fix | dcpx8c→dcpx10 19,997→24,753 (+23.8 %), pack 77.6→0.6 ms; the two halves were not separated (opt.readme.md:12-17) | Not applicable (DPA decode) |
| 3 | Prefill MTP | dcpx7b→dcpx8c −6 %, bundled with stats logging, not separated (opt.readme.md:36-37) | Present (prefill EAGLE) |
| 4 | PP4 prefill bundle | dcpx11→ppdc1 25,197→27,426 at C56 (`ppdc/README.md:7`). Bundled (`ppdc/opt.readme.md:4-5`); the bundle also included decode max-running 96 (row 7) and decode empty_cache with mem 0.88 (row 6). Only write_back was isolated: ppdc3 write_through gave 27,457, neutral (README:11,20). | Prefill parts present; decode empty_cache absent |
| 5 | Decode graph 96 + 13 capture sizes | +1.8 %, inside the spread; free memory 24.12→28.47 GB; DCP decode only (opt.readme.md:41-43) | Absent (256 / None) |
| 6 | index_share | llr1→llr3 +0.9 %, TPOT 20.5→19.8 ms (`llrepro README:12`) | Present |
| 7 | Decode mem 0.95 | llr2 ambiguous (README:11, `llrepro notes.md:3-6`); llr8 VOID | Absent (0.92) |
| 8 | Detokenizer 4 | llr4 37,601 vs llr3 37,743, neutral (README:13) | Absent. Note: ppda3 ran detok 4. |
| 9 | KF decode kernels | llr3→llr5 +8.0 %, TPOT 19.8→16.5 ms; ppda1→ppda2 34,288→40,780, TPOT 20.0→14.2 ms (`ppda-pp4-dpa4-kf/README.md:7-8`); KF engaged at rows 60..120, rows >128 stay on the original path (`kf notes.md:18`) | Present |
| 10 | Decode custom AR | llr5→llr6 +0.4 %, TPOT −4.8 %; prealloc wait frac>0 0.15→0.26 (`kf README:14`, `kf notes.md:27-28`) | Present |
| 11 | TOPK_V2=false (decode) | llr6→llr7 +1.8 % (`kf README:15`); it lets the KF top-k engage (`kf notes.md:15-17`) | Present on decode; kit default true (`worker.sh:42`) stays on the prefill |
| 12 | R2 on our stack | ppda1→ppda3 34,288→36,543, TTFT p90 22.95→10.04 s (`summary.tsv`) | Present |
| 13 | disaggregation_decode_extra_slots | Never measured | 0 (ppda3: 48; code reference in gap analysis:80) |
| 14 | Graph bs / max-running | Never isolated; the running cap rarely binds; graph memory 13.14 vs 6.56 GB (gap analysis:79-84) | 256 / 256 (ppda3: 64 / 96) |
| 15 | MC_IB_TIMEOUT / DEST_LOCAL_RAIL / MAX_MR_SIZE | Never isolated | Present (llying kit default) |
| 16 | HSA_USERPTR_FOR_PAGED_MEM=0 | Not measured in this period | Absent. ppda3 sets it on the prefill only (dryrun:18). |
| 17 | HiCache 260 vs 250 | Never isolated; no prefill-stage gap found (gap analysis:95) | 250 (ppda3: 260) |

Supporting evidence for some rows:
- **Item 12, R2 on our stack.**
  - ppda3 set `INFERA_R2_DECODE_DEMAND=on` (dryrun:29).
  - Per-rank decode request counts (`server_metrics_export.csv.gz:1012-1015`, the kv_transfer_bootstrap histogram) were 2405 / 2708 / 2451 / 2491 in ppda3, vs exactly 2,379 per rank in ppda1 (gap analysis:26).
- **Item 15, Mooncake rails.** llr7 uses 8 rails; ppda3 uses `MC_TE_FILTERS=ionic_0..3` (4 rails).
- **Item 16.** The dryrun matches for `HSA_USERPTR_FOR_PAGED_MEM=0` are on line 18 only, not on line 26 (the other line that carries the MC_* envs).

**Genuine ours-only differences of ppda3 vs llr7:**
- extra_slots 48
- graph bs 64 and max-running 96
- decode mem 0.95
- detok 4
- HSA_USERPTR=0 on the prefill
- HiCache 260
- 4 rails instead of 8
- conn / dcp_pack / deepseek_v2 mounts on the prefill (dryrun:18)
- overlay 93034eba91 vs 8dee3f6bdb, never compared at code level (`LED:1305`)

**Stale note:** `kf notes.md:37` ("the llying router's support is unchecked") is out of date; R2 is confirmed on.

## Open (UNVERIFIED)
- Whether the fdbafb60 image's Mooncake honours MC_IB_TIMEOUT / DEST_LOCAL_RAIL. `LED:1185` found these strings missing only in the v0522 DCP image's engine.so.
- Which router implementation and R2 path the ppda3 image contains.
- llr5-7 prefix-hit values, and why llr hit (0.41-0.50) is below ppda (0.70-0.74).
- The LED dcpx7b 14:11 entry (not read; opt.readme.md covers the numbers).
- The trace-based bottleneck analysis (user question 4). It depends on llr7r, which is running now; nothing above comes from its traces.

Scratch: none written (`/tmp/yihou/audit/` untouched by me).

## C_TOK

All checked claims are CONFIRMED and none are WRONG; three open items remain. On the user's question: decode tokenizer workers = 4 is in llr7, but decode detokenizer = 4 is not (llr7 runs it at 1). Detokenizer 4 ran only in llr4, measured neutral (-0.4%), and was not carried forward.

**1. Worker counts from the logged server args** (`tokenizer_worker_num` / `detokenizer_worker_num`)

| run | decode tok / detok | prefill0 tok / detok | source |
|---|---|---|---|
| llr3-c80 | 4 / 1 | 1 / 1 | `work/runs/llr3-c80/logs/yihou-llr-decode.log:23`, `prefill0.log:25` |
| llr4-c80 | 4 / 4 | 1 / 1 | `llr4-c80/.../decode.log:23`, `prefill0.log:25` |
| llr7-c80 | **4 / 1** | 1 / 1 | `llr7-c80/.../decode.log:25`, `prefill0.log:25` |
| g3x1 (our v0521 best) | 4 / 4 | 1 / 1 | `v0521dbg_20261010/runs/g3x1/server-logs/decode-0.a1.log:23`, `prefill-0.log:22` |

- The prefill gets 1/1 everywhere checked.
- llr7 has only a prefill0 log (`ls llr7-c80/logs`), and `llr7.site.env:8` sets `PREFILL_WORKERS=1`.

**2. Measurements**
- **llr4 vs llr3** (`spur006:~/dev/git/yihou_wt_tune_20261022/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/results/llr/summary.txt`, rows 4–5):

  | run | score | qps | ISL | TTFT p50 / p90 | TPOT | kvlat p50 / p90 | err |
  |---|---|---|---|---|---|---|---|
  | llr3 | 37,743 | 2.574 | 116,358 | 3.68 / 9.76 | 19.8 | 1.70 / 4.14 | 6 |
  | llr4 | 37,601 | 2.568 | 116,220 | 3.73 / 10.68 | 19.8 | 1.79 / 4.38 | 7 |

  - That is -0.4% score, the same TPOT and TTFT p90 +9%.
  - `README.md:13` says "neutral, not kept".
  - The ledger (`poll.md:1290-1291`) has only "llr4 running" and "llr4 profiling", with no result line for llr4.
- **Decode tokenizer 4 alone (tw90 vs pf90, node 089, C48):**
  - A1: 22,231 vs 21,859, +1.7% (`poll.md:369`).
  - A2: 22,597 vs 23,127, -2.3%, recorded as "no consistent effect" (`poll.md:371`).
- **The 4+4 rule** was a user order on 2026-10-15 ~08:55Z (`poll.md:511`), not a measured gain. It was wired in through overlay `config.yihou.tokw2.sh` (`poll.md:512`).
- **No completed isolated 4+4 A/B was found:**
  - ofg2h3 (`poll.md:520`): the A1 bench failed in client setup (`poll.md:530`), and the A1 C56 run was then lost when the burst hold was cancelled (`poll.md:539`).
  - ofg4h (`poll.md:513`, started `:531`): relaunched as ofg4h2 (`poll.md:536`), which was stopped at the census stage (`poll.md:544-545`).
  - ofg2h4 (G2K1 + tok4/detok4) was launched (`poll.md:576`).
  - **Open:** I did not check whether ofg2h4 or any later ofg*h round produced a clean 4+4 vs 1/1 comparison.

**3. Where the kit sets the counts, and how to add detokenizer 4**
- `kit/sglang_2p1d_glm5.2/engine/worker.sh:112` (decode only): `--mem-fraction-static 0.92 --cuda-graph-max-bs-decode 256 --tokenizer-worker-num 4`.
- The kit never passes `--detokenizer-worker-num`, so the decode runs at 1 (llr7 decode log above).
- `worker.sh` sets no worker-count flag for the prefill.
- `env.sh` and `up.sh` contain no worker-num setting.
- Extra args are appended in `worker.sh`: `read -ra extra <<<"$DECODE_EXTRA_ARGS"` (`:115`, prefill `:104`), then `args+=("${extra[@]}")` (`:117`).
- To add detokenizer 4: append `--detokenizer-worker-num 4` to `DECODE_EXTRA_ARGS` in the site env, exactly as `llr4.site.env:9` did. `llr7.site.env:9` `DECODE_EXTRA_ARGS` contains only the index_share override.

**Other site envs**
- `llr7r.site.env:1`: llr7 config unchanged, run as a replicate with profiling.
- `llr8.site.env:1`: llr7 + decode mem-fraction 0.95.
- **Open:** I read only these header comments, not the full args.

The scratch scripts I wrote for these checks are in `/tmp/yihou/audit/` (c.sh, c2.sh, c3.sh).