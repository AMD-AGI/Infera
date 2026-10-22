# GLM-5.2-MXFP4 v0.5.21 same-node P4D4 — OffloaderV2 ratio sweep at C48 and high concurrency (2026-10-15 03:10 → ~11:30 UTC)

Follow-up to `../glm52.v0521-offloaderv2.packup_20261015`, which covers the research, the patch, S0 (G8K1) and the correctness rounds of G4/G3/G2. This kit packs everything run after that kit was closed (03:10 UTC).
**Status: PARTIAL** — three rounds were still running when this was packed (see "In flight").

## Questions (user, 2026-10-15)

1. 01:35: "提高offload率观察性能的同时，堆高conc，正确" — run the 1/4, 1/3, 1/2 offload ratios.
2. 07:50: "我们是要测offload在高conc下能不能拿到更多的吞吐，baseline早就有了" — offload at C56/C64. The user ruled out new no-offload runs.
3. 08:5x: "decode侧cpu解码都是用4个worker" — clarified as decode `--tokenizer-worker-num 4 --detokenizer-worker-num 4`.
   - This applies to every new round from then on (overlay `config.yihou.tokw2.sh`).
   - Rounds already running finished without it.

## Results (score = `request_metrics.throughput.per_gpu.total_tput_tps`; all n = 1; full table `results/results.csv`)

Prefill KV per rank, from `logs/server/<r>/prefill-0.log*`:

| setting | KV tokens |
|---|---|
| no offload (pf90) | 2,541,376 |
| G4K1 (1/4) | 2,938,752 |
| G3K1 (1/3) | 3,092,288 |
| G2K1 (1/2) | 3,378,496 |

| run | node | offload | decode tok/detok | conc | stack | score | qps | TTFT mean/p95 (s) | TPOT (ms) | GPU hit | CPU hit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pf90 A1 / A2 (ref, earlier kit) | 089 | none | 1/1 | 48 | fresh / warm | 21,859 / 23,127 | 1.219 / 1.276 | 9.57/31.32 · 9.30/28.76 | 15.7 · 13.1 | 0.911 · 0.860 | 0.035 · 0.071 |
| ofg2 A1 | 288 | G2K1 | 1/1 | 48 | fresh | 21,157 | 1.171 | 12.32/29.13 | 14.5 | 0.960 | 0.003 |
| ofg4b A1 | 136 | G4K1 | 1/1 | 48 | fresh | 22,506 | 1.242 | 8.79/27.22 | 15.5 | 0.941 | 0.017 |
| **ofg4b A2** | 136 | G4K1 | 1/1 | 48 | warm (scored) | **23,771** | 1.307 | 9.71/28.77 | 12.1 | 0.925 | 0.018 |
| ofg4b A3 | 136 | G4K1 | 1/1 | **56** | warm (3rd) | 21,744 | 1.238 | 20.04/51.99 | 12.4 | 0.857 | 0.067 |
| ofg3b A1 | v3-020 | G3K1 | 1/1 | 48 | fresh | 22,484 | 1.237 | 9.27/23.86 | 15.0 | 0.954 | 0.008 |
| **ofg3b A2** | v3-020 | G3K1 | 1/1 | 48 | warm (scored) | **23,200** | 1.250 | 10.78/29.35 | 12.0 | 0.936 | 0.010 |

- **No like-for-like C56/C64 baseline exists.**
  - The only no-offload C56 points are tb56n 18,016 and tb56 17,611 (085). They ran with HiCache ratio 1.5 (~154 GB) and prefill mem 0.88, against 450 GB / 0.90 here.
  - The user chose not to run a new baseline.
  - The C56 numbers here can therefore only be compared across offload ratios.
- **Cross-node caveat:** every row is on a different node, and 136 runs its own image build (`9872bbda…`). The A2 spread across G4/G3/no-offload (23,771 / 23,200 / 23,127) is within what this kit cannot attribute.

## Failures and their causes (details in `notes.md`)

- **oai nodes released by the user at 05:00** ("TL gave the nodes to another team"). Two rounds were aborted with no JSON:
  - ofg4 A1 (252) was aborted ~76 min in.
  - ofg2 A2 (288) was aborted ~31 min in.
- **ofg2h / ofg2h2 on v3-026: PD warm-up hang.** Cause, from first-hand evidence (`results/pyspy/ofg2h2/`):
  - The reused cache key `026` held 4 of 16 aiter `.so`, so kernels JIT-built during warm-up.
  - A stack killed mid-build (pf90b) left `lock_module_norm`, and later schedulers waited on that lock.
  - Fixed by re-seeding cache/026 from cache/102.
- **ofg2h3 A1 client failure:** the same reused key had a broken client `uv-python`. The bad dirs were moved aside and the run continued on the live stack.
- **136 driver problems:**
  - `run.v3 stop` hung 50 min in sync_back; the stop PID was killed.
  - The login-host driver on spur007 vanished after ofg4h's prefill came up (cause unknown). It was relaunched as ofg4h2 from spur005.

## In flight at packing (~11:30 UTC)

| round | node | offload | decode tok/detok | plan |
|---|---|---|---|---|
| ofg3b | v3-020 | G3K1 | 1/1 | A3 C56 since 10:57:57 |
| ofg2h3 | v3-026 | G2K1 | 4/4 | A1 C56 since 10:31:29 → A2 C56 → C64 |
| ofg4h2 | 136 | G4K1 | 4/4 | C56/C56/C64; waiting for VRAM drain since 11:24 |

Refresh: `bash scripts/tools/packup_copy_highconc.yihou.sh` (copy-only). It regenerates `results/results.csv` via `scripts/session/results_csv_v15b.yihou.py highconc`.

## Navigate

`REPRODUCE.md` · `environment.md` · `notes.md` · `results/` (csv, bench dirs, rounds, dryruns+tokdiffs, py-spy, correctness/ofg4c) · `logs/server/<round>/` · `analysis/ledger/` · `scripts/`

## Update 2026-10-15 ~13:30 UTC (after the first packing at ~11:40)

No new score. Everything below lost a run.

| event | time (UTC) | effect |
|---|---|---|
| burst holds 217125 (v3-020) and 217126 (v3-026) CANCELLED (sacct State=CANCELLED, ExitCode -1:0; not cancelled by this session; amd-burst-qos PreemptMode=CANCEL) | 11:35:12-13 | ofg3b A3 C56 (from 10:57:57) and ofg2h3 A1 C56 (from 10:31:29) terminated, no JSON. Our containers on those nodes could not be checked afterwards (no job). |
| ofg4h2 (136): VRAM drain gate passed | 11:49:57 | stack up (bothlegs 12:12:30, router 12:12:35) |
| ofg4h2: pipeline stage `census` hung | 12:12 → 13:02 | killed; pipeline STOP at census; no bench ran |
| 136 not reachable by ssh from spur006 | ~13:20 | state of our ofg4h2 containers unverified at this update |

**Root cause of the census and stop hangs (first-hand).**
- `run.v3 sync_back` ran `ssh <node> tar | base64` into a bash variable, then `base64 -d <<< "$b64" | tar -x`.
- On the login host, that bash process sat in state D with wchan `mem_cgroup_handle_over_high`, i.e. memory-cgroup throttling.
- The node-side decode log was 22.9 MB.
- The user notes that login hosts enforce about 4 GB per user, with kill/evict above it. The spur007 driver that vanished at 10:54 is consistent with such a kill, but this is not verified.

**Fix prepared (not yet run).**
- `run.v3s.yihou.sh` streams `ssh tar | tar -x` with no bash variable.
- `pipeline.v3s.yihou.sh` and `chain_cap2ls.yihou.sh` are copies that call it.
- Created by `scripts/tools/mk_streamcopies.yihou.py`. The originals are untouched; the diff is the sync_back body and the two call sites.
