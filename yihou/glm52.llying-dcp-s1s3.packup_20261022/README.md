# GLM-5.2-MXFP4 — llying's best PD config on one MI355X node, then DCP4 decode, then OffloaderV2 prefill (S1/S2/S3), 2026-10-22

**What.** Three stages, one 30-min AgentX C80 run each, same-node P4D4 (prefill GPUs 0-3, decode GPUs 4-7):

| Stage | Change vs previous | Node | tok/s/chip (total / 8) | TPOT mean | TTFT mean |
|---|---|---|---|---|---|
| S1 | llying `2p1d-v0522` best config adapted to same-node P4D4: PP4 TP1 prefill + DPA4 decode; user overrides custom AR ON, index_share TRUE | OAI crsuse2-m2m-219 | **34,509** | 18.3 ms | 5.5 s |
| S2 | decode swapped to TP4 EP4 DCP4 (kv_b bf16 + split-triton indexer env, mem 0.85, graph/running 96) | OAI crsuse2-m2m-039 | **25,941** | 20.3 ms | 17.7 s |
| S3 | S2 + OffloaderV2 G4K1 on the PP4 prefill | OAI crsuse2-m2m-210 | **21,025** | 18.6 ms | 34.4 s |
| S3-aux | S3 on a burst node (amdgpu 6.16.13, `HSA_ENABLE_SDMA=0` on prefill) — **not comparable** | burst crsuse2-m2m-v3-028 | 16,472 | 16.3 ms | 56.4 s |

Reference (EXTERNAL, not comparable): llying 4+4 C80 = 37,310 tok/s/GPU (two nodes, 3600 s, custom AR off,
index_share false). Simulated acceptance `SGLANG_SIMULATE_ACC_LEN=3.61` in all runs: **no correctness claim**.

**Why.** User directive 2026-10-22: run llying's latest best config same-node (S1), swap the decode to DCP (S2),
then add OffloaderV2 4/1 on the PP prefill (S3). Plan of record: `spec/llying_dcp_plan.md`.

**Result vs goal (20k+ tok/s/chip):** S1/S2/S3 all clear 20k. DCP4 decode (S2) is 25 % below DPA4 decode (S1) in
this setup; adding OffloaderV2 to the prefill (S3) is a further 19 % lower. Queue readout (`results/RESULTS.md`): S1 has
no backed-up stage; S2/S3 have deep decode admission + transfer-in queues and prefill handshake/in-flight queues while
decode running stays under its cap of 96 and client-reported decode KV usage is 96-99 %. **Cause not established.**

## Navigate
- `REPRODUCE.md` — image build/load, kit, launch commands, analysis.
- `environment.md` — nodes, driver, image tag + ID + tar sha256, code SHAs, model path, secrets needed.
- `results/RESULTS.md` — full numbers, queues, step-time tables; per-run dirs hold the AgentX JSON, aiperf summary,
  `analysis.txt`, `engine-config.env`, `workers.json`, plot.
- `notes.md` — caveats (the /5 divisor bug, single-run protocol, what is not comparable, lost runs, open questions).
- `scripts/kit/` — the stage kit (llying orig + diff + edits); `scripts/image/` — image recipe; `patches/` — sglang PP fix.
- `spec/` — plan + the user-provided helper analysis of llying's config.

Related pack-ups: `../glm52.dcp4-pd.packup_20261022/` (the DCP4 r-series and DPA4-vs-DCP4 kernel A/B that preceded
this; same sglang branch), `../glm52.dcpx-xnode-dcp4decode.packup_20261022/` (other session, cross-node DCP4 decode).
