# Notes — caveats, wrong turns, open questions

## 1. The JSON `per_gpu` is total / 5 (all four runs)
- What: `kitedit.yihou.py` originally set `PREFILL_TP=1` in `engine/agentx/agentx.sh` metadata. InferenceX divides
  total throughput by prefill TP + decode TP, so `per_gpu` = total / 5.
- Fix: `PREFILL_TP=4` (llying's convention for a PP4 TP1 prefill: 4 GPUs). The kit in `scripts/kit/` already has the
  fix; the four runs here started before it. Report total / 8.

## 2. Protocol: one 30-min run per stage
User directive 2026-10-22: single runs, 1800 s profiling after a 10-request-per-lane warm-up, no A1/A2 repeat.
No run-to-run variance is measured; differences of a few percent are not resolved by this data.

## 3. What is and is not comparable
- S1/S2/S3 ran in parallel on three OAI nodes (amdgpu 6.14.14), same image, overlay, kit, dataset, seed. Each stage
  changes one block: S1->S2 the decode layout + its env (render diff: `--dp-size 4 --enable-dp-attention` mem 0.92
  graph 256 -> `--ep-size 4 --dcp-size 4` mem 0.85 graph/max-running 96 + 4 DCP env); S2->S3 only the prefill
  `--offload-*` flags. Different physical nodes per stage.
- S3-aux: burst node, amdgpu 6.16.13, `HSA_ENABLE_SDMA=0`. `launch_stage` passed it as PREFILL_ENV and DECODE_ENV,
  but `stages/S3.env` sets DECODE_ENV itself, so `engine-config.env` shows it on the prefill leg only. Not comparable.
- llying's 37,310 tok/s/GPU (4+4 C80): two nodes (cross-node transfer), 3600 s, custom AR off, index_share false.
  Here: one node, 1800 s, custom AR ON and `index_share_for_mtp_iteration` TRUE (user overrides, correctness ignored).
- Simulated acceptance (`SGLANG_SIMULATE_ACC_LEN=3.61`) in every run: outputs are not checked; no correctness claim.

## 4. Open: why S2/S3 decode admission backs up
Measured: decode admission 12.8-16.9 mean, transfer-in 22-35, running 23-31 (cap 96), client kv_usage 96-99 %.
Not measured: whether admission waits on KV pool capacity (DCP4 decode uses mem 0.85 vs DPA4's 0.92) or on transfer.
Single-variable runs that would separate them: S2 with only decode `--mem-fraction-static` raised; S2 with only the
context length capped. Related code-read finding (not measured here): Mooncake's transfer queue sharding
(`mooncake/conn.py:3112-3117`, by sum of dst ports) puts each prefill rank's sends to a DCP4 decode on one worker
(see `../glm52.dcp4-pd.packup_20261022/`).

## 5. Lost / failed attempts
- S1 on crsuse2-m2m-151 (11:26Z): node lost in the cluster-wide job end at 11:32Z. That log also shows an older
  `run_stage` line-33 `ETCD_PORT: unbound variable` (fixed: env.sh is sourced before the port preflight).
- The sglang PP fix (`patches/0001-...`) was needed before any PP4 prefill could start on this branch.

## 6. Operations
- Containers are started with `--init` and covered by `scripts/pd-tools/job_container_guard.yihou.sh` (guard logs
  `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/guard/{233019.219,233048.039,233080.210,233045.028}.log`).
- Stacks S1/S2/S3 were left up after the runs (pending the user); S3-aux was docker-stopped and its burst node
  released. Stop = `docker stop` only.
