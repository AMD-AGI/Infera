# Notes — what / why / how / context

Ledger: `analysis/working_process.md` (X1, k1, Q1, TB1-TB7, N1). Research: `analysis/research/`.

## 1. Where the time goes at C44-C52 (prefill side)

- What: TTFT 7.8 → 12.0 → 15.8 s from C44 → C48 → C52 on the q1 stack; decode running residence flat (~12-13 s).
- Why (Little's law, `results/analysis/q1_residence_c44_c48_c52.txt`): prefill queue 2.0 → 3.6 → 5.2 s and KV send
  (prefill inflight) 4.5 → 6.5 → 8.2 s grow; prefill cache hit falls (0.71 → 0.57) and the 450 GB host pool fills
  earlier (~50 → ~30 min). Every turn re-ships the whole ~145k-token context KV (~8 GB) because decode has no prefix
  cache (decode radix cache is incompatible with EAGLE in v0.5.21).
- How (next levers, not yet run): KV-send copy priority (`config.yihou.mc4340prio.sh`, 102-only image; env-only variant
  `GPU_MAX_HW_QUEUES=16 MC_HIP_NUM_STREAMS=8` works on the mc4340 image, microbench R043), larger prefill cache
  (HiCache 500 GB, prefill mem 0.90), decode mem 0.96 (q4 prepared), index_share TRUE (user: last).

## 2. Track B (decode in-flight KV cap) — status

- v1 (FIFO break): functional (cap held at 0.50), C40 warm 21,271 (−3.8 % vs 22,114) → regression. Cause (inferred):
  under `early` allocation a blocked request cannot start prefill and blocks everything behind it (~0.7 M-token prompts).
- v2 (skip + 5 s anti-starvation): functional (0.42 max, 64/64 ok), C40 fresh 20,680. **Scored A2 not run** — rerun
  `chain_pc40d.yihou.sh` (needs a fresh node; the chain waits for pc40c files — run its body without the wait).
- Not counted by the patch: nothing — it uses pool used − Σ running kv_allocated_len (covers in-flight, arrived-not-batched,
  deferred releases). Alternative native lever: `--disaggregation-decode-allocation-policy prefill_complete` (needs prefill
  HiCache write_back) — untested.

## 3. Operational gotchas (new)

- Burst QOS (`amd-burst-qos`, PreemptMode=cancel) cancels jobs after ~20 min under load — unusable for 3-h rounds.
- Login hosts are memory-policed: drivers on a login host died with it (pc40d A2, q4 at 17:18 2026-10-12). Drivers must
  run where they survive (setsid nohup is not enough if the host is lost).
- 102's job 207495 ended at 20:01 2026-10-12 (TIMEOUT 1-14:35 elapsed), earlier than the 2-day limit set before.
- `docker load` on 137 produced a different image ID (e8e2838f) than 102 (9d755dd6): add per-node IDs to the image config.
- Decode /metrics gauges (`decode_sum_seq_lens`, `kv_used_tokens`) only refresh on decode batches; during AgentX warmup
  (max_tokens=1) they read 0. `pending_prealloc_token_usage` is never assigned (always 0).

## 4. Open items / next steps (as of pack-up)

1. pc40d A2 (runshare v2, C40 warm) — the no-regression check the user asked for.
2. q4: q1 config + decode mem 0.96 at C48 (env/chain prepared: `env.q4-137.yihou.sh`, `chain_q4.yihou.sh`).
3. KV-send lever round (mc4340prio or env-only HW-queue variant) at C48.
4. index_share TRUE at the best config (user: last).
