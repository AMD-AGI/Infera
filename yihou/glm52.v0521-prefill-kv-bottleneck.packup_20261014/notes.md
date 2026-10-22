# Notes — gotchas, wrong turns, caveats (what / why / how / context)

## Caveats on the headline (read before quoting a number)

- **One run per arm, four nodes.** q1 ran on 137, x8d2/x8pc on 206, x128p on 085, x8p on 274. Node-to-node spread for
  identical configs was never measured in this session; the only same-node pair is x8d2 vs x8pc (206): 23,591 vs 22,489.
  Warm-replicate spread seen earlier on 102 for C40: 21,681-22,114 (~2 %) [sibling packup].
- **Reduced leg ≠ only more KV.** expert8/128 also cut MoE weight reads and expert compute (and expert8 forces all tokens
  to the same 8 experts, EP4). x8pc is the control for that on prefill: same 8-expert prefill but KV capped to the
  256-expert size → 22,489 ≈ q1 22,306. There is no equivalent control on decode (x8d2 changes both).
- **x8p vs x128p:** 25,905 vs 25,297 with 3.67 M vs 3.32 M prefill KV; also 8 vs 128 experts and EP4 vs EP1, different
  nodes. Not separable from these data.
- **A1 vs A2:** A1 = first run on a fresh stack (cold radix + HiCache), unscored; A2 = after flushing decode then
  prefill caches. A2 > A1 in every round here (+2.2 % … +4.8 %). q1's reference is its A3 (3rd run on the stack).
- **Correctness waived:** `SGLANG_SIMULATE_ACC_LEN=3.61` forces the MTP accept count; the synthetic models produce
  garbage. Scores are throughput only.

## x8d: a hardware fault, not a result

- What: x8d A2 died 08:39:57 with `Memory access fault by GPU node-5 … Reason: DRAM ECC failure / SRAM ECC failure` on
  both legs, then Fatal Python errors; `amd-smi metric --ecc` on 026: GPU3 (KFD node 5, PCI 0002:00:04.0, a **prefill**
  GPU) UMC correctable 13,198,148, deferred 2; others 0.
- Why it matters: x8d A1 (20,230, the lowest decode-expert8 number) ran on the same GPU → flagged suspect. The rerun
  x8d2 on 206 gave A1 22,511 / A2 23,591.
- How to avoid: check ECC counters on every GPU before a round (REPRODUCE §1). 026 was released (job 211170) and a new
  hold excluded it (`-x crsuse2-m2m-026`); that hold (211763) had 4 JobLaunchFailure restarts before landing on 206.

## x128p: 180 InvalidInferenceResultError per bench

- What: `request_accounting.records_error_dropped = 180` (`InvalidInferenceResultError`, empty content) in both A1 and A2;
  the aiperf profiling log reports `errors=0` for the phase. Other rounds: 0-5.
- Not investigated (cause unknown). Candidate check [not done]: grep the prefill/decode server logs and the aiperf
  `profile_export.jsonl` for the failing request ids (paths in `logs/LOG_INDEX.md`). The score still counts 5,030 ok requests.

## Harness gotchas that bit or nearly bit

- **Same `--model-path` on both legs.** The reduced leg gets its overlay `config.json` mounted OVER
  `$MODEL/config.json` inside that container only, so router / agentx_env / audit see one model path. A separate
  overlay model dir would make the router and AgentX recipe disagree.
- **Overlay config is re-sourced** by engine.sh and audit with the parent's exports; `config.yihou.expert8leg.sh`
  tolerates its own TSV already being set and refuses any other `*_MOUNTS_ADD` (merge TSVs first).
- **Never edit a config a running chain re-sources** — `config.yihou.mc4340n3.sh` is a copy of `n2` + node 206
  because x8d/x8p chains were live on n2.
- **Drivers on the compute node** (`spur exec … setsid nohup … &`) with the local ssh shim `$N/shimbin/ssh`
  (refuses any host but itself). Rationale: a login host was lost on 2026-10-12 17:18 and took the drivers with it.
  136-class plain-ssh nodes have read-only NFS → `chain_cap2l.yihou.sh` runs on the login host instead.
- **Flush order** decode (:29357) then prefill (:29101); prefill may return 400 while busy — chain_cap2 retries 60x.
- **x8p A3 (C52) intentionally not run**: the user approved reusing 274 for pc48 (prefill_complete) after x8p A2;
  `after_x8p_pc48.yihou.sh` killed the chain driver so no A3 started.
- **Bring-up time**: prefill healthy after ~14-15 min (HiCache 450 GB registration), decode ~10 min later; smoke ~27 min after start (x8p).

## Wrong turns / decisions

- First reading of "conc 52": the user remembered C52 as the best; the best score was C48 (q1: C44 21,862 / C48 22,306 /
  C52 21,418). Decision: A1/A2 at C48 and C52 as A3 on the same stack. Only x8d2 reached A3; C52 21,713 < C48 23,591
  as for q1.
- The "26k" the user referred to is v0.5.19 with 8 experts on BOTH legs at C56, `--max-total-tokens 3670016` both legs
  (r001-hc274 26,462; r001-hc450chunk512 26,071) — `analysis/research.res_expert8.md`. Not comparable to v0.5.21 numbers;
  it motivated the 3,670,016 default cap.
- x8pc was added (user "跑", ~15:00) after x8p to split "fewer prefill experts" from "more prefill KV"; x128p (~16:00) to
  see whether a milder weight cut gives most of the gain (3.32 M KV → 25,297).

## Telemetry notes

- `/metrics` sampler: 5 s, both legs, all DP ranks, into TSV `ts role dp metric value` (`scripts/session/tools/metrics_sampler.yihou.py`);
  `_error` rows = failed scrapes (x8p A2: 82 decode, 92 prefill). Decode gauges refresh only on decode batches.
- `analysis/metrics_digest.tsv` was computed off the cluster from the gzipped TSVs with `scripts/session/metrics_digest.yihou.py`
  (`token_usage` max/p90 per DP, queue/transfer/prealloc/inflight queues, retractions, HiCache host-used fraction).
  x8d A2 decode = 0 (decode dead early).
- GPU/CPU cache hit = `server_metrics.cache.{gpu,cpu}_cache_hit_rate` in the AgentX JSON (prefill radix vs HiCache host tier).
