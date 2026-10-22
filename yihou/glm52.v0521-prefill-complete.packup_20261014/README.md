# GLM-5.2-MXFP4 v0.5.21 1P1D P4D4 — decode allocation policy `prefill_complete` at C48 (HiCache 370, decode 0.96)

**Ran:** 2026-10-13 09:53 → 18:50 UTC on `crsuse2-m2m-274` (10.245.157.34, spur `amd-oai` job 211171). **Author:** yihou (with Claude).
**Packed:** 2026-10-14 (pack-up only; nothing was re-run for this kit).
**Status:** DONE (all three rounds finished). Policy works after a one-line validation fix; it did **not** produce a
consistent gain, and both arms sit >10 % below the C48 reference.

## Goal (spec/user_directives.md, user 2026-10-13 07:52 UTC, verbatim in Chinese there)

"v0.5.21 has a config that lets the decode side allocate GPU memory only after prefill has finished computing. Research how
to use it correctly, find a new machine, and test it with hicache=370, decode gmu=0.96, conc=48."

Success criteria (implicit in the directive; no numeric target was given): (1) the policy actually runs on both legs,
(2) a scored AgentX C48 number with HiCache 370 / decode mem 0.96. Added by us: (3) a same-node control with policy `early`
(user approved "跑", 15:09), so the policy can be separated from the other knobs.

## Result vs criteria

| criterion | status |
|---|---|
| policy runs on both legs | **PASS after a fix.** Stock v0.5.21 image rejects it for our TP4/DP4-attention shape (pc48: startup `ValueError`, a validation bug, see `patches/`). With the fix, server-info shows `disaggregation_decode_allocation_policy=prefill_complete` on both legs, prefill `optimistic_prefill_attempts=1`, `hicache_write_policy=write_back`, `hicache_size=370` |
| scored C48 number | **20,002 tok/s/chip** (pc48b A2, warm) |
| policy A/B (same node, same config otherwise) | **inconsistent sign**: A1 (fresh) −5.6 %, A2 (warm, scored) +4.4 % |

## Headline table (first-hand, `results/results.csv`; AgentX full 3600 s, score = `request_metrics.throughput.per_gpu.total_tput_tps`)

All rows: same-node 1P1D P4D4 (prefill GPU0-3, decode GPU4-7, TP4 DP4 DPA each), EAGLE MTP 5/1/6 with
`SGLANG_SIMULATE_ACC_LEN=3.61` (correctness waived), prefill mem 0.88, decode max_running 96, C48.

| run | node | decode alloc policy | prefill HiCache | decode mem | stack | tok/s/chip | qps | mean ISL | TTFT mean / p95 s | TPOT ms | GPU / CPU hit | ok / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **pc48b A2** | 274 | **prefill_complete** | 370 GB write_back | 0.96 | warm (scored) | **20,002** | 1.112 | 142,959 | 16.57 / 52.70 | 12.2 | 0.811 / 0.101 | 4027 / 3 |
| pc48e A2 | 274 | early (control) | 370 GB write_back | 0.96 | warm (scored) | 19,158 | 1.097 | 138,769 | 17.79 / 60.45 | 12.2 | 0.796 / 0.107 | 3980 / 2 |
| pc48b A1 | 274 | prefill_complete | 370 GB write_back | 0.96 | fresh | 18,225 | 1.040 | 139,272 | 17.42 / 68.59 | 15.3 | 0.865 / 0.062 | 3773 / 1 |
| pc48e A1 | 274 | early (control) | 370 GB write_back | 0.96 | fresh | 19,313 | 1.107 | 138,615 | 15.18 / 52.97 | 14.9 | 0.863 / 0.061 | 4012 / 1 |
| q1 A3 (reference) | 137 | early | 450 GB write_through | 0.95 | warm (3rd run) | 22,306 | 1.220 | 145,303 | 11.97 / 38.19 | 12.1 | 0.839 / 0.088 | 4425 / 2 |
| pc48 | 274 | prefill_complete | 370 GB write_back | 0.96 | — | died at prefill startup (ValueError) | | | | | | |

## Findings (measured; mechanisms not established)

1. **Validation bug in the image (and upstream main).** `parallel_hook.handle_deprecated_dp_attention` runs first and rewrites
   `--dp-size 4 --enable-dp-attention` into `attn_dp_size=4, dp_size=1`; `_validate_prefill_complete` then computes
   `cfg.dp_size if cfg.enable_dp_attention else 1` (image line 213) → 1 ≠ tp 4 → rejects "attention tensor parallelism
   greater than one". Upstream main still has the same expression (line 214; re-checked 2026-10-14 at `536d0653`). One-line
   fix mounted on both legs (`patches/0001-pcomplete-attndp-validation.diff`). Upstream PR candidate, **not filed**.
2. **Policy A/B on 274 is inconsistent in sign**: fresh A1 18,225 vs 19,313 (−5.6 %), warm A2 20,002 vs 19,158 (+4.4 %).
   With a single replicate per arm this is not a measured effect in either direction.
3. **Both arms are >10 % below q1** (A2: −10.3 % / −14.1 %). The four knobs that differ from q1 besides the policy —
   prefill HiCache **write_back** (forced by the policy), HiCache **370** vs 450, decode mem **0.96** vs 0.95, node **274** vs 137
   — were changed together; the drop has not been attributed to any one of them.
4. **Scheduler-log differences** (A1 and A2 windows, `analysis/logstats.pc48b-vs-pc48e.tsv`): with `prefill_complete` the decode
   leg shows `#prealloc-req` mean ≈2.5 (early: 0), `#transfer-req` mean ≈1.6 (early ≈3.8-4.1), `pre-allocated usage` mean
   ≈0.09 (early ≈0.22-0.23), `token usage` mean ≈0.33 / max 0.89-0.91 (early ≈0.46-0.47 / max 0.97-0.99); prefill shows
   `#optimistic-req` mean ≈1.2 (early 0), prefill `#queue-req` mean ≈3.1-3.4 in both arms. No retractions in any run.

## Folder map

- `README.md` (this), `REPRODUCE.md` (ordered steps), `environment.md` (HW/SW, image digest, gaps), `notes.md` (gotchas,
  wrong turns, open questions), `SHA256SUMS`.
- `spec/` — `user_directives.md` (verbatim user messages for this experiment), `mission.md` (session mission file, copy).
- `scripts/harness/` — verbatim harness: `bench-harness/` (engine.sh, launch.sh, agentx_bench.sh, stop.sh, config.sh, tools/),
  config chain (`config.yihou.pcomplete.sh` → `mc4340n2` → `E2-target` → `v0521-base`), env files of every round, mounts
  TSVs + `mounts-v0521/src/*`, drivers (`chain_cap2`, `cont_bench`, `after_x8p_pc48`, `stop_then_chain`, `wait_then`,
  `chain_cap2l`), `run.v3`, `pipeline.v3`, `tools/audit_live.v3.yihou.py`, `tokdiff.yihou.sh`, reference `dryrun.q1-137`.
- `scripts/session/` — node-local ssh shim, env template, `tools_cmp` / `tools_summ` / `collect_env`, session summary table.
- `scripts/tools/` — `/metrics` sampler, `logstats.yihou.awk` (log statistics), `results_csv.yihou.py`, `packup_copy.yihou.sh`.
- `patches/` — the attndp validation fix (orig / patched / diff), `README.md` (what/why/how/context), `UPSTREAM-PR-CANDIDATE.md`.
- `results/` — `results.csv`, per-bench dirs `pc48{b,e}-A{1,2}-C48/` (JSON, plots, aiperf summaries, gz benchmark log),
  `rounds/pc48{,b,e}/` (status, chain/pipeline/handover logs, audit, census, server-info gz, startup error log),
  `metrics/` (5-s `/metrics` TSVs, gz), `dryruns/` (+ `tokdiff.yihou.txt`), `env/` (collect_env captures).
- `analysis/` — `logstats.pc48b-vs-pc48e.tsv`, `research/` (prefill_complete research: `pc_synth`, `pc_semantics`,
  `pc_mounts`, `pc_upstream`; harness research `res_harness`), `ledger/` (`working_process.md`, `poll.md` copies).
- `logs/LOG_INDEX.md` — absolute paths + sizes of server logs and large aiperf artifacts that were **not** copied.

## Sibling packups (same session / same baseline)

- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-cap-c44c48.packup_20261013` — q1 reference (C48 22,306), warm-stack protocol.
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-decode-transfer-kvcap.packup_20261013` — decode in-flight KV cap (runshare).
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012` — baseline kit (e2g 22,114 C40), image recipe, node hold.
- Session-wide results table (all v0521cap2 rounds): `scripts/session/summary.v0521cap2.yihou.md`.

## Cold-read verification (2026-10-14, second agent)

- Scores re-read from the source JSONs (`$R/<round>/agentx-c48-<A>/agentx_conc48.json`): pc48b A1 18,225.47, A2 20,001.62;
  pc48e A1 19,313.40, A2 19,158.43; q1 A3 22,305.78 — match `results/results.csv` and `python3 -B $N/tools_summ.yihou.py`.
  Derived deltas (−5.6 %, +4.4 %, −10.3 %, −14.1 %, +9.7 %, −0.8 %) recomputed and match.
- Server-info (`results/rounds/pc48b/audit/*-server-info.json.gz`): both legs `prefill_complete`; prefill
  `optimistic_prefill_attempts=1`, `write_back`, 370, mem 0.88, 2,433,088 tok; decode mem 0.96, 2,816,896 tok.
- sha256 identical to the originals in `$H`: `bench-harness/{launch,engine}.sh`, `chain_cap2`, `cont_bench`,
  `config.yihou.pcomplete.sh`, `env.pc48b-274`, `mounts.yihou.pcomplete-attndp.tsv`, `tools/audit_live.v3.yihou.py`,
  shim `$N/shimbin/ssh`; patch files equal the NFS mount source and the image file.
- Added `scripts/harness/env.x8p-274.yihou.sh` (verbatim; sourced by `after_x8p_pc48.yihou.sh` to stop the preceding x8p stack).
- Note: pc48e A2 TPOT is 12.15 ms raw (shown 12.2 here, 12.1 in the session summary — rounding only).
- Still open: no ECC capture for 274; env capture ~7 h after the runs; JIT-seed command [inferred]; stray
  `spur006:/tmp/logstats.yihou.awk` (created by the builder, outside the kit, left in place).
