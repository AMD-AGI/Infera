# Notes — gotchas, wrong turns, open questions (what / why / how / context)

## 1. `prefill_complete` must be on BOTH legs and forces prefill HiCache `write_back`

- **What:** `--disaggregation-decode-allocation-policy prefill_complete` (sglang PR #40703, in the image) on prefill AND decode.
- **Why:** a leg mismatch starts fine but every request fails with `policy mismatch` (`srt/disaggregation/common/conn.py:2056-2063`
  in the image); `_validate_prefill_complete` rejects prefill HiCache other than L2 `write_back` (image `pd_disaggregation_hook.py:231-238`).
  Our base config uses `write_through`, so the policy drags a second variable (write policy) with it.
- **How:** `scripts/harness/config.yihou.pcomplete.sh` appends the flag to `PREFILL_EXTRA_ARGS` and `DECODE_EXTRA_ARGS` (idempotent,
  engine.sh re-sources the config) and sets `PREFILL_HICACHE_WRITE_POLICY:=write_back` before sourcing its base. sglang itself sets
  prefill `optimistic_prefill_attempts=1` (confirmed in server-info).
- **Context:** research `analysis/research/pc_synth.md` (A)-(B), `pc_semantics.md`, `pc_upstream.md`.

## 2. Validation bug: TP4 + attention-DP4 is rejected (round pc48)

- **What:** prefill died at startup: `ValueError: --disaggregation-decode-allocation-policy prefill_complete does not yet support
  attention tensor parallelism greater than one` (`results/rounds/pc48/prefill-0.startup-ValueError.log`, traceback at
  `pd_disaggregation_hook.py:244`).
- **Why (code read, first-hand):** in `srt/arg_groups/pipeline.py`, `handle_deprecated_dp_attention` (`parallel_hook.py:36-57`) runs
  before `handle_pd_disaggregation` and rewrites `--dp-size 4 --enable-dp-attention` into `attn_dp_size=4`, `dp_size=1`,
  `enable_dp_attention=False` (in the resolving view). `_validate_prefill_complete` then evaluates
  `attn_dp_size = cfg.dp_size if cfg.enable_dp_attention else 1` (line 213) = 1 ≠ `tp_size` 4. Our attention TP is actually 1.
  Upstream main has the identical line (line 214 at `536d0653`, 2026-10-14T02:00Z, re-checked for this pack-up).
- **How:** `patches/pd_disaggregation_hook.yihou-attndp.py` — one line: use `cfg.attn_dp_size` when > 1, else the old expression.
  Mounted read-only on both legs (`mounts.yihou.pcomplete-attndp.tsv` via `PREFILL_/DECODE_MOUNTS_ADD`) → round pc48b.
  Result: validation passed, policy active on both legs.
- **Context / wrong turn:** the pre-run research table (`pc_upstream.md` §3) marked this check "TP4/DP4 passes, 4 == 4 [inferred]".
  That inference was wrong — it read the expression without the earlier rewrite. The run, not the research, found it.

## 3. Audit false negative stopped pc48b's pipeline (self-inflicted)

- **What:** `AUDIT FAIL (2)` → `STOP at audit` at 10:24 although the stack was healthy (smoke ok, 93 PASS).
- **Why:** `tools/audit_live.v3.yihou.py` treats `*_MOUNTS_ADD` entries as role-only: a mount listed for prefill must NOT be on
  decode and vice versa. The same TSV was given to both roles, so each leg "has the other role's mount".
- **How:** verified server-info on both legs (policy, write_back, 370, optimistic attempts 1, decode `max_total_num_tokens`
  2,816,896), then continued A1 → flush → A2 on the live stack with `scripts/harness/cont_bench.yihou.sh` (appends to
  `pc48b.chain.log`). Next time: put both-leg mounts in `YIHOU_MOUNTS` (new config), not in `*_MOUNTS_ADD` [untested].
- **Context:** consequence for the data — pc48b A1 started 6 min after smoke via a different driver; the bench command is
  identical (`results/pc48b-A1-C48/benchmark_command.txt` vs pc48e).

## 4. The A/B is one replicate per arm, run sequentially on one node

- pc48b ran 10:30-13:18, pc48e 15:53-18:50 UTC on 274 (pc48b stack stopped by `stop_then_chain`, 10-min VRAM drain). Same image,
  same dryrun except the policy flag + hook mount (`results/dryruns/tokdiff.yihou.txt`).
- Signs disagree (A1 −5.6 %, A2 +4.4 %). The earlier warm-protocol replicate spread on 102 was ±1 % (baseline kit); the fresh
  (A1) spread on 102 was 19,335-20,371 (≈5 %). Whether ±4-6 % here is noise or effect needs ≥2 more replicates per arm.
- pc48b improved A1→A2 by +9.7 %; pc48e changed −0.8 %. Not explained.

## 5. Both arms are >10 % below q1 — not attributed

q1 A3 (22,306) differs from both arms in: HiCache write policy (write_through vs write_back), HiCache size (450 vs 370 GB),
decode mem (0.95 vs 0.96), node (137 vs 274), and run position (q1 A3 = 3rd run on its stack, C44 → C44 → C48; pc48 A2 = 2nd).
Measurements that would split it (none run): pc48e with write_through (early, 370, 0.96); then 450 vs 370 at fixed write policy;
decode 0.95 vs 0.96. Sibling result hc50 (HiCache 50 GB, C40, other node/build) scored 20,112 vs 22,114 — HiCache size matters at
C40, but that does not tell the size of the 450→370 step.

## 6. Scheduler-log observations (first-hand, `analysis/logstats.pc48b-vs-pc48e.tsv`)

Means over all 4 DPs within each bench window (A1 / A2):

| field | pc48b prefill_complete | pc48e early |
|---|---|---|
| decode `#prealloc-req` mean (max) | 2.53 / 2.58 (22 / 20) | 0 / 0 |
| decode `#transfer-req` mean (max) | 1.56 / 1.67 (13 / 15) | 3.81 / 4.06 (20 / 23) |
| decode `pre-allocated usage` mean (max) | 0.085 / 0.093 (0.83 / 0.69) | 0.218 / 0.232 (0.84 / 0.92) |
| decode `token usage` mean (max) | 0.333 / 0.321 (0.91 / 0.89) | 0.474 / 0.457 (0.99 / 0.97) |
| decode `#running-req` mean (max) | 3.89 / 3.51 (16 / 19) | 4.03 / 3.53 (19 / 16) |
| decode `#retracted-req` | 0 | 0 |
| prefill `#queue-req` mean (max) | 3.10 / 3.41 (20 / 19) | 3.12 / 3.41 (18 / 23) |
| prefill `#inflight-req` mean (max) | 1.46 / 1.63 (15 / 16) | 1.35 / 1.25 (15 / 12) |
| prefill `#optimistic-req` mean (max) | 1.22 / 1.24 (9 / 8) | 0 / 0 |
| prefill `#bootstrap-req` mean (max) | 0 / 0 | 0.07 / 0.11 (7 / 11) |
| prefill `token usage` mean (max) | 0.145 / 0.155 (1 / 1) | 0.139 / 0.142 (1 / 1) |

The session ledger quoted pc48b A1 as prefill `#queue-req` max 20 mean 3.48, `#inflight-req` max 15 mean 1.38, token usage max 1
mean 0.15; decode `#prealloc-req` max 22 mean 2.46, pre-allocated usage max 0.83 mean 0.08, token usage max 0.91 mean 0.32,
`#running-req` max 16 mean 3.84. Maxima match the recomputation above; the means differ slightly because the ledger's window was
not recorded (the table above uses the exact `status.tsv` bench windows). Under `prefill_complete` decode reaches lower KV
occupancy at C48; whether that headroom can be turned into throughput (e.g. at higher CONC) was not measured.

## 7. Other gotchas met in this experiment

- Prefill logs end with a uvicorn lifespan `Traceback` at stack **stop** time (pc48b 15:10:36, pc48e 2026-10-14 01:53:18) — shutdown
  noise, not a run fault. No `Memory access fault`, no `policy mismatch` lines.
- `spec_accept.after-bench.txt`: accept length 3.55-3.78 per DP = the forced `SGLANG_SIMULATE_ACC_LEN=3.61`; it is not a correctness
  result. Smoke output is garbled by construction.
- pc48 was launched by `after_x8p_pc48.yihou.sh` behind x8p A2 on 274 (user "好的" 09:32); the x8p stack's VRAM took 10 min to drain.
- `config.yihou.mc4340n2.sh` hard-codes the expected image ID per node; an unknown node aborts the config chain (intended).

## 8. Upstream PR candidate (not filed)

See `patches/UPSTREAM-PR-CANDIDATE.md`. Not filed because the `open-source-pr` workflow (strip local semantics, re-validate on a
current main build, CI) was not run and the user did not ask for it.

## 9. Open questions

1. Is the A1/A2 sign flip noise? → 2 more A1+A2 replicates per arm on one node.
2. Which of write_back / 370 / 0.96 / node explains the −10…−14 % vs q1? → the split runs in §5.
3. Does `prefill_complete` help at higher CONC where decode KV binds (q1 C52 had decode token usage max 0.98)? → C52/C56 A/B.
4. Upstream #43442 (reclaim parked optimistic prefills; explicitly skips `prefill_complete`) and #43719 are not in the image; their
   effect on these numbers is unknown.
