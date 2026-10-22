# r5 / r6 plan: final PD runs on 054 (stop by 00:55Z 10-22), drafted 20:15Z 10-21

Base: the r2/r4 stack (DPA4 prefill unchanged, decode TP4/EP4 + DCP4, mode N, sim 3.61, CUDA graph).
Source: our branch HEAD **9af2166f3f**. It contains:
- hang fix c4eda0343e
- idlefix 5fb03011d6
- offloader c9394c114f
- KEEP_KV_B_BF16 flag 636cc68dd0
- DP2 pad fix b17da53725
- split-triton 0347a6d47e
- seq-len log 9af2166f3f

All new knobs are env-gated and default off, so this sha with no new env set is r4-equivalent plus the hang fix.

Reference numbers:
- r2 A2b 17,053 tok/s/chip (C40, 3600 s): TPOT 21.5 ms; decode 66.5 ms/step at 19.4 running; custom AR off; decode mem 0.85; max-running 64, graph bs 64.
- r2 A1 (fresh stack): 16,670.
- r4 is r2 + custom AR on + NCCL_DEBUG off, fresh stack. Its result is due about 21:20Z.

## 1. Timeline (two runs fit only if r6 reuses the r5 stack)

| Run | What | Window |
|---|---|---|
| r5 | Restart decode and prefill with the r5 config; AgentX C40 3600 s (comparable with r2/r4) | start ~21:25Z, bring-up ~30 min (HiCache drain + prefill up), bench ~22:00-23:05Z |
| cancel check | `tools/pd_cancel_repro.yihou.py` against the r5 stack right after its AgentX (about 3 min; checks the hang fix) | ~23:10Z |
| r6 | **Same stack, client-only change**: AgentX C56 3600 s (warm), or C40 A2-style if r5 shows a problem | ~23:15-00:25Z |
| stop | docker stop by 00:50Z | |

r6 at C56 needs no restart: concurrency is a client knob, and max-running 64 / graph bs 64 already cover 56. Doing the experiment as a client change is the only way to get two scored runs.

## 2. r5 config: knobs to combine

| Knob | r5 setting | Expected effect (source) | Risk | Fallback |
|---|---|---|---|---|
| Custom AR + aiter AR+RMSNorm fusion (decode) | ON, **if r4's AgentX errors stay at r2's level** (r4 warmup: 0 errors at 510 s, 20:12Z) | About -3.5..-3.9 ms/step (mixrun decodebench DCP4 34K, N=8..32, measured). It also costs about 10% of KV (mixrun: 8.89M vs 9.76M tokens), which does not bind at C40 (r2 usage 0.43-0.59). | r3's ~30% "no content" errors. r4 is the test that separates custom AR from the decode-only restart. | `--disable-custom-all-reduce` (r2 setting). |
| Hang fix c4eda0343e (always on in the sha) | in | No step-time change intended. One extra 2-int gloo all-reduce per transfer poll on attn_tp=4; its cost is unmeasured (it runs inside the overlapped CPU loop). Prevents the post-cancel hang that killed r3. | If the queues still diverge for another reason, it raises "Decode transfer queues diverged across attn-TP ranks" instead of hanging. | Set `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=0` for the remaining runs. That trades the hang risk for the risk of a prefill write landing in an aborted request's freed pages. |
| Idlefix 5fb03011d6 | in (no knob) | Expected inert on a non-DPA decode: IDLE batches need attn-DP or a gathered buffer. | none seen | revert the sha |
| Decode `--mem-fraction-static` | **0.90** (r2: 0.85; other session: 0.95) | No step-time change at C40 (KV not binding); more headroom for C56. | 0.95 on our decode also has to fit the custom AR buffers, the DCP buffers and (if adopted) +6.5 GB from bf16 kv_b. Graph capture OOM risk is untested here. | 0.85 (r2). Use 0.95 only if r4's startup log shows at least about 15 GB free after graph capture. |
| Decode tokenizer/detokenizer workers 4/4 | ON, if r4's dryrun does not already have them (verify) | Decode-side CPU only; no measurement on our setup. The other session runs it as its rule since 10-15. | Startup/ordering problems with multi-worker detokenization (none known). | 1/1 |
| Graph bs vs max-running | `--cuda-graph-max-bs-decode 64` = `--max-running-requests 64` (unchanged) | Avoids eager steps for batches over 64. C56 fits. | none | — |
| NCCL_DEBUG | off (as r4) | about 0 (no per-step logging under graph replay; not measured) | — | — |
| `SGLANG_LOG_DECODE_SEQ_LENS=1` | ON | Diagnostic only: Σ/max context per decode log line, host-side, no sync. | log volume +1 field | off |
| `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1` (q-rep) | **only if benchrun's A/B is faster and its sim-off gate passes before ~21:20Z** | -2..-5 ms/step (**estimate**, design/qrep_bf16_kvb.md): Q all-gather -4..-7 ms, minus +1.4..2.2 ms of compute. | +6.5 GB/GPU (estimate). In a PD run, only the decode is DCP, so the prefill is unaffected. Check the startup line `prepared full-head Q weights for 78 MLA layers`. | Unset it (one env). |
| `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1` + `..._SPLIT_INDEXER_TRITON=1` | **only if benchrun shows the triton arm faster than ref at both 34K and 150K, and its sim-off gate passes** | **Estimate**: about -4..-5 ms vs split-torch at 150K, near 0 at 34K. Split-torch alone measured as ref ±1.5 ms at 150K and +6 ms at 34K (T7/T9). | The kernels are validated only in the CPU interpreter, and graph capture of a Triton sort is untested. It can change attention inputs if wrong, so the gate is mandatory. | Unset both. |

## 3. Expected result (rough; assumes throughput ∝ 1 / decode step, which ignores prefill limits)

| Combination | Step (from 66.5 ms) | tok/s/chip (from 17,053) |
|---|---|---|
| custom AR only | about 62.7 ms | about 18.1k |
| + bf16 kv_b (if validated) | about 58-61 ms | about 18.6-19.6k |
| + split-triton (if validated) | unknown at AgentX contexts (pdrun's context script decides how much is long-context) | — |

None of the validated knobs alone reaches 20k by this estimate. The known gap is still the decode step: PD 66.5 ms vs mixed about 53 ms at matched N. The candidates for it are in my 19:56Z note (context mix first; pdrun is measuring it).

## 4. Fallback config, if r5 misbehaves at bring-up or in the first warmup minutes

r2 exactly (custom AR off, mem 0.85, 1/1 workers, no new env) on sha 9af2166f3f. That is r2 + hang fix + idlefix, both of which are inert for the step. The hang-fix gloo reduce is the only code change on the hot loop; if suspected, use r2's mounted source (src/sglang-ae529c96bf, per ledger.pdrun P009) instead.

Rules for the run:
- Change one variable at a time when possible.
- If r5 must combine several knobs because of the time limit, keep `SGLANG_LOG_DECODE_SEQ_LENS=1` on, so the step time can be binned against context afterwards.
