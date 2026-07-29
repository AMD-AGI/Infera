> **[INVALIDATED 2026-07-29]** The premise of this document — that the
> degenerate output is an engine bug — was **falsified**. With the chat template
> applied and the model's own sampling (`temperature=1.0, top_p=0.95`), both
> MXFP4 and FP8 produce **0/128 degenerate at concurrency 128**. The symptom was
> caused by testing with raw base-LM completion at a forced `temperature=0`.
>
> Kept for the method and the raw data only. Every rate and every causal claim
> below is withdrawn. See
> `../../glm52.mxfp4.spur.mooncake.packup_20260729_degenerate_output/RETRACTIONS.md`.

# PD vs single-node mix: the degeneration is in DP-attention, not in PD

Date 2026-07-29. Two experiments run **in parallel**, differing in exactly one
variable: whether the deployment is PD-disaggregated or a single-node mix.

## First: a confounder found and removed

Before launching, I grepped the *running* `server_args` of the previous
"MTP off" arm and found:

```
MTP arm    : disable_custom_all_reduce=True
no-MTP arm : disable_custom_all_reduce=False     <-- 
```

`--disable-custom-all-reduce` was written **inside the MTP block** of
`pd_leg_spur.sh`, so `MTP=0` silently re-enabled the aiter custom all-reduce —
a kernel known to deadlock on gfx942/gfx950 under concurrency.

**The earlier "1.04 % with MTP vs 2.86 % without" comparison was therefore
two-variable and its 2.7× ratio must be retracted.** The flag has been moved
out of the MTP block in `pd_leg_spur.sh`, and added to `mix_leg.sh` (which
never had it at all).

## Setup

| | Exp1 | Exp2 |
|---|---|---|
| deployment | PD + mooncake (11232 prefill / 11233 decode) | single-node mix (9005) |
| `disaggregation_mode` | `'decode'` | `'null'` |
| `enable_dp_attention` | True | True |
| `speculative_algorithm` | **None** | **None** |
| `disable_custom_all_reduce` | **True** | **True** |
| source tree | clean upstream, 0 patches | clean upstream, 0 patches |

All flags verified by grepping the live `server_args` line, not inferred from
the launch script. 3 rounds × conc=128 × 512 tokens, `temperature=0`,
identical prompts, streaming tracker.

## Result

| arm | reqs | degenerate | rate | decode ranks hit |
|---|---|---|---|---|
| MTP + PD (patched, cAR off) | 384 | 4 | 1.04 % | 4/8 |
| noMTP + PD (clean, cAR **on**) | 384 | 11 | 2.86 % | 7/8 |
| **noMTP + PD (clean, cAR off)** | 384 | **7** | **1.82 %** | 5/8 |
| **noMTP + MIX (clean, cAR off)** | 384 | **7** | **1.82 %** | 5/8 |

**PD and single-node mix give exactly the same rate: 7/384.**

## Conclusions

1. **PD / mooncake is not the cause.** Removing disaggregation entirely
   changes nothing — same rate, same failure shape, same rank spread. The KV
   transfer path is exonerated.

2. **MTP is not the cause** (already shown, and unaffected by the confounder
   fix — both no-MTP arms still fail).

3. **Custom all-reduce is not the cause, but it is an aggravator.** Turning it
   off took the no-MTP PD arm from 2.86 % to 1.82 %. It makes things worse; it
   does not create the bug.

4. What remains common to every failing arm: **DP-attention + concurrency**.

## It is a race, not a prompt property

Failing prompt indices, by arm:

```
PD  : [69, 71, 77, 102, 110, 116, 126]
MIX : [31, 44, 51, 55, 83, 89, 121]
MTP : [22, 25, 33, 117]

PD ∩ MIX = {}          <- zero overlap
```

And within an arm, across 3 rounds, **no prompt failed twice**. 18 distinct
prompts out of 128 have failed at least once somewhere; none is reliably bad.
With `temperature=0` a deterministic engine would fail the same prompts every
time. It does not — so the corruption is injected by concurrent execution.

## The failure shape is invariant

Across all four arms, every degenerate request has the same structure:

```
token 0 : a digit          (legitimate -- the prompt ends "part 117.")
token 1 : '.'              <-- 21 of 21 degenerate requests, all arms
then    : a short periodic loop for the remaining ~510 tokens
```

Coherent requests that also start with a digit continue `':'` or `'\n'` and
then produce prose. The divergence is always at **token index 1**.

In the PD arms token 0 comes from prefill and token 1 is decode's first
output, which localised the fault to decode. The mix arm produces the
identical shape with no prefill/decode split at all, so the "first decode
step" framing is really "**the first token generated after the initial
forward**" — i.e. the transition into the decode loop, in both topologies.

Examples (all `temperature=0`, i.e. greedy):

```
3.2.1.1.1.1.1.1.1.1.1.1.1. ...      (PD)
1.1.3.1.1.3.1.1.3.1.1.3.1. ...      (MIX)
5.5.5.5.5.5.5.5.5.5.5.5.5. ...      (MIX)
1.2.3.4.5.6.7.8.9.10.11.12 ...      (MIX)
```

## Caveats

- Prefill-side DP rank is still not measured; `meta_info["dp_rank"]` is the
  decode/serving rank only, and `disagg_prefill_dp_rank` did not take effect
  when tried as a pin. No prefill-rank claim is made.
- The two PD arms differ in patch state as well (the MTP arm carries Bug
  1/2/5/6 + Variant B). The mix and no-MTP PD arms are both clean upstream and
  agree exactly, so the patches are not implicated either way.
- `first_bad_chunk` is a heuristic (it looks for a repeating cycle) and does
  not fire on counting sequences like `1.2.3.4.5...`, which are plainly
  degenerate. Where it did fire it reported index 0 every time. The raw token
  streams are stored in the jsonl for every degenerate request so the
  predicate can be re-judged offline.

## Next

The remaining variable is **DP-attention itself**. The split that isolates it
is a single-node mix with `--dp-size 1` (DPA off), same concurrency. If that
is clean, DP-attention is confirmed; if it still degenerates, the cause is
below DPA — in DSA decode or the sampler.
