# R4 — differential control: the unfixed build hangs on the same node, same traffic

Date 2026-07-29 18:12–18:25. **Same node (11429), same container, same config, same
router, same probe.** The only difference from R3 is that the fix was reverted
(verified: `_needs_eager_local` count in bytecode = 0; R1 probe retained = 3).

This is the 对拍 arm the project rules require. Without it, "it stopped hanging" is
an anecdote.

## Result: hung on the first request

```
[0] FAIL after 120.1s: TimeoutError
[1] FAIL after 120.1s: TimeoutError
[2] FAIL after   0.4s: HTTP 503   (router circuit opened after 0/1 timed out)
[3] FAIL after   0.4s: HTTP 503
0/4 ok
```

py-spy, twice 8 s apart, byte-identical → hard deadlock:

```
DP0: all_gather_into_tensor
DP1: broadcast
DP2: all_gather_into_tensor
DP3: init_forward_metadata (dsa_backend.py:785)   <-- the eager metadata path
DP4: all_gather_into_tensor
DP5: all_gather_into_tensor
DP6: all_gather_into_tensor
DP7: broadcast
```

## The prediction was exact

The guard records show `final` diverging on **exactly one** iteration — it=8, the last
one any rank reached:

| rank | final |
|---|---|
| **dp3** | **False → EAGER** |
| dp0,1,2,4,5,6,7 | True → GRAPH |

**dp3 is the rank py-spy found blocked in the eager `init_forward_metadata`.** Decision
and stack agree, independently, for the second time.

Note the victim rank **changed**: R1 was dp2, R4 is dp3. Same signature, different rank —
this is the race behaving as a race, and it rules out a rank-specific defect.

Per-rank draft-call totals are equal (8 each), so no rank was starved of iterations; the
divergence is in the decision, not the count.

## Side-by-side, everything else held constant

| | R4 control (no fix) | R3 (fix) |
|---|---|---|
| PD warmup | passed | passed |
| 4 × 24-token | **0/4 — deadlock on request 1** | 4/4 |
| 1 × 512-token | not reached | 512/512 |
| conc sweep 1/2/4/8/16/128/256 | not reached | **927/927, 0 failures** |
| `final`/`voted` uniform | **no** — diverged on the frozen iteration | **yes, 2992/2992** |
| draft graph used | yes (on the ranks that diverged into it) | yes, **98.4%** |
| victim rank | dp3 | — |

The control reproduces the failure at the first opportunity while the fixed arm survived
927 requests across seven concurrency levels. Same hardware, same image, same hour.

## What this closes

* The localization is not a timing artifact: reverting one line brings the hang straight
  back, on the same machine, within minutes.
* The mechanism is confirmed twice, with different victim ranks, by two independent
  signals that agree (the logged decision and the py-spy stack).
