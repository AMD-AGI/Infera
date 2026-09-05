# `m5_integration` — stage 5

Put m4's optimised kernel in front of the real service, decide whether it broke
or slowed anything, and pack the whole flow up.

A **non-leaf**: its work is its subgraph. Three leaves.

```
apply_patch          kernel_optimization + operator_workset + deploy_kit
                                                          → patch_overlay
integrate_and_verify patch_overlay + profiling_evidence + kernel_optimization
                     + deploy_kit  → stock.measurement
                                     patched.measurement
                                     integration_report
packup               all eight     → e2e_packup            (is_end)
```

Its inputs are the outputs of stages 1, 2, 3 and 4 — M5.1, *"首先强调输入是 1、2、
3、4 的输出"* — and each is a real datum rather than an ordering token:
`deploy_kit` for the image and the environment record, `profiling_evidence`
because the stock arm must reproduce m2's bench (M5.1.3.1) and because the
kernel's share of the profile is what the reconciliation needs (M5.1.3.2),
`operator_workset` because how a kernel goes back into sglang follows m3's
contract (M5.1.1), and `kernel_optimization` because it is the thing under test.

## The one structural change to know about

`integration-demo` had eight leaves. Five of them —
`serve_stock → measure_stock → serve_patched → measure_patched → compare` — are
one task here, because M5.2 forbids splitting bring-up from use across agents:
*"agent A 去把服务部署好，agent B 去使用：这是不被允许的"*. `seed_patch` is gone
too (M5.1); the input is m4's real optimisation.

**Three of those deleted edges carried an argument rather than a datum**, and the
one that mattered was `serve_patched ← measure_stock`: the patched bring-up's
first act is to destroy the stock deployment, so that edge said "the stock arm
has already been measured". It is now step 6 of one readme, which is weaker. Two
things replace it, and both are in the package:

- `round.sh` refuses to bring the patched arm up unless the stock arm's step
  record exists (`E2E_PRIOR_STEPS`);
- **`check_measurement_order`** refuses the pair afterwards if the arms overlap
  in time, ran different sequences, or contain a step that exited non-zero.

## Two containers, and this is the only stage allowed them

Modules 1–4 share one container on one held node (CONTRACT.md §5). This stage is
the designed exception, granted by mission G5.1: **a container holds one state
for its life**, which is the entire reason the two-arm design exists. Both arms
come up from the same image and the same `environment.fixed`, and differ in
exactly one thing — the set of bind mounts.

## Seven validators

| validator | dimension | what it is for |
|---|---|---|
| `check_environment` | completeness | the shared G5 rule, on all five kinds |
| `check_overlay_applies` | completeness | the overlay lands, compiles, and **differs** |
| `check_patch_live` | trustworthiness | the patched arm ran the patched **bytes** |
| `check_measurement_order` | trustworthiness | the deleted edge, as evidence |
| `check_acceptance` | completeness | correctness ran and is readable, frozen **and** ad-hoc |
| `check_bench_report` | completeness | each replay sent traffic and validates against m2's schema |
| `check_no_regression` | trustworthiness | the verdict recomputed from the raw numbers |
| `check_packup_shape` | usability | the export is a kit somebody who was not here can follow |

Three of them exist for failures that produce a **number rather than an error**,
which is the only kind worth a strong validator: a patch mounted but never
executed (`check_patch_live`), an overlay that changes nothing
(`check_overlay_applies`), and a report whose stated verdict is not the one its
numbers support (`check_no_regression`).

## Do not widen the bars

5% throughput and 10% latency. The within-arm round-to-round spread on a steady
node is ~2%. A previous round widened them to 35%/30% in response to two arms
measured fifteen minutes and one co-tenant apart; that was the wrong response and
the missing control is a comparability gate at bring-up (`../../../todo.md` T7).
`check_no_regression` now also refuses a report whose own declared bars are
looser than the validator's, so widening in the producer does not work either.
