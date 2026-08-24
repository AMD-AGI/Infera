# pr-verify.experiment — hand-off for the unfinished upstream PR validation

Validating and landing three upstream sglang GLM-5.2 PRs
([#33968](https://github.com/sgl-project/sglang/pull/33968),
[#33970](https://github.com/sgl-project/sglang/pull/33970),
[#33973](https://github.com/sgl-project/sglang/pull/33973)).
Started on n06-33 (8x MI355X / gfx950) on 2026-08-19; **the hardware validation is
unfinished** and has to continue on another cluster. This folder is everything needed
to resume without re-deriving any of it.

## Read in this order

| File | What it is |
|---|---|
| `context.md` | The task, the artifacts, the environment, and **nine traps** that each cost time here. Read before touching anything. |
| `plan.md` | The remaining work, in order, with the pass/fail criterion for each step. |
| `working_process.md` | Narrative of what happened: what was established first-hand, what failed, how it was resolved. |

## Where things stand

| # | Stage | State |
|---|-------|-------|
| 1 | Upstream status + scope | done |
| 2 | Rebase all three onto current `main` | done — no conflicts, defects re-confirmed present |
| 3 | Code review | not started |
| 4 | **gfx950 hardware validation** | **unfinished — this is the blocker** |
| 5 | Deep review (LSP + serena) | not started |
| 6 | Flip draft -> ready, update records | not started |

Per PR: **#33970** is nearest — the single-node TP4+TP4 approach is proven viable and
two concrete launch bugs need fixing (`plan.md` step 1). **#33973** is not started but
unblocked on any single gfx950 box. **#33968** is blocked on finding a machine that
reproduces the fault at all; n06-33 does not.

## Layout

```
context.md, plan.md, working_process.md
scripts/            validate_{A,B,C}.py, probe_host_devptr{,_sizes}.py,
                    mvp_mooncake_loopback.py
logs/               build / pull / probe / RDMA-MVP / validator logs from this session
rounds/r02-stock-positive-control/
                    scripts/up_singlenode.sh   single-node 1P1D bring-up, ARM=stock|patched
                    scripts/needle.py          needle-in-a-haystack probe for #33970
                    logs/up_stock.log          the failed r02 bring-up
```

## The one rule to carry over

**The positive control comes first.** Stock must reproduce the defect before a clean
result on the patched tree means anything. `validate_{A,B,C}.py` all PASS here, and that
is *not* evidence any fix works — they check equivalence and scope against the local
patch and never require the defect to reproduce.
