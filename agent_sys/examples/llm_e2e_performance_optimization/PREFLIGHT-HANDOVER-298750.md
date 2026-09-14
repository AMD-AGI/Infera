# The m1 -> `_off` handover preflight — the branch we wanted is still untested

**Written 2026-09-06T13:42:52Z by m35**, read-only from artefacts while
`20260906T130845-298750` was running. **No GPU touched.**

This is the event the launch existed to test: m1 sealed at 13:40:27Z and the
`_off` arm came up behind it, so the corrected card-preflight ran **at a real
handover** rather than on an idle node.

## 1. The verdict — `/data/yihou/e2e_flow/pmoff/preflight.json`, `measured_at 2026-09-06T13:40:22Z`

**Outside the run tree**, under `work_root`, in a per-arm directory. That is the
**third** preflight this run has written, each in a different place:

| bring-up | measured_at | file |
|---|---|---|
| m1's deploy | 13:21:46Z | in the sealed kit, `…/results/preflight.json` (+ 3 staged copies) |
| `check_deploy_serves` | 13:34:05Z | `/data/yihou/e2e_flow/validate/serves-ff0b8393/` |
| **`_off` arm (this one)** | **13:40:22Z** | **`/data/yihou/e2e_flow/pmoff/`** |

```json
"run_tag": "yihou_e2e_chain_pmoff",
"gpu_devices_taken": [0,1,2,3],
"gpu_busy_floor_bytes": 2147483648,
"gpu_holder_instruments": {
  "order": [
    "1. container labels (deciding) -- docker ps --filter label=deploy_kit_owner, which crosses container boundaries",
    "2. KFD process inspection (supplement, never deciding) -- rocm-smi --showpids plus /proc cgroup, which cannot see into another container",
    "3. rocm-smi VRAM per card (the reading that says a card is busy at all)"
  ],
  "containers_with_gpu_access": [
    "rc_26_7_902||/dev/kfd is mapped in",
    "xiaoming-dev||/dev/kfd is mapped in"
  ],
  "kfd_processes": "no KFD process is visible from this namespace (which is ALSO what a card held by one of our own containers looks like from here, so this reading decides nothing)",
  "seconds_waited_for_a_busy_card": 0,
  "verdict": "no card named in E2E_KIT_GPU_DEVICES read above the 2147483648-byte floor, so no holder classification was needed"
}
```

VRAM at the read: **297754624 B (~284 MiB) on every one of the eight cards**,
against a 2 GiB floor.

## 2. Was the hard branch exercised? **No. `seconds_waited_for_a_busy_card: 0`.**

**Third time today, and the third different reason it did not arrive:**

1. m1's deploy 13:21:46 — node idle;
2. `check_deploy_serves` 13:34:05 — node idle;
3. **the handover 13:40:22 — the predecessor had already torn down and settled.**

> **This is the case we most wanted and it still did not exercise the clause.**
> The cards were free by the time the preflight read them, so neither the wait
> path nor the ownership classification ran. **The ambiguous-holder branch that
> killed run 5 remains untested.**

## 3. Did it wait / which classification? **Not applicable — it never waited.**

Nothing to report for the leader's question 3, and reporting nothing is the
answer rather than an omission.

## 4. How much slack there was — **bounded, not measured, and the reason is a lost artefact**

Run 2 died on **1.8 seconds** of slack. This one had enough. **How much cannot be
determined after the fact:** the predecessor's containers
(`yihou_e2e_chain_serves-ff0b8393_sgl`, `aiperf_serves-ff0b8393`) are **gone from
`docker ps -a` entirely** — removed, not stopped — so their `FinishedAt` is
unreadable.

**Bound from my own observations:** they were alive at **13:37:48Z** (seen `Up 4
minutes`) and the preflight read at **13:40:22Z**. So teardown completed
somewhere in that window and the slack was **between 0 and ~154 seconds.**

**This is the fourth instance today of a fact only one process knew and nothing
recorded.** A teardown that removes its containers destroys the timestamp that
would price the next bring-up's margin. **The measurement that would fix it: have
teardown record its own completion time into an artefact**, the way the kit's
`teardown.json` already records its VRAM-settle check.

## 5. Defect, recorded and not fixed

```
yihou_e2e_chain_yihou_e2e_chain_pmoff_sgl
yihou_e2e_chain_yihou_e2e_chain_pmoff_etcd
```

**The `yihou_e2e_chain` prefix is applied twice** — the container name carries it
from both `--var container=` and the arm's own tagging. Cosmetic today; it
matters because **container names are how a human finds the right container in a
hurry**, and a doubled prefix is the kind of thing that makes a `grep` written
from memory miss. **Not fixed, as instructed.**

## 6. What the three zero-wait readings DO establish

Not nothing, and worth stating so the record is not read as three failures:

- the instrumentation runs, in the declared order, at three different bring-ups;
- the label query executes against two real foreign containers and correctly
  returns empty for both;
- **KFD's blindness is recorded as a property of the reading rather than as a
  zero** — the same sentence in all three files;
- **and the handover itself is clean**: the predecessor released the cards before
  the successor read them, which is the behaviour run 2 lacked.

**What none of them establishes is what the clause does when it has to decide.**
That needs a card genuinely held by an unlabelled container, which cannot be
manufactured honestly and has not occurred.
