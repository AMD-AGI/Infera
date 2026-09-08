# The card-preflight verdict, captured live from chain 6

**Written 2026-09-06T13:39:49Z by m35**, read-only from artefacts while
`20260906T130845-298750` was running. **No GPU touched, nothing in the chain
disturbed.**

## 0. There were TWO preflights, not one — and they are in different places

The containers quoted to me (`serves-ff0b8393_etcd` 13:34:05, `_sgl` 13:34:07)
are **not** m1's deployment. They are **`check_deploy_serves`'s own bring-up** —
the validator standing the sealed kit up to prove it runs. Confirmed by phase:
`deploy_and_prove: output_validating` while those containers were alive.

**Both bring-ups ran the preflight and both recorded a verdict, in different
locations:**

| bring-up | `measured_at` | verdict file |
|---|---|---|
| **m1's own deploy** | `13:21:46Z` | in the **sealed kit**: `handoffs/7eaa84e7…/v1/content/items/codes/qwen3-32b-mix-tp4.packup_20260906/results/preflight.json` |
| **`check_deploy_serves`** | `13:34:05Z` | **outside the run tree**, in `validate_work_root`: `/data/yihou/e2e_flow/validate/serves-ff0b8393/preflight.json` |

**The second one is the one matching the containers**, and it is **not under the
run directory** — anyone reconstructing this from `runs/` alone would find only
the first and conclude the 13:34 bring-up recorded nothing. It did; it recorded
it somewhere `--var validate_work_root` points.

Three such files exist there, one per `check_deploy_serves` invocation today
(`12:00:33`, `12:49:40`, `13:34:05`).

## 1. The verdict, verbatim — identical in both

```json
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

## 2. Which instruments ran, and what each returned

**All three ran. The order matches the corrected clause exactly.**

- **Container labels — ran, and returned empty for both foreign containers.**
  `rc_26_7_902||/dev/kfd is mapped in` — the empty field between the two `|`
  separators is where a `deploy_kit_owner` value would sit. **Neither carries the
  label**, which is correct: they are not ours.
- **KFD — consulted, and recorded as non-deciding in the artefact itself**:
  *"no KFD process is visible from this namespace (which is ALSO what a card held
  by one of our own containers looks like from here, so this reading decides
  nothing)."* **That is the clause's own rule, written into the output rather
  than left in the code** — the ambiguity is stated at the point of measurement,
  so a later reader cannot mistake the zero for evidence.
- **VRAM — the deciding reading.** 297754624 B ≈ 284 MiB used on every card,
  against a floor of 2147483648 B (2 GiB). **All eight well under.**

## 3. How long it waited: **0 seconds**, both times

`"seconds_waited_for_a_busy_card": 0` — the expected answer on free cards, and
the artefact says so explicitly rather than omitting the field.

## 4. What this does NOT establish — and it is most of what we wanted

> **The clause under test did not execute.** Its own verdict says so:
> *"no card named in `E2E_KIT_GPU_DEVICES` read above the … floor, **so no holder
> classification was needed**."*

- **The cards were free** (284 MiB against a 2 GiB floor), so the **wait path**
  never ran and the **ownership-classification path** never ran.
- **The ambiguous-holder branch — the one that killed run 5 — is UNTESTED.**
  Nothing here shows what happens when a card is busy and the holder cannot be
  identified. That branch aborts immediately by design; **whether it does, and
  whether it says something useful when it does, remains unobserved.**
- **What IS established: the instrumentation.** All three instruments ran in the
  declared order, the label query executed and returned a real (empty) answer for
  two real foreign containers, and KFD's uselessness-from-this-namespace was
  recorded as a property rather than as a zero. **That is the plumbing verified
  and the decision logic unexercised**, which is a weaker result than the m1→`_off`
  handover would have given and is the result we actually have.

## 5. The one that settles the untested branch

A preflight run while a card is genuinely held by a container carrying **no**
`deploy_kit_owner` label. **We cannot manufacture that honestly** — it needs a
foreign tenant to take a GPU, which has not happened all day. **Until it does,
the branch stays untested and should be described that way rather than as
"working".**
