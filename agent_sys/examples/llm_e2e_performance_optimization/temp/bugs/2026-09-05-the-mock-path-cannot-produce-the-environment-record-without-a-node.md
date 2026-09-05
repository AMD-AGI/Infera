# The mock path cannot produce the environment record without a node

**A constraint, not a defect — and it blocked Brief item 2 for two days.**

Recorded 2026-09-05 by the leader. Found by readme-cn while tracing why
`merge_profiling_evidence` hangs; the second of the two entries their
investigation produced. The first is
`2026-09-05-a-failing-program-task-is-recorded-succeeded-when-its-outputs-exist.md`
(`2a5b4e8`), which is what makes this one *invisible*. Read that one first.

## The constraint

The mock path exists so the five-stage chain can be exercised **without
hardware**. It has a hard dependency on reaching hardware anyway:

```
deploy_and_prove.task/mock_adapt.sh:70-72
  MOCK_IMAGE_ID="$(… remote.sh … docker image inspect "$E2E_IMAGE" --format {{.Id}} …)"
:74-80
  "mock_adapt: could not read a digest for '<image>' on the node." … exit 3
```

`remote.sh` asks **the node**. On a compute node that works. On the login node
there is no node to ask, the digest comes back empty, and `mock_adapt.sh`
refuses at `:74` — **correctly, loudly, and naming its own fix** — before ever
reaching `:127`'s `env_render --new --content-type code`.

So `items/codes/environment.yaml` is never written, and every downstream stage
that inherits from it (`env_render --inherit`) fails the same way.

**The refusal is right.** Its own message says why: an environment record naming
an image that may not exist on the node would fail later *"with something that
names neither this file nor the digest"*. It is refusing to fabricate a
measurement — the same rule the mission states for the real producer, that
`image_id` is discovered during bring-up and a variable holding it would be a
claim rather than a measurement.

## Why it took two days to see

**The refusal is discarded.** `entry.sh` captures `arc=3` and exits with it, but
`mock.sh` has already copied the kit, so the declared output exists and the task
is recorded `succeeded`. See `2a5b4e8`. Four bodies failed loudly in one run and
the run walked on; the only place it surfaced was
`merge_profiling_evidence`, the one task with no output to seal, which then hung
because a program body's escalation has no recipient.

## The other half: the corpus predates the record

Even with the render working, the replayed content has nothing to inherit:

```
/shared_nfs/yihou/agent_sys/cheat_for_mock/
  stage1-deploy      env.yaml=0   files=40
  stage2-profiling   env.yaml=0   files=140
  stage3-analyze     env.yaml=0   files=91
  stage4-kernel-opt  env.yaml=0   files=35
  stage5-integration env.yaml=0   files=136
```

**Zero `environment.yaml` in 442 files.** The corpus was sealed 2026-09-02; the
`environment` kind is this effort's Phase 0 machinery. `run_profiling_mode_off.task/entry.sh:33`
already says so — *"those artefacts predate `environment.yaml`, so the copy is
right and incomplete"* — which is exactly why `env_render` is on the mock path.
The design anticipated this. Only the render not landing was unanticipated.

## The workaround, measured

`mock_adapt.sh`'s own FIX line: set `MOCK_IMAGE_ID` explicitly, or pass
`--var image=` naming an image whose digest is readable where the body runs.

**Use a real digest, not a plausible one.** Taken from the 217 chain's sealed
kit rather than invented, so the record still describes an image that exists:

```sh
MOCK_IMAGE_ID=sha256:4601539c0f3d0f35a860e5a115510292ee4ca0ee854b56a8145e50e1932e59e2 \
python3 assets/lib/run_with_long_stall.py --stall-after 900 run \
    --package e2e-flow-noval --var mock_stages=all --var expect_ranks=2 …
```

Result, run `20260905T111302-99008d`, login node, no GPU:

| | before | after |
|---|---|---|
| `environment.yaml` in sealed handoffs | **0** | **18** |
| `merge_profiling_evidence` | hung at `running` | succeeded |
| stages reached | 2 of 5 | **5 of 5, `run complete`** |
| wall clock | — | **3 min 46 s** |

15 sealed handoffs, 408 files, `integration_report.json` present.
`m5_integration` had never reached `succeeded` in 45+ runs; the record was 4/5.

**Validation was disabled in that run** (`check_nothing` on every kind), so it
establishes reach and not correctness — and per `2a5b4e8` a `-noval` run cannot
distinguish "walked" from "walked over four silent failures". The `18` above is
the check that it was not the latter: it is a count of files that were absent
before and present after, not a verdict.

## What this changes about the loop

The whole five-stage graph is now exercisable in under four minutes on the login
node with no card. Every wiring question we have been paying a GPU hold and
forty minutes to ask can be asked here. `check_deploy_serves` is the one
validator that cannot come along — it does a real bring-up and a 180 s load, and
on the login node it sat for five minutes and had to be killed, leaving an
orphaned child that ran another six. `assets/lib/make_debug_package.py --keep`
takes the other twenty.

## Two things NOT to do about this

**Do not "fix" `mock_adapt.sh` to fabricate a digest.** The refusal is the
feature. A record naming a non-existent image is the failure it is preventing,
and it would surface two stages later with no reference to either.

**Do not widen the corpus gap into a corpus rewrite.** The sealed handoffs are
the only things here that ever produced a number. `MOCK_IMAGE_ID` plus
`env_render`'s existing `--inherit` chain is enough, and it leaves the corpus
untouched.
