# `stub_kit/` — a deployment-shaped stand-in, so an expensive validator can be exercised cheaply

Five scripts that satisfy `deploy_kit.layout.yaml`'s `runtime_contract` and
answer the eleven probes in `../probes.yaml`, **without a GPU, without an engine
image, and without loading a model.** `mock_adapt.sh` installs them into the
mocked `deploy_kit`, so `check_deploy_serves` does real work in a mock run
instead of being skipped.

## What it proves

Everything in `check_deploy_serves` that is not the engine:

- remote dispatch through `../../lib/remote.sh` to the held node;
- **bring-up from the kit's own `scripts/deploy.sh`** — the premise M2.3 leans on
  when it deletes module 2's `serve_baseline` step;
- the `runtime_contract` handshake: `deployment.json` written, parsed, and its
  required keys present;
- probe-plan resolution from `probes.yaml` and execution on the node by
  `probe_runner.py`;
- **the failure paths**, which is the half a passing fixture never reaches — a
  fatal probe skipping the load, a load that collects zero successful responses
  being refused rather than passed;
- teardown in the `finally`, on every path.

All of that was measured on `crsuse2-m2m-061` on 2026-09-03, and it **found a
real bug on its first run**: `build_plan` was expanding the oversize-prompt
probe's ~200 KB of filler into the plan, which travels to the node inside a
command string, so the call died with `OSError: [Errno 7] Argument list too
long: 'bash'` *after* a successful bring-up, naming neither the plan nor the
probe that grew it.

## What it does NOT prove, and this half matters more

**A stub mistaken for coverage is worse than no stub.** Two things this cannot
tell you, and neither is a detail:

1. **That a real engine answers the eleven probes the way `probes.yaml`
   predicts.** The stub answers them the way the *sealed kit's recorded
   evidence* says the real deployment answered them — `router_workers.json`,
   `router_models.json`, `chat_completion.json`. That is a faithful transcription
   of one run, not a second observation of the engine. Every probe's
   `direction:` describes a failure mode of a real server; none of them is
   exercised against one here.
2. **That a successful load clears `judge_load`'s floors.** Reaching them needs
   a load that succeeds, which needs a real engine. The stub's hand-rolled SSE is
   rejected by AIPerf's chat parser, so what the load phase exercises here is the
   **refusal** path — the validator reported `AIPERF_FAIL rc=1` with the reason
   and refused the handoff rather than passing on a run that collected zero
   successful responses. That path is genuinely exercised rather than merely
   written; the acceptance path is not.

A mock run that uses this kit has therefore **not proven m1**. It has proven
that `check_deploy_serves` works. Those are different claims and the run record
should carry the second, not the first.

## Two things a reader will assume and should not

**The stub does not compose with `E2E_KIT_ENGINE_EXTRA_ARGS` / `_ENV`.**
`check_deploy_serves` deliberately sets neither: it validates the kit **as the
kit is**, and a validator that varied the engine would be grading a
configuration nobody shipped. The stub reads them so that a kit carrying them
passes `check_deploy_kit`'s contract scan, and it ignores their contents.

**`deploy.sh` here starts processes, not containers.** Teardown is by pid file
under `$E2E_KIT_WORK_ROOT`, not by container name — so the "never `docker rm -f`
a name you did not create" rule is satisfied by there being nothing to remove.
The consequence is that `check_deploy_serves`'s writable-work-root probe, which
`docker exec`s into `runtime.container`, has nothing to exec into and reports
that; a real kit is where that check does its work.

## The rule this fixture exists to enforce, stated once

**A plan carries intent; a filler string is not intent.** Anything that crosses
the login-node → compute-node seam travels inside a command string, and a body
that puts *data* there rather than a *description* of the data will die on
`ARG_MAX` with an exception that names neither the payload nor the caller. Build
the payload on the far side from a parameter. `check.py` guards its plan at
128 KB so the next instance of this class fails with a sentence.

## Files

| file | what it is |
|---|---|
| `stub_env.sh` | The five `runtime_contract` parameters, each `: "${VAR:=…}"`. Named `stub_env.sh` and not `env.sh` so it never collides with the sealed kit's own. |
| `deploy.sh` | Starts two stub servers, waits, writes `deployment.json`. |
| `wait_ready.sh` | Polls `/health`; **exits non-zero on timeout**, which is the property a consumer's step 1 waits on. |
| `teardown.sh` | Stops what this run's tag created, and nothing else. |
| `stub_router.py` | Answers the probe set. Standard library only — the node's `python3` is whatever the image carries. |
