# `check_deploy_serves` — usability, **strong**, program

Mission M1.2.3, the real-run validator. **Program, not AI** (G4.1): every step is
an HTTP request with a stated criterion, and there is no judgement in it that a
model would make better than a table.

## The four phases, and what each is worth alone

1. **Bring-up, from the kit's own `scripts/deploy.sh`.** This phase alone is
   what M2.3 leans on when it deletes module 2's `serve_baseline` step on the
   grounds that *"module 1 的 output 已经包含了如何部署的全量信息"*. If a program
   cannot deploy from this handoff, that claim is false, and this is where it
   becomes false loudly instead of two stages later.
2. **The probes**, from `probes.yaml`, run on the node by `probe_runner.py`.
3. **The load** — 1k in / 1k out, concurrency 16, three minutes (M1.2.3.4).
4. **Teardown, on every path including every failure**, because what this body
   leaves behind on a shared node is a GPU nobody else can use.

## The probe set, and why it is a document

Mission M1.2.3.3: *"不允许临时发挥，网上调研几个常用的针对不同方向诊断的 curl"*.

`probes.yaml` is that set. Every row carries a `direction:` — the failure it
discriminates — and a `source:` naming where it comes from: the engine's
documented API, an upstream issue that measured a divergence, or first-hand
evidence in this repository. **A row with no source does not belong there**, and
`probe_runner.py` sends only what the yaml lists, which is what *不允许临时发挥*
reduces to once it has to be enforced rather than intended.

Three of the eleven are worth restating here because they are the ones that would
not have been invented:

- **`engine_generation_ready`** — `/health`, `/v1/models` and `/get_model_info`
  do not go through the scheduler's ZMQ path, so all three keep answering while
  generation is wedged. `POST /health_generate` runs an actual generation. A
  deployment that passes the health and identity probes and fails this one is up,
  is serving, and will hang the first real request.
  ([SGLang native APIs](https://docs.sglang.io/docs/basic_usage/native_api))
- **`context_limit_refused`** — SGLang's OpenAI-compatible endpoint returns HTTP
  **400** for a context overflow when `stream: false`, and HTTP **200 carrying an
  error payload** when `stream: true`; vLLM returns 400 for both. **A checklist
  that gates on the status line reports the streaming failure as a success.** So
  the stream is judged on its body, not its status.
  ([sgl-project/sglang#19996](https://github.com/sgl-project/sglang/issues/19996))
- **`worker_registered`** — the router answers `/health`, accepts a completion,
  and the request then queues, because no worker ever registered. Upstream shape:
  the router calls `/get_server_info` during `discover_metadata` with a
  hard-coded 10 s timeout, and a worker whose metadata call times out is never
  registered while everything else looks healthy.
  ([sgl-project/sglang#20836](https://github.com/sgl-project/sglang/issues/20836))

`severity: warn` is used where a hard gate would be dishonest rather than where
the check is unimportant: `/metrics` is absent unless the server was launched
with `--enable-metrics`, and `/get_server_info` has a known upstream hang. Warned
probes are printed and recorded, and they do not refuse the handoff.

## The load is not a third load generator

`assets/bench/aiperf_synthetic.sh` is
`../../../integration-demo/assets/bench/aiperf_replay.sh` with one block
changed: `--input-file … --fixed-schedule …` becomes
`--synthetic-input-tokens-* --output-tokens-* --concurrency
--benchmark-duration`. It is a sibling rather than a flag on that script because
the two are mutually exclusive — `--fixed-schedule` treats `--concurrency` as a
ceiling and paces from the trace, so a "duration and concurrency" run cannot be
expressed as an argument to a replay. Everything else, including the four
container settings that each fail quietly if dropped, is carried across verbatim
with the reasons recorded there. `summarise.py` is reused unchanged.

**The load's floors are deliberately low.** This validator's question is *does
this deployment serve*, not *is it fast*: the numbers that mean something are
module 2's, measured under controlled conditions. A floor here that encoded a
performance expectation would fail a healthy deployment on a busy node.

## Where it runs, and why it is shaped like this

The body runs on the login node, which has no GPU and no docker; the deployment
runs on the held compute node. Every phase is dispatched through
`../lib/remote.sh` — the one seam this package has for reaching the node — and
the probe plan is resolved on the login side and executed on the node by
`probe_runner.py`, which is **standard library only** because the node's
`python3` is whatever the image carries.

**Everything site-specific arrives through `args`, never the environment.** A
validator declares no agent, so this package's `env` block never reaches this
body; only the policy-derived environment does. That is measured, and the
previous stage records a run lost to it.

**Everything the kit recorded is read from the kit**, not from `args`: node,
model, image, ports and endpoint all come out of `codes/environment.yaml` and the
handshake. The kit is the thing under test, and a probe pointed at anything but
what the kit wrote is testing something else.

## Collision safety on a shared node

The run that produced the kit may still be up, and four other owners share these
two nodes. So this body binds nothing the producer bound: a fresh
`E2E_KIT_RUN_TAG`, its own `E2E_KIT_PORT_BASE`, its own `E2E_KIT_WORK_ROOT`.
Those three are `deploy_kit.layout.yaml`'s `runtime_contract`, which
`check_deploy_kit` has already proved the kit honours — the cheap validator makes
the expensive one safe to run, which is also why it runs first.

Teardown is in a `finally` and is `check=False`: a teardown that fails must be
reported and must not replace the verdict on the deployment. What it may never be
is skipped.

## Failure modes this body chooses on purpose

- **A criterion whose input is absent is reported as unevaluated, never as
  passed.** A load with no summary has not shown the deployment serves under
  load, however clean its exit code.
- **A fatal probe failure skips the load and says so.** Three minutes of GPU on a
  shared node cannot tell us anything a failed probe has not.
- **A probe whose `when:` is unmet is dropped and named**, not silently skipped
  and not failed: a deployment shape that does not publish the engine's own port
  is a legitimate kit, less diagnosable.
- **The body dying is a refusal**, not a vanishing: the exception becomes a
  fault with its type and message.

## What is not covered

**Comparability.** This validator proves a deployment is *live and usable*; it
does not prove it is comparable to another arm's measurement — node load at
measurement time is uncontrolled here. That is `todo.md` T7, and it belongs to
whichever stage compares two numbers, not to this one.
