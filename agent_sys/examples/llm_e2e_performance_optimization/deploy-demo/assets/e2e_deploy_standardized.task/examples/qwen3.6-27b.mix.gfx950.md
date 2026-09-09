# Exemplar — Qwen3.6-27B, mix mode, gfx950, TP1

Derived from a run of **this task** that passed both validators: the shape check
and a fresh Claude Code session that followed the kit and reproduced the
deployment from it alone. Sanitised per `README.md` beside this file — the
hostnames, ports, run tags and user paths were that site's and are gone; the
shape, the evidence set and the wording of the criteria are the point.

## The kit it produced

```
<model>-mix-<engine>-<arch>.packup_<YYYYMMDD>/
├── README.md          what it was, and a ## Result section
├── REPRODUCE.md       ordered commands + an Expected output section
├── environment.md     host class, GPU arch, image, model name AND path, versions
├── notes.md           the traps that bit, as rules
├── scripts/           env.sh  pick_params.sh  preflight.sh  start_etcd.sh
│                      start_router.sh  start_worker.sh  wait_ready.sh
│                      verify.sh  verify.py  collect_evidence.sh  teardown.sh
├── results/           chat_completion.json     router_workers.json
│                      router_models.json       router_health.json
│                      engine_health.json       worker_mode_line.txt
│                      verification.json        timings.json
│                      teardown.json            collision_refusal.json
│                      concurrent_deployments.json
│                      reasoning_budget_probe.json
└── logs/              worker + router logs as produced
```

Twelve scripts and thirteen evidence files is not a floor — the floor is four
documents, two directories and two `.json`. It is what a kit looks like when
every claim in `REPRODUCE.md` has a file behind it.

## What made it reproducible, in order of how much it mattered

**1. Every identifier comes from `pick_params.sh`, not from the text.** The kit
is re-pointed by running one script, which chooses a run tag and four free
ports and writes them to a file the other scripts source. Nothing in the kit
names a port. That is why a second copy could be started beside the first —
`results/concurrent_deployments.json` is that experiment, run by the producer
on purpose.

**2. `collision_refusal.json`.** The producer demonstrated that its own scripts
*refuse* a container name that already exists rather than `docker rm -f`-ing it.
A kit that says "we do not steal names" is worth less than one that shows the
refusal.

**3. The mode is read back twice and the two readings are compared in a file.**
`verification.json` carries `mode_readback.from_worker_log`,
`mode_readback.from_router_worker_listing`, and `agree: true`. Mix mode is
selected by *omitting* a flag, so the launch command is not evidence of it.

**4. The published model name is asserted, not assumed.** See the trap below.

## The `Expected output` section — the shape to copy

Numbered, in the order a reproducer meets them, each naming a file and a
condition. Abridged:

> 1. `preflight.sh` ends with `preflight OK` and exits 0.
> 2. `wait_ready.sh` ends with `router reports 1 active worker(s) after N s`.
>    *On that hardware N was 196.*
> 3. `verify.py` prints `12/12 checks passed`; no line begins with `FAIL`.
> 5. In `verification.json`: `"all_passed": true`,
>    `"published_model_name": "<org>/<Model>"`, both `mode_readback` lists are
>    `["mixed"]`, `agree` is `true`.
> 6. In `router_workers.json`: exactly one worker, `"disagg_mode": "mixed"`,
>    `"status": "active"`, `"model_name": "<org>/<Model>"`.
> 7. In `worker_mode_line.txt`: a line containing `disagg=DisaggMode.MIXED`.
> 8. In `chat_completion.json`: `"model"` is exactly `<org>/<Model>` — **a
>    filesystem path there is a failure, not a cosmetic difference** —
>    `finish_reason` is `stop`, and the answer contains the expected value.
> 9. `teardown.sh` ends with `teardown complete`, all ports report `free`, and
>    no container of this run remains.
>
> Anything else is a failure. In particular: a completion that comes back only
> from the engine's port and not from the router is a failure; a worker that
> registers under a filesystem path is a failure; and the two mode readings
> disagreeing is a failure even if both individually look plausible.
>
> **Not a failure, and do not report it as one:** a first attempt returning
> `finish_reason: "length"` with no answer in it. That is the reasoning-model
> budget trap; `verify.py` retries at four times the budget.

The last paragraph is the part most kits omit and the one that most often turns
a correct reproduction into a reported failure.

## The traps this run recorded

1. **`--served-model-name` does not appear in the engine's `--help`.** infera
   parses its own flags with `parse_known_args` and forwards the remainder to
   sglang's `ServerArgs`, so every sglang flag works and none is documented
   there. The worker registers under `served_model_name or model_path`, so
   omitting it publishes the *container's mount path* as the model id. An agent
   that greps `--help` will conclude the flag is unsupported and ship a kit that
   fails the served-name rule.
2. **A cold start is minutes of silence.** Weight load alone was ~150 s for
   ~51 GB read over a network filesystem, during which the health endpoint
   answers 503 and nothing is written. Reading from network storage can make
   this far longer — a sibling run of a 328 GB model took **910 s** to first
   health, against a kit that documented 200 s. Set the wait well above what
   you measured, and never treat repeated 503s as a hang.
3. **The reasoning preamble is not a bug.** Two identical prompts returned 159
   and 302 completion tokens; a budget sized for the smaller one truncates the
   larger mid-thought and returns `finish_reason: length`.
4. **VRAM lags `docker rm -f`.** Seven seconds after removal the card still
   reported 63 % used with no container and no process; it read 0 twenty-five
   seconds later. A teardown check that reads once reports a leak that does not
   exist.

## Numbers, marked as one site's

Cold start to `worker ready` 274 s (weight load 148 s of it); whole deploy plus
verify 205 s on a warm page cache; TP1 on one card of eight, ~51 GB of weights
against 288 GB per card. These are *shape*, not targets: the task makes no
throughput or latency claim.
