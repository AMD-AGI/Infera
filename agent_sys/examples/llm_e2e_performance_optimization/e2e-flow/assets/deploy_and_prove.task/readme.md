# `deploy_and_prove` — deploy the model you were given, then hand back the kit

Your instruction is in **`$E2E_INSTRUCTION`**. Read it first.

**The body of this brief is the `STEPS` section below** (mission G4.2.1): an
ordered list of commands, each with the criterion that says it worked. You
sequence them and read what comes back. **Do not invent a method** — where a
step says to read something, the answer is in the thing you are told to read,
not in your own judgement about what the flags should be.

Two deliverables and they are not the same thing. A run that works and is not
written down is worth nothing here; a kit that is written down and never ran is
worth less. **Do the work first, then pack up what you actually did.**

## What you were given

Facts about this site, not a recipe. None of them tells you which flags to use.

| variable | what it is |
|---|---|
| `$E2E_INSTRUCTION` | What you were asked to do, in plain words |
| `$E2E_MODEL_NAME` | The model's **published name** — `org/Model-Name`. This is what the served model must register as |
| `$E2E_MODEL_PATH` | The weights, **already on disk**, all shards present. Do not download anything |
| `$E2E_SERVED_NAME` | What to register the model as. Empty means `$E2E_MODEL_NAME`: resolve it with `: "${E2E_SERVED_NAME:=$E2E_MODEL_NAME}"` |
| `$E2E_IMAGE` | A docker image carrying **both** infera and a matching engine. Nothing needs building or pulling |
| `$E2E_ETCD_IMAGE` | An etcd image |
| `$E2E_DEPLOY_MODE` | The deployment shape |
| `$E2E_TP` | Tensor-parallel size to use |
| `$E2E_CTX` | The context window to configure |
| `$E2E_DSA_ARGS`, `$E2E_PARSER_ARGS` | Model-specific engine flag groups. The literal `none` means *this model wants neither* — which is different from nobody having said |
| `$E2E_CONTAINER`, `$E2E_PORT_*` | Starting points for the identifiers you bind. Starting points, not reservations: check before you bind |
| `$E2E_WORK_ROOT` | Where a container workdir may go. **Local disk** |
| `$E2E_JOBID`, `$E2E_NODE`, `$E2E_TRANSPORT` | Which node, and how to reach it. Use `assets/lib/remote.sh`'s `on "<command>"`; do not spell a scheduler command yourself |
| `$AGENT_SYS_OUTPUT_DEPLOY_KIT` | Where the kit goes. It exists and you are granted write on it |

**The repository's own `README.md`, `docs/` and `examples/` are the reference.**
There are worked launch scripts in `examples/` for models on this class of
hardware. Read them. Nothing in this readme is a substitute for them.

## Success, stated so it can fail

1. A server is up and `/health` answers.
2. A real chat completion comes back **through the router**, not only from the
   engine's own port.
3. The worker and the router both report the deployment mode you were asked for.
4. The model is registered under `$E2E_SERVED_NAME`, not under a filesystem path.
5. Every container you started is **torn down** by the time you finish.
6. The kit reproduces it, **and a program can drive it** — see step 6.

---

# STEPS

Each step is a command and the criterion that decides it. A step whose criterion
fails is not a step to work around: stop, read the log named in the criterion,
and fix the cause.

Two conventions used throughout. `on "<cmd>"` means

```sh
bash -c '. "$AGENT_SYS_TASK_PACKAGE/assets/lib/remote.sh"; on "$1"' _ "<cmd>"
```

which dispatches on `$E2E_TRANSPORT` — **do not write `spur` or `srun` yourself**
(M2.1: nothing site-specific in a spec). And `$KIT` means the packup directory
you create in step 5.

### 0. Mock, if this run is a mock run

```sh
bash "$AGENT_SYS_TASK_PACKAGE/assets/lib/mock.sh" stage1-deploy deploy_kit
```

**Criterion:** it prints `mock: stage1-deploy/deploy_kit -> deploy_kit (N files)`.
If it does, **stop here — the task is done.** If it prints
`... is not in E2E_MOCK_STAGES ... running for real` and exits 0, continue to
step 1. Any other exit is a fault; report it and stop.

### 1. Preflight — is this host able to run what you were asked for

```sh
on "docker image inspect $E2E_IMAGE --format '{{.Id}}'"
on "ls -1 $E2E_MODEL_PATH | head; du -sh $E2E_MODEL_PATH"
on "rocm-smi --showcomputepartition --showmeminfo vram --json" || on "rocm-smi"
on "ss -ltn | awk 'NR>1{print \$4}' | sed 's/.*://' | sort -n | uniq"
```

**Criteria, all four:**
- the image id comes back as a `sha256:` digest — **record it, it goes in
  `environment.yaml` as `fixed.image_id`**. A tag is not a reproduction;
- the weights directory lists shards and its size is plausible for the model;
- free VRAM per device exceeds the checkpoint size with room for the KV and any
  state pools, and you can say which device index you are taking;
- **the ports you intend to bind are not in that list.** They are shared with
  every other tenant on this node. If one is taken, move — do not wait for it.

Write what you found to `$KIT/results/preflight.json` later; capture it now.

### 2. Bring the deployment up

There is no single command here, because the launch flags are the model's and
the repository's, not this readme's. Read `examples/` and `docs/` for a model of
this class, write the launch as **scripts you can re-run**, and run them.

**Criterion:** the router answers within your own timeout, and the timeout is
sized from the weights and the storage you are actually reading from — not from
a number you copied. A cold start is minutes of JIT kernel builds and weight
load during which the log repeats a health-check failure. **That is not a hang.
Killing it and retrying is the single most expensive mistake available here.**

```sh
on "curl -sf -m 10 http://<node-ip>:<router-port>/health"
```

**Criterion:** HTTP 200. If it 503s, it is still starting; wait, and watch the
worker log grow rather than restarting anything.

### 3. Read the deployment back out of the running system

Do not trust the launch command. The deployment mode is selected by **omitting**
a flag, so the command line is not evidence of it.

```sh
on "curl -s http://<node-ip>:<router-port>/v1/workers"       > $KIT/results/router_workers.json
on "curl -s http://<node-ip>:<router-port>/v1/models"        > $KIT/results/router_models.json
on "docker logs <worker-container> 2>&1 | grep -m1 'worker ready:'" > $KIT/results/worker_mode_line.txt
```

**Criteria, all four:**
- `/v1/workers` lists at least one worker with `"status": "active"`;
- that worker's `"disagg_mode"` is the mode you were asked for — **the
  router's independent reading**;
- the worker's own `worker ready:` line carries `disagg=DisaggMode.…` — **the
  second, independent reading.** One component agreeing with itself is not
  evidence;
- every model id in `/v1/models` equals `$E2E_SERVED_NAME` and **none of them
  starts with `/`**. A model id that is a filesystem path is a machine's
  directory layout baked into the one field every caller copies. If you see one,
  the served-name flag is missing — and note that the flag may not appear in
  `--help`: infera parses its own flags with `parse_known_args` and forwards the
  rest to the engine's parser, so every engine flag works and none of them is
  documented in the infera help.

### 4. Prove it answers

```sh
on "curl -s http://<node-ip>:<router-port>/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"'$E2E_SERVED_NAME'\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer with one word.\"}],\"max_tokens\":2048,\"temperature\":0}'" \
  > $KIT/results/chat_completion.json
```

**Criteria, all three:**
- `choices[0].finish_reason` is `"stop"`. A `"length"` means **your request was
  too small**, not that the server is broken: a reasoning model emits a long
  preamble before the answer. Raise `max_tokens` and send it again;
- `choices[0].message.content` is non-empty. An empty `content` on a successful
  request is the wrong reasoning parser eating the answer — the request succeeds
  and the answer is gone;
- `model` in the response equals `$E2E_SERVED_NAME`.

### 5. Write the kit

Into `$AGENT_SYS_OUTPUT_DEPLOY_KIT`, in the layout
**`assets/schemas/deploy_kit.layout.yaml` defines**. That file is the authority
and `check_deploy_kit` is an interpreter for it, so read it — it names every
required file, every substance floor and every evidence rule, with the reason for
each. What follows is orientation, not a second copy of it.

```
README.md                                    ## Purpose, ## Interface, ## Boundary
items/codes/environment.yaml                 the flow's environment record
items/codes/<experiment-name>.packup_<YYYYMMDD>/
    README.md  REPRODUCE.md  environment.md  notes.md
    scripts/   results/   logs/
```

**`items/codes/` is not decoration.** This handoff's `content_type` is `code`,
which requires an item named `codes` and nothing else at the top level; a file
placed directly under `items/` is rejected before anyone reads it.

**`items/codes/environment.yaml` is the record**, and you do not write it by
hand — `assets/lib/env_render.py` builds it from this run's `E2E_*` variables
plus the facts only a bring-up can discover. **You are the flow's sole producer
of it**: every later stage inherits yours rather than re-deriving one, which is
what makes "modules 1–4 share one runtime" checkable instead of merely intended.

```sh
python3 "$AGENT_SYS_TASK_PACKAGE/assets/lib/env_render.py" --new \
        --content-type code --out "$AGENT_SYS_OUTPUT_DEPLOY_KIT" \
        --set fixed.gpu_arch=<the arch you read in step 1> \
        --set fixed.gpu_count=<count> \
        --set fixed.image_id=<the sha256 digest from step 1> \
        --set runtime.container=<the container you started> \
        --set runtime.endpoint=<the router URL you proved in step 2>
```

**Criterion:** it writes the file and exits 0 — it validates before it writes,
so a non-zero exit lists every problem with the field named. Confirm
independently:

```sh
python3 "$AGENT_SYS_TASK_PACKAGE/assets/lib/schema.py" \
        --schema environment --doc "$AGENT_SYS_OUTPUT_DEPLOY_KIT/items/codes/environment.yaml"
```

**Criterion:** `ok: … validates against environment`.

`environment.md` inside the packup is a **rendering** of that record for a human
to read, not a second record. The values the layout marks `must_render` must
appear in it verbatim, and `check_deploy_kit` compares them.

### 6. Make the kit callable, not only readable

This is the step that is new relative to a plain packup, and it is the reason
the next stage does not need a `serve_baseline` step of its own: **module 2
deploys from this handoff.** `deploy_kit.layout.yaml`'s `runtime_contract` says
what that takes, and it is small:

- `scripts/deploy.sh` brings the whole thing up, and `scripts/teardown.sh`
  removes what this run created **and only what this run created**;
- both honour three parameters, each in a defaulting form —
  `: "${E2E_KIT_RUN_TAG:=…}"`, `: "${E2E_KIT_PORT_BASE:=…}"`,
  `: "${E2E_KIT_WORK_ROOT:=…}"`. A bare `$NAME` is not enough: only the
  defaulting form lets the kit run with none of them set, which is the state the
  reproducer is in;
- on success `deploy.sh` writes `$E2E_KIT_WORK_ROOT/deployment.json` carrying at
  least `endpoint`, `container` and `run_tag`. `endpoint` is the **product**
  endpoint — the router, not the engine's port. Add `engine_endpoint` if the
  shape publishes one: it is what unlocks the engine-side diagnostic probes, and
  a kit without it is not refused, only less diagnosable.

**Criterion, and run it — do not reason about it:** with the three parameters
set to values *different from the ones your own run used*, `scripts/deploy.sh`
brings a **second** deployment up beside your first, writes its handshake, and
`scripts/teardown.sh` removes that second one and leaves the first running.

```sh
on "cd $KIT && E2E_KIT_RUN_TAG=selftest E2E_KIT_PORT_BASE=<a free band> \
    E2E_KIT_WORK_ROOT=$E2E_WORK_ROOT/selftest bash scripts/deploy.sh"
on "cat $E2E_WORK_ROOT/selftest/deployment.json"
on "cd $KIT && E2E_KIT_RUN_TAG=selftest E2E_KIT_WORK_ROOT=$E2E_WORK_ROOT/selftest bash scripts/teardown.sh"
```

If a second copy cannot run beside the first, the kit has frozen something it
binds — and `check_deploy_serves` will start a deployment from this kit **while
yours may still be up**, and it cannot edit your scripts.

### 7. Tear down, and prove it

```sh
on "cd $KIT && bash scripts/teardown.sh" > $KIT/results/teardown.json
on "docker ps --filter label=<your run label> --format '{{.Names}}'"
on "ss -ltn | grep -E '<your ports>'" || true
```

**Criteria:** no container carrying your run label remains, and none of your
ports is still listening. The kit is the deliverable, not a running process.

**Never `docker rm -f` a name you did not create.** That turns a collision into a
silent theft: it kills whatever was there — quite possibly your own still-running
server — and reports success. Fail on a name that is already taken; do not clear
it.

### 8. Self-check before you finish

Read `assets/schemas/deploy_kit.layout.yaml` once more against what you wrote,
and confirm each of these yourself:

- exactly one `<name>.packup_<YYYYMMDD>/` under `items/codes/`, beside
  `environment.yaml`;
- `README.md` has `## Result`; `REPRODUCE.md` has `Expected output` and at least
  five command lines inside code blocks;
- `results/` holds at least two non-empty `.json` files;
- no `TODO` / `TBD` / `FIXME` / `<placeholder>` in any of the four documents;
- nothing in `scripts/` assigns a value it then passes to `--name`, `--publish`,
  `--port`, `-p` or a mount without that value being `${…:=…}`-able.

`REPRODUCE.md`'s `Expected output` section is load-bearing: it is the only
criterion a reproducer is given. If it is vague, a correct reproduction can be
judged a failure and an incorrect one a success, and neither is recoverable.

---

## The traps, each of which has already cost somebody a day

- **Every identifier you bind on a shared host is a parameter, not a constant.**
  The host is shared and so is its docker daemon: container names, host ports,
  the container workdir and the GPU index sit in one namespace with everybody
  else. Check before you bind, record what you used, and write it so a caller
  can re-point it without editing the file.
- **The workdir is the one that survives fixing the names.** A fixed container
  workdir means two runs share one `logs/` and one `results/`, and your
  verification script writes its evidence into `results/`. The consequence is not
  mixed-up logs, which you would notice: it is **a verifier reading the other
  run's evidence and passing on it**, which you would not.
- **The container workdir must be on local disk.** A network home with
  `root_squash` maps a container's root to nobody and the engine then fails to
  write its logs **silently** — no error, just no log.
- **The served model name travels further than the host.** See step 3.
- **A cold start takes minutes and the log repeats a health-check failure the
  whole time.** Measured on this class of hardware: 186 s for ~51 GB of weights
  read from a local mount, and 910 s for ~328 GB over a network filesystem
  against a kit that documented 200 s. Size your wait from the weights and the
  storage you are actually reading from.
- **Reasoning models emit a long preamble before the answer.** See step 4.

## What is checked

Three validators, in cost order.

1. **`check_environment`** — the environment record every handoff in this flow
   carries.
2. **`check_deploy_kit`** — seconds, and exact. It is an interpreter for
   `assets/schemas/deploy_kit.layout.yaml`; read that file to know every rule,
   because there is no second list.
3. **`check_deploy_serves`** — expensive, and the one that decides whether this
   was real. It brings your deployment up **from your `scripts/deploy.sh`**,
   sends the diagnostic probe set in
   `assets/check_deploy_serves.validator/probes.yaml`, and then a 1k-in / 1k-out
   concurrency-16 three-minute load. **Read `probes.yaml`** — every probe states
   the failure direction it discriminates, and a kit that passes them is a
   deployment nobody has to re-diagnose downstream.

Do not claim in either `README.md` that anything was verified unless you ran it.

---

**This is a `readme.md` and the `entry.sh` beside it is only the mock path.**
That difference is the whole of what "an agent task" versus "a program task"
means in this system.
