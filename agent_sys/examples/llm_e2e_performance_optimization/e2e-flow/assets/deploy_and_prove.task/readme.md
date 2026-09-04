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
| `$E2E_IMAGE` | The docker image to serve with, carrying **both** infera and a matching engine. **It may not exist yet** — see step 1 |
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

**Criterion, by exit code — and read it as a code, not as a message:**

- **exit 0**, printing `mock: stage1-deploy/deploy_kit -> deploy_kit (N files)`:
  this is a mock run. **Stop here — the task is done.**
- **exit 3**, printing `... is not in E2E_MOCK_STAGES ...`: this stage is
  **not** mocked. **Continue to step 1 and do the real bring-up.** Exit 3 is
  *declining*, not failing.
- **anything else** is a fault. Report it and stop.

The distinction between 0 and 3 is the whole of this step and it is not
cosmetic: `mock.sh` used to exit 0 for both, and a caller that could not tell
them apart ran `mock.sh … && exit 0` — which made a task in **real** mode exit
successfully having written nothing. That is the "ten validators PASS over a run
in which every result was zero" failure. **This brief told you to read the
message rather than the code for one revision, which would have stopped a real
run at step 0.**

### 1. Preflight — is this host able to run what you were asked for

```sh
on "docker image inspect $E2E_IMAGE --format '{{.Id}}'"
on "ls -1 $E2E_MODEL_PATH | head; du -sh $E2E_MODEL_PATH"
on "rocm-smi --showcomputepartition --showmeminfo vram --json" || on "rocm-smi"
on "ss -ltn | awk 'NR>1{print \$4}' | sed 's/.*://' | sort -n | uniq"
```

**Criteria, all four:**
- the image id comes back as a `sha256:` digest — **record it, it goes in
  `environment.yaml` as `fixed.image_id`**. A tag is not a reproduction.

  **If `docker image inspect` says `No such image`, that is a fork in the road
  and not a failure.** This brief said "nothing needs building or pulling" for
  one revision, and that was an assumption from the one model it had been run
  against rather than a fact about models. Measured 2026-09-03: GLM-5.3-Flash is
  `Glm5NextForConditionalGeneration`, and **no released sglang image carries
  `glm5_next`** — support lives in an unmerged upstream PR, so the engine image
  has to be *built* before anything can be served. A second model met this on
  the first try; assume yours can.

  So: if the image is absent, build it, and then **`fixed.dockerfile` must be
  non-null and must name a path *inside this handoff*.** That field exists in
  `environment.schema.json` for exactly this case — *"or null when the image was
  pulled rather than built"* — and a Dockerfile that lives anywhere else is a
  path the reproducer will not have. Copy it into the kit beside the scripts,
  record the base image **by digest** as well as by tag, and say in `notes.md`
  what the build adds and why the released image was not enough.

  A build is minutes-to-an-hour and a large amount of disk. **Check the clock
  against your allocation before starting one**, and say in `notes.md` how long
  it took, because the next person's budget depends on it;
- the weights directory lists shards and its size is plausible for the model;
- **`fixed.gpu_count` is the number of devices that `rocm-smi` call LISTS** —
  `card<N>` keys in the `--json` form, `Device` rows in the table form. Count
  them. **Do not compute it from `$E2E_TP`, and do not reduce it by what a
  co-tenant is holding**: `environment.schema.json` defines the field as *cards
  PRESENT on the node, not cards this deployment could use*, so **a node with
  four cards busy still records 8.**

  This criterion exists because there was none. STEP 1 named a command for the
  digest and a command for the arch, and produced a device *index* — never a
  total — so `fixed.gpu_count` was the one required field no instruction
  generated. Rung 1 recorded `8` with four cards held by another tenant, and
  under this reading that was **correct**; it went in unchecked, and *that* is
  what this fixes, not the value (`todo.md` **T23**);
- free VRAM per device exceeds the checkpoint size with room for the KV and any
  state pools, and **you can say which device indices you are taking**.

  **That is a different number from the one above, and on a shared host it is
  the smaller one.** Both go in the record now: the count as `fixed.gpu_count`,
  and the indices you take as **`fixed.gpu_devices`** at STEP 5. Put them in
  `$KIT/results/preflight.json` as well, with the free-VRAM reading you based
  the choice on — the record says *what you took*, and `preflight.json` is where
  *why* survives.

  **`$E2E_GPU_DEVICES` may already answer this.** It is a comma-separated list
  or the literal `none`. `none` means **nobody said** — choose freely from what
  `rocm-smi` shows free, and say in `notes.md` what you chose and why. A list
  means **the operator has taken this decision out of your hands**: use exactly
  those indices, and if one of them is not free, **stop and report it** rather
  than substituting a free one. An operator naming cards is usually avoiding a
  co-tenant, and silently picking a different card defeats the only mechanism
  that fact has;
- **the ports you intend to bind are not in that list.** They are shared with
  every other tenant on this node. If one is taken, move — do not wait for it.

Write what you found to `$KIT/results/preflight.json` later; capture it now.

**Every conclusion you write in that file must quote the numbers it rests on.**
Not *"all eight were free"* — **"all eight free, ≤300 MB used each, per the
reading at `<timestamp>`"**, with the same figures that are in the structured
array beside it. This is one sentence of discipline and it is here because its
absence cost a deployment:

> Measured 2026-09-04. A kit's `preflight.json` recorded `gpu_cards[]` showing
> every card at **198 GiB used, 90 GiB free**, stamped `measured_at
> 07:26:58Z` — accurate, current, and correct for the moment it declared. Two
> keys away, `gpu_devices_rationale` read *"All eight were free (≤300 MB used
> each, no co-tenant)"* and `vram_headroom_note` read *"~288 GiB free per
> card"*. **The prose was wrong by a factor of three against the data directly
> above it**, the deployment bound four cards a co-tenant was mid-load on, and
> the run was stopped by hand.

**A timestamp did not prevent it and could not have** — the file had one and it
was right. What failed is that **a conclusion outlived the reading it came from
while the reading sat in the same document**, and nothing compares the two: no
validator reads `preflight.json`, and *"this sentence disagrees with that array"*
is not something the layout's regex rules can express (`todo.md` **T27**).

So the check has to be yours, at the moment you write it: **if a sentence claims
a fact about the machine, put the number in the sentence.** A conclusion that
restates its evidence cannot silently outlive it — and if the number you are
about to quote looks stale, that is the moment to re-read it, which is the whole
point.

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

**Mount what a later stage will need to measure in, not only what the engine
needs to serve.** Your container is not only a server: **m3 and m4 `docker exec`
into the one your record names**, because CONTRACT §5 gives its lifetime to you
and a consumer that created its own would be acquiring something it does not
own. **The consequence is that a consumer inherits your mounts and cannot add
any.** Measured 2026-09-04, from inside a container of this stage's that was
otherwise perfect — torch present, four cards visible:

```
/shared_nfs      : No such file or directory
the run root     : No such file or directory
```

It served correctly and **could not be measured in**. That is a hard stop at
rung 4 and it is invisible until then.

So the container must see the **run root** and the **workset root** read-write,
at the same path inside as out. **Use the two forms this cluster's docker
authorization plugin has been measured to accept, and no third:**

```
/shared_nfs/…      ->  -v /shared_nfs:/shared_nfs                OK
/home/<user>/…     ->  -v /home/<user>:/home/<user>              OK
                       -v /home:/home    denied [BH] by spur-authz
```

**Copy the derivation rather than writing one** — it is
`assets/build_workset.task/measure_in_container.sh:249-266`, and it already
carries the two corrections that cost m3 a run: derive the mount from the
**root** and never from `$HOME` (in a closed zone `$HOME` is `/home`, the one
form that is refused), and **refuse anything outside the two forms, naming
both**, rather than letting an authorization denial surface in the middle of
somebody else's measurement.

`deploy_kit.layout.yaml`'s `runtime_contract.measurement_visible` is the
contracted version of this paragraph.

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
        --set fixed.gpu_devices=<the indices you took, e.g. 4,5,6,7> \
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

- `scripts/deploy.sh` brings the whole thing up, `scripts/wait_ready.sh` blocks
  until it answers and **exits non-zero on timeout**, and `scripts/teardown.sh`
  removes what this run created **and only what this run created**. A readiness
  wait that exits 0 when it gave up turns "the model never loaded" into "the
  benchmark measured nothing", and the second is found three stages later;
- they honour five parameters, each in a defaulting form —
  `: "${E2E_KIT_RUN_TAG:=…}"`, `: "${E2E_KIT_PORT_BASE:=…}"`,
  `: "${E2E_KIT_WORK_ROOT:=…}"`. A bare `$NAME` is not enough: only the
  defaulting form lets the kit run with none of them set, which is the state the
  reproducer is in;
- and the two that let a later stage vary the engine without copying your
  launch: `: "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"`, appended **last** to the worker's
  argv so it overrides an earlier occurrence of the same flag, and
  `: "${E2E_KIT_ENGINE_EXTRA_ENV:=}"`, a space-separated `K=V` list exported into
  **the worker process** — not the container, because the router must not see
  it. Both empty by default, so a caller that sets neither gets your behaviour
  byte for byte. Module 2 brings your deployment up twice through these, and its
  two arms differ by nothing else;
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
- **`fixed.gpu_devices` is present and no longer than `fixed.gpu_count`.** The
  layout refuses a real bring-up that omits it, and it is refused *after* your
  deployment has already happened — a 40-minute bring-up graded in seconds. It
  costs nothing to confirm now;
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
