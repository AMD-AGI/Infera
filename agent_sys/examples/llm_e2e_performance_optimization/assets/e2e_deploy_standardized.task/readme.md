# e2e_deploy_standardized — deploy the model you were given, then hand back the kit

Your instruction is in **`$E2E_INSTRUCTION`**. Read it first; everything below
is how this system expects that instruction to be carried out, not a
replacement for it.

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
| `$E2E_IMAGE` | A docker image carrying **both** infera and a matching engine. Nothing needs building or pulling |
| `$E2E_ETCD_IMAGE` | An etcd image |
| `$E2E_DEPLOY_MODE` | The deployment shape — `mix` unless you were told otherwise |
| `$E2E_TP_SIZE` | Tensor-parallel size to use |
| `$E2E_WORK_ROOT` | Where a container workdir may go. **Local disk** |

**The repository's own `README.md`, `docs/` and `examples/` are the reference.**
There are worked launch scripts in `examples/` for models on this class of
hardware. Read them. Nothing in this readme is a substitute for them.

## What `mix` means, and how you will know you got it

`mix` is this repository's own word and it is **not** mixed precision. The repo
defines it; read `examples/` and `docs/` rather than taking my word for it. What
you are aiming at is one worker doing both phases, on a single node, with no
RDMA.

The criterion is checkable and you must check it: **the worker and the router
must independently report the disaggregation mode** — in the worker's own log
line and in the router's worker listing. Mix mode is selected by *omitting* a
flag, and an absent flag is not evidence of anything. Read the mode back out of
the running system twice, from two different components, and put both readings
in `results/`.

## Success, stated so it can fail

1. A server is up and `/health` answers.
2. A real chat completion comes back **through the infera router**, not only
   from the engine's own port.
3. The worker and the router both report the deployment mode you were asked for.
4. The model is registered under **`$E2E_MODEL_NAME`**, not under a filesystem
   path.
5. Every server, worker, router and container you started is **torn down** by
   the time you finish.
6. The kit reproduces it.

## The traps, each of which has already cost somebody a day

- **Every identifier you bind on a shared host is a parameter, not a constant.**
  The host is shared and so is its docker daemon: container names, host ports,
  the container workdir and the GPU index sit in one namespace with everybody
  else.
  - **Check before you bind, and record what you actually used.** The ports the
    docs use are frequently already held by somebody else.
  - **Let the kit be re-pointed without editing it.** Every such identifier must
    come from an environment variable with a default — `: "${VAR:=…}"`, never
    `export VAR=…` — so a second copy runs beside the first by exporting
    different values. `check_deploy_reproduces` starts another deployment from
    your kit while yours may still be up, and it cannot edit your scripts.
  - **Never take a name you did not create.** `docker rm -f <name>` before
    `docker run --name <name>` turns a collision into a silent theft: it kills
    whatever was there — quite possibly your own still-running server — and then
    reports success. Fail on a name that is already taken. Do not clear it.
  - **The workdir is the one that survives fixing the names.** A fixed container
    workdir means two runs share one `logs/` and one `results/`, and your
    verification script writes its evidence into `results/`. The consequence is
    not mixed-up logs, which you would notice: it is **a verifier reading the
    other run's evidence and passing on it**, which you would not.
- **The served model name travels further than the host.** Left unset, the
  engine registers the model under the filesystem path it loaded from, so every
  caller's `"model"` field then carries your machine's directory layout — baked
  into the one part of the kit a reader is most likely to copy. Set it
  explicitly to `$E2E_MODEL_NAME` and say in `environment.md` which name you
  used. `check_deploy_kit` refuses a kit whose evidence shows a path there.
- **The container workdir must be on local disk.** A home directory is often on
  a network filesystem with `root_squash`, where a container's root maps to
  nobody and the engine fails to write its logs **silently** — no error, just no
  log. Use `$E2E_WORK_ROOT`, and check what you are actually writing to.
- **A cold start takes minutes, and the log repeats a health-check failure the
  whole time.** That is a JIT compile, not a hang. Killing it and retrying is
  the single most expensive mistake available here. Graph capture can add
  minutes more.
- **Reasoning models emit a long preamble before the answer.** A small
  `max_tokens` truncates them mid-thought and the response comes back with a
  length-based finish reason. That is the request being too small, not the
  server being broken — and a correctness check that greps the output for one
  word will report a working server as broken. Budget tokens, or parse the
  reasoning, and say in the kit which you did.
- **Tear everything down when you finish.** The kit is the deliverable, not a
  running process. Tearing down is not a substitute for the parameter rule: it
  makes two runs safe in *time*, and the rule makes them safe in *space*. You
  need both, because you do not control when the reproducer starts.

## Where to write the kit, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_DEPLOY_KIT`**. That is
your output handoff's content directory; it already exists and you are granted
write on it. Do not create anything beside it — `claim/` and `manifest.yaml` are
the system's to write, and the manifest is what makes the version published.

```
README.md                                   with ## Purpose, ## Interface, ## Boundary
items/codes/<experiment-name>.packup_<YYYYMMDD>/
    README.md
    REPRODUCE.md
    environment.md
    notes.md
    scripts/
    results/
```

**`items/codes/` is not decoration.** This handoff's `content_type` is `code`,
and that type requires an item named `codes` and nothing else at the top level.
A file placed directly under `items/` is rejected before anyone reads it.

**Exactly one packup directory**, named `<experiment-name>.packup_<YYYYMMDD>`
with a real eight-digit date. Two of them means a reproducer has to guess which
kit is the one that worked.

### The packup itself

Use the `experiment-result-packup` skill. It defines the layout and it is the
authority for it; what follows is only the part this task makes mandatory.

| entry | what it must carry |
|---|---|
| `README.md` | What this was, and **a `## Result` section** saying whether it worked and how you know |
| `REPRODUCE.md` | Ordered, copy-pasteable commands from zero to the same result, and an **`Expected output`** section naming what success looks like and where to see it |
| `environment.md` | The exact hardware and software: host, **GPU architecture**, **image tag**, **model name and path**, driver and library versions. Pin things — this is the file whose absence breaks reproductions |
| `notes.md` | The gotchas and the wrong turns. Everything in the trap list above that bit you, plus everything that bit you and is not in it |
| `scripts/` | The scripts you actually ran, copied verbatim. Not paraphrases — byte-level flags have to survive |
| `results/` | The evidence, **machine-readable**: the completion you got back, the worker's mode line, the router's worker listing, timings. At least two of these must be `.json` — the next stage of this flow consumes results, and it consumes files rather than prose |

`patches/` and `logs/` are optional. Ship them if the work produced them and
leave them out if it did not — an empty directory is worse than an absent one.

**`REPRODUCE.md`'s `Expected output` section is load-bearing.** It is what
`check_deploy_reproduces` hands its reproducer as the criterion; there is no
other. If it is vague, a correct reproduction can be judged a failure and an
incorrect one a success, and neither is recoverable afterwards.

### Worked examples

`examples/` beside this readme holds **sanitised kits from runs of this task
that passed**, one per model. They are the shape to match, not the commands to
copy: their hostnames, ports and paths were stripped precisely because they were
one site's and yours are yours. Read one before you start writing, and read its
`environment.md` twice — that is the file most kits get wrong.

If `examples/` is empty, you are the first run of this task for this
deployment. Say so in `notes.md`.

### What is checked, and with what numbers

`check_deploy_kit` runs first and is exact. Its rules, in full:

- exactly one `<name>.packup_<YYYYMMDD>/` under `items/codes/`;
- all four files and both directories above present;
- `scripts/` and `results/` each hold at least one file, and `results/` holds at
  least **two `.json` files**;
- **content lines** — non-blank, not a heading, not a fence marker — at least
  **5** in `README.md`, **8** in `REPRODUCE.md`, **8** in `environment.md`,
  **3** in `notes.md`;
- no `TODO`, `TBD`, `FIXME`, `XXX`, `to be filled in`, or a leftover `<…>`
  template placeholder, in any of the four;
- at least **5 command lines inside code blocks** in `REPRODUCE.md`, and an
  `Expected output` section;
- a `## Result` heading in `README.md`;
- `environment.md` names a **GPU architecture**, an **image**, and the **model**,
  and carries at least one digit;
- no evidence file shows the model served under a **filesystem path**;
- no identifier is frozen in `scripts/` and then bound into a host-wide
  namespace — a container name, a published or listened-on port, the container
  workdir.

`check_deploy_reproduces` runs second and is the expensive one: a fresh Claude
Code session gets a copy of your kit, follows `REPRODUCE.md` for real, and
reports whether it reproduced. It must be able to do that from your kit
**alone** — it does not get this readme, it does not get your transcript, and it
does not get the machine in the state you left it.

### `README.md`, the handoff's own

Three sections, all required by the `code` content type, all checked for being
non-empty and for not being a placeholder:

- **`## Purpose`** — what this handoff is: a delivery kit for one model's
  end-to-end deployment.
- **`## Interface`** — how it is consumed: point at the packup directory and at
  `REPRODUCE.md` as the entry point, and say what a reproducer needs to have
  before they start.
- **`## Boundary`** — what is **not** here. Be specific: throughput, latency and
  accuracy are not in scope unless you measured them; say what you did not try
  and what you left open. An honest boundary is worth more than a confident one.

Do not claim in either README that anything was verified unless you ran it.
`check_deploy_reproduces` will disagree with you, in public, and its transcript
is kept.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That one file's
difference is the whole of what "an agent task" versus "a program task" means in
this system.
