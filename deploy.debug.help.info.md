# `deploy-demo` — debug notes, run of 2026-09-02

Module: **deploy-demo** (stage 1, `e2e_deploy_standardized`).
Node: job `101078`, `crsuse2-m2m-080`, GPUs 0–1. Port band 8100–8119.
Worktree HEAD at start: `532da57` ("stage 1 moves into `deploy-demo/`").

Prior art: this package passed 3+3 consecutive runs on 2026-09-01 on
`crsuse2-m2m-020` / `-188`. Run roots under
`/shared_nfs/yihou/agent_sys_debug/runroot/{qwen,glm}-s[123]`.
Driver: `/shared_nfs/yihou/agent_sys_debug/ws/run_agent_sys.sh`.

---

## 1. The move into `deploy-demo/` did not break the load — confirmed

`agent-sys show --package .../deploy-demo` with the four required site vars:

```
package  loaded 1 task package(s) from .../deploy-demo
closure  e2e_deploy_standardized: agent 'e2e_deployer', 0 in, 1 out
closure  main: agent None (a non-leaf declares none), 0 in, 1 out
  graph  2 tasks: 1 root and 1 subtasks
  phase  main: output validation runs 2
  phase  e2e_deploy_standardized: output validation runs 2
```

**2 tasks, 2 closures, 2 validators on each of two output phases.** Same as
before the move.

Two things worth knowing for anyone moving a package directory:

- **Without the four site vars, `show` REJECTs** and prints
  `deploy.yaml:69:5::$[0].env.E2E_MODEL_NAME: no value for ${model_name}`.
  That is the package working as designed, not the move breaking it. Do not
  read a rejection as a load failure until you have supplied
  `model_name`, `model_path`, `image`, `packup_skill`.
- **The exec bits survived the move.** `git ls-files -s` shows `100755` on
  both `check_*.validator/entry.sh`. This matters: `agent/gate.py` requires
  `script`/`command`/`entry` handoff items to be executable, and a lost
  `chmod +x` is reported as a *timeout*, not as a permission error (BRIEF
  known-limit 2). `git mv` preserves mode; a `cp -r`-style move would not
  necessarily.

---

## 2. Localisation trap: the image the README names does not exist on this node

`infera/engine-sglang:gfx950-local` was built on 2026-09-01 on
`crsuse2-m2m-020`/`-188`. **Docker images are per-node on this cluster** —
there is no shared registry pull-through for it. On `crsuse2-m2m-080` it is
absent.

What *is* on `crsuse2-m2m-080` (2026-09-02 ~08:5x UTC), all other tenants':

```
infera/engine-sglang:test-local            (built 7 h ago, ON this node, full infera Dockerfile
                                            build: infera-router, mooncake, libionic)
infera/engine-vllm:test-local
lmsysorg/sglang:v0.5.12-rocm720-mi35x      (note: .12, NOT the .17 the brief names)
naqin/sglang-m3-specforge:v0.5.17-rocm720-mi35x
qiongzhu/tile-skills-sglang:v0.5.17-rocm720-mi35x-profilerfix-*
```

`lmsysorg/sglang:v0.5.17-rocm720-mi35x` — the base the brief says `gfx950-local`
was committed onto — **is also not here**.

Being investigated: whether `infera/engine-sglang:test-local` carries the
`qwen3_5.py` model definition Qwen3.6-27B needs. If it does, `--var image=`
points at it and no build is needed.

**Lesson for the other four modules: `docker images` on YOUR node before you
plan around any image tag. An image built yesterday on another node of the
same type is not available to you.**

---

## 3. `infera/engine-sglang:test-local` is a drop-in for `gfx950-local` — verified, not assumed

Rather than rebuild, I probed the image another tenant had already built on
this node 7 h earlier. All four checks run inside
`docker run --rm --entrypoint bash infera/engine-sglang:test-local -c ...`:

| what | result |
|---|---|
| `sglang.__version__` | `0.5.17` — the version the brief names |
| `srt/models/qwen3_5.py` | **present** (with `qwen3_5_mtp`, `qwen3_5_text`) → Qwen3.6-27B loads |
| `srt/models/glm5*` | **absent** (only `glm4_moe*`, `glm_ocr*`) → GLM-5.3-Flash would still need a build, same as `gfx950-local` |
| `import infera` | `/opt/infera/infera/__init__.py` |
| `python3 -m infera.engine.sglang --help` | **rc=0**, prints the ~20 infera flags |
| `infera-router --help` | present at `/usr/local/bin/infera-router` |

That last pair is the one that actually matters, because the kit's
`start_worker.sh` invokes `python3 -m infera.engine.sglang` and the router as
the `infera-router` binary. **`command -v infera` returns rc=1 — there is no
`infera` console script.** Anyone reasoning "the image has infera" from a
`which infera` will conclude wrongly in both directions.

Two differences from `gfx950-local` that did not bite but could:

- `test-local` is a **Dockerfile build** (infera-router built from Rust,
  mooncake, libionic), not a `docker commit`. It therefore has an
  `ENTRYPOINT ["/usr/local/bin/infera-inject-host-ionic"]`, which copies the
  host libionic provider over the container's and then execs the command. It is
  benign for a mix deployment (no RDMA), but **`docker run` without
  `--entrypoint` will run it**, so a probe that expects a bare shell must pass
  `--entrypoint bash`.
- It is **another tenant's image**. Read it, never `docker rmi` it.

---

## 4. The reusable spur-side run driver

`/shared_nfs/yihou/agent_sys_debug/ws2/run_deploy.sh`, derived from the
2026-09-01 `ws/run_agent_sys.sh` that produced six green runs. **The other four
modules should take this file's four load-bearing blocks rather than
re-deriving them**, because each one cost a debugging round on 2026-09-01:

1. **`HOME` and `PATH`.** `spur exec` gives `HOME=/opt/spur` and a PATH without
   `~/.local/bin`. Without `export HOME=/home/yihou; export
   PATH="$HOME/.local/bin:$PATH"`, `claude` says "Not logged in" and
   `agent-sys` says "the 'claude' backend is not on PATH".
2. **The credential block.** `env_mgr.harness` forwards a name listed in
   `~/.claude/settings.json` **only if that name already has a value in the
   live environment**, and a `spur exec` shell has none. So the driver reads
   the operator's own settings file and exports the `ANTHROPIC_*` /
   `CLAUDE_CODE_*` names into its own process. No value reaches a command line,
   a log or an artefact.
3. **`--var claude_cli="$(command -v claude)"`.** A validation zone has an empty
   `HOME` and a policy-derived `PATH`, so `check_deploy_reproduces` finds no
   binary and fails the kit for it. This is BRIEF known-limit 4.
4. **Run in a standalone clone, not the worktree.** `env_mgr` sets
   `extensions.preciousObjects` on the repo it cuts workspaces from; in a
   worktree that key lands in the **shared** common config and would make
   `git gc` refuse in the main checkout and every other worktree. Sync with
   `git -C /shared_nfs/yihou/agent_sys_debug/repo fetch <worktree> <branch> &&
   git checkout -q FETCH_HEAD`.

Plus one delta forced by the move: **`--package` must now name
`.../llm_e2e_performance_optimization/deploy-demo`**, not the root. Pointing at
the root enumerates all five nested `main.yaml` files into one namespace; that
is exactly what commit `532da57` fixed.

### A bash trap in the driver itself

Site constraints are delivered through the package's `instruction` variable
(the package exposes it precisely so a site need not edit the brief). Writing
that multi-line default as

```bash
: "${INSTRUCTION:=... ${WORK_ROOT} ...}"      # WRONG
```

fails the **whole file** with `unexpected EOF while looking for matching '}'`:
the `}` closing the inner `${WORK_ROOT}` is taken for the outer one. Use a
plain `if [ -z "${INSTRUCTION:-}" ]; then INSTRUCTION="..."; fi`. `bash -n` on
the driver before launching catches it in a second; discovering it through
`spur exec` costs a round trip.

### Steering a shared node through `instruction`, not through edits

The node is shared three ways, so the instruction carries: GPUs 0–1 only, host
ports 8100–8119 only, container names prefixed `dbg_deploy_`, workdir
`/mnt/m2m_nobackup/yihou/deploy`. This is the right seam — the package brief
already tells the agent that every bound identifier is a parameter; the site
only has to say *which* values are its own.

---

## 5. Run `deploy-d1` — launched

```
run root : /shared_nfs/yihou/agent_sys_debug/ws2/runroot/deploy-d1
package  : deploy-demo @ 532da57
model    : Qwen/Qwen3.6-27B <- /shared_nfs/yihou/models/Qwen3.6-27B
image    : infera/engine-sglang:test-local
tp_size  : 1        deploy_mode : mix
host     : crsuse2-m2m-080     started 2026-09-02T08:33:57Z
```

The run root was created fresh, so `runs/` under it holds exactly one stamp and
"my run" needs no `ls -1td` guess (BRIEF known-limit 7).

Load banner confirms the move is fine at run time too:
`loaded 1 task package(s) from .../deploy-demo`, both closures resolved,
`e2e_deploy_standardized: input_validating -> running`.

---

## 6. How to see whether a running agent is working, without touching it

`agent-sys run` prints a phase line and then nothing for tens of minutes. The
zone's `logs/` stays **empty** — an `ai` agent's output goes to the SDK, not to
a file — so `tail`ing the console tells you nothing between phases.

Two live signals, both read-only:

**(a) the agent's own session transcript.** `env_mgr` redirects
`CLAUDE_CONFIG_DIR` into the zone, so the transcript is at

```
<rundir>/zones/task.<root>.0.<h>/task.<leaf>.0.<h>/config/projects/<mangled-cwd>/<session>.jsonl
```

Find it with `find "$ZONE/config" -name '*.jsonl'`. Its last records carry a
`toolStats` block — `{"readCount":4,"searchCount":0,"bashCount":54,...}` — and
a timestamp. Rising `bashCount` with a recent timestamp is proof of life; a
frozen one is the thing to escalate. Note the agent's `cwd` is
`<zone>/package/assets`, **not** the workspace.

**(b) the side effects on the host.** `docker ps | grep dbg_deploy_` and
`find /mnt/m2m_nobackup/yihou/deploy/`. These stay empty for a long while —
measured on this run, **18 minutes elapsed with 54 bash calls and still no
container**, because the agent reads the brief, both exemplars, the packup
skill and the repo before it binds anything. Do not read an empty `docker ps`
at minute 15 as a stall.

The zone `workspace/` is a **checkout of the infera repo**, cut by `env_mgr`;
it is not empty at the start and its mtimes are all the zone-creation instant,
so it is a poor progress signal.

---

## 7. THE BIG ONE — `crsuse2-m2m-080` is CPX-partitioned: 64 × 36 GiB, not 8 × 288 GiB

Run `deploy-d1` was launched with `tp_size=1`, straight from the prior art. At
minute 14 its agent reported, from inside the container:

> HIP enumerates 64 CPX partitions of 36 GiB each — not 288 GiB cards.
> A 52 GB bf16 model cannot fit TP1 in 36 GiB.

I confirmed it first-hand rather than take the agent's word:

```
$ rocm-smi --showcomputepartition   ->  GPU[0], GPU[8], GPU[16], GPU[24], ... : Compute Partition: CPX
$ rocm-smi --showmemorypartition    ->  Memory Partition: NPS1
$ torch.cuda.device_count()         ->  64
$ per-device total_memory           ->  36.0 GiB, every one
```

**The shared BRIEF's "8 × MI355X gfx950, 288 GiB each" is not true on this
node.** Every plan sized against 288 GiB per GPU is sized against a number that
does not exist here. This was raised to the leader for the other four modules
the moment it was confirmed.

### The partition → card mapping, measured

`torch.cuda.get_device_properties(i).uuid` is **identical within each group of
eight**:

| HIP devices | physical card |
|---|---|
| 0–7 | card 0 |
| 8–15 | card 1 |
| 16–23 | card 2 |
| 24–31 | card 3 |

So on a shared CPX node **"you get GPUs 0–1" means HIP devices 0–15**, and the
three-way split of `101078` is really 0–15 / 16–31 / 32–39. Anybody who reads
"GPU 0–1" as `HIP_VISIBLE_DEVICES=0,1` has taken two partitions of one card and
left fourteen of their own idle — and anybody who reads it as physical indices
while setting `HIP_VISIBLE_DEVICES` has stepped on a neighbour.

### The arithmetic, and why the run was restarted

`model.safetensors.index.json` `metadata.total_size` → **55.6 GB** of bf16
weights.

| TP | partitions × 36 GiB | weights/partition | verdict |
|---|---|---|---|
| 1 | 36 | 55.6 GB | **impossible** |
| 2 | 72 | 27.8 GB | fits, ~2.8 GiB left for KV at `mem-fraction-static 0.85` — tight |
| 4 | 144 | 13.9 GB | comfortable, and 0–3 are all on **one physical card** |

`deploy-d1` was killed at minute 25 (`pkill -f "agent-sys run"`; no container
had been bound yet, so nothing leaked) and relaunched as `deploy-d2` with
`TP_SIZE=4`.

**Why kill rather than let it adapt:** the agent brief presents `$E2E_TP_SIZE`
as "Tensor-parallel size to use" — a directive. An agent told TP1 and shown
TP1 is arithmetically impossible may well stop and report impossibility, which
is the correct behaviour and also a wasted hour. The site knew the right
answer; the site should have supplied it.

### What went into the instruction the second time

The site constraints in the driver are now specific rather than nominal —
CPX stated as a measurement, HIP devices `0-15` named as the allocation,
`0,1,2,3` recommended for the TP group so the links stay intra-card, and
`8-15` explicitly reserved **because `check_deploy_reproduces` starts a second
deployment from the kit while the first may still be up**. The agent is also
told to record the CPX finding in `environment.md` and `notes.md`, since a
reproducer on an SPX host would otherwise have no idea why TP4.

---

## 8. Hazard: the standalone clone is shared between all five modules

`deploy-d2`'s banner reads `package : deploy-demo @ 4011eb7`, not `532da57` —
another module had `git checkout`ed the clone at
`/shared_nfs/yihou/agent_sys_debug/repo` between my two launches.

Checked before trusting the run:

```
$ git diff --stat 532da57 4011eb7 -- .../deploy-demo/     # empty
$ git diff --stat 532da57 4011eb7                          # only analyze-demo/
```

**deploy-demo's subtree is byte-identical**, so the run is valid and its
provenance is "tree `4011eb7`, deploy-demo subtree unchanged from `532da57`".

But this is luck, not design. **Five modules sharing one clone that each of
them `git checkout`s is a race**: a checkout mid-run can change the package
under a running agent, and `agent-sys` reads the package at load only for the
spec — the zone's `package/` copy is cut per task, so the window is the load
instant. If a module needs its own commits live, it should cut its own clone
(`git clone --shared` is enough) rather than move the shared one.

---

## 9. Job `101078` was CANCELLED mid-run — the hold is not a given

At 08:56 UTC, ~6 minutes into `deploy-d2`'s agent phase:

```
$ spur exec 101078 bash -lc '...'
Error: exec failed
Caused by: job 101078 is not running (state: CANCELLED)
```

Not cancelled by me. `squeue -u yihou` then showed only `101052`
(`crsuse2-m2m-079`) and `101053` (`crsuse2-m2m-276`) — and both at **~39
minutes elapsed**, not the several hours an "8 h hold started ~08:2x UTC"
would imply. Something is reaping these allocations well short of walltime.

**Consequences and the lesson:**

- `101078` carried **three** modules (deploy, analyze, kernel-opt). One
  cancellation took out three modules at once. Co-tenanting three teams on one
  hold concentrates risk that partitioning the GPUs does not address.
- **Checkpoint into the notes file continuously, not at the end.** Everything
  in sections 1–8 survived the cancellation because it was written as it was
  measured. The run did not.
- **Verify the job is alive before every long `spur exec`,** and treat
  `job ... is not running` as the first hypothesis when a remote command fails
  oddly, rather than debugging the command.
- A run root on `/shared_nfs` survives the node; a workdir on
  `/mnt/m2m_nobackup` does not, since it is per-node local scratch. Nothing of
  value should end a run only there.

Nothing leaked: no container of `deploy-d2` had been bound when the node went
away, and the node itself is gone.

### Restarting on a fresh node — the whole procedure

```bash
# 1. Is the new node CPX or SPX? This decides tp_size and nothing else does.
spur exec <newjob> bash -lc 'rocm-smi --showcomputepartition | head -3'

# 2. Is the image there? Images do not travel between nodes (section 2).
spur exec <newjob> bash -lc 'docker images | grep -Ei "infera|sglang"'

# 3. Run. TP_SIZE=4 on a CPX node, 1 on an SPX node with 288 GiB cards.
spur exec <newjob> bash -lc \
  'cd /shared_nfs/yihou/agent_sys_debug/ws2 && RUN_TAG=deploy-d3 TP_SIZE=4 \
   setsid nohup bash run_deploy.sh > .../runroot/deploy-d3/driver.log 2>&1 &'
```

The driver also hard-codes the HIP device band in its instruction text; on a
node that is not shared three ways, that stanza should be relaxed rather than
copied.

---

## 10. The move did not break the validator BODY either — all four controls reproduce

`agent-sys show` proves a package *loads*. It does not prove a validator body
still runs from its new path: `entry.sh` resolves `check.py` through
`$AGENT_SYS_TASK_PACKAGE`, and `check.py` imports `assets/lib/zone.py`. Both
are relative to a package root that moved.

With no node available (section 9), this was the productive thing to do. The
body was run **offline** against the four control kits the `deploy-demo/README.md`
documents verdicts for. Harness:
`/shared_nfs/yihou/agent_sys_debug/ws2/valcheck/run_check.sh`.

| control | kit | README says | measured after the move |
|---|---|---|---|
| A | the Qwen kit from the run that passed both validators | PASS | **PASS** |
| B | the GLM kit from the run that passed | PASS | **PASS** |
| C | `glm53flash.mix.packup_20260830`, human-written | FAIL, 5 faults | **FAIL, 7 faults** — the original 5 (3 unfilled placeholders, no `Expected output`, 0 `.json` in `results/`) **plus** the two rules added later, `require_mode_readback` and `require_completion_evidence`. The README's "5" predates those rules; 7 is the consistent number |
| D | control A minus `chat_completion.json`, `verification.json`, `worker_mode_line.txt` | FAIL on exactly the 2 new rules | **FAIL on exactly those 2, nothing else** |

So the shape check is still correct *and* still non-vacuous from
`deploy-demo/`.

### Reconstructing a validation zone offline — the recipe

Useful for any module that wants to test a validator without burning a run.
`ScriptBodyRunner` gives the body a cwd containing three files and nothing else:

```
args.json       the validator's `args:` block from the step yaml, verbatim
inputs.json     ["<handoff-id>"]
materials.json  {"<handoff-id>": "<absolute path to the STAGED CONTENT DIR>"}
```

then runs `/bin/sh $AGENT_SYS_TASK_PACKAGE/assets/<name>.validator/entry.sh`
with `AGENT_SYS_TASK_PACKAGE` and `AGENT_SYS_DEMO_PYTHON` set. It writes
`verdict.json` into the cwd.

**The one trap: the staged path IS the content directory.** `zone.content_of`
says so explicitly — "`env_mgr` narrowed `stage` to `content/`, so there is no
`content/` hop below it". Point `materials.json` at
`<handoff>/v1/content`, **not** at `<handoff>/v1`. Getting this wrong yields
`no items/codes directory` and a `false` verdict that looks like a real
failure of the kit.

### The second validator's body was smoke-tested the same way

`check_deploy_reproduces` cannot be exercised offline for real — it brings a
model up. But the question after a directory move is only *does the body still
resolve*, so it was run against control A with `timeout_seconds: 20`:

```
check_deploy_reproduces: ...: FAIL: the reproducer did not finish inside 20s
```

That one line answers it. To print it, the body had to resolve `check.py` and
`assets/lib/zone.py` from `deploy-demo/`, read `materials.json`, find the kit,
find `claude` via the `claude_cli` arg, and actually launch a reproducer — then
hit the deadline I set. No import error, no path error, and specifically **not**
the "no claude binary" failure that was `temp/bugs/2026-09-01-a-validation-zone-
cannot-find-the-claude-cli.md`. Harness: `ws2/valcheck/run_check2.sh`.

A 20-second `FAIL` is the *expected* result of this test, not a fault. Anyone
repeating it should not report it as one.

---

## 11. Partition mode varies BETWEEN nodes — check yours, do not inherit a number

Replacement node `crsuse2-m2m-019` (job `101155`), probed 09:09 UTC:

```
rocm-smi --showcomputepartition  ->  GPU[0]: SPX,  GPU[1]: SPX, ...
```

**SPX**, not CPX. So on `-019` the BRIEF's "8 × MI355X, 288 GiB each" is
correct and Qwen3.6-27B fits at **tp=1** — the exact opposite of `-080`, where
tp=1 is arithmetically impossible.

The rule that follows is short: **`rocm-smi --showcomputepartition | head -3`
is a per-node question and it changes your `tp_size`.** Do not carry my TP4
onto an SPX node, and do not carry the BRIEF's 288 GiB onto a CPX one. Run it
on the node you actually got, every time you get a new one.

### And check the images on the new node too

`-019` carries only `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705`. I
looked inside it: sglang `0.5.14.dev20260705`, and **`qwen3_5.py` is not
there** — the file appears in 0.5.17. Qwen3.6-27B cannot be served from that
image, and it carries no infera either.

So the two things a node must have are independent and both are per-node:

1. an engine image new enough for the model's architecture, **with infera in it**;
2. a partition mode that makes the model fit at the `tp_size` you plan.

`-080` had (1) and failed (2) at tp=1. `-019` has (2) and fails (1).

### Docker images persist on a node across allocations

Worth knowing when a hold dies: the image store is the node's, not the job's.
`infera/engine-sglang:test-local` should still be on `-080` even though job
`101078` was cancelled — which makes `--nodelist crsuse2-m2m-080` on a new hold
far cheaper than rebuilding elsewhere. Requested from the leader on that basis.

---

## 12. Building an infera engine image from scratch — the whole recipe, ~4 minutes

Needed on `-019`, which had no usable image. This is how `gfx950-local` was
made and it is cheap because **infera is pure Python** — no Rust build.

```bash
docker pull lmsysorg/sglang:v0.5.17-rocm720-mi35x        # most layers already present -> fast

docker run -d --name dbg_deploy_build --entrypoint bash \
  -v /shared_nfs/yihou/agent_sys_debug/repo:/src:ro \
  lmsysorg/sglang:v0.5.17-rocm720-mi35x -c "sleep 3600"

docker exec dbg_deploy_build bash -lc "cp -r /src /build && cd /build && pip install --no-cache-dir ."
docker exec dbg_deploy_build bash -lc "rm -rf /build"

docker commit -c 'ENTRYPOINT []' -c 'CMD ["/bin/bash"]' \
  dbg_deploy_build infera/engine-sglang:gfx950-deploy
docker rm -f dbg_deploy_build

docker pull quay.io/coreos/etcd:v3.5.14                  # the package's default etcd_image
```

Four things that cost a round trip each:

1. **`pip install /src` against a read-only bind mount fails.** setuptools
   writes `amd_infera.egg-info` **into the source tree**, so a `:ro` mount gives
   `error: could not create 'amd_infera.egg-info': Read-only file system` and
   then `Failed to build 'file:///src'`. **Copy the tree into the container
   first** (`cp -r /src /build`) and install from the copy. Mounting `:rw`
   would work too but would litter the shared clone that four other modules
   are using.
2. **`infera-router` is a Rust binary and you do not need it.** The kit's
   `start_router.sh` runs `python3 -m infera.server`. A `pip install` image has
   no `/usr/local/bin/infera-router` and that is fine; only the full Dockerfile
   build produces one. Do not go build Rust because `command -v infera-router`
   came back empty.
3. **Commit `ENTRYPOINT []`.** Otherwise the committed image inherits whatever
   the base had, and every `docker run` in the kit has to remember
   `--entrypoint`.
4. **The etcd image is a separate pull.** `etcd_image` defaults to
   `quay.io/coreos/etcd:v3.5.14` and a node that has never run infera will not
   have it. It is small; pull it at the same time as the engine image so the
   agent does not discover the gap mid-bring-up.

Verify before committing, not after:

```
python3 -c "import infera; print(infera.__file__)"     -> /opt/venv/lib/python3.10/site-packages/infera/__init__.py
python3 -m infera.engine.sglang --help | head -1       -> usage: __main__.py ...
python3 -m infera.server --help | head -1              -> usage: __main__.py ...
ls .../sglang/srt/models/qwen3_5.py                    -> present in v0.5.17
```

---

## 13. Run `deploy-d3` — third launch, and the one that had a working node

```
run root : /shared_nfs/yihou/agent_sys_debug/ws2/runroot/deploy-d3
host     : crsuse2-m2m-019  (job 101155, SPX)
image    : infera/engine-sglang:gfx950-deploy   (built here, section 12)
tp_size  : 1        deploy_mode : mix
started  : 2026-09-02T09:14:19Z
```

Run history for the record, since two earlier attempts are part of the story:

| run | node | tp | outcome |
|---|---|---|---|
| `deploy-d1` | `-080` | 1 | killed at 25 min — CPX makes tp=1 impossible (§7) |
| `deploy-d2` | `-080` | 4 | died at 6 min — job `101078` CANCELLED (§9) |
| `deploy-d3` | `-019` | 1 | the run |

The driver's site stanza is now `${GPU_NOTE}`, set by a thin per-node wrapper
(`run_deploy_019.sh`) rather than edited into the driver, because the stanza is
the one part that is genuinely per-node — SPX vs CPX, which devices are yours,
which are the neighbour's.

---

## 14. Two real bugs the agent found in its own kit, both worth knowing about

Recorded because they are traps in *any* multi-tenant bring-up script, not just
this one, and because they are the reason `deploy-d3`'s agent ran three pairs
of deployments rather than one. It was converging, not looping.

**(a) `exec 3>&-` on an unopened fd kills a `set -e` shell.** The kit's port
picker probed a candidate port by opening fd 3, and closed it unconditionally
afterwards. On a **busy** port the open fails, so fd 3 was never opened, so the
close returns non-zero, so `set -e` terminated the picker — meaning the port
picker died the first time it met a port somebody else held. On an empty box it
never fires; on a shared one it fires immediately.

**(b) A free-port check is not a free-port guarantee, and etcd's `/health` will
lie to you.** Run B's picker took port 8102 during the **~50-second gap**
between run A's worker being launched and its engine actually calling `bind()`.
Worse, infera's readiness probe then hit `/health` on 8102 and got a 200 —
**from etcd**, which was already listening there — and concluded the engine was
up. Two independent failures composing into a false "ready".

The agent's fix is the right shape and worth copying: **verify that the port
serves the thing you expect, not merely that something answers.** After the
fix, run B skipped 8100–8103 even though 8102 and 8103 were not yet bound.

The general lesson for every module on a shared node: a `ss -ltn` check tells
you about *this instant*. Between your check and your `bind()` there is a
window, and on this cluster it is tens of seconds wide because engines are slow
to start. Reserve by *creating* something, or re-verify identity after binding.

**Cold start on `-019`, measured:** 443 s from launch to the deployment being
up (15 shards, 55.6 GB, read over NFS) — well above the exemplar's 274 s on
other hardware, and a reminder that the wait in a kit must be sized generously.

---

## 15. Job `101155` CANCELLED too — the second node lost in 80 minutes

At 10:18 UTC, with `deploy-d3` minutes from the validation phase:

```
Error: exec failed
Caused by: job 101155 is not running (state: CANCELLED)
```

Same failure as §9, a different node, 65 minutes after that one. **Two of my
three nodes have now been reaped mid-run, neither at walltime.** Meanwhile
`101052` and `101053` have run 2 h 01 m untouched. Whatever the mechanism, it
is not hitting every hold equally, and any module planning a >40-minute run on
a freshly issued hold should assume it may not survive.

### What survived, and it is most of the value

The kit lives on `/shared_nfs` (that is what `--demo-root` on a network path
buys), so the cancellation took the *run*, not the *artefact*:

```
README.md 83 lines   REPRODUCE.md 200   environment.md 97   notes.md 286
results/  28 non-empty .json across two run subdirectories
scripts/  13, incl. teardown.sh, prove_no_name_theft.sh, snapshot_concurrency.sh
logs/
```

**And it passes `check_deploy_kit`** — run offline with the §10 harness, same
body and same `args.json` the run would have used:

```
check_deploy_kit: ...: packup qwen3.6-27b-mix-sglang-gfx950.packup_20260902
check_deploy_kit: {'...': True}
```

**Be precise about what that is and is not.** It is a real verdict from the
real validator on content the run really produced. It is **not** a run report,
the handoff was never sealed, and `check_deploy_reproduces` never ran. Do not
write this up as "both validators PASS".

Secured immediately to `/shared_nfs/yihou/agent_sys/debugging/deploy/content`
(62 files) before doing anything else, on the principle that survived the last
cancellation: checkpoint the artefact the moment you have it.

### `deploy-d4`, on the third node

`101576` / `crsuse2-m2m-260`, probed before use:

- **SPX** → tp=1;
- **`infera/engine-sglang:test-local` already present** and verified good here
  too (sglang 0.5.17, `qwen3_5.py`, `/opt/infera`, `python3 -m infera.server`
  runs) — note its digest differs from `-080`'s image of the same tag, so
  *verify per node, do not trust the tag*.

No build needed. Launched 10:19:44Z. Its `GPU_NOTE` also hands the agent the
two script traps from §14 up front, so it need not rediscover them.

| run | node | tp | outcome |
|---|---|---|---|
| `deploy-d1` | `-080` | 1 | killed by me at 25 min — CPX makes tp=1 impossible (§7) |
| `deploy-d2` | `-080` | 4 | died at 6 min — job `101078` CANCELLED (§9) |
| `deploy-d3` | `-019` | 1 | died at 64 min — job `101155` CANCELLED; **kit survived and passes the shape check** |
| `deploy-d4` | `-260` | 1 | running |

---

## 16. Job `101576` CANCELLED too — three nodes in two hours, and the staging lesson

`deploy-d4` died at 10:54, 34 minutes into its hold, with the deployment
verified **19/19** and the agent midway through its final clean pair.

```
job 101576 is not running (state: CANCELLED)
```

| job | node | mine for | fate |
|---|---|---|---|
| `101078` | `-080` | ~30 min | CANCELLED |
| `101155` | `-019` | ~70 min | CANCELLED |
| `101576` | `-260` | ~34 min | CANCELLED |
| `101052` | `-079` | (not mine) | still running at 2 h 37 m |
| `101053` | `-276` | (not mine) | still running at 2 h 37 m |

Three of three. The two long-lived holds are the two that were issued in the
original batch; every hold issued *later* was reaped. Whatever the cause, the
operational conclusion for anyone reading this later is: **on this cluster a
freshly issued hold is not a resource you can plan a 60-minute job around.**

### The staging lesson, which is the one that actually cost me the second kit

`deploy-d3` and `deploy-d4` were killed at similar maturity. **One kit survived
and one did not**, and the difference was entirely where the agent staged it:

- **`deploy-d3`** wrote the kit *directly into the handoff directory* —
  `<rundir>/handoffs/<id>/v1/content/items/codes/<packup>/` — which is under
  `--demo-root`, which I had put on `/shared_nfs`. The node died; **62 files
  survived**, and they pass the shape check.
- **`deploy-d4`** built the kit in a staging area on `$E2E_WORK_ROOT`
  (`/mnt/m2m_nobackup`, **per-node local scratch**) intending to copy it in at
  the end. The node died before the copy. What is left in its handoff is a
  single `README.md`, 4 KB. Everything else — the scripts, the 19-check
  verification, the evidence — went with the node.

Both are reasonable things for an agent to do and the brief does not forbid
either. But on a cluster that reaps holds, **the handoff directory is the only
durable place**, and a "stage locally, copy at the end" pattern converts a
survivable interruption into a total loss.

If a later revision of this package wants to be robust here, the brief should
say: *write into the handoff as you go; local scratch is for the container's
logs, not for the deliverable.* Recorded as an observation rather than changed,
since it is a package-content decision and not mine to make unasked.

---

## 17. Outcome

**Delivered:** `/shared_nfs/yihou/agent_sys/debugging/deploy/`

- `content/` — the `deploy-d3` handoff content, 62 files, one packup
  `qwen3.6-27b-mix-sglang-gfx950.packup_20260902`.
- `PROVENANCE.md` — run id, package commit, exact command line, the evidence
  fields I opened by hand, every substitution, and an explicit statement of
  what did **not** happen.

**Status of the two validators, stated exactly:**

| validator | ran | verdict |
|---|---|---|
| `check_deploy_kit` | yes — **offline**, real body, real args, after the node died | **PASS** |
| `check_deploy_reproduces` | **no** | — |

The handoff was never sealed and **there is no run report**. This is not
"both validators PASS" and must not be written up as such.

**Step 1 of the assignment — did the move to `deploy-demo/` break anything —
is answered in full, and negatively:**

- the package loads: 2 tasks, 2 closures, 2 validators per output phase (§1);
- the exec bits survived the move (§1);
- `check_deploy_kit`'s body still resolves and is still non-vacuous — all four
  documented controls reproduce (§10);
- `check_deploy_reproduces`'s body still resolves and reaches its reproducer
  (§10);
- a real run drove the package end to end on two different nodes and produced a
  conforming kit (§13, §15).

Nothing in this module's failures was caused by the move. Every one was the
cluster: a missing image, a partition mode, and three cancelled holds.

---

## 18. `deploy-d5` on `-079`, and the delivery format corrections

Fourth node. `101052` / `crsuse2-m2m-079`, **`amd-primus-qos`** — the class that
has not been reaped today, unlike the three `amd-burst-qos` holds that took
`deploy-d2`, `d3` and `d4` with them. Verified on the node before launching, per
§11's rule:

- **SPX** → tp=1;
- **`infera/engine-sglang:gfx950-local` already present**, built there by
  `profiling`, and verified in place: `qwen3_5.py`, `import infera`,
  `python3 -m infera.server --help`. No build, no pull.
- ports 8100–8119 all free, no containers running.

HIP device 4, with 5–7 left for the reproduction check's second deployment.
Started 10:57:31Z. Run root under `/shared_nfs/yihou/agent_sys/temp/deploy/`
per the user's 2026-09-02 instruction that working files live there.

Its `GPU_NOTE` carries two things the earlier runs had to learn:

- the two script traps from §14, so the agent need not rediscover them;
- **"write the kit directly into the handoff directory as you go; do not stage
  it on local scratch and copy at the end"** — the §16 lesson, stated as an
  instruction rather than left to chance.

### Delivery format — three corrections that arrived after `deploy-d3`

Recorded here because getting any of them wrong produces a fixture that looks
fine and fails at the consumer.

1. **Copy `<hid>/v<N>/`, keeping the `<hid>` level — not `content/` alone.** A
   store wants `<root>/<uuid>/v<N>/manifest.yaml`. Deliver the bare content tree
   and there is no `manifest.yaml`, no `claim/`, no `validation.yaml`, and a
   version without a manifest is unpublished by definition: `store.py` then
   returns `versions() == []`, `kind_of() == ''`, `content_dir() is None`.
2. **Never `chmod -R 777` a delivery.** `handoff/digest.py:82` folds each file's
   **exec bit** into a git-shaped tree digest, and `handoff/store.py:284`
   recomputes it on every consumption. A recursive chmod does not read oddly —
   it raises `DigestMismatch` and fails the consuming task. Use `cp -a`, then
   widen directories only: `find <dir> -type d -exec chmod 777 {} +`. A
   directory enters the digest as a constant, so that is free.
3. **Verify the digest yourself and record it**, because nothing downstream
   will: the digest is checked when a handoff is *produced*
   (`handoff.store.copy_out`, from `agent/gate.py:226`) and **not** when it is
   consumed — input staging uses `env_mgr.fs.layout.copy_out`, a plain
   `shutil.copytree`. So post-seal damage stages silently.

   ```python
   import yaml; from handoff import digest
   m = yaml.safe_load((v / "manifest.yaml").read_text())
   assert digest.tree_digest(str(v / "content")).hex() == m["digest"]["sha256"]
   ```

   Beware NFS when checking across hosts: a mode change can show stale for about
   a minute on one node while another already sees it. Re-stat before believing
   a cross-host mismatch.

An **unsealed** kit — like `deploy-d3`'s — has no manifest, so it must **not**
go into a `store/` layout. It ships as `content/` plus a `PROVENANCE.md` that
says so. If a sealed one arrives later, both are kept and PROVENANCE says which
is which.

---

## 19. The `items_schema` audit — all five packages, no drift

Run while `deploy-d5` was in its agent phase, after `analyze` found that a kind
declaring **six** items with `additionalProperties: false` against a producer
writing **ten** makes a handoff that can never seal — and that
`FilesystemStore.seal`'s refusal is filed under `seal_refused`, which has no
reader outside tests, so the symptom is a clock rather than a reason.

**Result: no drift in any of the five.** Method was to run against **real
sealed content** where it exists and to read the producer where it does not.

| package | evidence | finding |
|---|---|---|
| profiling | **all seven kinds sealed** in its delivered store | six `reproducible` each carry exactly `[command, env, logs, result, watchout]`; `profile_packup` carries `[codes]` |
| kernel-opt | two sealed kinds | `[codes]` each; neither declares an `items_schema` |
| analyze | sealed `analyze_packup` | ten items, matching the widened schema |
| deploy | this module's kit | `[codes]`; `deploy_kit` declares no `items_schema` |
| integration | **no content yet** — producers read | ten kinds, every one matching: 7 × `reproducible` at exactly the five, `kernel_patch` and `integration_packup` at `[codes, watchout]`, `integration_report` at `[report.md, schema, text.json]` |

### The method warning, which is the part worth keeping

**Grepping for `items/<name>` is unsound on these bodies and produced two false
positives** before I caught them by reading the files: `measure.sh` appeared to
omit the required `logs`, and `seed.py` appeared to omit the required
`watchout`. Neither is true. They build item paths through per-arm shell
variables — `A="$OUT_ACCEPT/items"`, `B="$OUT_BENCH/items"`,
`A_ITEMS="$OUT_AIPERF/items"`, `ITEMS="$OUT/items"` — so the literal string
never appears in the source.

If you must audit statically, grep for the **variable assignments** first and
then for `$VAR/`; better, check real content with `handoff.content.check_items`
and only fall back to reading when no content exists.

### The structural point

Four of the five were never at risk because their at-risk kinds declare **no**
`items_schema` at all and fall back to the content type's own rules, where
producer and type agree by construction. `deploy-demo`'s `steps/deploy.yaml`
already argues this explicitly: an `items_schema` on `deploy_kit` "could only
restate `codes`", and a closed one would additionally reject the `logs` and
`watchout` that the `code` type itself lists as optional.

That generalises, and it is the real lesson of `analyze`'s bug: **a closed
`items_schema` that merely restates its content type buys nothing and carries
the whole risk.** `integration-demo`'s ten schemas are all of that shape. They
happen to be correct today; they are ten opportunities to drift tomorrow.

---

## 20. `deploy-d5` — GREEN. Both validators PASS, from the run report

Fifth run, fourth node, and the first to reach a validation phase.

```
     final  e2e_deploy_standardized: succeeded
     final  main: succeeded
   handoff  deploy_kit slot v0: valid
   verdict  check_deploy_kit: PASS          completeness / strong    PASS
   verdict  check_deploy_reproduces: PASS   usability / weak         PASS
      done  run complete; ... 0 validation(s) dropped
```

`crsuse2-m2m-079` (`101052`, **`amd-primus-qos`** — the class that survives),
SPX, tp=1, HIP device 4, image `infera/engine-sglang:gfx950-local` already on
the node. Started 10:57:31Z, sealed 11:44:05Z, `check_deploy_reproduces`
returned 11:59:24Z. **~62 minutes end to end**, the fastest of the five.

`check_deploy_reproduces` really ran: it stood up `dbg_deploy_sgl_20260902-115107`
plus its etcd from the kit alone and tore them down.

### Why this run was faster than the ones before it

Everything the earlier runs discovered was handed to this one up front through
`GPU_NOTE`: the SPX fact, the device band, the two script traps from §14, and
the anti-staging instruction from §16. Its recon took **5 minutes** against
12–15 for `deploy-d1`. The agent acknowledged the staging rule in its own
words — *"write the scripts directly into the handoff (per the site constraint
about host reclamation)"* — and wrote the kit into
`handoffs/<id>/v1/content/` on `/shared_nfs` from the first minute.

**Being told about a trap did not prevent it — it made it diagnosable in one
step.** The agent hit the port race anyway at 11:27 and said so: *"the exact
trap the site warned about, caught live."* `deploy-d3`'s agent needed 09:49 to
10:02 to find the same thing from first principles. That is the realistic value
of writing traps into a site instruction: not immunity, fast attribution.

And it fixed it better than `deploy-d3` did — **real port reservation via a
lock**, plus an engine-identity check in `wait_ready.sh` — then re-ran with
deployment B started deliberately *inside* A's cold-start window to prove it.
`results/port_reservation_incident.json` is that incident with a timeline, root
causes, the fixes, and a stated residual limitation. Its teardown correctly
**refused to declare success** while the thief's etcd still held the port, and
that refusal is kept as evidence.

### Verifying the sealed handoff — what I actually opened

| file | field | value |
|---|---|---|
| `manifest.yaml` | kind / digest | `deploy_kit` / `f4d3a253…`, algorithm `agent_sys.handoff.tree.v1` |
| `validation.yaml` | both entries | `result: true`, timestamps 11:44:05Z and 11:59:24Z — agrees with the printed report |
| `results/chat_completion.json` | `endpoint` | `router /v1/chat/completions (the product path)` |
| | `response.model` | `Qwen/Qwen3.6-27B` — a name, **not** a path |
| | `finish_reason` / `content` | `stop` / 554 chars ending `Paris` |
| `results/verification.json` | | `all_passed: true`; `mode_readback` `["mixed"]` from the worker log (raw `DisaggMode.MIXED`) **and** from the router listing, `agree: true` |
| `results/router_workers.json` | | one worker, `active`, `disagg_mode: mixed` |
| `results/worker_mode_line.txt` | | `disagg=DisaggMode.MIXED` |

14 non-empty `.json` in `results/`, 13 scripts all executable, four documents
totalling 597 lines.

### The digest gotcha

```python
digest.tree_digest(str(path))          # TypeError: can't concat str to bytes
digest.tree_digest(os.fsencode(path))  # correct
```

`tree_digest` takes a **bytes** path — its own docstring says `pathlib.Path`
rejects bytes and mixing the two makes a name comparison silently `False`. Any
circulated verification snippet using `str(...)` will crash rather than report
a mismatch, which at least fails loudly.

Verified after copying, since the digest is checked when a handoff is
**produced** (`handoff.store.copy_out` from `agent/gate.py:226`) and **never**
when it is consumed (input staging is a plain `shutil.copytree`):

```
manifest / recomputed: f4d3a2531a868a72d47f46dde734dcff028c96198e9380fec6cd0fce9358c9c7   MATCH
```

### Delivery

```
/shared_nfs/yihou/agent_sys/debugging/deploy/
  store/d54829ae-52a3-4fa9-9d62-12f2f1cd8929/v1/   claim/ content/ manifest.yaml validation.yaml
  unsealed-deploy-d3/                              the earlier, unsealed kit — kept, not the deliverable
  PROVENANCE.md
```

`cp -a`, then **directories only** widened
(`find store -type d -exec chmod 777 {} +`) — a recursive `chmod` would fold
new exec bits into the digest and raise `DigestMismatch` at the consumer.

Proved it reads as a store rather than assuming:

```
AGENT_SYS_DEMO_STORE=<...>/deploy/store
versions('d54829ae-…')    -> [1]
kind_of('d54829ae-…')     -> 'deploy_kit'
content_dir('d54829ae-…') -> .../v1/content   items -> ['codes']
```

That is exactly the test a bare `content/` tree fails, which is why the
unsealed `deploy-d3` kit is kept **outside** `store/` rather than dressed up to
look like a version it never became.

### Final run history

| run | node | QOS | tp | outcome |
|---|---|---|---|---|
| `deploy-d1` | `-080` | burst | 1 | killed by me at 25 min — CPX makes tp=1 impossible (§7) |
| `deploy-d2` | `-080` | burst | 4 | node CANCELLED at 6 min (§9) |
| `deploy-d3` | `-019` | burst | 1 | node CANCELLED at 64 min; **kit survived, passes the shape check offline** (§15) |
| `deploy-d4` | `-260` | burst | 1 | node CANCELLED at 34 min; kit lost — it was staged on local scratch (§16) |
| `deploy-d5` | `-079` | **primus** | 1 | **GREEN, both validators PASS** |

Four of four `amd-burst-qos` holds died mid-run. The one `amd-primus-qos` node
carried the run to completion. On this cluster that distinction is the single
best predictor of whether a run finishes.

---

## 21. Three rules this module earned, stated as rules

Written as rules rather than anecdotes because each was observed twice, in
different runs, by different agents.

### What ships must be what generated the evidence

Both `deploy-d3` and `deploy-d5` reached a passing kit, noticed they had edited
a script *after* the evidence was produced, and re-ran the whole thing from the
shipped bytes unprompted. `deploy-d5`'s wording: *"The scripts changed after the
run, so I'll re-run the whole thing from the shipped kit to make sure the
evidence comes from exactly these bytes."*

That is correct and it should be the default. A kit whose `results/` were
produced by a slightly different `scripts/` is a kit that documents a
deployment nobody performed — and the divergence is invisible, because every
file is present and every check passes. **If you touch a script after the
evidence, the evidence is stale. Re-run.**

### A written-down trap buys fast attribution, not immunity

`deploy-d5` was handed the port race in its site instruction and **hit it
anyway** — *"the exact trap the site warned about, caught live."* The naive
conclusion is that writing traps down does not work. The measured conclusion is
the opposite: `deploy-d3` spent 09:49–10:02 discovering the same race from
first principles, and `deploy-d5` recognised it on contact and went straight to
a better fix.

**Warnings do not prevent the failure; they collapse the diagnosis from a
bisection to a single step.** That is most of the value, and it is worth saying
plainly so nobody stops writing them down after the first time one is ignored.

The corollary the agent got right on its own: **capture before fixing.**
`results/port_reservation_incident.json` exists because it recorded the
incident *before* repairing it. A fix applied first leaves nothing to hand over.

### A redundant closed `items_schema` is pure downside

From §19's audit, and the sharpest design lesson in this package set. A kind
carrying **no** `items_schema` cannot drift, because producer and content type
agree by construction. The exposure comes entirely from declaring a **narrow
closed** schema over a **rich** producer — which is what stopped
`analyze_packup` sealing.

`deploy-demo`'s own comment on `deploy_kit` generalises it: an `items_schema`
there "could only restate `codes`", and a closed one would additionally reject
the `logs` and `watchout` that the `code` type itself lists as optional. **A
closed schema that merely restates its content type buys nothing and carries
the whole risk.**

A related heuristic worth keeping: **a kind is exposed in proportion to how
many things its producer writes at the top level of `items/`.** One fat item is
safe by construction; a flat spread is where drift hides. That is why
`deploy_kit`, `profile_packup`, `workset`, `kernel_optimization` and
`integration_packup` — all `code`-typed, everything inside `items/codes/` —
were never at risk.
