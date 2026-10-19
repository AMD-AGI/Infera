# 01 · hello world — a script measures, an AI agent checks

Level 1 of the [demo ladder](../README.md). The smallest package that shows every
moving part of `agent_sys` once:

```
main                  a non-leaf task: no body, only a subgraph
├─ env_facts          PROGRAM agent — collect.py measures this machine
└─ env_check          AI agent      — re-checks each fact with its own commands
```

`env_facts` publishes the **`env_facts`** handoff; `check_env_facts` validates it
before `env_check` may start. `env_check` publishes **`env_report`**;
`check_env_report` judges the agent — every fact checked with a real command,
agreement stated honestly, and `hostname` / `in_container` re-measured by the
validator itself, so a fabricated answer fails.

## The files

| file | what it is |
|---|---|
| `main.yaml` | the root task and its subgraph (`env_facts` → `env_check`) |
| `steps/facts.yaml` | program agent, `env_facts` kind, its validator, the task |
| `steps/check.yaml` | AI agent, `env_report` kind, its validator, the task |
| `assets/env_facts.task/` | `entry.sh` + `collect.py`: the program task's body |
| `assets/env_check.task/readme.md` | the AI task's brief — this *is* the prompt |
| `assets/check_*.validator/` | each validator's `entry.sh` + `check.py` |
| `assets/lib/vio.py` | the validator side of the contract: inputs, args, content, verdict |

A body is found by convention: `assets/<name>.task/` and `assets/<name>.validator/`.

## Prerequisites

- `pip install -e "agent_sys[claude]"` from the repository root.
- A working `claude` CLI with credentials (`claude -p hi` answers). The run
  checks this first and refuses with a clear message if it does not.

## Run it on the host

From the repository root:

```bash
# 1. Load and type-check only; dispatches nothing, takes about a second.
PYTHONPATH=$PWD/agent_sys python3 -m cli.main show \
  --package agent_sys/examples/01_hello_world

# 2. Run it.
PYTHONPATH=$PWD/agent_sys python3 -m cli.main run \
  --package agent_sys/examples/01_hello_world \
  --demo-root /tmp/$USER/hello_world
```

(`agent-sys show|run` is the same CLI once installed; `PYTHONPATH` makes sure the
code of *this* checkout runs.)

Expected end of the output:

```
     final  env_check: succeeded
     final  env_facts: succeeded
     final  main: succeeded
   verdict  check_env_facts: PASS    completeness / strong
   verdict  check_env_report: PASS    trustworthiness / strong
```

**A green exit code is not the result — open the handoffs.** Each run gets a
directory under `--demo-root`; the two handoffs are:

```bash
R=$(ls -d /tmp/$USER/hello_world/runs/* | tail -1)
cat $R/handoffs/handoff.env_facts.*/v0/content/items/text.json    # what the script measured
cat $R/handoffs/handoff.env_report.*/v0/content/items/text.json   # what the agent checked, and how
cat $R/handoffs/handoff.env_report.*/v0/validation.yaml           # the verdict
```

## Run it in docker

The control-plane image carries `agent_sys`, the `claude` CLI and Python; build
it once (needs BuildKit — see [`deploy/docker/agent-sys/README.md`](../../../deploy/docker/agent-sys/README.md)):

```bash
deploy/docker/agent-sys/build.sh --tag infera/agent-sys:latest
```

Then run the package through the CLI's docker mode. It starts a container,
mounts your `~/.claude` read-only for credentials and the run directory at the
same path, and runs the package **from the copy baked into the image** — rebuild
after editing it:

```bash
export INFERA_AGENT_SYSTEM_WORKROOT=/tmp/$USER/agent_sys_runs
PYTHONPATH=$PWD/agent_sys python3 -m cli.main run --docker \
  --docker-image infera/agent-sys:latest \
  --docker-name hello-world-$USER \
  --package agent_sys/examples/01_hello_world \
  --demo-root $INFERA_AGENT_SYSTEM_WORKROOT/hello_world
```

Always pass your own `--docker-name`: a running container with the same name is
reused, and the default name is shared by everyone on the host. To work inside the
container instead: `docker exec -it -w /opt/Infera hello-world-$USER bash`, then
`agent-sys run --package agent_sys/examples/01_hello_world`.

In the report you should now see `"in_container": true` for both the script and
the agent.

**If docker says `error while creating mount source path ... permission denied`**:
the docker daemon cannot read your home directory (typical for an NFS home with
mode 0700). Point `HOME` at a local directory for the launch and give it a copy of
your Claude settings:

```bash
export HOME=/var/tmp/$USER/agent_sys_home
mkdir -p $HOME/.claude && install -m 600 ~/.claude/settings.json $HOME/.claude/
# ... run the docker command above, then remove the copy:
rm $HOME/.claude/settings.json
```

## What to try next

- Break the agent on purpose: change the brief to ask it to report
  `"agrees": true` for everything, and watch `check_env_report` fail.
- Level 2 runs a real model: [`02a_sglang_serve_qwen36`](../02a_sglang_serve_qwen36/).
