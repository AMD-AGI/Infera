# agent_sys control-plane image

This image runs `agent_sys` tasks.  It contains:

- `agent_sys[claude,dev]`
- Claude Code CLI and `claude-agent-sdk`
- Python 3.12 in `/opt/venv/agent`
- git, ssh, rsync, uv and the Docker client

It does **not** contain inference engines, model weights, or task-specific
dependencies (Magpie, compilers, aiperf, etc.).  Those are declared in each
task package's environment recipe and installed by `env_mgr` at runtime.

## Build

No SSH keys or credentials are needed at build time.

Two requirements that are easy to trip over:

**BuildKit is required.**  The Dockerfile uses `# syntax=`, `RUN --mount=type=cache`
and `COPY --chmod`, none of which the legacy builder understands.  Docker ships
BuildKit through the `buildx` CLI plugin, and a host can have a modern daemon
without it — `docker buildx version` answering `unknown command` is the symptom,
and `build.sh` then fails with *"BuildKit is enabled but the buildx component is
missing or broken"*.  You do not need to change anything system-wide to get one:
drop the plugin into a config directory you own and point `DOCKER_CONFIG` at it.

```bash
mkdir -p ~/my-docker-config/cli-plugins
curl -fsSL -o ~/my-docker-config/cli-plugins/docker-buildx \
  https://github.com/docker/buildx/releases/download/v0.20.1/buildx-v0.20.1.linux-amd64
chmod +x ~/my-docker-config/cli-plugins/docker-buildx
DOCKER_CONFIG=~/my-docker-config deploy/docker/agent-sys/build.sh
```

**Build from the main checkout, not from a `git worktree`.**  A worktree's `.git`
is a one-line pointer file naming a path on your machine; `COPY` carries the
pointer and not what it points at, so inside the image it names nothing.  The
build stops early and says so.  `.git` itself is deliberately *not* ignored —
`env_mgr/workspace.py` runs `git clone --shared` against `/opt/Infera`.

```bash
deploy/docker/agent-sys/build.sh
deploy/docker/agent-sys/build.sh --tag infera/agent-sys:v1
```

Or directly:

```bash
docker build -f deploy/docker/Dockerfile.agent-sys -t infera/agent-sys:latest .
```

## Usage

### Standard: start the container, work inside

```bash
# Start the container
docker run -d --name agent-sys \
  --user "$(id -u):$(id -g)" \
  --network host \
  -e HOME=/home/agent \
  -v "$HOME/.ssh:/home/agent/.ssh:ro" \
  -v "$HOME/.claude:/home/agent/.claude" \
  -e INFERA_AGENT_SYSTEM_WORKROOT="$HOME/.agent_sys_runs" \
  -v "$HOME/.agent_sys_runs:$HOME/.agent_sys_runs" \
  infera/agent-sys:latest \
  sleep infinity

# Work inside
docker exec -it agent-sys bash

# Or pass commands through
docker exec agent-sys agent-sys show --package /opt/Infera/agent_sys/examples/demo
```

A convenience script is provided at the repository root:

```bash
./run_agent_sys_container.sh
```

### Automated: env_mgr --docker

If `agent_sys` is installed on the host, `env_mgr` can manage the container
lifecycle automatically:

```bash
agent-sys run --docker --package /path/to/task
```

**The o11y panel runs inside the container, and `--docker` does not start one on
the host.**  A panel can only ingest transcripts it can see, and under `--docker`
the run writes them into the *container's* prefix, which is not mounted on the
host.  `--agentsview-port` and `--no-agentsview` are forwarded, so both answers
still reach it.

Before that was fixed, the failure was silent and it was worse than "no panel":
the host daemon took the port, the container's skipped saying *"port N is in use
by something else"* — us — and the transcripts sat unread.  They are not lost:
**a later run's panel ingests them under the directory-name fallback**, so a
project appears named `task.directions.e42df9d9_…` instead of its run.  The
symptom is therefore *mislabelled history*, discovered long afterwards by
whoever wonders where that project name came from.

If you reproduce any of this by hand, note that `ensure_running` launches
`agentsview serve --background --replace`, and `--replace` replaces the daemon
**for that data directory** — not for that port.  Starting a panel on any port
against the default prefix will stop a colleague's daemon that is sharing it.
Point `AGENT_SYS_HOME` at a prefix of your own first.

Options (all default to **on**):

| Flag | Effect |
|------|--------|
| `--detect-and-copy-host-ssh-config` | Mount host `~/.ssh` into container (read-only) |
| `--detect-and-copy-host-claude-config` | Mount host `~/.claude` into container |
| `--no-detect-and-copy-host-ssh-config` | Skip SSH config detection |
| `--no-detect-and-copy-host-claude-config` | Skip Claude config detection |

## Container requirements

- A writable `HOME` (`-e HOME=/home/agent`)
- The caller's UID/GID when a shared filesystem is used
- A run root mounted at the same absolute path on every host that reads it

Credentials (SSH keys, Claude config) must **not** be stored in the image.
Mount them at runtime.

## Smoke tests

```bash
docker exec agent-sys python3 -m pytest -q /opt/Infera/agent_sys

docker exec agent-sys \
  agent-sys run --dry-run \
    --package /opt/Infera/agent_sys/examples/demo
```

## Exit codes

| Code | Meaning                                          |
| ---: | ------------------------------------------------ |
|  `0` | Run completed according to the package contract  |
|  `1` | Load error                                       |
|  `2` | Precondition failure                             |
|  `3` | An expected failure did not occur                |
|  `4` | A declared expectation was not reached           |
|  `5` | The task graph did not complete                  |
