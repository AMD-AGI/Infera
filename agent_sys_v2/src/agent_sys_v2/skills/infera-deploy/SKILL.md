---
name: infera-deploy
description: Bring an Infera kit (examples/<kit>) up or down on the task's machines, audit the configuration that is really live, and guard the run
---

# infera-deploy

A **kit** is `repos/Infera/examples/<kit>/`: one site file plus `up / smoke / eval / bench|agentx / down` bash
scripts, all driven by environment variables (`CONC=24 MTP_AL=3.33 bash engine/up.sh` overrides one run).
`examples/atom_4p4d_glm5.2` is the model to follow; the SGLang kits (`sglang_1p1d_glm5.2`, ...) keep one long-lived
container per node and start each leg with `docker exec` (`engine/leg.sh`), which makes restarting one leg cheap.

## Bring it up

1. Only on the machines this task was given (`$AGENT_SYS_MACHINES`). Before anything, run the checks of
   `infera-hygiene` (free VRAM, no foreign containers on the GPUs, free ports).
2. Read the kit's `README.md` and `env.sh` (or the site file). Override with environment variables or the escape
   hatches, never by editing the kit: `PREFILL_EXTRA_ARGS`, `DECODE_EXTRA_ARGS`, `ROUTER_EXTRA_ARGS` are appended to
   the command lines; `PREFILL_EXTRA_ENV`, `DECODE_EXTRA_ENV` to the containers' environment (ATOM kit; SGLang kits:
   `EXTRA_ENGINE_ARGS`).
3. `bash engine/up.sh` with `RUN_ID=<label>` so the run directory is named after the experiment. The ATOM kit writes
   `$WORK_DIR/runs/<RUN_ID>/engine-config.env` (every setting used) and `workers.json`; smoke, eval, agentx and down
   write into the same directory. Keep that path: it is the evidence of the result.
4. A cold start takes about 10 minutes (weights, AITER/FlyDSL JIT, graph capture). `Health check failed` repeated
   during it is not a hang. Wait with a background command that exits when `/health` answers or a container dies
   (`wait_health` in `engine/common.sh`), not with a sleep loop.

## Audit what is really live (do not trust the scripts)

SGLang kit wrappers forward only the variables on their list; the rest are silently ignored (e.g. glm5.2's `up.sh`
drops `MAX_RUNNING CUDA_GRAPH_BS HICACHE_GB ...`). After every bring-up write `effective-config.json` next to the
run directory from:

- `docker inspect <ctr>` (env and command), and `ps -o args` of the engine process inside the container;
- `curl http://<worker>/get_server_info` (SGLang: the arguments in effect); `curl http://<router>/v1/workers`;
- the engine log head (parallel sizes, graph sizes, MTP, attention backend).

Diff it against what you intended: an unexpected difference voids the measurement. Also check that the KV transfer
is RDMA (`installTransport, type=rdma` in the logs, no `MC_FORCE_TCP`, `PORT_ACTIVE` > 0) and that DP attention is
the same on both legs (a mismatch garbles output silently).

## Guard it

Long runs fail quietly. Start a guard next to the deployment (in the remote chain, not only in the agent's session):
stop the client when the engine log shows `Fatal Python error`, `crashed with exit code` or an HSA error, or when no
request completed for 900 s; snapshot `/metrics` before teardown.

## Take it down

`bash engine/down.sh` saves `logs/<ctr>.log` and `<ctr>.inspect.json`, removes the containers and waits until
every GPU holds < 4 GiB VRAM (`wait_vram_free`; 15-40 minutes, longer with HiCache). Never `docker rm` a container
this task did not create. A task that ends with its deployment still up says so in progress.md.
