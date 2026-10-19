# 02a · serve Qwen3.6 with sglang

Level 2 of the [demo ladder](../README.md). An AI agent deploys **Qwen3.6-27B**
on **sglang** behind the **infera** router, proves it answers, and hands back a
**deploy kit**: the scripts that bring the service up, wait for it and tear it
down, plus an `environment.yaml` recording exactly where and how it ran. Three
validators then judge the kit — its layout, its environment record, and whether
it really serves by running it again and putting load on it.

```
main
└─ m1_deploy                 (e2e-flow stage 1)
   └─ deploy_and_prove       AI agent  -> deploy_kit
```

This is e2e-flow's stage 1 on its own. The package writes only `main.yaml`, its
root readme and this README; `steps/`, `shared.yaml` and `assets/<object>` are
links into [`../llm_e2e_performance_optimization/e2e-flow`](../llm_e2e_performance_optimization/e2e-flow/),
whose [`CONTRACT.md`](../llm_e2e_performance_optimization/e2e-flow/CONTRACT.md)
defines the kit. The agent's brief is
`assets/deploy_and_prove.task/readme.md`.

## Prerequisites

- Level 1 runs (`pip install -e "agent_sys[claude]"`, a working `claude`).
- One free GPU. Qwen3.6-27B in bf16 is about 52 GB of weights and fits on one
  MI300X with `tp=1`.
- The weights on local disk, and an image carrying **infera and sglang** (the
  kit runs the infera router): e.g. `infera:sglang-gfx942-*` on MI300X.
- `quay.io/coreos/etcd:v3.5.14` and `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`
  pulled (the router's registry, and the load `check_deploy_serves` applies).

## Run it on one host

Every value below is a fact about *your* machine; nothing has a site default.

```bash
PYTHONPATH=$PWD/agent_sys python3 -m cli.main run \
  --package agent_sys/examples/02a_sglang_serve_qwen36 \
  --demo-root /var/tmp/$USER/l2a \
  --var model_name=Qwen3.6-27B --var model_path=/data/models/Qwen3.6-27B \
  --var image=infera:sglang-gfx942-glm52 \
  --var tp=1 --var gpu_devices=6 --var context_length=32768 \
  --var "parser_args=--reasoning-parser qwen3" \
  --var container=${USER}_l2a \
  --var port_router=18101 --var port_worker=18102 --var port_etcd=18103 \
  --var validate_port_base=18140 \
  --var work_root=/var/tmp/$USER/l2a/work --var scratch_root=/var/tmp/$USER/l2a/scratch \
  --var transport=local --var node=$(hostname) --var node_ip=127.0.0.1 --var jobid=local \
  --var mock_stages=none
```

What the less obvious ones mean:

| var | why |
|---|---|
| `gpu_devices` | the cards to use. On a shared host pick free ones (`rocm-smi --showuse`) |
| `parser_args` | Qwen3.6 thinks before it answers; without a reasoning parser the thinking lands in the answer |
| `transport=local` | run on this host. Without it the flow looks for `spur`/`srun` and drives a cluster node |
| `node`, `node_ip`, `jobid` | the cluster allocation, in cluster mode; on one host any consistent values |
| `mock_stages=none` | run the stage for real (the flow can replay a sealed corpus instead) |
| `container`, `port_*`, `validate_port_base` | names and ports bound on the host — the deployment's, and the validator's own re-deployment's; make them yours |

`show` with the same arguments loads everything and dispatches nothing.

## What a run looked like

Measured on one MI300X (`gpu_devices=6`, `tp=1`, `context_length=32768`): the
agent deployed and wrote the kit in about 33 minutes for $12.05 of model use; then
`check_deploy_kit` and `check_environment` passed, and `check_deploy_serves`
brought the kit up again on its own ports and ran 180 s of load (1024 in / 1024
out, concurrency 16): 616 output tok/s, inter-token latency 23.8 ms, TTFT p50
2.2 s, 112 requests completed.

## What you get

```bash
R=$(ls -d /var/tmp/$USER/l2a/runs/* | tail -1)
ls  $R/handoffs/handoff.deploy_kit.*/v*/content/items/codes/*/            # the kit
cat $R/handoffs/handoff.deploy_kit.*/v*/content/items/codes/environment.yaml
cat $R/handoffs/handoff.deploy_kit.*/v*/validation.yaml                    # three verdicts
```

The kit's `scripts/deploy.sh`, `wait_ready.sh` and `teardown.sh` bring the same
service up again anywhere the environment record allows — which is exactly what
level 2b does before it profiles.

## Next

[`02b_profiling_qwen36`](../02b_profiling_qwen36/) profiles this deployment.
