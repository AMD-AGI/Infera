# GLM-5.2-MXFP4 — ATOM 8P4D with DCP on two MI355X nodes

Two prefills and one decode, each **TP4 + DCP4**, on two 8×MI355X (gfx950)
nodes: the [4P4D kit](../atom_4p4d_glm5.2_dcp/) on the head node plus a second
prefill on another node. This is a thin kit: the recipe, smoke, GSM8K and AgentX
scripts are the 4P4D kit's, run with this kit's `env.sh`. Read the 4P4D README
first; its recipe, image and gotchas all apply.

| node | runs | GPUs |
|---|---|---|
| head (`NODE_IP`) | etcd, router, prefill0, decode, AgentX / GSM8K clients | prefill0 0-3, decode 4-7 |
| second (`PREFILL1_HOST`, `PREFILL1_IP`) | prefill1 | 0-3 (`PREFILL1_GPUS`); 4-7 idle |

Both prefills keep the 4P4D settings (256 GiB LMCache CPU tier per rank) and
register with the head node's etcd; the router (round-robin with
`--session-affinity prefill`) keeps each agent session on one prefill. KV goes
from either prefill to the decode by Mooncake over the ionic RDMA rails.

## Run

Everything runs on the head node. Fill in [`env.sh`](env.sh) (head node IP,
the second node's ssh destination and IP, weights, image, scratch dir), then:

```bash
bash engine/up.sh        # etcd -> prefill0 + decode here, prefill1 there -> router (2 prefill / 1 decode)
bash engine/smoke.sh     # 4P4D smoke checks through the router
bash engine/eval.sh      # GSM8K, 5-shot, 200 questions
bash engine/down.sh      # both nodes: save logs, remove containers, wait for VRAM
```

Benchmark points, cold start each (forced MTP acceptance, performance only):

```bash
CONC=84 MTP_AL=2.99 bash engine/up.sh
bash engine/agentx.sh    # DURATION=3600 WARMUP_PER_LANE=10 by default
bash engine/down.sh
CONC=96 MTP_AL=2.99 bash engine/up.sh
bash engine/agentx.sh
bash engine/down.sh
```

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit**; sources the 4P4D `env.sh` for everything else |
| [`engine/prefill1.sh`](engine/prefill1.sh) | `check`, `up` or `down` for prefill1: the 4P4D `worker.sh` / `down.sh` with `DOCKER_HOST=ssh://$PREFILL1_HOST` |
| `engine/up.sh`, `down.sh`, `smoke.sh`, `eval.sh`, `agentx.sh` | run the 4P4D script of the same name with this `env.sh` |

The 4P4D `up.sh` and `down.sh` start and remove prefill1 through
`REMOTE_PREFILLS`; `agentx.sh` reads the worker list from the router, so
`PREFILL_NUM_WORKERS=2` and every worker's `/metrics` go to AgentX. Containers
are `<CTR>-*` on the head node and `<CTR>-p1-prefill` on the second node; logs
of both land in the run directory under `WORK_DIR`.

## Requirements

- The head node logs in to the second node by ssh without a password, and
  `DOCKER_HOST=ssh://<PREFILL1_HOST> docker ps` works from the head node.
- Both kits, `MODEL` and `WORK_DIR` have the same paths on both nodes (a shared
  file system). Do not share `WORK_DIR` with a running 4P4D kit: the current
  run is recorded there.
- The same image ID on both nodes: build once, `docker save | docker load` to
  the other node. `up.sh` refuses to start otherwise.
- The ports of the 4P4D README, and Mooncake's RPC ports 15000-17000, reachable
  in both directions between the nodes.

## Validation status

Correctness was checked on two nodes, with real MTP acceptance, on a build of
the same base without the ATOM DCP fix of the 4P4D README.

| item | status |
|---|---|
| correctness | **passed**: router 2 prefill / 1 decode, smoke 6/6, both prefills served requests, cross-node KV over RDMA |
| GSM8K | **0.985** |
| AgentX C84, C96 | **not measured yet**; the commands above are the suggested points |

## Notes and gotchas

**1. Throughput per GPU is over 12 GPUs**, 2 × 4 prefill + 1 × 4 decode. The
second node's GPUs 4-7 are idle and not counted.

**2. One prefill per node.** The LMCache CPU tier is GTT memory (4P4D gotcha 4),
which has a machine-wide cap of about 1376 GiB on the reference nodes; 4 ×
256 GiB fits once. Two prefills on one node would need `PREFILL_OFFLOAD_GB=128`
or less and halve each CPU tier.

**3. Cross-node KV uses the same rails.** prefill1 writes to the decode over
the same-numbered ionic rails as prefill0 (`ATOM_MOONCAKE_MATCHED_RAILS=auto`);
the two nodes' rails must be on the same fabric. prefill0 also goes over RDMA
(`MC_DISABLE_HIP=1`), so both paths use the same transport.

**4. Decode KV is the likely limit at C96.** One decode serves both prefills;
at C48 in 4P4D its KV use already reached about 60% at p99. Watch the decode log
for `GPU KV cache usage`, preemption and `PD backpressure`.

**5. Session affinity balances sessions, not tokens.** A long agent session
stays on one prefill, so the two prefills can be uneven for a while. smoke's
long-prompt pair carries one session id, so the repeat hits the prefill that
cached it.

**6. VRAM waits only cover the kit's GPUs.** `down.sh` and `up.sh` wait on GPUs
0-7 of the head node and `PREFILL1_GPUS` of the second node. JIT cache volumes
are named by `CTR`, role and image, so the first 8P4D cold start recompiles the
JIT kernels on both nodes.

## Source

- Base kit: [`atom_4p4d_glm5.2_dcp`](../atom_4p4d_glm5.2_dcp/)
- Walkthrough: [`manual/examples/atom_8p4d_glm5.2_dcp.md`](../../manual/examples/atom_8p4d_glm5.2_dcp.md)
