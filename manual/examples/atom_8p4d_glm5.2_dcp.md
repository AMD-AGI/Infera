# Example: ATOM 8P4D with DCP — GLM-5.2-MXFP4 on two MI355X nodes

Two **TP4 + DCP4 prefills** and one **TP4 + DCP4 decode** on two 8×MI355X
nodes. The head node runs the [4P4D deployment](atom_4p4d_glm5.2_dcp.md)
unchanged (prefill0 on GPUs 0-3, decode on GPUs 4-7, etcd, router); the second
node runs prefill1 on GPUs 0-3. The runnable kit is
[`examples/atom_8p4d_glm5.2_dcp/`](https://github.com/AMD-AGI/Infera/tree/main/examples/atom_8p4d_glm5.2_dcp);
it reuses the 4P4D kit's scripts and starts prefill1 with the same `worker.sh`
through `DOCKER_HOST=ssh://<second node>`.

## 1. Prepare both nodes

- Build the image as in the [4P4D example](atom_4p4d_glm5.2_dcp.md), then
  `docker save | docker load` it to the second node: the image ID must match.
- Passwordless ssh from the head node to the second node.
- The kits, the weights and the scratch directory at the same paths on both
  nodes.

## 2. Fill in `env.sh` and run

In `examples/atom_8p4d_glm5.2_dcp/env.sh` set `NODE_IP` (head node),
`PREFILL1_HOST` (ssh destination of the second node), `PREFILL1_IP`, `MODEL`,
`IMAGE` and `WORK_DIR`. Then, on the head node:

```bash
cd examples/atom_8p4d_glm5.2_dcp
bash engine/up.sh       # router must list 2 prefill / 1 decode
bash engine/smoke.sh
bash engine/down.sh     # removes the containers on both nodes
```

Smoke and GSM8K (0.985) pass on two nodes; the AgentX points below are not
measured yet on this image (forced MTP acceptance, performance only):

```bash
CONC=96 MTP_AL=2.99 bash engine/up.sh
bash engine/agentx.sh   # throughput per GPU over 12 GPUs
bash engine/down.sh
```

## 3. Limits

- One prefill per node: the LMCache CPU tier is GTT memory with a machine-wide
  cap of about 1376 GiB, which fits one 4 × 256 GiB pool.
- Cross-node KV moves over the same-numbered ionic rails as on one node.
- One decode serves both prefills; its KV capacity is the likely limit at C96.

The kit README lists the gotchas.
