# Example: ATOM 4P4D with DCP — GLM-5.2-MXFP4 on one MI355X node

A PD-disaggregated GLM-5.2-MXFP4 deployment on a single 8×MI355X node: one
**TP4 + DCP4 prefill** (GPUs 0-3) with an LMCache CPU KV tier, one **TP4 + DCP4
decode** (GPUs 4-7), KV moved by Mooncake over the ionic RDMA rails, and the
Infera Python router (round-robin with `--session-affinity prefill`). The runnable kit is
[`examples/atom_4p4d_glm5.2_dcp/`](https://github.com/AMD-AGI/Infera/tree/main/examples/atom_4p4d_glm5.2_dcp);
its README carries the full recipe, reference numbers and gotchas.

For the concepts, read [PD disaggregation](../features/pd_disaggregation.md) first.

## 1. Build the image

The kit needs the ATOM nightly `661ef766` base plus an ATOM DCP fix and an
LMCache fix, enabled by one opt-in build argument of `Dockerfile.atom`. From the
repo root:

```bash
docker build -f deploy/docker/Dockerfile.atom \
  --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:nightly_202610131505 \
  --build-arg BUILD_MOONCAKE=0 \
  --build-arg ATOM_NIGHTLY=1 \
  -t <infera-atom-image> .
```

`ATOM_NIGHTLY=1` applies `deploy/docker/patches/atom_nightly/`: the ATOM sparse
DCP padded-row fix (without it the prefill hits a `Memory access fault` under
long-prefix load), the LMCache ROCm pinned-pool fix (the pool becomes GTT, kit
README gotcha 4), an AITER kernel prebuild and a `protobuf<7` pin.
`BUILD_MOONCAKE=0` keeps the base's Mooncake 0.3.13; the kit sets its
`MC_DISABLE_HIP=1` so same-host P/D on disjoint GPUs goes over RDMA.

## 2. Fill in `env.sh`

`env.sh`, at the kit root, is the only file to edit; the scripts live in
`engine/`. Replace the placeholders:

| variable | value |
|---|---|
| `NODE_IP` | this node's data-plane IP |
| `MODEL` | GLM-5.2-MXFP4 weights directory |
| `IMAGE` | the image from step 1 |
| `WORK_DIR` | host scratch directory for runs and caches |

Check `MOONCAKE_DISABLE_HIP_DMABUF` for your driver (`1` = peer-memory,
validated at mem util 0.90) and that the ports in section 3 are free.

## 3. Bring it up and check it

```bash
cd examples/atom_4p4d_glm5.2_dcp
bash engine/up.sh       # etcd -> prefill + decode -> router; ~10 min cold start
bash engine/smoke.sh    # PD registration, completion, chat, stream, 66k-token prefix reuse
bash engine/eval.sh     # GSM8K, 5-shot, 200 questions
```

`engine/up.sh` fails unless the router lists exactly one prefill and one decode
worker. Clients use `http://<NODE_IP>:18000` with the OpenAI API and the
resolved weights path as the model name.

## 4. Benchmark

AgentX reference points start the engines with forced MTP acceptance, which
gives wrong text and is for performance runs only:

```bash
bash engine/down.sh
CONC=48 MTP_AL=2.99 bash engine/up.sh
bash engine/agentx/agentx.sh   # InferenceX 918524ff, 10 warm-up requests per lane, 3600 s
```

With this image, CONC 48 reached 26325 total tokens/s/GPU with a P90
interactivity of 66.52 tokens/s/user, and CONC 64 30821 with 56.83. The kit
README lists both, and C24 to C80 on the previous image.

## 5. Tear down

```bash
bash engine/down.sh     # saves logs into the run directory, waits for VRAM to drain
```

For two prefills and one decode on two nodes, see
[ATOM 8P4D DCP](atom_8p4d_glm5.2_dcp.md).
