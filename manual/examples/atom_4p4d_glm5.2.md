# Example: ATOM 4P4D — GLM-5.2-MXFP4 on one MI355X node

A PD-disaggregated GLM-5.2-MXFP4 deployment on a single 8×MI355X node: one
**TP1 × PP4 prefill** (GPUs 0-3, ATOM's GLM-5.2 AgentX recipe) with a Mooncake
Store CPU KV tier, one **TP4 + DCP4 decode** (GPUs 4-7), KV moved by Mooncake
over the ionic RDMA rails, and the Infera Rust router (round-robin with
`--session-affinity prefill`). `PREFILL_PP=1` runs a TP4 + DCP4 prefill instead,
with the LMCache tier: ATOM's Store offload does not support DCP. The runnable
kit is
[`examples/atom_4p4d_glm5.2/`](https://github.com/AMD-AGI/Infera/tree/main/examples/atom_4p4d_glm5.2);
its README carries the full recipe, reference numbers and gotchas.

For the concepts, read [PD disaggregation](../features/pd_disaggregation.md) first.

## 1. Build the image

The kit needs the ATOM nightly `661ef766` base plus Infera's ATOM, aiter and
LMCache patches for it, enabled by one opt-in build argument of
`Dockerfile.atom`. From the repo root:

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
README gotcha 4), the decode patches (ATOM API-process GC, DCP exchanges on
aiter's custom IPC kernels, tuned MLA bmm configs and FP8 GEMM rows), an AITER
kernel prebuild and a `protobuf<7` pin.
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

Check `MOONCAKE_DISABLE_HIP_DMABUF` for your driver (`1` = peer-memory) and
that the ports in section 3 are free. The Store needs THP (`madvise` or
`always`) and `STORE_OWNER_GB` (768 GiB) free on each of the two NUMA nodes;
on smaller hosts, or hosts where its startup probe fails, set
`PREFILL_CPU_TIER=lmcache` (kit README gotcha 14).

## 3. Bring it up and check it

```bash
cd examples/atom_4p4d_glm5.2
bash engine/up.sh       # etcd -> Store -> prefill + decode -> router; ~10 min cold start
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
CONC=96 MTP_AL=2.99 SESSION_AFFINITY=off ROUTER_EXTRA_ARGS="--breaker-failure-threshold 0" \
  PREFILL_EXTRA_ARGS="--spec-decode-acceptance-length 2.99" bash engine/up.sh
bash engine/agentx/agentx.sh   # InferenceX 918524ff, 10 warm-up requests per lane, 3600 s
```

With the kit's defaults, C56, C72, C96 and C112 reached 32925, 40075, 46234 and
48154 input tokens/s/GPU (InferenceX CI's AgentX client). The kit README lists
them, the LMCache tier, and the PP and DCP prefills under the previous
defaults.

## 5. Tear down

```bash
bash engine/down.sh     # saves logs into the run directory, waits for VRAM to drain
```
