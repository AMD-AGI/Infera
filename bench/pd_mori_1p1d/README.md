# 1P1D gpt-oss-120b over MoRI, with kvd L3 pairing

Two-node prefill/decode-disaggregated stack for **gpt-oss-120b**, KV transfer
over **MoRI-IO (AINIC RDMA)** through 8 ionic NICs, with an optional **kvd L3
tier on the decode worker**. Throughput-benchmarks the PD path and the
kvd-in-PD pairing.

Adapted from the proven DeepSeek-R1 1P1D mooncake harness (mooncake→mori,
DeepSeek/MLA/EAGLE→gpt-oss/MoE/GQA).

## Topology

| role | node | data-plane IP | GPUs |
|---|---|---|---|
| etcd + router + prefill | `$PREFILL_NODE` | `$PREFILL_IP` | TP=8 |
| decode (+ kvd L3 opt) | `$DECODE_NODE` | `$DECODE_IP` | TP=8 |

Both nodes need: the `rocm/sgl-dev:*-mi35x-*` image (has `import mori`), 8
ACTIVE ionic NICs, a shared NFS mount (weights + this checkout), and a persistent
container `pd_mori_sgl` started with `--network=host --privileged
--device=/dev/infiniband --ulimit memlock=-1`.

## Prereqs (once per node)

```bash
# libionic ABI fix: the sgl-dev image bundles an OLDER ionic rdma provider
# (libionic .149) than the host kernel ionic_rdma driver (.184), so MoRI's
# RdmaManager sees ZERO devices ("availDevices.size() > 0" assert) and the
# engine aborts. Override the container's provider with the host's matched
# build by bind-mounting the host's resolved libionic over BOTH the rdma
# provider path and the libionic.so.1 soname. (MORI_AINIC_ON_AMD.md §1c.)
SRC=$(readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so)
docker run -d --name pd_mori_sgl --network=host --privileged --ipc=host \
    --ulimit memlock=-1 --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video -v /mnt/shared:/mnt/shared -e HF_HUB_OFFLINE=1 \
    -v ${SRC}:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
    -v ${SRC}:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
    rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260601 sleep infinity
docker exec pd_mori_sgl pip install -q msgpack msgspec xxhash nats-py prometheus-client
# verify: must print 8
docker exec pd_mori_sgl ibv_devices | grep -c ionic
```

## Bring up

```bash
# baseline — PD over MoRI, no kvd L3
INFERA_KVD_L3=0 bash up.sh

# kvd L3 on decode (repeated-prefix workloads benefit)
INFERA_KVD_L3=1 bash up.sh
```

`up.sh` starts etcd → router → prefill (`$PREFILL_NODE`) → decode (`$DECODE_NODE`).
Cold start ~8-12 min (weights + cuda-graph capture). Watch:

```bash
ssh "$PREFILL_NODE" 'docker exec pd_mori_sgl tail -f $INFERA_SRC/bench/pd_mori_1p1d/logs/pd_prefill_*'
curl -s "http://$PREFILL_IP:8000/v1/workers" | python3 -m json.tool   # expect 2
```

## Smoke test

```bash
curl -s "http://$PREFILL_IP:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"/PATH/TO/gpt-oss-120b",
       "messages":[{"role":"user","content":"Name three primary colors."}],
       "max_tokens":32,"temperature":0}'
```

## Benchmark

```bash
SWEEP="16 32 64 128 256" ISL=1024 OSL=1024 bash bench.sh
# results → results/<TAG>/*.json  (sglang bench_serving format)
```

Run the sweep twice (`INFERA_KVD_L3=0` then `=1`, re-`up.sh` between) to get
the kvd-in-PD comparison. For the kvd L3 effect, use a workload with repeated
prefixes (warm cache) so D's L3 actually serves — a pure-unique sweep writes
L3 but never reads it.

## Teardown

```bash
bash down.sh
```

## Key knobs (`launch_engine.sh`)

| env | default | meaning |
|---|---|---|
| `ROLE` | — | `prefill` \| `decode` |
| `DATA_PLANE_IP` | — | this node's data-plane IP (MoRI/bootstrap bind) |
| `ETCD_ENDPOINT` | — | `<prefill-ip>:12379` |
| `MODEL_PATH` | gpt-oss-120b | model dir |
| `TP_SIZE` | 8 | tensor-parallel per engine |
| `INFERA_KVD_L3` | 0 | decode-side kvd HiCacheStorage offload |
| `PAGE_SIZE` | 64 | KV page (MoRI chunks by page; 1-token wastes RDMA) |

## Notes / gotchas

- **MoRI wants ALL ionic NICs** (pairs NIC↔NIC by GID subnet) — opposite of
  Mooncake where you drop the flag. `MORI_IB_GID_INDEX=1` (ULA, not link-local).
- **Bind the data-plane IP** (`SGLANG_HOST_IP`/`HOST_IP`) — else
  `get_local_ip_auto()` grabs the public NIC and cross-node bootstrap fails.
- **No MLA/EAGLE flags** — gpt-oss is MoE+GQA; the speculative-decode flags
  from the DeepSeek harness are MLA-specific and would error here.
- **Radix cache stays ON** on decode even with kvd L3 — L3 backs the radix L1.
- PD launch preflight rejects configs prone to silent TCP fallback (`infera/common/disagg_preflight.py`);
  it validates the chosen backend/connector name, but does not probe whether RDMA was actually negotiated.
