# Results & diagnostic evidence

## Cross-node PD smoke — after HIP-gate fix (image = patched mooncake)

### mooncake 1P1D (chi2879 prefill → chi2865 decode)
```
workers: prefill 10.2.122.10:30000 active ; decode 10.2.122.52:30000 active
REPLY: Paris
REPLY: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

### mori 1P1D (same nodes; all 8 ionic NICs)
```
workers: prefill 10.2.122.10:30000 active ; decode 10.2.122.52:30000 active
REPLY: Paris
REPLY: Okay! Here we go: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10!
```

## Differential evidence — mooncake WITHOUT the fix (stock base mooncake)

Same nodes, same model, same kit; only the image's mooncake differs.

Prefill (chi2879):
```
E hip_transport.cpp:70 HipTransport: hipIpcOpenMemHandle failed (Error code: 17 - invalid device pointer)
[TP0] Session 10.2.122.52:15577 failed.
[TP0] Prefill transfer failed ... KVTransferError: Failed to send kv chunk ... to 10.2.122.52
```
Decode (chi2865):
```
[TP0] Decode transfer failed ... KVTransferError: Failed to get kvcache from prefill instance, it might be dead
POST /v1/chat/completions -> 500 Internal Server Error
```
Router: `WARNING infera.router.disagg: prefill worker returned 500 (decode may fail)`

→ single variable (HIP-gate on/off) flips the result. Causal link closed.

## Root-cause evidence

Base mooncake commit (the regression):
```
$ cd /sgl-workspace/Mooncake && git log -1 --oneline
01d1eb2a [TE] Support rdma+hip multi-protocol segments for single-node disaggregation (#2682)
```

HIP transport presence across images (`strings engine*.so | grep -c 'HIP transport installed'`):
```
infera 20260711 (old sglang)     -> 0   (no HIP transport; cross-node PD worked)
infera sglang-v0.1.0-rc6         -> 0
lmsysorg/sglang:v0.5.15.post1    -> 1   (new: unconditional HIP transport)
```

Transport selection priority (`multi_transport.cpp:468`): `hip=4 > rdma=2` — the
dual-registered `rdma,hip` KV pool always picks hip, even cross-node.

## Build validation (the fix, end-to-end in `docker build`)

```
docker build -f deploy/docker/Dockerfile.sglang -t infera/engine-sglang:pd-final .
  [mc-hip-gate] patched .../transfer_engine_impl.cpp
  [mc-hip-gate] cmake configure ; ninja build (engine module)
  [43/43] Linking CXX shared module .../engine.cpython-310-x86_64-linux-gnu.so
  [mc-hip-gate] DONE — HIP transport gated
  ... pip install amd-infera ... naming to infera/engine-sglang:pd-final done
post-build verify: engine.so gate count = 1 ; `from mooncake.engine import TransferEngine` OK
```
