# Launch recipes per mode (validated on spur crsuse2-m2m)

Concrete env + flags for each mode, grafted onto a native `sglang.launch_server`
PD leg. All validated on `crsuse2-m2m` (8×MI355X gfx950, DSv4-Pro). Run **inside
the engine container**, one leg per role. Env keys not shown here (the R4 perf env:
`SGLANG_USE_AITER=1 … SGLANG_OPT_USE_FUSED_COMPRESS=true …`, `--attention-backend
dsv4`, etc.) come from the project's PD launch script — this doc covers only the
**mode-specific KV-registration** knobs.

## Shared (every mode)

```bash
export MC_DISABLE_HIP_TRANSPORT=1        # cross-node PD MUST stay RDMA
unset  MC_ENABLE_HIP_TRANSPORT           # never on for cross-node (and not for 2-process single-node)
export NCCL_IB_DISABLE=1                  # keep collectives off the KV rail
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
# pin host IP + gloo to the data-plane netdev of the chosen KV NIC:
export SGLANG_HOST_IP=$MY_IP HOST_IP=$MY_IP SGLANG_LOCAL_IP_NIC=$NETDEV GLOO_SOCKET_IFNAME=$NETDEV
```

`--disaggregation-transfer-backend mooncake` on every leg; prefill adds
`--disaggregation-bootstrap-port <p>`.

## Mode A — bare ibv_reg_mr + peer-mem (stock image)

Peer-mem present → register GPU pages directly, no pin, **all rails usable** (let
mooncake auto-discover per-GPU affine NICs; do **not** pass `--disaggregation-ib-device`).

```bash
export MOONCAKE_DISABLE_HIP_DMABUF=1     # force the bare ibv_reg_mr path
export MC_GID_INDEX=<routable RoCEv2 idx> # from the probe (e.g. 1)
# no MC_MS_FILTERS, no --disaggregation-ib-device: every rail carries KV
```

This is the in-repo default (`infera/engine/rocm_rdma_env.py` set-defaults
`MOONCAKE_DISABLE_HIP_DMABUF=1`).

## Mode B — ibv_reg_dmabuf_mr on the ODP NIC (dma-buf image)

No peer-mem, one ODP NIC (spur: `mlx5_0`, GID idx 3). Force **all** KV onto it so
non-ODP rails never pin. Needs the dma-buf rebuild.

```bash
export MOONCAKE_DISABLE_HIP_DMABUF=0     # allow the dma-buf registration path
export MC_MS_AUTO_DISC=0 MC_MS_FILTERS=mlx5_0
export MC_GID_INDEX=3
export RDMAV_FORK_SAFE=1                  # ionic rails on the box need it; harmless for mlx5
# launch flag:
#   --disaggregation-ib-device mlx5_0
```

Validated: 2-node TP8 1P1D, KV **not** doubled (decode steady 237/288 GiB = weights
+ 1×KV), all traffic on mlx5, zero ionic. **Perf note:** all KV rides one 200G mlx5
instead of eight 400G ionic — accepted bandwidth tradeoff for no-pin.

## Mode C — cap-KV (STUB)

No peer-mem, no ODP. dma-buf would pin+double; bare would EFAULT. Left for a
follow-up: run dma-buf with a KV pool small enough that weights + 2×KV-pin fit, or
the driver-bug workaround. Do not launch on the stub — surface the blocker.

## Single-node PD (same box, two P/D processes)

Even same-node, KV goes over the RDMA NIC in **loopback** — do **not** enable HIP
transport. `hipIpcOpenMemHandle` can't open another *process's* GPU handle, so
`MC_ENABLE_HIP_TRANSPORT=1` gives warmup-OK but every real request 500s
(`Requested address … not found`).

- **mlx5 loopback** (Mode B, `NIC_DEV=mlx5_0 GID=3`, no hip): **stable**, KV not
  doubled. Recommended for single-node dma-buf.
- **ionic loopback**: 1st request OK then the mooncake session dies (`remote
  mooncake session not alive`) — no ODP. Avoid.

TP4 note: DSv4 weights are ~210 GiB/GPU at TP4 → `--mem-fraction-static ≥ 0.76`
(vs TP8's 159 GiB/GPU).

## Verify KV is NOT doubled

The whole point of A / B-on-ODP is no duplication. Confirm at decode steady state:
`VRAM ≈ weights + 1×KV`, not `weights + 2×KV`. On spur: 2-node TP8 decode 237/288,
single-node TP4 mlx5 243/288. If you see ~2×KV, dma-buf pinned on a non-ODP NIC —
wrong NIC filter or wrong mode.
