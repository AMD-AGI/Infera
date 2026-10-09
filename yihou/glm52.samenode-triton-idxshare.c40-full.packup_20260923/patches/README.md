# Patches — what differs from the tracked repo, and why

`scripts/bench-harness/` is a **vendored copy** of `bench/glm5p2_pd/`.
`diff -rq` against the tracked tree shows **exactly three** files differ. The
tracked repo was never modified. Every change is a **no-op for a cross-node
deployment**, so the same harness still serves the existing cross-node work.

| diff | file | what it fixes |
|---|---|---|
| `tools_topology.py.diff` | `tools/topology.py` | `load()` raised on `node in nodes or data_ip in ips`, which one host cannot satisfy. Relaxed to reject only an exactly-duplicated `(role, node, ip)` row; role whitelist, IPv4 check and the index/instance numbering are preserved — every port derives from the row index. |
| `launch.sh.diff` | `launch.sh` | Two same-node fixes. (a) `ENGINE_PORT_STRIDE` on the engine port only: SGLang derives an internal block from `--port` (`server_args.py:836,846-861`) and reserves `port_base+0..NUM_DERIVED_PORTS-1`; at stride 1 the legs got 29001/29002 and overlapped 9 of 10 ports, killing prefill with `zmq.error.ZMQError: Address already in use`. Default 1. (b) the same-node start gate, which waits for an already-started leg's `/health` before starting the next — **necessary but, as this run showed for the second time, not sufficient**; see `../notes.md`. |
| `tools_agentx_env.py.diff` | `tools/agentx_env.py` | Mirrors `ENGINE_PORT_STRIDE` in `load_topology`, which independently re-derives each worker URL from `ENGINE_PORT_BASE + index`. Without it the benchmark client addresses the decode leg at the wrong port. Easy to miss — the stride must be applied in both places. |

`SHA256SUMS.txt` covers every file in the kit.

## Not here

The **image** carries two baked-in patches, not applied at run time: the GLM
NextN shared-experts fusion fix and sglang PR #37152 (HiCache ROCm copy-round
widening — open upstream, never merged, its AMD ROCm CI red). Both were verified
present by import in the 20260922 packup. **#37152 is exercised in this run**,
unlike the HiCache-off runs, because prefill HiCache is on.
